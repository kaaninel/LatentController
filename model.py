"""
ANT — 937K parameter byte-level transformer with persistent hierarchical memory.

Components: RMSNorm, RoPE, SiLUFFN, AddrNet, Attention, MemoryAttention,
TransformerBlock, StaticKVCache, ANT.

The model itself stores NO knowledge — all knowledge lives in the external trie.
The engine (engine.py) orchestrates per-token READ→PROCESS→WRITE cycles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ---------------------------------------------------------------------------
# Static KV Cache
# ---------------------------------------------------------------------------

class StaticKVCache:
    """Pre-allocated KV cache for generation (no torch.cat per step)."""
    __slots__ = ('k', 'v', 'pos')

    def __init__(self, n_layers: int, batch: int, n_heads: int,
                 max_seq: int, head_dim: int, device, dtype=None):
        dtype = dtype or torch.float32
        shape = (n_layers, batch, n_heads, max_seq, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
              pos: int, n: int):
        self.k[layer_idx, :, :, pos:pos + n, :] = k
        self.v[layer_idx, :, :, pos:pos + n, :] = v

    def read(self, layer_idx: int, end_pos: int):
        return (self.k[layer_idx, :, :, :end_pos, :],
                self.v[layer_idx, :, :, :end_pos, :])


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (self.weight * x_float * rms).to(dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# SiLU FFN
# ---------------------------------------------------------------------------

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


# ---------------------------------------------------------------------------
# AddrNet — Hierarchical address generation co-processor
# ---------------------------------------------------------------------------

class AddrNet(nn.Module):
    """Generates an 8-level hierarchical address from a hidden state.

    Each cycle: project → logits → pick bin → condition on choice.
    Training uses Gumbel-softmax for differentiable address selection.
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 16,
                 n_bins: int = 256, depth: int = 8):
        super().__init__()
        self.depth = depth
        self.n_bins = n_bins
        self.proj_in = nn.Linear(d_model, hidden_dim)
        self.bin_embed = nn.Embedding(n_bins, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_bins)

    def forward(self, hidden_state: torch.Tensor, temperature: float = 1.0):
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        h = self.proj_in(hidden_state)
        bins = []
        for _ in range(self.depth):
            logits = self.out(h)
            if self.training and temperature > 0:
                soft = F.gumbel_softmax(logits, tau=temperature, hard=True)
                bin_idx = soft.argmax(dim=-1)
                h = h + (soft @ self.bin_embed.weight)
            else:
                bin_idx = logits.argmax(dim=-1)
                h = h + self.bin_embed(bin_idx)
            h = F.silu(self.mlp(h))
            bins.append(bin_idx)
        return torch.stack(bins, dim=1)  # (B, depth)


# ---------------------------------------------------------------------------
# Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.q = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)

    def forward(self, x, mask, cos, sin, kv_cache=None,
                _layer_idx=-1, _cache_position=-1):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if isinstance(kv_cache, StaticKVCache):
            pos = _cache_position
            kv_cache.write(_layer_idx, k, v, pos, T)
            k, v = kv_cache.read(_layer_idx, pos + T)
            new_cache = kv_cache
        elif kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = (k, v)

        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        elif isinstance(mask, str) and mask == "causal":
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True)
        else:
            attn_mask = mask.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out), new_cache


# ---------------------------------------------------------------------------
# Memory Cross-Attention
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    """Cross-attention to trie-retrieved key-value memory.

    Per-head learnable inverse temperature for sharper entity discrimination.
    Optional top-k sparse attention with straight-through estimator.
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int, topk: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        inner = n_heads * head_dim

        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)

        self.inv_temp = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor, mem_keys: torch.Tensor,
                mem_values: torch.Tensor,
                mem_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        S = mem_keys.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(mem_keys).view(B, S, H, D).transpose(1, 2)
        v = self.v(mem_values).view(B, S, H, D).transpose(1, 2)

        temp = self.inv_temp.view(1, H, 1, 1)
        q = q * temp

        effective_topk = self.topk
        if effective_topk <= 0 or effective_topk >= S:
            attn_mask = None
            if mem_mask is not None:
                attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
                attn_mask.masked_fill_(~mem_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            if mem_mask is not None:
                pad_mask = ~mem_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float('-inf'))
            full_attn = F.softmax(scores, dim=-1)
            k_clamped = min(effective_topk, S)
            topk_vals, _ = scores.topk(k_clamped, dim=-1)
            threshold = topk_vals[..., -1:]
            topk_mask = (scores >= threshold).float()
            sparse_attn = full_attn * topk_mask
            sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-8)
            attn = full_attn + (sparse_attn - full_attn).detach()
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out)


# ---------------------------------------------------------------------------
# Transformer Block: Self-Attn → Tag-Attn → Mem-Attn → FFN
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)

        self.use_tag = cfg.use_tag_system
        if self.use_tag:
            self.norm_tag = RMSNorm(cfg.d_model)
            self.tag_head = nn.Linear(cfg.d_model, cfg.d_model)
            self.tag_gate = nn.Linear(cfg.d_model, 1)

        self.norm_mem = RMSNorm(cfg.d_model)
        self.mem_attn = MemoryAttention(
            cfg.d_model, cfg.n_heads, cfg.head_dim,
            topk=cfg.memory_topk)

        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SiLUFFN(cfg.d_model, cfg.ffn_dim)

    def forward(self, x, mask, cos, sin, kv_cache=None,
                mem_keys=None, mem_values=None, mem_mask=None,
                tag_register=None,
                _layer_idx=-1, _cache_position=-1):
        # 1. Self-attention
        attn_out, new_cache = self.attn(
            self.norm1(x), mask, cos, sin, kv_cache,
            _layer_idx=_layer_idx, _cache_position=_cache_position)
        x = x + attn_out

        # 2. Tag cross-attention (GRU-style gated update)
        if self.use_tag and tag_register is not None:
            normed = self.norm_tag(x)
            new_tag = torch.tanh(self.tag_head(normed))
            gate = torch.sigmoid(self.tag_gate(normed))
            tag_context = gate * new_tag + (1 - gate) * tag_register.unsqueeze(1)
            x = x + tag_context

        # 3. Memory cross-attention (trie-retrieved vectors)
        if mem_keys is not None:
            x = x + self.mem_attn(self.norm_mem(x), mem_keys, mem_values, mem_mask)

        # 4. FFN
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# ANT Model
# ---------------------------------------------------------------------------

class ANT(nn.Module):
    """ANT — 937K param byte-level transformer.

    Weights store computation only. Knowledge lives in the external trie.
    The engine (engine.py) handles trie interaction — this module is pure NN.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model,
                                  padding_idx=cfg.pad_id)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

        # Halt head: continue(0) / halt(1) decision
        self.halt_head = nn.Linear(cfg.d_model, 2, bias=True)
        nn.init.constant_(self.halt_head.bias, 0.0)

        # 3 AddrNets for hierarchical address generation
        self.addr_nets = nn.ModuleList([
            AddrNet(cfg.d_model, cfg.addr_hidden_dim,
                    cfg.addr_n_bins, cfg.addr_depth)
            for _ in range(cfg.n_addr_nets)
        ])

        # V_proj: hidden → stored value (decouples representation from storage)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # RoPE
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def make_cache(self, batch_size: int, max_seq: int = 0,
                   device=None, dtype=None) -> StaticKVCache:
        if max_seq == 0:
            max_seq = self.cfg.max_seq_len
        device = device or self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        return StaticKVCache(
            self.cfg.n_layers, batch_size, self.cfg.n_heads,
            max_seq, self.cfg.head_dim, device, dtype)

    def compute_addresses(self, hidden: torch.Tensor, temperature: float = 1.0):
        """hidden: (B, d) or (d,) → list of N tensors, each (B, depth) int64."""
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        return [net(hidden, temperature) for net in self.addr_nets]

    def compute_value(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to memory value vector."""
        return self.v_proj(hidden)

    def forward(self, token_ids, mem_keys=None, mem_values=None,
                mem_mask=None, tag_register=None, return_hidden=False,
                kv_cache=None, cache_position=0):
        """
        Forward pass through the transformer.

        Args:
            token_ids:      (B, T) byte token IDs
            mem_keys:       (B, S, d) trie-retrieved vectors as keys
            mem_values:     (B, S, d) trie-retrieved vectors as values
            mem_mask:       (B, S) bool mask for valid memory slots
            tag_register:   (B, d) persistent context register
            return_hidden:  if True, also return hidden states
            kv_cache:       StaticKVCache for generation
            cache_position: RoPE position offset for incremental decode

        Returns:
            logits:      (B, T, V) next-byte prediction logits
            halt_logits: (B, T, 2) continue/halt logits
            [hidden]:    (B, T, d) if return_hidden=True
            [kv_cache]:  updated cache if kv_cache was provided
        """
        B, T = token_ids.shape
        x = self.embed(token_ids)

        use_static = isinstance(kv_cache, StaticKVCache)
        is_incremental = kv_cache is not None and cache_position > 0

        if is_incremental:
            cos = self.rope_cos[cache_position:cache_position + T]
            sin = self.rope_sin[cache_position:cache_position + T]
            mask = None
        else:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]
            mask = "causal"

        if use_static:
            for i, layer in enumerate(self.layers):
                x, _ = layer(x, mask, cos, sin, kv_cache=kv_cache,
                             mem_keys=mem_keys, mem_values=mem_values,
                             mem_mask=mem_mask, tag_register=tag_register,
                             _layer_idx=i, _cache_position=cache_position)
            kv_cache.pos = cache_position + T
            new_cache = kv_cache
        else:
            new_cache = []
            for i, layer in enumerate(self.layers):
                layer_cache = (kv_cache[i] if kv_cache is not None
                               and i < len(kv_cache) else None)
                x, layer_kv = layer(x, mask, cos, sin, kv_cache=layer_cache,
                                    mem_keys=mem_keys, mem_values=mem_values,
                                    mem_mask=mem_mask,
                                    tag_register=tag_register)
                new_cache.append(layer_kv)

        hidden = self.norm(x)
        logits = F.linear(hidden, self.embed.weight)
        halt_logits = self.halt_head(hidden)

        outputs = [logits, halt_logits]
        if return_hidden:
            outputs.append(hidden)
        if kv_cache is not None:
            outputs.append(new_cache)
        return tuple(outputs)
