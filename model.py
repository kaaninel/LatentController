import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


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
    """Returns (cos, sin) each of shape (max_seq_len, head_dim//2)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)           # (max_seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x   : (B, n_heads, T, head_dim)
    cos : (T, head_dim//2)
    sin : (T, head_dim//2)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)     # (1, 1, T, half)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


# ---------------------------------------------------------------------------
# SiLU FFN
# ---------------------------------------------------------------------------

class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        d = cfg.d_model
        self.q = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(d, cfg.n_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple | None = None,
    ) -> tuple:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        # RoPE on Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV-cache: append new K,V to cached K,V
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_cache = (k, v)

        # Use PyTorch's fused SDPA (dispatches to Flash Attention when available).
        # Pass the additive float mask (0 = attend, -1e9 = block) directly.
        attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_kv)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,  # we provide our own asymmetric mask
        )

        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out), new_cache


# ---------------------------------------------------------------------------
# Memory Cross-Attention
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    """Cross-attention to external key-value memory.

    Full Q/K/V/O projections so each head can specialize in reading
    different aspects of memory (entity names, locations, relations, etc.).
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim

        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, mem_keys: torch.Tensor,
                mem_values: torch.Tensor,
                mem_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        S = mem_keys.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)          # (B, H, T, D)
        k = self.k(mem_keys).view(B, S, H, D).transpose(1, 2)   # (B, H, S, D)
        v = self.v(mem_values).view(B, S, H, D).transpose(1, 2)  # (B, H, S, D)

        attn_mask = None
        if mem_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~mem_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, H * D)
        return self.o(out)





# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = Attention(cfg)

        self.use_mem_attn = getattr(cfg, 'use_memory_cross_attention', False)
        if self.use_mem_attn:
            self.norm_mem = RMSNorm(cfg.d_model)
            self.mem_attn = MemoryAttention(
                cfg.d_model, cfg.n_heads, cfg.head_dim
            )

        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SiLUFFN(cfg.d_model, cfg.ffn_dim)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple | None = None,
        mem_keys: torch.Tensor | None = None,
        mem_values: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> tuple:
        attn_out, new_cache = self.attn(self.norm1(x), mask, cos, sin, kv_cache)
        x = x + self.drop(attn_out)

        if self.use_mem_attn and mem_keys is not None:
            x = x + self.drop(self.mem_attn(
                self.norm_mem(x), mem_keys, mem_values, mem_mask))

        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_cache


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LoopedLatentController(nn.Module):
    def __init__(self, cfg: ModelConfig, use_checkpoint: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_checkpoint = use_checkpoint

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

        # Halt head: bias initialized to [0, 0] (50/50 CONTINUE/HALT)
        self.halt_head = nn.Linear(cfg.d_model, 2, bias=True)
        nn.init.constant_(self.halt_head.bias, 0.0)

        # Address heads
        self.addr_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.addr_dim, bias=False)
            for _ in range(cfg.n_addr_heads)
        ])

        # Temporal embedding for memory slots (chunk ordering)
        if getattr(cfg, 'use_memory_cross_attention', False):
            max_temporal = getattr(cfg, 'max_temporal_chunks', 32)
            self.temporal_emb = nn.Embedding(max_temporal, cfg.d_model)

        # RoPE cache (full context: mem + text — must cover max possible sequence)
        total_pos = cfg.n_mem_positions + cfg.max_seq_len
        cos, sin = precompute_rope(cfg.head_dim, total_pos, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Pre-compute attention masks as registered buffers so that
        # torch.compile / CUDAGraphs can reference them as stable tensors.
        neg_inf = -1e9

        # text-only causal mask — used in Phase 1 (no memory)
        n_text_max = cfg.max_seq_len
        text_only = torch.triu(
            torch.full((n_text_max, n_text_max), neg_inf), diagonal=1
        )
        self.register_buffer("text_only_mask", text_only)

        # mem+text asymmetric mask — used in Phases 3/4
        n_mem_max  = cfg.n_mem_positions    # 11  (<MEM> + slots + </MEM>)
        n_text_max2 = cfg.max_seq_len       # 512 (absolute max text length)
        T_max = n_mem_max + n_text_max2     # 523
        mem_text = torch.zeros(T_max, T_max)
        # Memory rows: block all text columns
        mem_text[:n_mem_max, n_mem_max:] = neg_inf
        # Text rows: causal over text
        mem_text[n_mem_max:, n_mem_max:] = torch.triu(
            torch.full((n_text_max2, n_text_max2), neg_inf), diagonal=1
        )
        self.register_buffer("mem_text_mask", mem_text)

    # ------------------------------------------------------------------
    # Memory-only forward (for split KV-cache)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_memory(self, memory_vectors: torch.Tensor) -> list:
        """
        Process memory positions through all layers independently.

        Since memory rows cannot attend to text (masked), the K,V computed
        here are identical to those in a joint [mem+text] forward pass.

        memory_vectors : (B, n_mem_slots, d_model) — float tensor
        Returns list of (k, v) per layer, each k,v shape (B, H, n_mem_pos, D).
        """
        B = memory_vectors.shape[0]
        device = memory_vectors.device

        mem_start = self.embed(
            torch.full((B, 1), self.cfg.mem_start_id, dtype=torch.long, device=device)
        )
        mem_end = self.embed(
            torch.full((B, 1), self.cfg.mem_end_id, dtype=torch.long, device=device)
        )
        x = torch.cat([mem_start, memory_vectors, mem_end], dim=1)  # (B, n_mem, d)

        n_mem = x.shape[1]
        # Memory-only block of the asymmetric mask (all-to-all, no text)
        mask = self.mem_text_mask[:n_mem, :n_mem]
        cos = self.rope_cos[:n_mem]
        sin = self.rope_sin[:n_mem]

        mem_kvs = []
        for layer in self.layers:
            x, layer_kv = layer(x, mask, cos, sin)
            mem_kvs.append(layer_kv)

        return mem_kvs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        memory_vectors: torch.Tensor | None = None,
        memory_keys: torch.Tensor | None = None,
        memory_values: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        kv_cache: list | None = None,
        cache_position: int = 0,
        bidirectional: bool = False,
    ):
        """
        token_ids      : (B, T_text)
        memory_vectors : (B, n_mem, d_model) float — raw embeddings (legacy)
        memory_keys    : (B, n_mem, inner) — pre-computed keys from write heads
        memory_values  : (B, n_mem, inner) — pre-computed values from write heads
        memory_mask    : (B, n_mem) bool — True for valid memory slots
        kv_cache       : list of (k, v) tuples per layer, or None
        cache_position : position offset for RoPE when using KV-cache
        bidirectional  : if True, use full (non-causal) self-attention mask.
                         Use during memory encoding so all tokens see each other.
        Returns (logits, halt_logits[, hidden][, new_kv_cache]).
        When kv_cache is not None, returns new_kv_cache as last element.
        logits cover text positions only.
        """
        B, T_text = token_ids.shape
        device = token_ids.device
        use_cross_attn = getattr(self.cfg, 'use_memory_cross_attention', False)

        x = self.embed(token_ids)   # (B, T_text, d_model)

        # Memory handling: cross-attention mode vs prepend mode
        n_mem = 0
        cross_keys = None
        cross_vals = None
        cross_mask = memory_mask  # (B, S) bool or None
        if use_cross_attn:
            if memory_keys is not None and memory_values is not None:
                cross_keys, cross_vals = memory_keys, memory_values
        elif memory_vectors is not None:
                # Legacy prepend: [<MEM> mem1..N </MEM> text]
                n_mem = memory_vectors.shape[1] + 2
                mem_start = self.embed(
                    torch.full((B, 1), self.cfg.mem_start_id, dtype=torch.long, device=device)
                )
                mem_end = self.embed(
                    torch.full((B, 1), self.cfg.mem_end_id, dtype=torch.long, device=device)
                )
                x = torch.cat([mem_start, memory_vectors, mem_end, x], dim=1)

        T_new = x.shape[1]  # tokens being processed this call

        # Determine if this is a prefill (first call, populating cache) or incremental
        is_prefill = kv_cache is not None and cache_position == 0
        is_incremental = kv_cache is not None and cache_position > 0

        if is_incremental:
            cos = self.rope_cos[cache_position:cache_position + T_new]
            sin = self.rope_sin[cache_position:cache_position + T_new]
            T_total = cache_position + T_new
            mask = torch.zeros(T_new, T_total, device=device)
            for i in range(T_new):
                pos = cache_position + i
                mask[i, pos + 1:] = -1e9
        else:
            cos = self.rope_cos[:T_new]
            sin = self.rope_sin[:T_new]
            if bidirectional:
                # Full attention: all tokens see all others (for encoding)
                mask = torch.zeros(T_new, T_new, device=device)
            elif n_mem == 0:
                mask = self.text_only_mask[:T_text, :T_text]
            else:
                mask = self.mem_text_mask[:T_new, :T_new]

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None and i < len(kv_cache) else None
            if self.use_checkpoint and self.training:
                x, _ = checkpoint(layer, x, mask, cos, sin, None,
                                  cross_keys, cross_vals, cross_mask,
                                  use_reentrant=False)
                new_kv_cache.append(None)
            else:
                x, layer_kv = layer(x, mask, cos, sin, kv_cache=layer_cache,
                                    mem_keys=cross_keys, mem_values=cross_vals,
                                    mem_mask=cross_mask)
                new_kv_cache.append(layer_kv)

        hidden = self.norm(x)

        # Text positions only (n_mem=0 in cross-attention mode)
        text_hidden = hidden[:, n_mem:, :]              # (B, T_text, d_model)

        logits = F.linear(text_hidden, self.embed.weight)
        halt_logits = self.halt_head(text_hidden)       # (B, T_text, 2)

        if kv_cache is not None:
            if return_hidden:
                return logits, halt_logits, hidden, new_kv_cache
            return logits, halt_logits, new_kv_cache
        if return_hidden:
            return logits, halt_logits, hidden
        return logits, halt_logits

    # ------------------------------------------------------------------
    # Address computation
    # ------------------------------------------------------------------

    def compute_addresses(self, hidden_state: torch.Tensor):
        """
        hidden_state : (d_model,) or (1, d_model)
        Returns list of 3 int8 tensors each of shape (addr_dim,).
        """
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.squeeze(0)
        addresses = []
        for head in self.addr_heads:
            raw = head(hidden_state)                    # (addr_dim,)
            # Scale to int8 range
            scale = raw.abs().max().clamp(min=1e-6)
            addr = (raw / scale * 127.0).round().clamp(-128, 127).to(torch.int8)
            addresses.append(addr)
        return addresses

def compute_addresses_batch(self, hidden_states: torch.Tensor):
        """
        Vectorized address computation for a batch.
        hidden_states : (B, d_model)
        Returns list of 3 int8 tensors each of shape (B, addr_dim).
        """
        addresses = []
        for head in self.addr_heads:
            raw = head(hidden_states)                   # (B, addr_dim)
            scale = raw.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
            addr = (raw / scale * 127.0).round().clamp(-128, 127).to(torch.int8)
            addresses.append(addr)
        return addresses
