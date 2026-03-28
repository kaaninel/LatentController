"""
Looped Latent Controller - Model
Decoder-only transformer with RoPE, RMSNorm, halt head, address heads.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config import ModelConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_cache(max_seq_len: int, head_dim: int, theta: float = 10000.0):
    """Precompute sin/cos tables for RoPE."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(0, max_seq_len, dtype=torch.float32)
    angles = torch.outer(pos, freqs)  # (max_seq_len, half)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos_cache, sin_cache, offset=0):
    """
    Apply rotary positional embeddings to x.
    x: (batch, seq_len, n_heads, head_dim)
    """
    seq_len = x.size(1)
    half = x.size(-1) // 2
    cos = cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
    sin = sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with RoPE and mask support."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, cos_cache, sin_cache, mask, rope_offset=0):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K only
        q = apply_rope(q, cos_cache, sin_cache, rope_offset)
        k = apply_rope(k, cos_cache, sin_cache, rope_offset)

        # (B, n_heads, S, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """FFN with SiLU activation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos_cache, sin_cache, mask, rope_offset=0):
        x = x + self.dropout(self.attn(self.norm1(x), cos_cache, sin_cache, mask, rope_offset))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class LoopedLatentController(nn.Module):
    """
    Main model: decoder-only transformer with memory, halt head, address heads.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (shared with LM head via tied weights)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model)

        # Halt head for ACT (Phase 4)
        # Outputs 2 logits: [HALT, CONTINUE]
        self.halt_head = nn.Linear(config.d_model, 2, bias=True)
        # CRITICAL: Initialize bias to favor CONTINUE
        with torch.no_grad():
            self.halt_head.bias.copy_(torch.tensor([-1.0, 1.0]))

        # Address heads for memory (Phase 2, frozen after training)
        self.addr_heads = nn.ModuleList([
            nn.Linear(config.d_model, config.addr_dim, bias=False)
            for _ in range(config.n_addr_heads)
        ])

        # Precompute RoPE cache
        cos_cache, sin_cache = precompute_rope_cache(
            config.max_seq_len, config.head_dim, config.rope_theta
        )
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)

        self._use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._use_gradient_checkpointing = False

    def build_attention_mask(self, seq_len: int, n_mem: int, device):
        """
        Build the asymmetric attention mask.
        Memory positions (0..n_mem-1): attend to memory only
        Text positions (n_mem..seq_len-1): attend to all memory + causal text
        Memory CANNOT attend to text.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Memory attends to memory
        if n_mem > 0:
            mask[:n_mem, :n_mem] = True

        # Text attends to all memory
        if n_mem > 0:
            mask[n_mem:, :n_mem] = True

        # Text attends causally to text
        text_len = seq_len - n_mem
        if text_len > 0:
            causal = torch.tril(
                torch.ones(text_len, text_len, dtype=torch.bool, device=device)
            )
            mask[n_mem:, n_mem:] = causal

        # (1, 1, seq_len, seq_len) for broadcasting over batch and heads
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, token_ids, memory_vectors=None, return_hidden=False):
        """
        Full forward pass.

        Args:
            token_ids: (batch, text_seq_len) token IDs
            memory_vectors: (batch, n_mem_slots, d_model) float or None
            return_hidden: if True, also return last-position hidden state

        Returns:
            logits: (batch, text_seq_len, vocab_size) - text positions only
            halt_logits: (batch, 2) - from last position
            hidden: (batch, d_model) - only if return_hidden
        """
        B, T = token_ids.shape
        device = token_ids.device

        # Embed text tokens
        text_embeds = self.embed(token_ids)  # (B, T, d_model)

        # Build input: optionally prepend memory
        if memory_vectors is not None:
            mem_start = self.embed(
                torch.full((B, 1), self.config.mem_start_id,
                           device=device, dtype=torch.long)
            )
            mem_end = self.embed(
                torch.full((B, 1), self.config.mem_end_id,
                           device=device, dtype=torch.long)
            )
            # Pad memory vectors to n_mem_slots if needed
            n_provided = memory_vectors.size(1)
            if n_provided < self.config.n_mem_slots:
                pad = torch.zeros(
                    B, self.config.n_mem_slots - n_provided,
                    self.config.d_model, device=device, dtype=memory_vectors.dtype
                )
                memory_vectors = torch.cat([memory_vectors, pad], dim=1)
            elif n_provided > self.config.n_mem_slots:
                memory_vectors = memory_vectors[:, :self.config.n_mem_slots, :]

            # Memory vectors injected as raw embeddings (NOT token IDs)
            x = torch.cat([mem_start, memory_vectors, mem_end, text_embeds], dim=1)
            n_mem = self.config.n_mem_positions  # 11
        else:
            x = text_embeds
            n_mem = 0

        S = x.size(1)

        # Attention mask
        mask = self.build_attention_mask(S, n_mem, device)

        # Run transformer layers
        for layer in self.layers:
            if self._use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, self.cos_cache, self.sin_cache, mask, 0,
                    use_reentrant=False
                )
            else:
                x = layer(x, self.cos_cache, self.sin_cache, mask)

        x = self.final_norm(x)

        # LM logits from text positions only
        if n_mem > 0:
            text_hidden = x[:, n_mem:, :]
        else:
            text_hidden = x

        # Tied LM head
        logits = F.linear(text_hidden, self.embed.weight)  # (B, T, vocab_size)

        # Halt logits from last position
        last_hidden = x[:, -1, :]  # (B, d_model)
        halt_logits = self.halt_head(last_hidden)  # (B, 2)

        if return_hidden:
            return logits, halt_logits, last_hidden
        return logits, halt_logits

    def compute_addresses(self, hidden_state):
        """
        Compute 3 addresses from a hidden state.
        hidden_state: (batch, d_model) or (d_model,)
        Returns: list of 3 tensors, each int8
        """
        addresses = []
        for head in self.addr_heads:
            raw = F.linear(hidden_state.float(), head.weight.float())
            addr = raw.clamp(-128, 127).round().to(torch.int8)
            addresses.append(addr)
        return addresses

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
