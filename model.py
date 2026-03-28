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
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q(x).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)

        # RoPE on Q and K
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Use PyTorch's fused SDPA (dispatches to Flash Attention when available).
        # Pass the additive float mask (0 = attend, -1e9 = block) directly.
        attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,  # we provide our own asymmetric mask
        )

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
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SiLUFFN(cfg.d_model, cfg.ffn_dim)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask, cos, sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


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

        # Halt head: bias initialized to [-1, +1] (favor CONTINUE=0)
        self.halt_head = nn.Linear(cfg.d_model, 2, bias=True)
        nn.init.constant_(self.halt_head.bias, 0.0)
        self.halt_head.bias.data[0] = -1.0
        self.halt_head.bias.data[1] = 1.0

        # Address heads
        self.addr_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.addr_dim, bias=False)
            for _ in range(cfg.n_addr_heads)
        ])

        # RoPE cache (full context: mem + text)
        total_pos = cfg.n_mem_positions + cfg.n_text_positions
        cos, sin = precompute_rope(cfg.head_dim, total_pos, cfg.rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # Attention mask cache: keyed by (n_mem, n_text)
        self._mask_cache = {}

    # ------------------------------------------------------------------
    # Asymmetric attention mask
    # ------------------------------------------------------------------

    def _build_mask(self, n_mem: int, n_text: int, device) -> torch.Tensor:
        """
        Memory positions : attend only to other memory positions (block → text).
        Text positions   : attend to ALL memory + causal over text.
        Returns (T, T) additive mask (0 = keep, -inf = mask).
        """
        T = n_mem + n_text
        # Start with all-attend
        mask = torch.zeros(T, T, device=device)
        neg_inf = -1e9

        # Memory rows: block text columns
        if n_mem > 0 and n_text > 0:
            mask[:n_mem, n_mem:] = neg_inf

        # Text rows: causal over text (vectorized, no Python loop)
        if n_text > 0:
            text_causal = torch.ones(n_text, n_text, device=device).triu(diagonal=1) * neg_inf
            mask[n_mem:, n_mem:] = text_causal

        return mask

    def _get_mask(self, n_mem: int, n_text: int, device) -> torch.Tensor:
        """Return cached mask for the given (n_mem, n_text) dimensions."""
        key = (n_mem, n_text)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._build_mask(n_mem, n_text, device)
        return self._mask_cache[key]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        memory_vectors: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """
        token_ids      : (B, T_text)
        memory_vectors : (B, n_mem, d_model) float — raw 512-dim embeddings
        Returns (logits, halt_logits) or (logits, halt_logits, hidden).
        logits cover text positions only.
        """
        B, T_text = token_ids.shape
        device = token_ids.device

        x = self.embed(token_ids)   # (B, T_text, d_model)

        n_mem = 0
        if memory_vectors is not None:
            n_mem = memory_vectors.shape[1] + 2  # +2 for <MEM> and </MEM> tokens
            mem_start = self.embed(
                torch.full((B, 1), self.cfg.mem_start_id, dtype=torch.long, device=device)
            )
            mem_end = self.embed(
                torch.full((B, 1), self.cfg.mem_end_id, dtype=torch.long, device=device)
            )
            x = torch.cat([mem_start, memory_vectors, mem_end, x], dim=1)

        T = x.shape[1]
        mask = self._get_mask(n_mem, T_text, device)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, mask, cos, sin, use_reentrant=False)
            else:
                x = layer(x, mask, cos, sin)

        hidden = self.norm(x)

        # Text positions only
        text_hidden = hidden[:, n_mem:, :]              # (B, T_text, d_model)

        logits = F.linear(text_hidden, self.embed.weight)
        halt_logits = self.halt_head(text_hidden)       # (B, T_text, 2)

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
