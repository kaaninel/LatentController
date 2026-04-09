"""
ANT Engine — Per-token READ→PROCESS→WRITE cycle.

This is the heart of ANT. It orchestrates:
  1. AddrNets → 3 hierarchical trie addresses
  2. Trie READ → up to 25 ancestor vectors
  3. Transformer forward → hidden states with memory cross-attention
  4. V_proj → value vector
  5. Trie WRITE → accumulate knowledge
  6. Halt head → continue/halt decision
  7. LM head → next byte logits

Two modes:
  - encode(): Training — parallel read-all, forward, write-all per sequence
  - generate(): Inference — true per-token cycle with live trie updates
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig, MemoryConfig
from model import ANT, StaticKVCache

try:
    import ant_memory
except ImportError:
    raise ImportError(
        "ant_memory Rust extension not found. "
        "Build with: cd ant_memory && maturin develop --release")


class ANTEngine:
    """Wraps ANT model + Rust trie memory for per-token operation."""

    def __init__(self, model: ANT, mem_cfg: MemoryConfig, device: str = "cpu"):
        self.model = model
        self.cfg = model.cfg
        self.mem_cfg = mem_cfg
        self.device = device

        self.memory = ant_memory.MemorySystem(
            mem_cfg.data_path,
            d_model=mem_cfg.d_model,
            depth_cap=mem_cfg.depth_cap,
            alpha_base=mem_cfg.ema_alpha_base,
            alpha_min=mem_cfg.ema_alpha_min,
            flush_interval=mem_cfg.flush_interval,
        )

        # Persistent tag register (per batch slot — reset between sequences)
        self._tag_register = None

    def reset_state(self, batch_size: int = 1):
        """Reset tag register for new sequences."""
        self._tag_register = torch.zeros(
            batch_size, self.cfg.d_model, device=self.device)

    def _addrs_to_numpy(self, addr_tensors: list[torch.Tensor]) -> list[list[np.ndarray]]:
        """Convert AddrNet outputs to numpy arrays for Rust trie.

        addr_tensors: list of N tensors, each (B, depth) int64
        Returns: list of B lists, each containing N numpy uint8 arrays
        """
        B = addr_tensors[0].shape[0]
        N = len(addr_tensors)
        batch_addrs = []
        for b in range(B):
            addrs = []
            for n in range(N):
                addr = addr_tensors[n][b].cpu().numpy().astype(np.uint8)
                addrs.append(addr)
            batch_addrs.append(addrs)
        return batch_addrs

    def _read_memory(self, hidden: torch.Tensor,
                     temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read from trie using AddrNet-generated addresses.

        Args:
            hidden: (B, d) hidden states to generate addresses from
            temperature: Gumbel-softmax temperature for address generation

        Returns:
            mem_keys:   (B, S, d) retrieved vectors as cross-attention keys
            mem_values: (B, S, d) retrieved vectors as cross-attention values
            mem_mask:   (B, S) bool mask for valid slots
        """
        B = hidden.shape[0]
        S = self.cfg.n_mem_slots

        # Generate addresses
        addr_tensors = self.model.compute_addresses(hidden, temperature)
        batch_addrs = self._addrs_to_numpy(addr_tensors)

        # Batch read from trie
        vecs_np, mask_np = self.memory.read_batch(batch_addrs, max_vectors=S)

        # Convert to tensors on the correct device
        mem_vecs = torch.from_numpy(vecs_np).to(self.device)   # (B, S, d)
        mem_mask = torch.from_numpy(mask_np).to(self.device)    # (B, S)

        return mem_vecs, mem_vecs, mem_mask

    def _write_memory(self, hidden: torch.Tensor,
                      temperature: float = 1.0):
        """Write to trie using V_proj values at AddrNet addresses.

        Args:
            hidden: (B, d) hidden states
            temperature: Gumbel-softmax temperature for address generation
        """
        # Generate value vectors
        values = self.model.compute_value(hidden)  # (B, d)
        values_np = values.detach().cpu().numpy().astype(np.float32)

        # Generate addresses
        addr_tensors = self.model.compute_addresses(hidden.detach(), temperature)
        batch_addrs = self._addrs_to_numpy(addr_tensors)

        # Batch write to trie
        self.memory.write_batch(batch_addrs, values_np)

    def encode(self, token_ids: torch.Tensor,
               temperature: float = 1.0,
               write_to_trie: bool = True) -> dict:
        """Training-mode encoding: parallel read → forward → parallel write.

        This is a parallel approximation of the per-token cycle for efficiency.
        Within a sequence, self-attention provides token-to-token flow.
        Between training steps, the trie accumulates knowledge.

        Args:
            token_ids:     (B, T) byte token IDs
            temperature:   Gumbel-softmax temperature
            write_to_trie: whether to write hidden states to trie

        Returns dict with:
            logits:      (B, T, V) next-byte prediction logits
            halt_logits: (B, T, 2) continue/halt logits
            hidden:      (B, T, d) hidden states
            mem_mask:    (B, S) memory mask (for loss masking if needed)
        """
        B, T = token_ids.shape

        if self._tag_register is None or self._tag_register.shape[0] != B:
            self.reset_state(B)

        # Step 1: Read from trie using initial hidden state (embedding)
        # Use the mean embedding as the address query
        with torch.no_grad():
            init_embed = self.model.embed(token_ids).mean(dim=1)  # (B, d)

        mem_keys, mem_values, mem_mask = self._read_memory(
            init_embed, temperature)

        # Step 2: Forward pass with memory cross-attention
        logits, halt_logits, hidden = self.model(
            token_ids,
            mem_keys=mem_keys,
            mem_values=mem_values,
            mem_mask=mem_mask,
            tag_register=self._tag_register,
            return_hidden=True,
        )

        # Step 3: Update tag register (use last token's hidden state)
        self._tag_register = hidden[:, -1, :].detach()

        # Step 4: Write to trie (using per-position hidden states)
        if write_to_trie:
            # Write every position's hidden state to trie
            for t in range(T):
                self._write_memory(hidden[:, t, :], temperature)

        return {
            "logits": logits,
            "halt_logits": halt_logits,
            "hidden": hidden,
            "mem_mask": mem_mask,
        }

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_tokens: int = 256,
                 temperature: float = 0.8, top_k: int = 40,
                 top_p: float = 0.9) -> list[int]:
        """Generate tokens autoregressively with per-token trie interaction.

        True per-token cycle: read → process → write → output → repeat.

        Args:
            prompt_ids: list of byte token IDs (prompt)
            max_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = disabled)
            top_p: nucleus sampling threshold

        Returns: list of generated token IDs
        """
        self.model.eval()
        cfg = self.cfg
        device = self.device

        # Encode prompt into trie
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        B = 1

        self.reset_state(B)

        # Prefill: process prompt through model with trie
        kv_cache = self.model.make_cache(B, max_seq=len(prompt_ids) + max_tokens,
                                         device=device)

        # Read memory for prompt
        init_embed = self.model.embed(prompt).mean(dim=1)
        mem_keys, mem_values, mem_mask = self._read_memory(init_embed)

        # Prefill forward
        logits, halt_logits, hidden, kv_cache = self.model(
            prompt,
            mem_keys=mem_keys, mem_values=mem_values, mem_mask=mem_mask,
            tag_register=self._tag_register,
            return_hidden=True,
            kv_cache=kv_cache, cache_position=0,
        )

        # Write prompt tokens to trie
        for t in range(prompt.shape[1]):
            self._write_memory(hidden[:, t, :])

        self._tag_register = hidden[:, -1, :]
        cache_pos = prompt.shape[1]

        # Sample first token
        next_logits = logits[:, -1, :]
        next_token = self._sample(next_logits, temperature, top_k, top_p)
        generated = [next_token]

        # Autoregressive generation
        for _ in range(max_tokens - 1):
            if next_token == cfg.eos_id:
                break

            tok = torch.tensor([[next_token]], dtype=torch.long, device=device)

            # Per-token: read from trie using previous hidden state
            prev_hidden = hidden[:, -1, :]
            mem_keys, mem_values, mem_mask = self._read_memory(prev_hidden)

            # Forward one token
            logits, halt_logits, hidden, kv_cache = self.model(
                tok,
                mem_keys=mem_keys, mem_values=mem_values, mem_mask=mem_mask,
                tag_register=self._tag_register,
                return_hidden=True,
                kv_cache=kv_cache, cache_position=cache_pos,
            )
            cache_pos += 1

            # Write to trie
            self._write_memory(hidden[:, 0, :])

            # Update tag register — always (B, d)
            self._tag_register = hidden[:, -1, :]

            # Handle halt head (multi-cycle fetch)
            halt_probs = F.softmax(halt_logits[:, -1, :], dim=-1)
            if halt_probs[0, 1] > 0.5:  # Halt probability > 50%
                # Model wants more memory cycles — re-read with updated state
                mem_keys, mem_values, mem_mask = self._read_memory(hidden[:, -1, :])
                logits, halt_logits, hidden, kv_cache = self.model(
                    tok,
                    mem_keys=mem_keys, mem_values=mem_values, mem_mask=mem_mask,
                    tag_register=self._tag_register,
                    return_hidden=True,
                    kv_cache=kv_cache, cache_position=cache_pos - 1,
                )

            # Sample next token
            next_logits = logits[:, -1, :]
            next_token = self._sample(next_logits, temperature, top_k, top_p)

            if next_token == cfg.noop_id:
                continue  # NOOP = nothing to output yet

            generated.append(next_token)

        return generated

    def _sample(self, logits: torch.Tensor, temperature: float = 0.8,
                top_k: int = 40, top_p: float = 0.9) -> int:
        """Sample a token from logits with temperature, top-k, and top-p."""
        if temperature <= 0:
            return logits.argmax(dim=-1).item()

        logits = logits / temperature

        if top_k > 0:
            v, _ = logits.topk(min(top_k, logits.size(-1)))
            logits[logits < v[..., -1:]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def flush(self):
        """Flush trie to disk."""
        self.memory.flush()

    def reset_memory(self):
        """Reset all trie memory."""
        self.memory.reset()

    def memory_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "total_nodes": self.memory.total_nodes(),
            "total_entries": self.memory.total_entries(),
        }
