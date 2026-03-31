"""
Agent — token stream processor.

One public method: process_token(token_id) → output_token_id | None
"""

from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from model import LoopedLatentController
from memory import MemorySystem


# ── Shared utilities (also used by train_phase3) ──

def addr_bytes(addr_tensor: torch.Tensor) -> bytes:
    """Convert int8 address tensor → bytes for TrieIndex."""
    return addr_tensor.cpu().numpy().tobytes()


def memory_vecs_to_tensor(
    vecs: list,
    d_model: int,
    device,
) -> torch.Tensor:
    """Convert list of int8 numpy arrays to float tensor (1, n, d_model)."""
    out = np.stack(vecs, axis=0).astype(np.float32) / 127.0
    return torch.from_numpy(out).unsqueeze(0).to(device)


class Agent:
    def __init__(
        self,
        model: LoopedLatentController,
        memory: MemorySystem,
        device,
        max_act_steps: int = 6,
        emit_threshold: float = 0.3,
    ):
        self.model = model
        self.memory = memory
        self.device = device
        self.max_act_steps = max_act_steps
        self.emit_threshold = emit_threshold
        self.cfg = model.cfg

        # Internal state
        self.h = torch.zeros(self.cfg.d_model, device=device)
        self.context_buffer: List[int] = []  # rolling, max n_text_positions tokens

        # KV-cache: stored as combined [mem|text] to avoid split/combine overhead.
        # Only split on ACT re-read (rare path).
        self.kv_cache: list | None = None     # per-layer (k, v) combined
        self.n_mem_positions: int = 0         # memory prefix length in cache
        self.text_cache_len: int = 0          # text positions in cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_memory(self):
        """Read up to 9 memory vectors based on current hidden state h."""
        addrs = self.model.compute_addresses(self.h)
        ab    = [addr_bytes(a) for a in addrs]
        return self.memory.read_memory(ab)

    def _write_memory(self):
        """Write current hidden state to memory."""
        vec_np = self.h.detach().float().cpu().numpy()
        addrs  = self.model.compute_addresses(self.h.detach())
        ab     = [addr_bytes(a) for a in addrs]
        self.memory.write_memory(ab, vec_np)

    def _build_input(self) -> torch.Tensor:
        """Build (1, T) token-id tensor from context buffer."""
        ids = self.context_buffer[:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _build_mem_tensor(self, mem_vecs) -> torch.Tensor:
        return memory_vecs_to_tensor(mem_vecs, self.cfg.d_model, self.device)

    def _invalidate_cache(self):
        self.kv_cache = None
        self.n_mem_positions = 0
        self.text_cache_len = 0

    # ------------------------------------------------------------------
    # feed_prefill — bulk context processing in one forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def feed_prefill(self, token_ids: list):
        """
        Process a full token sequence in one forward pass.
        Reads memory, builds KV cache for the whole sequence, writes memory once.
        ~100× faster than feeding token-by-token for context/passage.
        """
        if not token_ids:
            return

        # Read memory based on current hidden state
        mem_vecs = self._read_memory()

        # Update context buffer (may truncate to window size)
        self.context_buffer.extend(token_ids)
        if len(self.context_buffer) > self.cfg.n_text_positions:
            self.context_buffer = self.context_buffer[-self.cfg.n_text_positions:]

        # Build memory + full input
        mem_tensor = self._build_mem_tensor(mem_vecs)
        inp = self._build_input()

        # Single forward pass — no ACT loop (context doesn't need variable halting)
        logits, halt_logits, hidden, self.kv_cache = self.model(
            inp, memory_vectors=mem_tensor, return_hidden=True,
            kv_cache=[], cache_position=0,
        )

        n_mem = mem_tensor.shape[1] + 2
        self.n_mem_positions = n_mem
        self.text_cache_len = len(self.context_buffer)
        self.h = hidden[0, -1, :].detach()

        # Write final hidden state to memory
        self._write_memory()

    # ------------------------------------------------------------------
    # process_token
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_token(self, token_id: int, temperature: float = 1.0,
                      top_k: int = 0, repetition_penalty: float = 1.0,
                      recent_tokens: list | None = None,
                      force_emit: bool = False) -> Optional[int]:
        """
        1. Compute addresses from self.h → read memory (up to 9 vectors).
        2. Append token to context_buffer (cap at 501).
        3. Forward pass with ACT hard halting (exit when p(HALT) > 0.5).
           Re-read memory between ACT steps using split KV-cache.
        4. Write hidden state (int8) to memory at 3 addresses.
        5. Emit: if top1_prob > emit_threshold → sample and return token; else None.
        
        Split KV-cache: memory K,V and text K,V are cached separately.
        ACT re-read only recomputes memory K,V (11 positions) + reprocesses
        the last text token (1 position) instead of full context (11+T).
        """
        HALT = 1

        # Step 1: read memory
        mem_vecs = self._read_memory()

        # Step 2: update context buffer
        self.context_buffer.append(token_id)
        if len(self.context_buffer) > self.cfg.n_text_positions:
            self.context_buffer = self.context_buffer[-self.cfg.n_text_positions:]
            self._invalidate_cache()

        # Step 3: ACT hard halting
        final_logits = None
        for act_step in range(self.max_act_steps):
            has_cache = self.kv_cache is not None

            if has_cache and act_step == 0:
                # INCREMENTAL: pass combined cache directly — no split/combine
                cache_pos = self.n_mem_positions + self.text_cache_len
                inp = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                logits, halt_logits, hidden, self.kv_cache = self.model(
                    inp, memory_vectors=None, return_hidden=True,
                    kv_cache=self.kv_cache, cache_position=cache_pos,
                )
                self.text_cache_len += 1

            elif has_cache and act_step > 0:
                # ACT RE-READ: replace memory prefix with fresh K,V
                mem_tensor = self._build_mem_tensor(mem_vecs)
                n_mem = mem_tensor.shape[1] + 2
                new_mem_kv = self.model.forward_memory(mem_tensor)

                # Rebuild cache: new_mem_kv + old text (minus last position)
                n_old_mem = self.n_mem_positions
                text_end = n_old_mem + self.text_cache_len - 1  # exclude last
                rebuilt = [
                    (torch.cat([mk, k[:, :, n_old_mem:text_end, :]], dim=2),
                     torch.cat([mv, v[:, :, n_old_mem:text_end, :]], dim=2))
                    for (mk, mv), (k, v) in zip(new_mem_kv, self.kv_cache)
                ]
                cache_pos = n_mem + self.text_cache_len - 1

                last_token = self.context_buffer[-1]
                inp = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
                logits, halt_logits, hidden, self.kv_cache = self.model(
                    inp, memory_vectors=None, return_hidden=True,
                    kv_cache=rebuilt, cache_position=cache_pos,
                )
                self.n_mem_positions = n_mem
                # text_cache_len unchanged

            else:
                # PREFILL: first token or cache invalidated — full forward
                mem_tensor = self._build_mem_tensor(mem_vecs)
                n_mem = mem_tensor.shape[1] + 2
                inp = self._build_input()
                logits, halt_logits, hidden, self.kv_cache = self.model(
                    inp, memory_vectors=mem_tensor, return_hidden=True,
                    kv_cache=[], cache_position=0,
                )
                self.n_mem_positions = n_mem
                self.text_cache_len = len(self.context_buffer)

            final_logits = logits

            # Update internal hidden state from last text position
            self.h = hidden[0, -1, :].detach()

            halt_prob = F.softmax(halt_logits[0, -1, :], dim=-1)[HALT].item()
            if halt_prob > 0.5 or act_step == self.max_act_steps - 1:
                break

            # Re-read memory for next ACT step (addresses change with updated h)
            mem_vecs = self._read_memory()

        # Step 4: write hidden state to memory
        self._write_memory()

        # Step 5: decide whether to emit a token
        if final_logits is None:
            return None

        # Move to CPU for sampling (avoids MPS sync overhead per .item())
        last_logits = final_logits[0, -1, :].float().cpu()

        # Apply repetition penalty to recently emitted tokens
        if repetition_penalty != 1.0 and recent_tokens:
            penalty_ids = set(recent_tokens[-64:])  # last 64 tokens
            for tid in penalty_ids:
                if last_logits[tid] > 0:
                    last_logits[tid] /= repetition_penalty
                else:
                    last_logits[tid] *= repetition_penalty

        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            last_logits = last_logits / temperature

        probs = F.softmax(last_logits, dim=-1)
        top1_prob = probs.max().item()

        if top1_prob > self.emit_threshold or force_emit:
            if top_k > 0:
                topk_probs, topk_ids = probs.topk(min(top_k, probs.size(-1)))
                topk_probs = topk_probs / topk_probs.sum()
                idx = torch.multinomial(topk_probs, 1).item()
                return topk_ids[idx].item()
            else:
                return probs.argmax().item()
        return None
