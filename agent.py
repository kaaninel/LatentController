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

        # Split KV-cache: memory and text cached separately
        self.mem_kv_cache: list | None = None   # per-layer (k, v) for memory positions
        self.text_kv_cache: list | None = None  # per-layer (k, v) for text positions
        self.n_mem_positions: int = 0           # always 11 when cache is active
        self.text_cache_len: int = 0            # number of text positions cached

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
        self.mem_kv_cache = None
        self.text_kv_cache = None
        self.n_mem_positions = 0
        self.text_cache_len = 0

    def _combine_kv(self, mem_kv, text_kv):
        """Combine separate memory and text K,V caches into a single cache."""
        return [
            (torch.cat([mk, tk], dim=2), torch.cat([mv, tv], dim=2))
            for (mk, mv), (tk, tv) in zip(mem_kv, text_kv)
        ]

    def _split_kv(self, combined_kv, n_mem):
        """Split combined K,V cache into memory and text portions."""
        mem_kvs, text_kvs = [], []
        for k, v in combined_kv:
            mem_kvs.append((k[:, :, :n_mem, :], v[:, :, :n_mem, :]))
            text_kvs.append((k[:, :, n_mem:, :], v[:, :, n_mem:, :]))
        return mem_kvs, text_kvs

    # ------------------------------------------------------------------
    # process_token
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_token(self, token_id: int, temperature: float = 1.0,
                      top_k: int = 0) -> Optional[int]:
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

        # Step 3: ACT hard halting with split KV-cache
        final_logits = None
        for act_step in range(self.max_act_steps):
            mem_tensor = self._build_mem_tensor(mem_vecs)
            n_mem = mem_tensor.shape[1] + 2  # +2 for <MEM>, </MEM>

            has_cache = self.text_kv_cache is not None

            if has_cache and act_step == 0:
                # INCREMENTAL: fresh memory K,V + cached text K,V + 1 new token
                new_mem_kv = self.model.forward_memory(mem_tensor)
                combined = self._combine_kv(new_mem_kv, self.text_kv_cache)
                cache_pos = n_mem + self.text_cache_len

                inp = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                logits, halt_logits, hidden, returned_cache = self.model(
                    inp, memory_vectors=None, return_hidden=True,
                    kv_cache=combined, cache_position=cache_pos,
                )

                self.mem_kv_cache = new_mem_kv
                _, self.text_kv_cache = self._split_kv(returned_cache, n_mem)
                self.n_mem_positions = n_mem
                self.text_cache_len += 1

            elif has_cache and act_step > 0:
                # ACT RE-READ: fresh memory K,V + cached text K,V (minus last)
                # + reprocess last text token. Cost: 11 + 1 positions vs 11+T.
                new_mem_kv = self.model.forward_memory(mem_tensor)

                # Strip last text token's K,V (will be recomputed with new memory context)
                stripped_text_kv = [
                    (k[:, :, :-1, :], v[:, :, :-1, :])
                    for k, v in self.text_kv_cache
                ]
                combined = self._combine_kv(new_mem_kv, stripped_text_kv)
                cache_pos = n_mem + self.text_cache_len - 1

                last_token = self.context_buffer[-1]
                inp = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
                logits, halt_logits, hidden, returned_cache = self.model(
                    inp, memory_vectors=None, return_hidden=True,
                    kv_cache=combined, cache_position=cache_pos,
                )

                self.mem_kv_cache = new_mem_kv
                _, self.text_kv_cache = self._split_kv(returned_cache, n_mem)
                self.n_mem_positions = n_mem
                # text_cache_len unchanged (stripped last, re-added it)

            else:
                # PREFILL: first token or cache invalidated — full forward
                inp = self._build_input()
                logits, halt_logits, hidden, returned_cache = self.model(
                    inp, memory_vectors=mem_tensor, return_hidden=True,
                    kv_cache=[], cache_position=0,
                )

                self.mem_kv_cache, self.text_kv_cache = self._split_kv(returned_cache, n_mem)
                self.n_mem_positions = n_mem
                self.text_cache_len = len(self.context_buffer)

            final_logits = logits

            # Update internal hidden state from last text position
            self.h = hidden[0, -1, :].detach()

            halt_prob = F.softmax(halt_logits[0, -1, :], dim=-1)[HALT].item()
            if halt_prob > 0.5 or act_step == self.max_act_steps - 1:
                break

            # Re-read memory for next ACT step — cache is preserved!
            mem_vecs = self._read_memory()

        # Step 4: write hidden state to memory
        self._write_memory()

        # Step 5: decide whether to emit a token
        if final_logits is None:
            return None

        last_logits = final_logits[0, -1, :]            # (vocab_size,)

        # Apply temperature
        if temperature > 0 and temperature != 1.0:
            last_logits = last_logits / temperature

        probs = F.softmax(last_logits, dim=-1)
        top1_prob = probs.max().item()

        if top1_prob > self.emit_threshold:
            # Top-k sampling
            if top_k > 0:
                topk_probs, topk_ids = probs.topk(min(top_k, probs.size(-1)))
                topk_probs = topk_probs / topk_probs.sum()
                idx = torch.multinomial(topk_probs, 1).item()
                return topk_ids[idx].item()
            else:
                return probs.argmax().item()
        return None
