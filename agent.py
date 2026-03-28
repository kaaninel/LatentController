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
from train_phase3 import memory_vecs_to_tensor, addr_bytes


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

    # ------------------------------------------------------------------
    # process_token
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_token(self, token_id: int) -> Optional[int]:
        """
        1. Compute addresses from self.h → read memory (up to 9 vectors).
        2. Append token to context_buffer (cap at 501).
        3. Forward pass with ACT hard halting (exit when p(HALT) > 0.5).
           Re-read memory between ACT steps.
        4. Write hidden state (int8) to memory at 3 addresses.
        5. Emit: if top1_prob > emit_threshold → sample and return token; else None.
        """
        HALT = 1

        # Step 1: read memory
        mem_vecs = self._read_memory()

        # Step 2: update context buffer
        self.context_buffer.append(token_id)
        if len(self.context_buffer) > self.cfg.n_text_positions:
            self.context_buffer = self.context_buffer[-self.cfg.n_text_positions:]

        inp = self._build_input()

        # Step 3: ACT hard halting
        final_logits = None
        for act_step in range(self.max_act_steps):
            mem_tensor = self._build_mem_tensor(mem_vecs)
            logits, halt_logits, hidden = self.model(
                inp, memory_vectors=mem_tensor, return_hidden=True
            )
            final_logits = logits

            # Update internal hidden state from last text position
            self.h = hidden[0, -1, :].detach()

            halt_prob = F.softmax(halt_logits[0, -1, :], dim=-1)[HALT].item()
            if halt_prob > 0.5 or act_step == self.max_act_steps - 1:
                break

            # Re-read memory for next ACT step
            mem_vecs = self._read_memory()

        # Step 4: write hidden state to memory
        self._write_memory()

        # Step 5: decide whether to emit a token
        if final_logits is None:
            return None

        last_logits = final_logits[0, -1, :]            # (vocab_size,)
        probs = F.softmax(last_logits, dim=-1)
        top1_prob, top1_id = probs.max(dim=-1)

        if top1_prob.item() > self.emit_threshold:
            return top1_id.item()
        return None
