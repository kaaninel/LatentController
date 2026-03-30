"""
Orchestrator — agent lifecycle manager.
"""

from typing import List, Optional

import torch

from model import LoopedLatentController
from memory import MemorySystem
from agent import Agent
from tokenizer_utils import encode


class Orchestrator:
    def __init__(
        self,
        model: LoopedLatentController,
        memory: MemorySystem,
        tokenizer,
        device,
    ):
        self.model     = model
        self.memory    = memory
        self.tokenizer = tokenizer
        self.device    = device

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def create_agent(self, max_act_steps: int = 6, emit_threshold: float = 0.3) -> Agent:
        return Agent(
            self.model, self.memory, self.device,
            max_act_steps=max_act_steps,
            emit_threshold=emit_threshold,
        )

    # ------------------------------------------------------------------
    # Feed tokens from text into an agent
    # ------------------------------------------------------------------

    def feed(self, agent: Agent, text: str) -> List[int]:
        """Tokenise text and feed each token to the agent; collect outputs."""
        ids = encode(self.tokenizer, text)
        outputs = []
        for tid in ids:
            out = agent.process_token(tid)
            if out is not None:
                outputs.append(out)
        return outputs

    # ------------------------------------------------------------------
    # Generate text from an agent
    # ------------------------------------------------------------------

    def generate(self, agent: Agent, max_tokens: int = 500,
                 temperature: float = 0.8, top_k: int = 50,
                 repetition_penalty: float = 1.3) -> str:
        """
        Drive the agent to emit tokens until EOS or max_tokens.
        Uses temperature sampling with top-k and repetition penalty.
        Detects degenerate n-gram loops and stops early.
        """
        eos_id = self.tokenizer.token_to_id("<eos>") or 1

        emitted: List[int] = []

        for _ in range(max_tokens):
            if emitted:
                out = agent.process_token(emitted[-1], temperature=temperature,
                                          top_k=top_k,
                                          repetition_penalty=repetition_penalty,
                                          recent_tokens=emitted)
            else:
                if agent.context_buffer:
                    out = agent.process_token(
                        agent.context_buffer[-1], temperature=temperature,
                        top_k=top_k, repetition_penalty=repetition_penalty,
                        recent_tokens=emitted,
                    )
                else:
                    out = agent.process_token(
                        self.tokenizer.token_to_id("<bos>") or 2,
                        temperature=temperature, top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        recent_tokens=emitted,
                    )
            if out is None:
                continue
            if out == eos_id:
                break
            emitted.append(out)

            # Detect degenerate n-gram loop (e.g. "kid kid kid kid")
            if len(emitted) >= 12:
                for n in (1, 2, 3):
                    tail = emitted[-6*n:]
                    pattern = tail[-n:]
                    repeats = sum(1 for i in range(0, len(tail) - n + 1, n)
                                 if tail[i:i+n] == pattern)
                    if repeats >= 5:
                        return self.tokenizer.decode(emitted[:-n*4])

        return self.tokenizer.decode(emitted)

    # ------------------------------------------------------------------
    # Pipe: agent A feeds agent B
    # ------------------------------------------------------------------

    def pipe(self, source: Agent, dest: Agent, n_steps: int):
        """Run source for n_steps, forwarding its outputs to dest."""
        bos_id = self.tokenizer.token_to_id("<bos>") or 2
        feed_id = bos_id
        for _ in range(n_steps):
            out = source.process_token(feed_id)
            if out is not None:
                feed_id = out
                dest.process_token(out)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def query(self, text: str, think_budget: int = 5, max_output: int = 500,
             temperature: float = 0.8, top_k: int = 50) -> str:
        """
        Feed the text into a fresh agent (think_budget ACT steps), then generate.
        """
        agent = self.create_agent(max_act_steps=think_budget)
        self.feed(agent, text)
        return self.generate(agent, max_tokens=max_output,
                             temperature=temperature, top_k=top_k)

    def ingest_document(self, text: str):
        """
        Feed a document into a dedicated agent so its content is written to memory.
        """
        agent = self.create_agent(max_act_steps=2, emit_threshold=1.1)  # never emits
        self.feed(agent, text)

    def background_consolidation(self, n_steps: int = 1000):
        """
        Run a pair of agents in a loop to allow memory consolidation.
        Agent A generates tokens → Agent B reads them.
        """
        a = self.create_agent(max_act_steps=4, emit_threshold=0.2)
        b = self.create_agent(max_act_steps=4, emit_threshold=0.2)
        self.pipe(a, b, n_steps=n_steps)
