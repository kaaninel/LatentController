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

    def generate(self, agent: Agent, max_tokens: int = 500) -> str:
        """
        Drive the agent to emit tokens until EOS or max_tokens.
        We re-feed emitted tokens back into the agent to build context.
        """
        eos_id = self.tokenizer.token_to_id("<eos>") or 1
        bos_id = self.tokenizer.token_to_id("<bos>") or 2

        emitted: List[int] = []
        # Seed with BOS
        agent.process_token(bos_id)

        for _ in range(max_tokens):
            # Use the agent's last emitted token (or BOS) as next input
            feed_id = emitted[-1] if emitted else bos_id
            out = agent.process_token(feed_id)
            if out is None:
                continue
            emitted.append(out)
            if out == eos_id:
                break

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

    def query(self, text: str, think_budget: int = 5, max_output: int = 500) -> str:
        """
        Feed the text into a fresh agent (think_budget ACT steps), then generate.
        """
        agent = self.create_agent(max_act_steps=think_budget)
        self.feed(agent, text)
        return self.generate(agent, max_tokens=max_output)

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
