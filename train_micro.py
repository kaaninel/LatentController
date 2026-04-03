#!/usr/bin/env python3
"""
Micro Prototype — ~1M param memory-recall experiment.

Self-contained: builds tokenizer, generates dataset, trains, evaluates.
Proves memory architecture works on extractive QA before scaling.
Runs on MPS/CPU in ~30 minutes.

Usage:
    python train_micro.py                    # full pipeline
    python train_micro.py --eval_only        # eval last checkpoint
    python train_micro.py --device cpu       # force CPU
"""

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import MicroModelConfig, MemoryConfig
from model import LoopedLatentController
from memory import MemorySystem
from agent import addr_bytes, memory_vecs_to_tensor

# ============================================================================
# Verbose Logging Utilities
# ============================================================================

class TrainingTracker:
    """Tracks running statistics for verbose logging."""

    def __init__(self, window=50):
        self.window = window
        self.losses = []
        self.grad_norms = []
        self.step_times = []
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.eval_history = []  # (step, acc, breakdown)
        self._last_time = None

    def tick(self):
        now = time.time()
        if self._last_time is not None:
            self.step_times.append(now - self._last_time)
        self._last_time = now

    def add_loss(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss

    def add_grad_norm(self, norm):
        self.grad_norms.append(norm)

    def add_eval(self, step, acc, breakdown=None):
        self.eval_history.append((step, acc, breakdown))
        if acc > self.best_acc:
            self.best_acc = acc

    @property
    def avg_loss(self):
        w = self.losses[-self.window:]
        return sum(w) / len(w) if w else 0

    @property
    def avg_grad_norm(self):
        w = self.grad_norms[-self.window:]
        return sum(w) / len(w) if w else 0

    @property
    def steps_per_sec(self):
        w = self.step_times[-self.window:]
        return 1.0 / (sum(w) / len(w)) if w else 0

    @property
    def loss_trend(self):
        """Arrow showing loss direction over last window."""
        if len(self.losses) < self.window:
            return "─"
        first_half = self.losses[-self.window:-self.window // 2]
        second_half = self.losses[-self.window // 2:]
        avg1 = sum(first_half) / len(first_half)
        avg2 = sum(second_half) / len(second_half)
        diff = avg2 - avg1
        if diff < -0.01:
            return "↓"  # improving
        elif diff > 0.01:
            return "↑"  # worsening
        return "→"  # flat


def log_model_summary(model, cfg, label="Model"):
    """Print detailed model configuration and parameter counts."""
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ┌─── {label} Architecture ───")
    print(f"  │ d_model={cfg.d_model}  n_heads={cfg.n_heads}  head_dim={cfg.head_dim}")
    print(f"  │ n_layers={cfg.n_layers}  ffn_dim={cfg.ffn_dim}  max_seq={cfg.max_seq_len}")
    max_slots = getattr(cfg, "max_memory_slots", cfg.n_mem_slots)
    print(f"  │ mem_slots={cfg.n_mem_slots}  max_memory_slots={max_slots}  addr_heads={cfg.n_addr_heads}  addr_dim={cfg.addr_dim}")
    print(f"  │ params: {n_total:,} total ({n_total/1e6:.2f}M), {n_train:,} trainable")
    # Parameter breakdown by module
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n or "tok_emb" in n)
    layers = sum(p.numel() for n, p in model.named_parameters() if "layers." in n)
    heads = sum(p.numel() for n, p in model.named_parameters() if "addr_" in n or "halt_" in n)
    lm_head = sum(p.numel() for n, p in model.named_parameters() if "lm_head" in n or "output" in n)
    print(f"  │ breakdown: embed={embed:,}  layers={layers:,}  heads={heads:,}  lm_head={lm_head:,}")
    print(f"  └────────────────────")


def log_memory_diagnostics(mem_vecs, label="Memory"):
    """Print memory vector statistics."""
    B, S, D = mem_vecs.shape
    norms = mem_vecs.norm(dim=-1)  # (B, S)
    # Pairwise cosine similarity between slots
    normed = F.normalize(mem_vecs, dim=-1)
    sims = torch.bmm(normed, normed.transpose(1, 2))  # (B, S, S)
    # Mask diagonal
    mask = ~torch.eye(S, device=sims.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    off_diag_sim = sims[mask].reshape(B, -1)

    print(f"  {label}: shape=({B},{S},{D})")
    print(f"    norms:  mean={norms.mean():.2f}  std={norms.std():.2f}  "
          f"min={norms.min():.2f}  max={norms.max():.2f}")
    print(f"    cosine: mean={off_diag_sim.mean():.3f}  std={off_diag_sim.std():.3f}  "
          f"max={off_diag_sim.max():.3f}")
    # Variance across slots (high = diverse, low = redundant)
    slot_var = mem_vecs.var(dim=1).mean()  # average variance across slots
    print(f"    slot_var={slot_var:.4f} (higher=more diverse)")


def log_gradient_stats(model):
    """Compute and return gradient norm, also log per-module stats."""
    total_norm = 0.0
    module_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            pn = p.grad.data.norm(2).item() ** 2
            total_norm += pn
            # Group by first two path components
            parts = name.split(".")
            module = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            module_norms[module] = module_norms.get(module, 0) + pn
    total_norm = total_norm ** 0.5
    return total_norm, {k: v ** 0.5 for k, v in module_norms.items()}

# ============================================================================
# Micro Tokenizer — simple word-level, ~200 tokens
# ============================================================================

SPECIAL_TOKENS = {
    "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
    "<mem_start>": 4, "<mem_end>": 5, "<noop>": 6,
}

NAMES = [
    "mary", "john", "daniel", "sandra", "fred",
    "bill", "julie", "emma", "bob", "alice",
]
LOCATIONS = [
    "garden", "kitchen", "office", "bedroom", "bathroom",
    "hallway", "park", "school", "cinema", "library",
]
VERBS = ["went", "moved", "journeyed", "travelled", "ran", "walked"]
PREPOSITIONS = ["to", "the", "in", "is", "a"]
QUESTION_WORDS = ["where", "?"]
PUNCTUATION = ["."]
CONNECTORS = ["then", "after", "that"]
ANSWER_MARKER = ["<ans>"]  # marks start of answer region

def build_vocab():
    """Build word→id and id→word mappings."""
    vocab = dict(SPECIAL_TOKENS)
    idx = len(vocab)
    for group in [NAMES, LOCATIONS, VERBS, PREPOSITIONS,
                  QUESTION_WORDS, PUNCTUATION, CONNECTORS, ANSWER_MARKER]:
        for w in group:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab

VOCAB = build_vocab()
ID2WORD = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def tokenize(text: str) -> list[int]:
    """Simple whitespace+punctuation tokenizer."""
    # Split keeping punctuation as separate tokens
    text = text.replace(".", " .").replace("?", " ?")
    words = text.lower().split()
    return [VOCAB.get(w, VOCAB["<unk>"]) for w in words]


def detokenize(ids: list[int]) -> str:
    words = [ID2WORD.get(i, "<unk>") for i in ids
             if i not in (VOCAB["<pad>"], VOCAB["<bos>"], VOCAB["<eos>"],
                          VOCAB["<noop>"], VOCAB["<ans>"])]
    text = " ".join(words)
    text = text.replace(" .", ".").replace(" ?", "?")
    return text


# ============================================================================
# Dataset Generator — bAbI-style extractive QA
# ============================================================================

@dataclass
class QAExample:
    """A single memory-recall QA example."""
    passage: str       # "Mary went to the garden . John went to the kitchen ."
    question: str      # "Where is Mary ?"
    answer: str        # "garden"
    answer_entity: str # the name being asked about
    facts: dict        # {"mary": "garden", "john": "kitchen"}


def generate_single_fact() -> QAExample:
    """One person, one location."""
    name = random.choice(NAMES)
    loc = random.choice(LOCATIONS)
    verb = random.choice(VERBS)
    passage = f"{name} {verb} to the {loc} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc, name, {name: loc})


def generate_two_facts() -> QAExample:
    """Two people, ask about one."""
    names = random.sample(NAMES, 2)
    locs = random.sample(LOCATIONS, 2)
    verb1, verb2 = random.choice(VERBS), random.choice(VERBS)
    passage = f"{names[0]} {verb1} to the {locs[0]} . {names[1]} {verb2} to the {locs[1]} ."
    target = random.randint(0, 1)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_three_facts() -> QAExample:
    """Three people, ask about one."""
    names = random.sample(NAMES, 3)
    locs = random.sample(LOCATIONS, 3)
    parts = []
    for n, l in zip(names, locs):
        v = random.choice(VERBS)
        parts.append(f"{n} {v} to the {l} .")
    passage = " ".join(parts)
    target = random.randint(0, 2)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_temporal() -> QAExample:
    """Person moves twice — answer is LAST location."""
    name = random.choice(NAMES)
    loc1, loc2 = random.sample(LOCATIONS, 2)
    v1, v2 = random.sample(VERBS, 2)
    passage = f"{name} {v1} to the {loc1} . then {name} {v2} to the {loc2} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc2, name, {name: loc2})


def generate_distractor() -> QAExample:
    """Two people, one moves twice. Ask about either."""
    n1, n2 = random.sample(NAMES, 2)
    l1, l2, l3 = random.sample(LOCATIONS, 3)
    v1, v2, v3 = random.choices(VERBS, k=3)
    passage = (f"{n1} {v1} to the {l1} . {n2} {v2} to the {l2} . "
               f"then {n1} {v3} to the {l3} .")
    facts = {n1: l3, n2: l2}
    target = random.choice([n1, n2])
    question = f"where is {target} ?"
    return QAExample(passage, question, facts[target], target, facts)


GENERATORS = [
    (generate_single_fact, 0.15),
    (generate_two_facts, 0.30),
    (generate_three_facts, 0.20),
    (generate_temporal, 0.20),
    (generate_distractor, 0.15),
]


def generate_dataset(n: int, seed: int = 42) -> list[QAExample]:
    """Generate n QA examples with weighted type distribution."""
    random.seed(seed)
    gens, weights = zip(*GENERATORS)
    examples = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        examples.append(gen())
    return examples


# ============================================================================
# PyTorch Datasets
# ============================================================================

class ContextQADataset(Dataset):
    """
    For warmup/LM training: passage + question + answer all in context.
    Shifted autoregressive format:
      full = <bos> passage <ans> question answer <eos>
      inp  = full[:-1]
      tgt  = [PAD...context...] answer <eos>  (PAD=ignored in loss)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 128):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos] + passage_ids + [ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            # Only compute loss on answer tokens — use pad_id for context so they're ignored
            n_context = 1 + len(passage_ids) + 1 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids), f"{len(inp_ids)} != {len(tgt_ids)}"

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MemoryQADataset(Dataset):
    """
    For memory training: passage is ONLY in memory, not in context.
    Shifted autoregressive format:
      full    = <bos> <ans> question answer <eos>
      inp     = full[:-1]
      tgt     = [PAD...context...] answer <eos>  (PAD=ignored in loss)
      passage = padded passage tokens (for memory feeding)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 128,
                 max_passage_len: int = 64):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos, ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            n_context = 2 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids)

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            # Pad passage to fixed length for batching
            p_ids = passage_ids[:max_passage_len]
            while len(p_ids) < max_passage_len:
                p_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                torch.tensor(p_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Helpers
# ============================================================================

def hidden_to_int8(hidden: torch.Tensor) -> np.ndarray:
    arr = hidden.detach().float().cpu().numpy()
    scale = np.abs(arr).max()
    if scale < 1e-6:
        return np.zeros(arr.shape, dtype=np.int8)
    return np.clip(np.round(arr / scale * 127.0), -128, 127).astype(np.int8)


def batch_read_memory(model, hidden_states, memory, device):
    """Read memory for a batch using hidden states to compute addresses."""
    B = hidden_states.size(0)
    d = model.cfg.d_model
    addr_heads = model.compute_addresses_batch(hidden_states)
    addr_cpu = [h.cpu().numpy() for h in addr_heads]
    batch_addresses = []
    for b in range(B):
        sample_addrs = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
        batch_addresses.append(sample_addrs)
    mem_np = memory.read_memory_batch(batch_addresses)
    mem_tensor = torch.from_numpy(
        mem_np.astype(np.float32) / 127.0
    ).to(device, non_blocking=True)
    return mem_tensor


def write_memory_batch(model, hidden, memory, positions=None):
    """Write hidden states at specified positions to memory."""
    B, T, D = hidden.shape
    if positions is None:
        positions = [T - 1]  # default: write last position only

    n_writes = 0
    for pos in positions:
        h_batch = hidden[:, pos, :].detach()
        vecs_np = h_batch.float().cpu().numpy()
        addr_heads = model.compute_addresses_batch(h_batch)
        addr_cpu = [h.cpu().numpy() for h in addr_heads]
        for b in range(B):
            ab = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
            memory.write_memory(ab, vecs_np[b])
            n_writes += 1
    return n_writes


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Phase A: Warmup LM on QA patterns (passage in context)
# ============================================================================

def train_phase_a(model, cfg, device, examples, steps=500, lr=3e-4,
                  batch_size=64, max_len=128):
    """Train LM on QA format: passage + question → answer. No memory."""
    print("\n" + "=" * 60)
    print("  Phase A: Warmup LM (QA patterns, no memory)")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)

    t0 = time.time()
    tracker = TrainingTracker()
    for step in range(1, steps + 1):
        tracker.tick()
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        logits, halt_logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm, _ = log_gradient_stats(model)
        tracker.add_loss(loss.item())
        tracker.add_grad_norm(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [A {step:>4}/{steps}] loss={loss.item():.4f} "
                  f"avg={tracker.avg_loss:.4f}{tracker.loss_trend} "
                  f"gnorm={grad_norm:.2f} lr={lr_now:.1e} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase A done in {elapsed:.0f}s, final loss={loss.item():.4f}")
    return loss.item()


# ============================================================================
# Phase B: Address head contrastive training
# ============================================================================

def train_phase_b(model, cfg, device, examples, steps=300, lr=1e-3,
                  batch_size=128, max_len=128):
    """
    Train address heads to produce distinct addresses for different entities.
    Same entity → similar address, different entity → different address.
    """
    print("\n" + "=" * 60)
    print("  Phase B: Address Head Contrastive Training")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")
    print(f"  Only training: addr_heads (backbone frozen)")

    # Generate hidden states for various entities
    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Only train address heads
    for p in model.parameters():
        p.requires_grad_(False)
    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for head in model.addr_heads for p in head.parameters()],
        lr=lr, weight_decay=0.01,
    )

    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp = inp.to(device)

        with torch.no_grad():
            _, _, hidden = model(inp, return_hidden=True)

        # Use last position hidden states
        h = hidden[:, -1, :]  # (B, d_model)
        B = h.size(0)

        # Compute addresses from all heads
        loss = torch.tensor(0.0, device=device)
        for head in model.addr_heads:
            raw = head(h)  # (B, addr_dim)
            # Contrastive: pull same-batch neighbors apart (diversity),
            # but encourage spread across address space
            # Use pairwise distance — maximize average distance
            dists = torch.cdist(raw.unsqueeze(0), raw.unsqueeze(0)).squeeze(0)  # (B, B)
            # Encourage large pairwise distances (negative = push apart)
            margin = 4.0
            spread_loss = F.relu(margin - dists).mean()

            # Entropy: encourage each dimension to use full range
            dim_std = raw.std(dim=0)
            target_std = 15.0
            entropy_loss = F.mse_loss(dim_std, torch.full_like(dim_std, target_std))

            loss = loss + spread_loss + 0.1 * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for head in model.addr_heads for p in head.parameters()], 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [B {step}/{steps}] loss={loss.item():.4f} ({elapsed:.0f}s)")

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.time() - t0
    print(f"  Phase B done in {elapsed:.0f}s")


# ============================================================================
# Phase C: ACT Curriculum (learn when to think more)
# ============================================================================

def streaming_act_forward(model, inp, memory_system, mem_tensor,
                          max_steps, temperature, device):
    """ACT forward with soft halting and optional memory re-read."""
    B, T = inp.shape
    HALT = 1

    remaining = torch.ones(B, T, device=device)
    weighted_logits = None
    expected_steps = torch.zeros(B, T, device=device)
    halt_counts = Counter()

    for i in range(max_steps):
        logits, halt_logits, hidden = model(
            inp, memory_vectors=mem_tensor, return_hidden=True
        )
        halt_prob = F.softmax(halt_logits / max(temperature, 1e-6), dim=-1)[..., HALT]

        if i < max_steps - 1:
            w = remaining * halt_prob
        else:
            w = remaining

        if weighted_logits is None:
            weighted_logits = w.unsqueeze(-1) * logits
        else:
            weighted_logits = weighted_logits + w.unsqueeze(-1) * logits

        expected_steps = expected_steps + (i + 1) * w
        remaining = (remaining - w).clamp(min=0.0)
        halt_counts[i + 1] += (halt_prob > 0.5).sum().item()

        # Re-read memory between ACT steps
        if i < max_steps - 1 and memory_system is not None:
            h_last = hidden[:, -1, :].detach()
            mem_tensor = batch_read_memory(model, h_last, memory_system, device)

    return weighted_logits, expected_steps, halt_counts, hidden


def train_phase_c(model, cfg, device, examples, steps=500, lr=1e-4,
                  batch_size=64, max_len=128):
    """
    ACT curriculum: ramp max_steps and ponder cost.
    Uses context QA (passage in context) so model can learn to halt properly.
    """
    print("\n" + "=" * 60)
    print("  Phase C: ACT Curriculum")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")
    print(f"  Curriculum: ramp max_steps 2→4, ponder 0→0.005")

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    # Curriculum: (step_threshold, max_act, ponder_weight, temperature)
    curriculum = [
        (0,   2, 0.0,   1.0),
        (100, 2, 0.001, 1.0),
        (200, 4, 0.002, 0.5),
        (350, 4, 0.005, 0.1),
    ]

    def get_curriculum_params(step):
        max_act, pw, temp = 2, 0.0, 1.0
        for thresh, ma, p, t in curriculum:
            if step >= thresh:
                max_act, pw, temp = ma, p, t
        return max_act, pw, temp

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        max_act, ponder_w, temperature = get_curriculum_params(step)

        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        weighted_logits, expected_steps, halt_counts, _ = streaming_act_forward(
            model, inp, None, None, max_act, temperature, device
        )

        lm_loss = F.cross_entropy(
            weighted_logits.reshape(-1, weighted_logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )
        ponder_loss = ponder_w * expected_steps.mean() if ponder_w > 0 else 0.0
        loss = lm_loss + ponder_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            total_halt = sum(halt_counts.values())
            avg_halt = sum(k * v for k, v in halt_counts.items()) / max(total_halt, 1)
            print(f"  [C {step}/{steps}] loss={loss.item():.4f} "
                  f"act={max_act} halt={avg_halt:.1f} pw={ponder_w:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase C done in {elapsed:.0f}s")


# ============================================================================
# Phase D: Memory QA — streaming video-frame encoder
# ============================================================================

def _find_entity_positions(token_ids, pad_id):
    """
    Parse passage tokens to find per-entity key positions.
    Returns list of (name_pos, loc_pos, period_pos) tuples, one per entity.
    Positions are offsets into the token_ids list (0-indexed).
    """
    name_ids = {VOCAB[n] for n in NAMES if n in VOCAB}
    loc_ids = {VOCAB[l] for l in LOCATIONS if l in VOCAB}
    period_id = VOCAB.get(".", -1)

    ids = [t for t in token_ids if t != pad_id]
    entities = []
    cur_name_pos = None
    cur_loc_pos = None

    for i, tid in enumerate(ids):
        if tid in name_ids:
            cur_name_pos = i
            cur_loc_pos = None
        elif tid in loc_ids and cur_name_pos is not None:
            cur_loc_pos = i
        elif tid == period_id and cur_name_pos is not None:
            entities.append((
                cur_name_pos,
                cur_loc_pos if cur_loc_pos is not None else i,
                i,  # period position
            ))
            cur_name_pos = None
            cur_loc_pos = None

    return entities



def encode_streaming_memory(model, passages, device, chunk_size=8,
                            slots_per_chunk=2, differentiable=False):
    """Video-frame streaming encoder — per-token memory.

    Processes passage chunks sequentially. Each chunk's forward pass includes
    cross-attention to ALL previously emitted token hidden states, so each
    new "frame" is encoded with full awareness of what came before.

    Every content token becomes its own memory vector (no pooling).
    This preserves per-token entity/location information that pooling destroys.

    Returns: (keys, values, mask) -- memory bank for downstream QA.
      keys:   (B, n_tokens, d_model) -- per-token hidden + temporal signal
      values: (B, n_tokens, d_model) -- per-token hidden states
      mask:   (B, n_tokens) bool     -- True for valid positions
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]

    # Per-example content lengths (exclude padding)
    content_lens = []
    for b in range(B):
        clen = sum(1 for t in passages[b].tolist() if t != pad_id)
        content_lens.append(clen)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    n_chunks = (max_clen + chunk_size - 1) // chunk_size

    # Accumulate per-token memory across chunks
    all_keys = []    # list of (B, d_model) tensors, one per token
    all_values = []
    all_mask = []    # list of (B,) bool tensors

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, max_clen)
        chunk_len = end - start

        # Build chunk input: [BOS, chunk_tokens...] per example
        inp = torch.full((B, chunk_len + 1), pad_id, dtype=torch.long, device=device)
        per_token_valid = torch.zeros(B, chunk_len, dtype=torch.bool, device=device)

        for b in range(B):
            clen = content_lens[b]
            if start >= clen:
                continue
            inp[b, 0] = bos_id
            actual_end = min(end, clen)
            actual_len = actual_end - start
            inp[b, 1:actual_len + 1] = passages[b, start:actual_end]
            per_token_valid[b, :actual_len] = True

        # Prepare accumulated memory from previous chunks
        if all_keys:
            mem_keys = torch.stack(all_keys, dim=1)
            mem_vals = torch.stack(all_values, dim=1)
            mem_mask = torch.stack(all_mask, dim=1)
        else:
            mem_keys, mem_vals, mem_mask = None, None, None

        # Forward pass — each chunk sees all previous tokens via cross-attention
        if differentiable:
            _, _, hidden = model(
                inp, memory_keys=mem_keys, memory_values=mem_vals,
                memory_mask=mem_mask, return_hidden=True)
        else:
            with torch.no_grad():
                _, _, hidden = model(
                    inp, memory_keys=mem_keys, memory_values=mem_vals,
                    memory_mask=mem_mask, return_hidden=True)

        # Skip BOS (position 0), keep per-token hidden states
        content_h = hidden[:, 1:, :]  # (B, chunk_len, d_model)

        # Temporal embedding for this chunk
        t_idx = torch.tensor(ci, dtype=torch.long, device=device)
        t_idx = t_idx.clamp_max(model.temporal_emb.num_embeddings - 1)
        t_emb = model.temporal_emb(t_idx)  # (d_model,)

        # Each token becomes its own memory entry
        for ti in range(chunk_len):
            tok_h = content_h[:, ti, :]       # (B, d_model)
            tok_valid = per_token_valid[:, ti]  # (B,)
            all_keys.append(tok_h + t_emb)
            all_values.append(tok_h)
            all_mask.append(tok_valid)

    # Stack into memory bank tensors
    keys = torch.stack(all_keys, dim=1)    # (B, n_tokens, d_model)
    values = torch.stack(all_values, dim=1)
    mask = torch.stack(all_mask, dim=1)

    return keys, values, mask


# ---------------------------------------------------------------------------
# Encoder-Decoder Memory: Multi-iteration bidirectional encoder
# ---------------------------------------------------------------------------

def encode_enc_dec(model, passages, device, n_iterations=3,
                   differentiable=False):
    """Encoder-decoder memory: multi-iteration bidirectional encoder.

    Two-phase architecture:
      ENCODER: Processes the full passage with bidirectional self-attention
               across multiple iterations. Each iteration cross-attends to
               the previous iteration's output, enabling iterative refinement.
               No token output — pure understanding.
      DECODER: (handled by normal model.forward) Cross-attends to encoder
               output via memory, produces answer tokens with causal attention.

    Key insight: memory KEYS use raw token embeddings (entity identity),
    while memory VALUES use refined contextual hidden states (understanding).
    This separates "what to attend to" (entity matching) from
    "what to retrieve" (contextual answer), solving the entity-routing problem.

    Returns: (keys, values, mask) for decoder cross-attention.
      keys:   (B, n_tokens, d_model) -- raw embeddings (identity-focused)
      values: (B, n_tokens, d_model) -- refined hidden states (context-focused)
      mask:   (B, n_tokens) bool     -- True for valid positions
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]

    # Per-example content lengths
    content_lens = []
    for b in range(B):
        clen = sum(1 for t in passages[b].tolist() if t != pad_id)
        content_lens.append(clen)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    # Build full passage input: [BOS, passage_tokens...]
    inp = torch.full((B, max_clen + 1), pad_id, dtype=torch.long, device=device)
    token_valid = torch.zeros(B, max_clen, dtype=torch.bool, device=device)
    for b in range(B):
        clen = content_lens[b]
        inp[b, 0] = bos_id
        inp[b, 1:clen + 1] = passages[b, :clen]
        token_valid[b, :clen] = True

    # Raw token embeddings for memory KEYS (entity identity, no context mixing)
    with torch.no_grad():
        raw_emb = model.embed(inp[:, 1:])  # (B, max_clen, d_model), skip BOS

    # Multi-iteration bidirectional encoding for memory VALUES
    mem_keys_iter, mem_vals_iter, mem_mask = None, None, None

    ctx = torch.enable_grad if differentiable else torch.no_grad
    with ctx():
        for iteration in range(n_iterations):
            # Full forward: bidirectional self-attn + cross-attn to prev iteration
            _, _, hidden = model(
                inp,
                memory_keys=mem_keys_iter,
                memory_values=mem_vals_iter,
                memory_mask=mem_mask,
                return_hidden=True,
                bidirectional=True,
            )

            # Skip BOS, take per-token content hidden states
            content_h = hidden[:, 1:, :]  # (B, max_clen, d_model)

            # This iteration's output becomes next iteration's cross-attn target
            mem_keys_iter = content_h
            mem_vals_iter = content_h
            mem_mask = token_valid

    # Keys = raw embeddings (identity), Values = refined hidden (context)
    return raw_emb, content_h, token_valid


def encode_kv_memory_chunked(model, passages, device, chunk_size=8,
                             slots_per_chunk=2, **kwargs):
    """Non-differentiable streaming memory encoding."""
    return encode_streaming_memory(
        model, passages, device, chunk_size, slots_per_chunk,
        differentiable=False)


def encode_kv_memory_chunked_differentiable(model, passages, device,
                                             chunk_size=8, slots_per_chunk=2,
                                             **kwargs):
    """Differentiable streaming memory encoding -- gradients flow through."""
    return encode_streaming_memory(
        model, passages, device, chunk_size, slots_per_chunk,
        differentiable=True)


# ---------------------------------------------------------------------------
# Sentence-boundary streaming encoder
# ---------------------------------------------------------------------------

def encode_sentence_memory(model, passages, device, differentiable=False):
    """Sentence-boundary streaming encoder — per-token memory.

    Like the video-frame streaming encoder, but splits at sentence boundaries
    (period tokens) instead of fixed chunk sizes. This guarantees each chunk
    contains exactly one entity-location fact, preventing the boundary-splitting
    problem that caps accuracy at 73%.

    Each chunk's forward pass includes cross-attention to ALL previously
    emitted token hidden states (streaming awareness).

    Returns: (keys, values, mask) -- memory bank for downstream QA.
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]
    period_id = VOCAB["."]

    # Per-example: find sentence boundaries and content lengths
    example_sentences = []  # list of lists of (start, end) per example
    content_lens = []
    for b in range(B):
        tokens = passages[b].tolist()
        clen = sum(1 for t in tokens if t != pad_id)
        content_lens.append(clen)

        # Split at periods
        sents = []
        sent_start = 0
        for i in range(clen):
            if tokens[i] == period_id:
                sents.append((sent_start, i + 1))  # include period
                sent_start = i + 1
        # Trailing tokens without period
        if sent_start < clen:
            sents.append((sent_start, clen))
        example_sentences.append(sents)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    max_n_sents = max(len(s) for s in example_sentences)

    # Accumulate per-token memory across sentences
    all_keys = []
    all_values = []
    all_mask = []

    for si in range(max_n_sents):
        # Find max sentence length for this sentence index
        sent_lens = []
        for b in range(B):
            if si < len(example_sentences[b]):
                s, e = example_sentences[b][si]
                sent_lens.append(e - s)
            else:
                sent_lens.append(0)
        max_slen = max(sent_lens) if sent_lens else 0
        if max_slen == 0:
            continue

        # Build sentence input: [BOS, sentence_tokens...]
        inp = torch.full((B, max_slen + 1), pad_id, dtype=torch.long, device=device)
        per_token_valid = torch.zeros(B, max_slen, dtype=torch.bool, device=device)

        for b in range(B):
            if si >= len(example_sentences[b]):
                continue
            s, e = example_sentences[b][si]
            slen = e - s
            inp[b, 0] = bos_id
            inp[b, 1:slen + 1] = passages[b, s:e]
            per_token_valid[b, :slen] = True

        # Prepare accumulated memory from previous sentences
        if all_keys:
            mem_keys = torch.stack(all_keys, dim=1)
            mem_vals = torch.stack(all_values, dim=1)
            mem_mask_prev = torch.stack(all_mask, dim=1)
        else:
            mem_keys, mem_vals, mem_mask_prev = None, None, None

        # Forward pass — causal within sentence, cross-attn to prev sentences
        if differentiable:
            _, _, hidden = model(
                inp, memory_keys=mem_keys, memory_values=mem_vals,
                memory_mask=mem_mask_prev, return_hidden=True)
        else:
            with torch.no_grad():
                _, _, hidden = model(
                    inp, memory_keys=mem_keys, memory_values=mem_vals,
                    memory_mask=mem_mask_prev, return_hidden=True)

        # Skip BOS, keep per-token hidden states
        content_h = hidden[:, 1:, :]  # (B, max_slen, d_model)

        # Temporal embedding for this sentence index
        t_idx = torch.tensor(si, dtype=torch.long, device=device)
        t_idx = t_idx.clamp_max(model.temporal_emb.num_embeddings - 1)
        t_emb = model.temporal_emb(t_idx)

        # Pool: last valid token per example as single memory slot per sentence
        pooled_h = torch.zeros(B, d_model, device=device)
        sent_valid = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            if si >= len(example_sentences[b]):
                continue
            slen = example_sentences[b][si][1] - example_sentences[b][si][0]
            pooled_h[b] = content_h[b, slen - 1, :]
            sent_valid[b] = True
        all_keys.append(pooled_h + t_emb)
        all_values.append(pooled_h)
        all_mask.append(sent_valid)

    # Stack into memory bank tensors
    keys = torch.stack(all_keys, dim=1)
    values = torch.stack(all_values, dim=1)
    mask = torch.stack(all_mask, dim=1)

    return keys, values, mask


def encode_sentence_frozen(model, passages, device, **kwargs):
    """Non-differentiable sentence-boundary memory encoding."""
    return encode_sentence_memory(model, passages, device, differentiable=False)


def encode_sentence_differentiable(model, passages, device, **kwargs):
    """Differentiable sentence-boundary memory encoding."""
    return encode_sentence_memory(model, passages, device, differentiable=True)


def encode_enc_dec_frozen(model, passages, device, **kwargs):
    """Non-differentiable encoder-decoder memory."""
    return encode_enc_dec(model, passages, device, n_iterations=3,
                          differentiable=False)


def encode_enc_dec_differentiable(model, passages, device, **kwargs):
    """Differentiable encoder-decoder memory — gradients flow through all iterations."""
    return encode_enc_dec(model, passages, device, n_iterations=3,
                          differentiable=True)


def train_phase_d(model, cfg, device, train_examples, val_examples,
                  memory_dir, steps=3000, lr=1e-4, batch_size=16,
                  max_len=128, eval_interval=200, encoder_mode='enc_dec'):
    """
    Memory-dependent QA with encoder-decoder architecture.

    encoder_mode='enc_dec': Multi-iteration bidirectional encoder (default).
      The encoder completely consumes the passage through 3 iterations of
      bidirectional self-attention + cross-attention to previous iteration.
      No token output during encoding — pure understanding.
      The decoder then reads from encoder memory via cross-attention.

    encoder_mode='streaming': Legacy streaming chunk encoder (single-pass).

    D1: frozen encoding (teaches decoder to use memory)
    D2: differentiable encoding (encoder + decoder co-adapt)
    """
    print("\n" + "=" * 60)
    print(f"  Phase D: Memory QA (encoder={encoder_mode})")
    print("=" * 60)

    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    context_fade_start = int(steps * 0.2)

    # Select encoder functions based on mode
    if encoder_mode == 'enc_dec':
        encode_frozen = encode_enc_dec_frozen
        encode_diff = encode_enc_dec_differentiable
        enc_label = "enc-dec (3-iter bidir)"
    elif encoder_mode == 'sentence':
        encode_frozen = encode_sentence_frozen
        encode_diff = encode_sentence_differentiable
        enc_label = "sentence-boundary (bidir per sentence)"
    else:
        encode_frozen = lambda m, p, d, **kw: encode_kv_memory_chunked(
            m, p, d, chunk_size, slots_per_chunk)
        encode_diff = lambda m, p, d, **kw: encode_kv_memory_chunked_differentiable(
            m, p, d, chunk_size, slots_per_chunk)
        enc_label = f"streaming (chunk={chunk_size})"

    print(f"  Steps:          {steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  LR:             {lr}")
    print(f"  D1 (frozen):    steps 1-{context_fade_start}")
    print(f"  D2 (diff):      steps {context_fade_start+1}-{steps}")
    print(f"  Eval every:     {eval_interval} steps")
    print(f"  Encoder:        {enc_label}")
    print(f"  d_model:        {d_model}")

    train_ds = MemoryQADataset(train_examples, max_len=max_len)
    val_ds = MemoryQADataset(val_examples, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    model.train()
    loader_iter = iter(train_loader)
    t0 = time.time()
    best_acc = 0.0
    best_step = 0
    tracker = TrainingTracker(window=50)

    # Initial memory diagnostics
    print("\n  Initial memory encoding diagnostics:")
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        model.eval()
        sample_keys, sample_vals, sample_mask = encode_frozen(
            model, sample_batch[2].to(device), device)
        n_valid = sample_mask.sum(dim=1).float().mean().item()
        print(f"  Avg valid slots: {n_valid:.1f} / {sample_keys.size(1)}")
        log_memory_diagnostics(sample_keys, "  Init keys")
        log_memory_diagnostics(sample_vals, "  Init vals")
        model.train()
    del sample_batch

    print(f"\n  {'─' * 56}")
    print(f"  Training started at {time.strftime('%H:%M:%S')}")
    print(f"  {'─' * 56}")

    for step in range(1, steps + 1):
        tracker.tick()

        try:
            inp, tgt, passages = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inp, tgt, passages = next(loader_iter)

        inp, tgt, passages = inp.to(device), tgt.to(device), passages.to(device)

        lr_now = get_lr(step, 200, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # D1: frozen encoding (teaches decoder to read from memory)
        # D2: differentiable encoding (encoder + decoder co-adapt end-to-end)
        if step >= context_fade_start:
            mem_keys, mem_vals, mem_mask = encode_diff(
                model, passages, device)
        else:
            model.eval()
            mem_keys, mem_vals, mem_mask = encode_frozen(
                model, passages, device)
            model.train()

        # Memory-only QA (no context passage in input)
        inp_d = inp
        tgt_d = tgt

        # Decoder forward
        logits, halt_logits, hidden = model(
            inp_d, memory_keys=mem_keys, memory_values=mem_vals,
            memory_mask=mem_mask, return_hidden=True
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_d.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Track gradient norms
        grad_norm, module_norms = log_gradient_stats(model)
        tracker.add_grad_norm(grad_norm)
        tracker.add_loss(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Logging ──
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            eta = elapsed / step * (steps - step) if step > 0 else 0
            phase_tag = "D1" if step < context_fade_start else "D2*"
            trend = tracker.loss_trend

            print(f"  [{phase_tag} {step:>5}/{steps}] "
                  f"loss={loss.item():.4f} avg={tracker.avg_loss:.4f}{trend} "
                  f"gnorm={grad_norm:.2f} "
                  f"lr={lr_now:.1e} "
                  f"spd={tracker.steps_per_sec:.1f}it/s "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # Detailed diagnostics every 250 steps
        if step % 250 == 0:
            print(f"\n  ┌─── Diagnostics @ step {step} ───")
            model.eval()
            with torch.no_grad():
                diag_keys, diag_vals, diag_mask = encode_frozen(
                    model, passages[:4], device)
                n_valid = diag_mask[:4].sum(dim=1).float().mean().item()
                print(f"  │ Valid slots: {n_valid:.1f}/{diag_keys.size(1)}")
                log_memory_diagnostics(diag_keys, "  │ Keys")
                log_memory_diagnostics(diag_vals, "  │ Vals")
            model.train()
            # Gradient breakdown
            top_modules = sorted(module_norms.items(), key=lambda x: -x[1])[:5]
            print(f"  │ Top grad modules: " +
                  " ".join(f"{n}={v:.3f}" for n, v in top_modules))
            # Loss statistics
            print(f"  │ Loss: best={tracker.best_loss:.4f} "
                  f"avg50={tracker.avg_loss:.4f} "
                  f"last={loss.item():.4f}")
            if tracker.eval_history:
                last_step, last_acc, _ = tracker.eval_history[-1]
                print(f"  │ Last eval: step={last_step} acc={last_acc:.1%} "
                      f"best={tracker.best_acc:.1%}")
            print(f"  └────────────────────\n")

        # Evaluation
        if step % eval_interval == 0 or step == steps:
            acc, breakdown = evaluate_memory_qa(
                model, cfg, val_examples, device, max_examples=200,
                return_breakdown=True, encoder_mode=encoder_mode)
            tracker.add_eval(step, acc, breakdown)

            # Show detailed eval results
            delta = acc - tracker.eval_history[-2][1] if len(tracker.eval_history) > 1 else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1%})" if len(tracker.eval_history) > 1 else ""
            print(f"\n  ╔══ EVAL @ step {step} ══")
            print(f"  ║ Accuracy: {acc:.1%}{delta_str}  (best: {tracker.best_acc:.1%})")
            if breakdown:
                for n_facts, (corr, tot) in sorted(breakdown.items()):
                    bar = "█" * int(corr / max(tot, 1) * 20)
                    print(f"  ║   {n_facts}-fact: {corr:>3}/{tot:>3} = "
                          f"{corr/max(tot,1):.1%} {bar}")
            print(f"  ╚{'═' * 30}\n")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                save_path = os.path.join(memory_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "vocab": VOCAB,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

            # Early stopping: no improvement for 5 evals (1000 steps)
            if step > context_fade_start and step - best_step >= eval_interval * 5:
                print(f"\n  ⚠ Early stopping at step {step} (no improvement since step {best_step})")
                # Reload best model
                ckpt = torch.load(os.path.join(memory_dir, "best_model.pt"),
                                  map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model"])
                break

    elapsed = time.time() - t0
    print(f"  Phase D done in {elapsed:.0f}s, best accuracy={best_acc:.1%}")
    return best_acc


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_memory_qa(model, cfg, examples, device, max_examples=200,
                       return_breakdown=False, encoder_mode='enc_dec'):
    """
    Evaluate memory recall with specified encoder.
    encoder_mode='enc_dec': multi-iteration bidirectional encoder
    encoder_mode='streaming': legacy chunked streaming encoder
    """
    model.eval()
    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    pad_id = VOCAB["<pad>"]

    correct = 0
    total = 0
    type_correct = Counter()
    type_total = Counter()

    for i, ex in enumerate(examples[:max_examples]):
        passage_ids = tokenize(ex.passage)

        p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)

        if encoder_mode == 'enc_dec':
            mem_keys, mem_vals, mem_mask = encode_enc_dec_frozen(
                model, p_tensor, device)
        elif encoder_mode == 'sentence':
            mem_keys, mem_vals, mem_mask = encode_sentence_frozen(
                model, p_tensor, device)
        else:
            mem_keys, mem_vals, mem_mask = encode_kv_memory_chunked(
                model, p_tensor, device, chunk_size, slots_per_chunk)

        # Build question input
        question_ids = tokenize(ex.question)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        logits, _, hidden = model(inp, memory_keys=mem_keys,
                                   memory_values=mem_vals,
                                   memory_mask=mem_mask,
                                   return_hidden=True)

        pred_id = logits[0, -1, :].argmax().item()
        expected_id = VOCAB.get(ex.answer, -1)
        is_correct = pred_id == expected_id

        if is_correct:
            correct += 1
        total += 1

        n_facts = len(ex.facts)
        type_total[n_facts] += 1
        if is_correct:
            type_correct[n_facts] += 1

    accuracy = correct / max(total, 1)

    for n_facts in sorted(type_total.keys()):
        t_corr = type_correct[n_facts]
        t_tot = type_total[n_facts]
        print(f"    {n_facts}-fact: {t_corr}/{t_tot} = {t_corr/max(t_tot,1):.1%}")

    model.train()

    if return_breakdown:
        breakdown = {k: (type_correct[k], type_total[k]) for k in type_total}
        return accuracy, breakdown
    return accuracy


@torch.no_grad()
def evaluate_context_qa(model, cfg, examples, device, max_examples=200):
    """Evaluate with passage in context (no memory needed). Baseline test."""
    model.eval()
    correct = 0
    total = 0

    for ex in examples[:max_examples]:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)

        bos = VOCAB["<bos>"]
        ans_marker = VOCAB["<ans>"]
        inp_ids = [bos] + passage_ids + [ans_marker] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        logits, _ = model(inp)
        pred_id = logits[0, -1, :].argmax().item()

        expected_id = VOCAB.get(ex.answer, -1)
        if pred_id == expected_id:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    model.train()
    return accuracy


def detailed_eval(model, cfg, examples, device, n=10, encoder_mode='enc_dec'):
    """Print detailed examples with specified encoder."""
    model.eval()
    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    print("\n" + "-" * 60)
    print("  Detailed Examples")
    print("-" * 60)

    for i, ex in enumerate(examples[:n]):
        passage_ids = tokenize(ex.passage)

        p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if encoder_mode == 'enc_dec':
                mem_keys, mem_vals, mem_mask = encode_enc_dec_frozen(
                    model, p_tensor, device)
            elif encoder_mode == 'sentence':
                mem_keys, mem_vals, mem_mask = encode_sentence_frozen(
                    model, p_tensor, device)
            else:
                mem_keys, mem_vals, mem_mask = encode_kv_memory_chunked(
                    model, p_tensor, device, chunk_size, slots_per_chunk)

        # Question input
        question_ids = tokenize(ex.question)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_mem, _, hidden = model(inp, memory_keys=mem_keys,
                                           memory_values=mem_vals,
                                           memory_mask=mem_mask,
                                           return_hidden=True)
            n_valid = mem_mask[0].sum().item()
            avg_norm = mem_keys[0, :n_valid].norm(dim=-1).mean().item() if n_valid > 0 else 0.0
            # Without memory
            logits_no_mem, _ = model(inp)

        pred_mem = logits_mem[0, -1, :].argmax().item()
        pred_no_mem = logits_no_mem[0, -1, :].argmax().item()

        # Top-5 with memory
        top5_vals, top5_ids = logits_mem[0, -1, :].topk(5)
        top5_probs = F.softmax(top5_vals, dim=0)
        top5_words = [(ID2WORD.get(idx.item(), "?"), f"{p.item():.3f}")
                      for idx, p in zip(top5_ids, top5_probs)]

        expected = ex.answer
        mark = "✓" if ID2WORD.get(pred_mem, "") == expected else "✗"

        print(f"\n  {mark} Example {i+1}:")
        print(f"    Passage:  {ex.passage}")
        print(f"    Question: {ex.question}")
        print(f"    Expected: {expected}")
        print(f"    With mem: {ID2WORD.get(pred_mem, '?')} | "
              f"No mem: {ID2WORD.get(pred_no_mem, '?')}")
        print(f"    Top-5:    {top5_words}")
        print(f"    Mem: {n_valid} slots, avg_norm={avg_norm:.2f}")

    model.train()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Micro Prototype Training")
    parser.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--output_dir", default="./checkpoints/micro",
                        help="Where to save checkpoints")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--n_train", type=int, default=20000,
                        help="Number of training examples")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation examples")
    parser.add_argument("--phase_a_steps", type=int, default=500)
    parser.add_argument("--phase_b_steps", type=int, default=300)
    parser.add_argument("--phase_c_steps", type=int, default=500)
    parser.add_argument("--phase_d_steps", type=int, default=3000)
    parser.add_argument("--encoder_mode", default="sentence",
                        choices=["enc_dec", "streaming", "sentence"],
                        help="Memory encoder: sentence (sentence-boundary bidir), enc_dec (multi-iter bidir), or streaming (chunked)")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Override chunk_size for streaming encoder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("  MICRO PROTOTYPE — Memory Recall Experiment")
    print("=" * 60)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = MicroModelConfig()
    cfg.vocab_size = VOCAB_SIZE  # override with actual vocab size
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    print(f"  Device:     {device}")
    print(f"  Vocab:      {VOCAB_SIZE} words")

    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)
    n_params = count_params(model)
    log_model_summary(model, cfg)

    # Generate data
    print(f"\n  Generating {args.n_train} train + {args.n_val} val examples...")
    train_examples = generate_dataset(args.n_train, seed=args.seed)
    val_examples = generate_dataset(args.n_val, seed=args.seed + 1)

    # Show data stats
    type_counts = Counter()
    for ex in train_examples[:1000]:
        type_counts[len(ex.facts)] += 1
    print(f"  Type distribution (first 1000): {dict(type_counts)}")
    print(f"  Example: {train_examples[0].passage}")
    print(f"           {train_examples[0].question} → {train_examples[0].answer}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save vocab
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(VOCAB, f, indent=2)

    if args.eval_only:
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: No checkpoint at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint (step={ckpt['step']}, acc={ckpt['accuracy']:.1%})")

        print("\n  Context QA (baseline, no memory needed):")
        ctx_acc = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc:.1%}")

        print("\n  Memory QA (passage from memory only):")
        mem_acc = evaluate_memory_qa(model, cfg, val_examples, device)
        print(f"  → {mem_acc:.1%}")

        detailed_eval(model, cfg, val_examples, device, n=10)
        return

    # ===== Full Training Pipeline =====
    t_start = time.time()
    print("\n" + "─" * 60)
    print(f"  Pipeline: A({args.phase_a_steps}) → B({args.phase_b_steps}) "
          f"→ C({args.phase_c_steps}) → D({args.phase_d_steps})")
    print("─" * 60)

    # Phase A: Warmup LM
    loss_a = train_phase_a(model, cfg, device, train_examples,
                           steps=args.phase_a_steps, batch_size=64)

    # Quick context QA check
    print("\n  Context QA after Phase A:")
    ctx_acc_a = evaluate_context_qa(model, cfg, val_examples, device)
    print(f"  → {ctx_acc_a:.1%}")

    # Phase B: Address heads
    train_phase_b(model, cfg, device, train_examples,
                  steps=args.phase_b_steps, batch_size=128)

    # Phase C: ACT curriculum
    train_phase_c(model, cfg, device, train_examples,
                  steps=args.phase_c_steps, batch_size=64)

    # Quick context QA check
    print("\n  Context QA after Phase C:")
    ctx_acc_c = evaluate_context_qa(model, cfg, val_examples, device)
    print(f"  → {ctx_acc_c:.1%}")

    # Memory QA baseline (before Phase D)
    print("\n  Memory QA BEFORE Phase D (random baseline):")
    pre_d_acc = evaluate_memory_qa(model, cfg, val_examples, device,
                                   max_examples=200,
                                   encoder_mode=args.encoder_mode)
    print(f"  → {pre_d_acc:.1%}")

    # Phase D: Memory QA (the real test)
    mem_dir = os.path.join(args.output_dir, "memory")
    os.makedirs(mem_dir, exist_ok=True)
    best_acc = train_phase_d(
        model, cfg, device, train_examples, val_examples, mem_dir,
        steps=args.phase_d_steps, lr=1e-4, batch_size=32,
        eval_interval=200, encoder_mode=args.encoder_mode,
    )

    # Final report
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(f"  Total time:      {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Parameters:      {n_params:,}")
    print(f"  Context QA:      {ctx_acc_c:.1%} (passage in context)")
    print(f"  Memory QA:       {best_acc:.1%} (passage from memory)")
    print(f"  Target:          >80% memory QA")
    print(f"  ACT fix:         halt bias [0,0] (50/50 init)")

    # Detailed eval
    detailed_eval(model, cfg, val_examples, device, n=10,
                  encoder_mode=args.encoder_mode)

    # Save final report
    report = {
        "total_time_s": total_time,
        "n_params": n_params,
        "context_qa_acc": ctx_acc_c,
        "memory_qa_acc": best_acc,
        "phases": {
            "A": {"steps": args.phase_a_steps, "final_loss": loss_a},
            "B": {"steps": args.phase_b_steps},
            "C": {"steps": args.phase_c_steps, "context_qa": ctx_acc_c},
            "D": {"steps": args.phase_d_steps, "best_acc": best_acc},
        },
        "config": {
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "vocab_size": VOCAB_SIZE,
        },
        "device": device,
    }
    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
