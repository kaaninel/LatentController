"""
Shared training utilities.
"""

import math
import time
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step: int, loss: float, path: str, extra: Optional[dict] = None):
    payload = {
        "step": step,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"Saved checkpoint → {path}  (step={step}, loss={loss:.4f})")


def load_checkpoint(model, optimizer, path: str, device) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    print(f"Loaded checkpoint from {path}  (step={step})")
    return step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = 100) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    for inp, tgt in val_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        logits, _ = model(inp)
        # logits: (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=0,
            reduction="sum",
        )
        mask = tgt.reshape(-1) != 0
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        n_batches += 1
        if n_batches >= max_batches:
            break
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    model.train()
    return {"loss": avg_loss, "perplexity": ppl}


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(
    model,
    tokenizer,
    prompt_ids,
    max_tokens: int = 200,
    device: str = "cpu",
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    model.eval()
    ids = list(prompt_ids)
    eos_id = tokenizer.token_to_id("<eos>") or 1

    for _ in range(max_tokens):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)

        # Top-p nucleus sampling
        probs = F.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

        ids.append(next_token)
        if next_token == eos_id:
            break

    model.train()
    return tokenizer.decode(ids)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = time.perf_counter()
        self._last = self._start

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        return dt

    def reset(self):
        self._start = time.perf_counter()
        self._last = self._start
