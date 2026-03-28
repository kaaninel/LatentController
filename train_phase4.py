"""
Phase 4 — Adaptive Computation Time (ACT) training.

Start from Phase 3 best + frozen address heads.
Differentiable soft halting with ponder curriculum.
"""

import os
import argparse
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, Phase4Config, MemoryConfig
from model import LoopedLatentController
from memory import MemorySystem
from dataset import prepare_data
from utils import get_lr, save_checkpoint, load_checkpoint, Timer
from train_phase3 import memory_vecs_to_tensor, addr_bytes


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def get_curriculum(step: int, curriculum: list) -> tuple:
    """
    Returns (max_steps, ponder_weight, temperature) for the given training step.
    Curriculum is a sorted list of (threshold, max_steps, ponder_weight, temperature).
    """
    max_steps, ponder_weight, temperature = curriculum[0][1], curriculum[0][2], curriculum[0][3]
    for threshold, ms, pw, temp in curriculum:
        if step >= threshold:
            max_steps, ponder_weight, temperature = ms, pw, temp
        else:
            break
    return max_steps, ponder_weight, temperature


# ---------------------------------------------------------------------------
# Soft ACT forward (differentiable)
# ---------------------------------------------------------------------------

def act_forward(
    model: LoopedLatentController,
    inp: torch.Tensor,
    mem_tensor: torch.Tensor | None,
    max_steps: int,
    temperature: float,
) -> tuple:
    """
    Soft halting ACT.
    Returns (weighted_logits, expected_steps, halt_step_counts).
    halt_step_counts: Counter of which step caused halting (for logging).
    """
    B, T = inp.shape
    device = inp.device
    HALT = 1  # index in halt_logits corresponding to HALT action

    remaining = torch.ones(B, T, device=device)
    weighted_logits = None
    expected_steps  = torch.zeros(B, T, device=device)
    halt_step_counts = Counter()

    for i in range(max_steps):
        logits, halt_logits, _ = model(inp, memory_vectors=mem_tensor, return_hidden=True)
        # halt_logits: (B, T, 2)
        halt_prob = F.softmax(halt_logits / max(temperature, 1e-6), dim=-1)[..., HALT]  # (B, T)

        if i < max_steps - 1:
            w = remaining * halt_prob
        else:
            # Last step gets all remaining weight
            w = remaining

        if weighted_logits is None:
            weighted_logits = w.unsqueeze(-1) * logits
        else:
            weighted_logits = weighted_logits + w.unsqueeze(-1) * logits

        expected_steps = expected_steps + (i + 1) * w
        remaining = remaining - w
        remaining = remaining.clamp(min=0.0)

        # Track halting for histogram
        halt_step_counts[i + 1] += (halt_prob > 0.5).sum().item()

    return weighted_logits, expected_steps, halt_step_counts


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(checkpoint_dir: str, data_dir: str, resume: bool = False):
    cfg   = ModelConfig()
    pcfg  = Phase4Config()
    mcfg  = MemoryConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 4 training — device: {device}")

    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    phase3_dir = os.path.join(checkpoint_dir, "phase3")
    phase4_dir = os.path.join(checkpoint_dir, "phase4")
    os.makedirs(phase4_dir, exist_ok=True)

    train_mem_dir = os.path.join(phase3_dir, "train_memory")

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_ds, val_ds, tokenizer = prepare_data(tok_path, data_dir, seq_len=cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=pcfg.micro_batch, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=pcfg.micro_batch, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=True).to(device)

    # Load Phase 3 best
    best_p3 = os.path.join(phase3_dir, "best.pt")
    if os.path.exists(best_p3):
        load_checkpoint(model, None, best_p3, device)
    else:
        print("WARNING: No Phase 3 checkpoint — trying Phase 1")
        best_p1 = os.path.join(checkpoint_dir, "phase1", "best.pt")
        if os.path.exists(best_p1):
            load_checkpoint(model, None, best_p1, device)

    # Ensure halt head bias is [-1, +1]
    model.halt_head.bias.data[0] = -1.0
    model.halt_head.bias.data[1] =  1.0

    # Load + freeze Phase 2 address heads
    addr_heads_path = os.path.join(phase2_dir, "addr_heads.pt")
    if os.path.exists(addr_heads_path):
        ckpt = torch.load(addr_heads_path, map_location=device)
        for i, sd in enumerate(ckpt["addr_heads"]):
            model.addr_heads[i].load_state_dict(sd)
    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(False)

    # Memory
    train_memory = MemorySystem(train_mem_dir, mcfg) if os.path.exists(train_mem_dir) else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=pcfg.lr,
        betas=pcfg.betas,
        weight_decay=pcfg.weight_decay,
    )

    step = 0
    best_loss = float("inf")
    tokens_seen = 0

    if resume:
        latest_path = os.path.join(phase4_dir, "latest.pt")
        if os.path.exists(latest_path):
            step = load_checkpoint(model, optimizer, latest_path, device)

    tokens_per_step = pcfg.micro_batch * pcfg.grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,}")

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad()
    accum_loss  = 0.0
    halt_hist   = Counter()

    try:
        while step < total_steps:
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            max_act, ponder_w, temperature = get_curriculum(step, pcfg.ponder_curriculum)

            for micro_step in range(pcfg.grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp, tgt = inp.to(device), tgt.to(device)

                # Read memory
                mem_tensor = None
                if train_memory is not None:
                    with torch.no_grad():
                        _, _, hid_nm = model(inp, return_hidden=True)
                        h0 = hid_nm[0, -1, :]
                        addrs = model.compute_addresses(h0)
                        ab    = [addr_bytes(a) for a in addrs]
                        mvecs = train_memory.read_memory(ab)
                        mem_tensor = memory_vecs_to_tensor(mvecs, cfg.d_model, device)
                        mem_tensor = mem_tensor.expand(inp.size(0), -1, -1)

                weighted_logits, expected_steps, counts = act_forward(
                    model, inp, mem_tensor, max_act, temperature
                )
                halt_hist.update(counts)

                lm_loss = F.cross_entropy(
                    weighted_logits.reshape(-1, weighted_logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=cfg.pad_id,
                )
                ponder_loss = ponder_w * expected_steps.mean() if ponder_w > 0 else torch.zeros_like(lm_loss)
                loss = lm_loss + ponder_loss

                (loss / pcfg.grad_accum).backward()
                accum_loss += loss.item() / pcfg.grad_accum
                tokens_seen += inp.numel()

            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 500 == 0:
                dt = timer.lap()
                hist_str = " ".join(f"{k}:{v}" for k, v in sorted(halt_hist.items()))
                print(
                    f"step {step:6d}/{total_steps}  loss={accum_loss:.4f}  "
                    f"lr={lr:.2e}  max_act={max_act}  pw={ponder_w:.4f}  "
                    f"T={temperature:.2f}  halt=[{hist_str}]  "
                    f"tokens={tokens_seen/1e9:.3f}B"
                )
                accum_loss = 0.0
                halt_hist.clear()

            if step % pcfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, accum_loss,
                    os.path.join(phase4_dir, "latest.pt"),
                    extra={"curriculum_step": step},
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving…")
        save_checkpoint(
            model, optimizer, step, accum_loss,
            os.path.join(phase4_dir, "latest.pt"),
        )

    print(f"Phase 4 complete.  Steps: {step}  Tokens: {tokens_seen/1e9:.2f}B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir, args.resume)
