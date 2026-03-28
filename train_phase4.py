"""
Phase 4 — Adaptive Computation Time (ACT) training.

Start from Phase 3 best + frozen address heads.
Differentiable soft halting with ponder curriculum.
"""

import os
import argparse
import math
from collections import Counter

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import ModelConfig, Phase4Config, MemoryConfig
from hardware import detect_hardware, print_hardware_report
from model import LoopedLatentController
from memory import MemorySystem
from dataset import prepare_data
from utils import (
    get_lr, save_checkpoint, load_checkpoint,
    Timer, vram_report, peak_vram, format_eta, format_time,
)
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
    hw   = detect_hardware()
    print_hardware_report(hw)
    ov   = hw['overrides']

    cfg   = ModelConfig()
    pcfg  = Phase4Config()
    mcfg  = MemoryConfig()
    device = hw['device']

    # Hardware-aware hyperparameters
    micro_batch = ov['phase4']['micro_batch']
    grad_accum  = ov['phase4']['grad_accum']
    num_workers = ov['num_workers']
    pin_memory  = ov['pin_memory']
    use_amp     = ov['use_amp'] and device == 'cuda'
    amp_dtype   = hw['amp_dtype']
    use_scaler  = use_amp and amp_dtype == torch.float16
    use_ckpt    = ov.get('gradient_checkpointing', True)

    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    phase3_dir = os.path.join(checkpoint_dir, "phase3")
    phase4_dir = os.path.join(checkpoint_dir, "phase4")
    os.makedirs(phase4_dir, exist_ok=True)

    train_mem_dir = os.path.join(phase3_dir, "train_memory")

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_ds, val_ds, tokenizer = prepare_data(tok_path, data_dir, seq_len=cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=use_ckpt).to(device)

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
    scaler = GradScaler(enabled=use_scaler)

    step = 0
    best_loss = float("inf")
    tokens_seen = 0

    if resume:
        latest_path = os.path.join(phase4_dir, "latest.pt")
        if os.path.exists(latest_path):
            step = load_checkpoint(model, optimizer, latest_path, device)

    tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step
    total_tokens_fmt = f"{pcfg.total_tokens / 1e9:.2f}B"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_label = f"{'BF16' if amp_dtype == torch.bfloat16 else 'FP16'} (autocast + GradScaler)" if use_amp else "FP32"

    print("=" * 64)
    print("  PHASE 4: Adaptive Computation Time (ACT) Training")
    print("=" * 64)
    print(f"  Model:           {total_params / 1e6:.1f}M parameters ({trainable_params / 1e6:.1f}M trainable)")
    print(f"  Effective Batch: {micro_batch * grad_accum} (micro={micro_batch} × accum={grad_accum})")
    print(f"  Total Tokens:    {total_tokens_fmt} → {total_steps:,} optimizer steps")
    print(f"  Precision:       {amp_label}")
    if hw['gpu_name']:
        print(f"  GPU:             {hw['gpu_name']} ({hw['gpu_vram_gb']:.1f} GB VRAM)")
    print(f"  Gradient Ckpt:   {'Enabled' if use_ckpt else 'Disabled'}")
    print("=" * 64)
    print(f"Total steps: {total_steps:,}")

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    accum_loss  = 0.0
    halt_hist   = Counter()

    try:
        while step < total_steps:
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            max_act, ponder_w, temperature = get_curriculum(step, pcfg.ponder_curriculum)

            for micro_step in range(grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

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

                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
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
                    loss = (lm_loss + ponder_loss) / grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens_seen += inp.numel()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 500 == 0:
                elapsed = timer.elapsed()
                pct = 100.0 * step / total_steps
                eta_secs = (total_steps - step) * (elapsed / max(step, 1))
                hist_str = " ".join(f"{k}:{v}" for k, v in sorted(halt_hist.items()))
                # Compute avg halt steps from histogram
                total_halt = sum(halt_hist.values())
                avg_halt = sum(k * v for k, v in halt_hist.items()) / max(total_halt, 1)
                print(
                    f"[Phase 4] Step {step}/{total_steps} ({pct:.1f}%) | "
                    f"LM Loss: {accum_loss:.4f} | "
                    f"Halt: {avg_halt:.1f} steps | "
                    f"max_s={max_act} pw={ponder_w:.3f} T={temperature:.1f} | "
                    f"VRAM: {vram_report()} | ETA: {format_eta(eta_secs)} | "
                    f"Elapsed: {format_time(elapsed)} | "
                    f"Tokens: {tokens_seen/1e9:.3f}B/{total_tokens_fmt}"
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

    wall_time = timer.elapsed()
    print("=" * 64)
    print("  PHASE 4 COMPLETE")
    print("=" * 64)
    print(f"  Total Steps:     {step:,}")
    print(f"  Total Tokens:    {tokens_seen/1e9:.2f}B")
    print(f"  Wall Time:       {format_time(wall_time)}")
    avg_tps = tokens_seen / max(wall_time, 1)
    print(f"  Avg Throughput:  {avg_tps/1e3:.1f}k tok/s")
    print(f"  Peak VRAM:       {peak_vram():.1f} GB")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir, args.resume)
