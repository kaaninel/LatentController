"""
Phase 5 — Unified streaming training.

All capabilities active: Memory read/write + ACT + unfrozen address heads.
Loads Phase 4 best checkpoint + Phase 2 address heads (unfrozen at 0.3× LR).

Every token goes through the same unified loop:
  read memory → ACT forward (with per-step memory re-reads) → write memory → predict next token

NOOP targets teach the model when to absorb (stay silent) vs emit (produce output).
The model has no modes — it learns this autonomously from the data structure.
"""

import os
import argparse
import math
import shutil
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import ModelConfig, Phase5Config, MemoryConfig
from hardware import (
    detect_hardware, print_hardware_report,
    auto_calibrate_batch_size, build_trial_fn, handle_oom,
)
from model import LoopedLatentController
from memory import MemorySystem
from dataset import prepare_data
from utils import (
    get_lr, save_checkpoint, load_checkpoint,
    Timer, vram_report, peak_vram, format_eta, format_time,
)
from agent import addr_bytes, memory_vecs_to_tensor
from train_phase3 import hidden_to_int8, batch_read_memory
from train_phase4 import get_curriculum


# ---------------------------------------------------------------------------
# Streaming ACT forward — re-reads memory between ACT steps
# ---------------------------------------------------------------------------

def streaming_act_forward(
    model: LoopedLatentController,
    inp: torch.Tensor,
    memory_system: Optional[MemorySystem],
    mem_tensor: Optional[torch.Tensor],
    max_steps: int,
    temperature: float,
    cfg: ModelConfig,
) -> tuple:
    """
    ACT forward that re-reads memory per step, matching agent.process_token().

    On each ACT step:
      1. Forward pass with current memory
      2. Compute halt weights (soft halting with temperature)
      3. Re-read memory from updated hidden states for next step

    Returns (weighted_logits, expected_steps, halt_counts, final_hidden).
    """
    B, T = inp.shape
    device = inp.device
    HALT = 1

    remaining = torch.ones(B, T, device=device)
    weighted_logits = None
    expected_steps = torch.zeros(B, T, device=device)
    halt_step_counts = Counter()
    final_hidden = None

    for i in range(max_steps):
        logits, halt_logits, hidden = model(
            inp, memory_vectors=mem_tensor, return_hidden=True
        )
        final_hidden = hidden
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
        halt_step_counts[i + 1] += (halt_prob > 0.5).sum().item()

        # Re-read memory for next ACT step (addresses shift as hidden evolves)
        if i < max_steps - 1 and memory_system is not None:
            h_last = hidden[:, -1, :].detach()  # (B, d_model)
            mem_tensor = batch_read_memory(
                model, h_last, memory_system, cfg, device)

    return weighted_logits, expected_steps, halt_step_counts, final_hidden


# ---------------------------------------------------------------------------
# Multi-position memory writes within a sequence
# ---------------------------------------------------------------------------

def write_memory_multi_position(
    model: LoopedLatentController,
    hidden: torch.Tensor,
    memory_system: MemorySystem,
    cfg: ModelConfig,
    write_every: int = 64,
):
    """Write hidden states to memory at multiple positions within the sequence."""
    text_start = cfg.n_mem_positions
    B, T, D = hidden.shape
    text_len = T - text_start
    if text_len <= 0:
        return 0

    # Collect all write positions
    positions = list(range(text_start + write_every, T, write_every))
    last_pos = T - 1
    if not positions or positions[-1] != last_pos:
        positions.append(last_pos)

    n_writes = 0
    for pos in positions:
        h_batch = hidden[:, pos, :].detach()  # (B, d_model)
        vecs_np = h_batch.float().cpu().numpy()  # (B, d_model)
        addr_heads = model.compute_addresses_batch(h_batch)
        addr_cpu = [h.cpu().numpy() for h in addr_heads]
        for b in range(B):
            ab = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
            memory_system.write_memory(ab, vecs_np[b])
            n_writes += 1

    return n_writes


# ---------------------------------------------------------------------------
# Evaluation with streaming ACT + memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_streaming(
    model: LoopedLatentController,
    val_loader: DataLoader,
    eval_memory: Optional[MemorySystem],
    device: str,
    cfg: ModelConfig,
    max_act_steps: int = 4,
    temperature: float = 0.5,
    max_batches: int = 50,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (inp, tgt) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)

        # Read memory (batch vectorized)
        mem_tensor = None
        if eval_memory is not None:
            _, _, hid = model(inp, return_hidden=True)
            h_last = hid[:, -1, :]  # (B, d_model)
            mem_tensor = batch_read_memory(model, h_last, eval_memory, cfg, device)
            del hid, h_last  # free before ACT loop

        # Streaming ACT forward
        weighted_logits, _, _, _ = streaming_act_forward(
            model, inp, eval_memory, mem_tensor, max_act_steps, temperature, cfg
        )

        loss = F.cross_entropy(
            weighted_logits.reshape(-1, weighted_logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=cfg.pad_id,
            reduction="sum",
        )
        mask = tgt.reshape(-1) != cfg.pad_id
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        del weighted_logits, loss, mem_tensor

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    model.train()
    return {"loss": avg_loss, "perplexity": ppl}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    checkpoint_dir: str,
    data_dir: str,
    resume: bool = False,
    dataset_name: Optional[str] = None,
    text_column: str = "text",
    context_column: Optional[str] = None,
    phase_config: Optional[Phase5Config] = None,
    freeze_memory: bool = False,
    preseed_memory_dir: Optional[str] = None,
):
    hw = detect_hardware()
    print_hardware_report(hw)
    ov = hw['overrides']

    cfg = ModelConfig()
    pcfg = phase_config or Phase5Config()
    mcfg = MemoryConfig()
    device = hw['device']

    # Hardware-aware hyperparameters
    micro_batch = ov['phase5']['micro_batch']
    grad_accum = ov['phase5']['grad_accum']
    num_workers = ov['num_workers']
    pin_memory = ov['pin_memory']
    use_amp = ov['use_amp'] and device == 'cuda'
    amp_dtype = hw['amp_dtype']
    use_scaler = use_amp and amp_dtype == torch.float16
    use_ckpt = ov.get('gradient_checkpointing', True)

    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    phase3_dir = os.path.join(checkpoint_dir, "phase3")
    phase4_dir = os.path.join(checkpoint_dir, "phase4")
    phase5_dir = os.path.join(checkpoint_dir, "phase5")
    os.makedirs(phase5_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    ds_name = dataset_name or "roneneldan/TinyStories"
    train_ds, val_ds, tokenizer = prepare_data(
        tok_path, data_dir,
        seq_len=cfg.max_seq_len,
        dataset_name=ds_name,
        text_column=text_column,
        context_column=context_column,
    )

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=use_ckpt).to(device)

    # Load best checkpoint: Phase 4 > Phase 3 > Phase 1
    loaded_from = None
    for phase, name in [(phase4_dir, "Phase 4"), (phase3_dir, "Phase 3"),
                        (os.path.join(checkpoint_dir, "phase1"), "Phase 1")]:
        best_path = os.path.join(phase, "best.pt")
        if os.path.exists(best_path):
            load_checkpoint(model, None, best_path, device)
            loaded_from = name
            break
    if loaded_from:
        print(f"Loaded backbone from {loaded_from}")
    else:
        print("WARNING: No prior checkpoint found — training from scratch")

    # Ensure halt head has sensible initialization
    if hasattr(model, 'halt_head'):
        model.halt_head.bias.data[0] = -1.0
        model.halt_head.bias.data[1] = 1.0

    # Load Phase 2 address heads
    addr_heads_path = os.path.join(phase2_dir, "addr_heads.pt")
    if os.path.exists(addr_heads_path):
        ckpt = torch.load(addr_heads_path, map_location=device, weights_only=False)
        for i, sd in enumerate(ckpt["addr_heads"]):
            model.addr_heads[i].load_state_dict(sd)

    # Freeze address heads when memory is frozen — addresses must stay consistent
    # with the pre-seeded memory store. Unfreeze in Phase 5b when memory writes resume.
    if freeze_memory:
        for head in model.addr_heads:
            for p in head.parameters():
                p.requires_grad_(False)
        print("Loaded Phase 2 address heads (FROZEN — consistent with pre-seeded memory)")
    else:
        for head in model.addr_heads:
            for p in head.parameters():
                p.requires_grad_(True)
        print("Loaded Phase 2 address heads (UNFROZEN for Phase 5)")

    # ------------------------------------------------ auto-calibrate batch size
    max_curriculum_steps = max(s[1] for s in pcfg.ponder_curriculum)
    target_effective = micro_batch * grad_accum
    if device == 'cuda':
        trial = build_trial_fn(model, cfg, device, use_amp, amp_dtype,
                               has_memory=True, act_steps=max_curriculum_steps)
        micro_batch, grad_accum = auto_calibrate_batch_size(
            trial, device, micro_batch,
            target_effective=target_effective,
            target_vram_frac=0.90,
        )

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, micro_batch // 4), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # ----------------------------------------------------------- memory
    # Bootstrap from pre-seeded memory, Phase 3 memory, or start fresh
    train_mem_dir = os.path.join(phase5_dir, "train_memory")
    eval_mem_dir = os.path.join(phase5_dir, "eval_memory")

    if preseed_memory_dir and os.path.exists(preseed_memory_dir):
        if not os.path.exists(train_mem_dir):
            print(f"Bootstrapping memory from pre-seeded store: {preseed_memory_dir}")
            shutil.copytree(preseed_memory_dir, train_mem_dir)
    elif not os.path.exists(train_mem_dir):
        p3_mem_dir = os.path.join(phase3_dir, "train_memory")
        if os.path.exists(p3_mem_dir):
            print(f"Bootstrapping memory from Phase 3: {p3_mem_dir}")
            shutil.copytree(p3_mem_dir, train_mem_dir)
        else:
            os.makedirs(train_mem_dir, exist_ok=True)

    train_memory = MemorySystem(train_mem_dir, mcfg)

    if not os.path.exists(eval_mem_dir):
        shutil.copytree(train_mem_dir, eval_mem_dir)
    eval_memory = MemorySystem(eval_mem_dir, mcfg)

    if freeze_memory:
        print(f"  Memory FROZEN — no writes during training ({train_memory.total_entries():,} entries)")
    else:
        print(f"  Memory ACTIVE — writes enabled ({train_memory.total_entries():,} entries)")

    # --------------------------------------------------------- optimizer
    # Two param groups: main params (full LR) + address heads (0.3× LR)
    addr_params = set()
    for head in model.addr_heads:
        for p in head.parameters():
            addr_params.add(id(p))

    main_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in addr_params]
    addr_param_list = [p for p in model.parameters()
                       if p.requires_grad and id(p) in addr_params]

    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": pcfg.lr},
        {"params": addr_param_list, "lr": pcfg.lr * pcfg.addr_head_lr_mult},
    ], betas=pcfg.betas, weight_decay=pcfg.weight_decay)

    scaler = GradScaler(enabled=use_scaler)

    step = 0
    best_loss = float("inf")
    tokens_seen = 0

    tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step

    if resume:
        latest_path = os.path.join(phase5_dir, "latest.pt")
        if os.path.exists(latest_path):
            ckpt = load_checkpoint(model, optimizer, latest_path, device)
            step = ckpt.get("step", 0)
            tokens_seen = ckpt.get("tokens_seen", step * tokens_per_step)
            best_loss = ckpt.get("best_loss", float("inf"))
            if use_scaler and "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])

    total_tokens_fmt = f"{pcfg.total_tokens / 1e9:.2f}B"
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_label = (
        f"{'BF16' if amp_dtype == torch.bfloat16 else 'FP16'} (autocast + GradScaler)"
        if use_amp else "FP32"
    )

    print("=" * 64)
    print("  PHASE 5: Unified Streaming Training")
    print("=" * 64)
    print(f"  Model:           {total_params / 1e6:.1f}M parameters ({trainable_params / 1e6:.1f}M trainable)")
    print(f"  Dataset:         {ds_name}")
    if context_column:
        print(f"  Context Column:  {context_column} (NOOP targets enabled)")
    print(f"  Effective Batch: {micro_batch * grad_accum} (micro={micro_batch} × accum={grad_accum})")
    print(f"  Total Tokens:    {total_tokens_fmt} → {total_steps:,} optimizer steps")
    print(f"  Precision:       {amp_label}")
    print(f"  Address Head LR: {pcfg.lr * pcfg.addr_head_lr_mult:.2e} ({pcfg.addr_head_lr_mult}× base)")
    print(f"  Streaming ACT:   {'Enabled' if pcfg.streaming_act else 'Disabled'}")
    print(f"  Mem Write Every: {pcfg.write_every_n_positions} positions")
    if hw['gpu_name']:
        print(f"  GPU:             {hw['gpu_name']} ({hw['gpu_vram_gb']:.1f} GB VRAM)")
    print(f"  Gradient Ckpt:   {'Enabled' if use_ckpt else 'Disabled'}")
    print(f"  Loaded From:     {loaded_from or 'scratch'}")
    print("=" * 64)
    print(f"Total steps: {total_steps:,}")

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    halt_hist = Counter()
    total_mem_writes = 0

    try:
        while step < total_steps:
            # LR schedule — same schedule for both param groups (ratio preserved)
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * pcfg.addr_head_lr_mult

            max_act, ponder_w, temperature = get_curriculum(step, pcfg.ponder_curriculum)

            for micro_step in range(grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                try:
                    # Initial memory read (batch vectorized)
                    mem_tensor = None
                    with torch.no_grad():
                        _, _, hid_init = model(inp, return_hidden=True)
                        h_last = hid_init[:, -1, :]  # (B, d_model)
                        mem_tensor = batch_read_memory(
                            model, h_last, train_memory, cfg, device)

                    # Streaming ACT forward (re-reads memory between steps)
                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        if pcfg.streaming_act:
                            weighted_logits, expected_steps, counts, hidden = streaming_act_forward(
                                model, inp, train_memory, mem_tensor,
                                max_act, temperature, cfg,
                            )
                        else:
                            weighted_logits, expected_steps, counts, hidden = streaming_act_forward(
                                model, inp, None, mem_tensor,
                                max_act, temperature, cfg,
                            )
                        halt_hist.update(counts)

                        lm_loss = F.cross_entropy(
                            weighted_logits.reshape(-1, weighted_logits.size(-1)),
                            tgt.reshape(-1),
                            ignore_index=cfg.pad_id,
                        )
                        ponder_loss = (
                            ponder_w * expected_steps.mean()
                            if ponder_w > 0 else torch.zeros_like(lm_loss)
                        )
                        loss = (lm_loss + ponder_loss) / grad_accum

                    scaler.scale(loss).backward()
                    accum_loss += loss.item()
                    tokens_seen += inp.numel()

                    # Memory writes at multiple positions (skip if frozen)
                    if hidden is not None and not freeze_memory:
                        with torch.no_grad():
                            n_writes = write_memory_multi_position(
                                model, hidden.detach(), train_memory, cfg,
                                write_every=pcfg.write_every_n_positions,
                            )
                            total_mem_writes += n_writes

                except torch.cuda.OutOfMemoryError:
                    micro_batch, grad_accum, rebuild = handle_oom(
                        micro_batch, grad_accum, target_effective)
                    if rebuild:
                        train_loader = DataLoader(
                            train_ds, batch_size=micro_batch, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            drop_last=True, persistent_workers=num_workers > 0)
                        loader_iter = iter(train_loader)
                        tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
                        total_steps = pcfg.total_tokens // tokens_per_step
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    halt_hist.clear()
                    break

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Logging
            if step % pcfg.log_interval == 0:
                elapsed = timer.elapsed()
                pct = 100.0 * step / total_steps
                eta_secs = (total_steps - step) * (elapsed / max(step, 1))
                total_halt = sum(halt_hist.values())
                avg_halt = sum(k * v for k, v in halt_hist.items()) / max(total_halt, 1)
                tok_per_sec = tokens_seen / max(elapsed, 1e-6)
                print(
                    f"[Phase 5] Step {step}/{total_steps} ({pct:.1f}%)\n"
                    f"  Loss: {accum_loss:.4f} | Halt: {avg_halt:.1f} steps\n"
                    f"  max_s={max_act} pw={ponder_w:.3f} T={temperature:.2f}\n"
                    f"  MemWrites: {total_mem_writes:,}\n"
                    f"  VRAM: {vram_report()} | Peak: {peak_vram():.1f} GB\n"
                    f"  Tokens: {tokens_seen/1e9:.3f}B/{total_tokens_fmt} ({tok_per_sec:.0f} tok/s)\n"
                    f"  ETA: {format_eta(eta_secs)} | Elapsed: {format_time(elapsed)}",
                    flush=True,
                )
                accum_loss = 0.0
                halt_hist.clear()

            # Evaluation
            if step % pcfg.eval_interval == 0:
                # Free training VRAM before eval (ACT loop needs ~4× forward pass memory)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Refresh eval memory periodically
                if step % pcfg.eval_memory_refresh_interval == 0:
                    shutil.rmtree(eval_mem_dir, ignore_errors=True)
                    shutil.copytree(train_mem_dir, eval_mem_dir)
                    eval_memory = MemorySystem(eval_mem_dir, mcfg)
                    print(f"  Refreshed eval memory from train memory")

                metrics = evaluate_streaming(
                    model, val_loader, eval_memory, device, cfg,
                    max_act_steps=max_act, temperature=temperature,
                )
                print(
                    f"  [Eval] Loss: {metrics['loss']:.4f} | "
                    f"PPL: {metrics['perplexity']:.2f} | "
                    f"Best: {best_loss:.4f}"
                )

                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    save_checkpoint(
                        model, optimizer, step, best_loss,
                        os.path.join(phase5_dir, "best.pt"),
                        extra={
                            "tokens_seen": tokens_seen,
                            "best_loss": best_loss,
                            "scaler": scaler.state_dict() if use_scaler else None,
                            "addr_heads": [h.state_dict() for h in model.addr_heads],
                        },
                    )

            # Save checkpoint
            if step % pcfg.save_interval == 0:
                train_memory.flush_to_disk()
                save_checkpoint(
                    model, optimizer, step, accum_loss,
                    os.path.join(phase5_dir, "latest.pt"),
                    extra={
                        "tokens_seen": tokens_seen,
                        "best_loss": best_loss,
                        "scaler": scaler.state_dict() if use_scaler else None,
                        "addr_heads": [h.state_dict() for h in model.addr_heads],
                    },
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving…")
        train_memory.flush_to_disk()
        save_checkpoint(
            model, optimizer, step, accum_loss,
            os.path.join(phase5_dir, "latest.pt"),
            extra={
                "tokens_seen": tokens_seen,
                "best_loss": best_loss,
                "scaler": scaler.state_dict() if use_scaler else None,
                "addr_heads": [h.state_dict() for h in model.addr_heads],
            },
        )

    train_memory.flush_to_disk()
    wall_time = timer.elapsed()
    print("=" * 64)
    print("  PHASE 5 COMPLETE")
    print("=" * 64)
    print(f"  Total Steps:     {step:,}")
    print(f"  Total Tokens:    {tokens_seen/1e9:.2f}B")
    print(f"  Memory Writes:   {total_mem_writes:,}")
    print(f"  Best Eval Loss:  {best_loss:.4f}")
    print(f"  Wall Time:       {format_time(wall_time)}")
    avg_tps = tokens_seen / max(wall_time, 1)
    print(f"  Avg Throughput:  {avg_tps/1e3:.1f}k tok/s")
    print(f"  Peak VRAM:       {peak_vram():.1f} GB")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--context_column", default=None)
    parser.add_argument("--freeze_memory", action="store_true",
                        help="Don't write to memory during training (read-only)")
    parser.add_argument("--preseed_memory_dir", default=None,
                        help="Path to pre-seeded memory store to bootstrap from")
    args = parser.parse_args()
    train(
        args.checkpoint_dir, args.data_dir, args.resume,
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        context_column=args.context_column,
        freeze_memory=args.freeze_memory,
        preseed_memory_dir=args.preseed_memory_dir,
    )
