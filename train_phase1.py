"""
Phase 1 — Baseline language model training on TinyStories.

No memory, no ACT.  Pure next-token prediction.
"""

import os
import math
import argparse

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import ModelConfig, Phase1Config
from hardware import detect_hardware, print_hardware_report
from model import LoopedLatentController
from dataset import prepare_data, train_tokenizer_from_tinystories
from utils import (
    get_lr, save_checkpoint, load_checkpoint, evaluate, generate_sample,
    Timer, vram_report, peak_vram, format_eta, format_time,
)


def train(checkpoint_dir: str, data_dir: str, resume: bool = False):
    hw  = detect_hardware()
    print_hardware_report(hw)
    ov  = hw['overrides']

    cfg  = ModelConfig()
    pcfg = Phase1Config()
    device = hw['device']

    # Hardware-aware hyperparameters
    micro_batch = ov['phase1']['micro_batch']
    grad_accum  = ov['phase1']['grad_accum']
    num_workers = ov['num_workers']
    pin_memory  = ov['pin_memory']
    use_amp     = ov['use_amp'] and device == 'cuda'
    amp_dtype   = hw['amp_dtype']
    use_scaler  = use_amp and amp_dtype == torch.float16  # bf16 doesn't need loss scaling
    use_compile = ov.get('use_compile', False) and hasattr(torch, 'compile')
    use_ckpt    = ov.get('gradient_checkpointing', True)

    save_dir = os.path.join(checkpoint_dir, "phase1")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        train_tokenizer_from_tinystories(tok_path, vocab_size=16384)

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
    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if use_compile:
        print("Compiling model with torch.compile (this takes ~30-60s on first batch)...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pcfg.lr,
        betas=pcfg.betas,
        weight_decay=pcfg.weight_decay,
    )
    scaler = GradScaler(enabled=use_scaler)

    # ---------------------------------------------------------------- state
    step = 0
    best_loss = float("inf")
    tokens_seen = 0

    if resume:
        latest_path = os.path.join(save_dir, "latest.pt")
        if os.path.exists(latest_path):
            step = load_checkpoint(model, optimizer, latest_path, device)

    # Compute total steps
    tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step
    total_tokens_fmt = f"{pcfg.total_tokens / 1e9:.2f}B"

    amp_label = f"{'BF16' if amp_dtype == torch.bfloat16 else 'FP16'} (autocast + GradScaler)" if use_amp else "FP32"

    print("=" * 64)
    print("  PHASE 1: Baseline Language Model Training")
    print("=" * 64)
    print(f"  Model:           {total_params / 1e6:.1f}M parameters ({trainable_params / 1e6:.1f}M trainable)")
    print(f"  Dataset:         {len(train_ds):,} train / {len(val_ds):,} val samples")
    print(f"  Effective Batch: {micro_batch * grad_accum} (micro={micro_batch} × accum={grad_accum})")
    print(f"  Sequence Length: {cfg.max_seq_len}")
    print(f"  Total Tokens:    {total_tokens_fmt} → {total_steps:,} optimizer steps")
    print(f"  Optimizer:       AdamW (lr={pcfg.lr:.1e} → {pcfg.min_lr:.1e}, warmup={pcfg.warmup_steps})")
    print(f"  Precision:       {amp_label}")
    if hw['gpu_name']:
        print(f"  GPU:             {hw['gpu_name']} ({hw['gpu_vram_gb']:.1f} GB VRAM)")
    print(f"  Gradient Ckpt:   {'Enabled' if use_ckpt else 'Disabled'}")
    print(f"  torch.compile:   {'Enabled' if use_compile else 'Disabled'}")
    print("=" * 64)

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    try:
        while step < total_steps:
            # Update LR
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            for micro_step in range(grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    logits, _ = model(inp)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        tgt.reshape(-1),
                        ignore_index=cfg.pad_id,
                    )
                    loss = loss / grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens_seen += inp.numel()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Logging
            if step % pcfg.log_interval == 0:
                elapsed = timer.elapsed()
                dt = timer.lap()
                tps = tokens_per_step * pcfg.log_interval / max(dt, 1e-6)
                pct = 100.0 * step / total_steps
                ppl = math.exp(min(accum_loss, 100))
                steps_remaining = total_steps - step
                eta_secs = steps_remaining * (elapsed / max(step, 1))
                print(
                    f"[Phase 1] Step {step}/{total_steps} ({pct:.1f}%) | "
                    f"Loss: {accum_loss:.4f} | PPL: {ppl:.2f} | "
                    f"LR: {lr:.2e} | Throughput: {tps/1e3:.1f}k tok/s | "
                    f"VRAM: {vram_report()} | ETA: {format_eta(eta_secs)} | "
                    f"Elapsed: {format_time(elapsed)} | "
                    f"Tokens: {tokens_seen/1e9:.3f}B/{total_tokens_fmt}"
                )
                accum_loss = 0.0

            # Evaluation
            if step % pcfg.eval_interval == 0:
                metrics = evaluate(model, val_loader, device, amp_dtype=amp_dtype if use_amp else None)
                print(
                    f"  [eval] val_loss={metrics['loss']:.4f}  "
                    f"ppl={metrics['perplexity']:.2f}"
                )

                # Sample generation
                bos_id = tokenizer.token_to_id("<bos>") or cfg.bos_id
                sample = generate_sample(model, tokenizer, [bos_id], device=device)
                print(f"  [sample] {sample[:200]}")

                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    save_checkpoint(
                        model, optimizer, step, best_loss,
                        os.path.join(save_dir, "best.pt"),
                    )
                model.train()

            # Periodic checkpoint
            if step % pcfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, accum_loss,
                    os.path.join(save_dir, "latest.pt"),
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint…")
        save_checkpoint(
            model, optimizer, step, accum_loss,
            os.path.join(save_dir, "latest.pt"),
        )

    wall_time = timer.elapsed()
    print("=" * 64)
    print("  PHASE 1 COMPLETE")
    print("=" * 64)
    print(f"  Total Steps:     {step:,}")
    print(f"  Total Tokens:    {tokens_seen/1e9:.2f}B")
    print(f"  Best Val Loss:   {best_loss:.3f} (PPL {math.exp(min(best_loss, 100)):.2f})")
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
