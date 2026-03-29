"""
Phase 3 — Memory integration training.

Start from Phase 1 best + frozen Phase 2 address heads.
Memory builds live during training.
"""

import os
import argparse
import math
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from config import ModelConfig, Phase3Config, MemoryConfig
from hardware import (
    detect_hardware, print_hardware_report,
    auto_calibrate_batch_size, build_trial_fn, handle_oom,
)
from model import LoopedLatentController
from memory import MemorySystem
from dataset import prepare_data
from utils import (
    get_lr, save_checkpoint, load_checkpoint, evaluate, generate_sample,
    Timer, vram_report, peak_vram, format_eta, format_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hidden_to_int8(hidden: torch.Tensor) -> np.ndarray:
    """Scale a (d_model,) float tensor to int8 numpy array."""
    arr = hidden.detach().float().cpu().numpy()
    scale = np.abs(arr).max()
    if scale < 1e-6:
        return np.zeros(arr.shape, dtype=np.int8)
    return np.clip(np.round(arr / scale * 127.0), -128, 127).astype(np.int8)


def memory_vecs_to_tensor(
    vecs: list,
    d_model: int,
    device,
) -> torch.Tensor:
    """
    Convert list of int8 numpy arrays (each (512,)) to float tensor (1, n, d_model).
    """
    n = len(vecs)
    out = np.stack(vecs, axis=0).astype(np.float32) / 127.0  # (n, 512)
    return torch.from_numpy(out).unsqueeze(0).to(device)     # (1, n, d_model)


def addr_bytes(addr_tensor: torch.Tensor) -> bytes:
    """Convert int8 torch tensor → bytes for TrieIndex."""
    return addr_tensor.cpu().numpy().tobytes()


def batch_read_memory(model, hidden_states, memory, cfg, device):
    """
    Vectorized memory read for a batch.
    hidden_states: (B, d_model) tensor — last hidden position per sample.
    Returns: (B, n_mem_slots, d_model) float tensor on device.
    """
    B = hidden_states.size(0)
    # Batch address computation — 3 heads, each (B, addr_dim) on GPU
    addr_heads = model.compute_addresses_batch(hidden_states)
    # Transfer to CPU once, convert to bytes
    addr_cpu = [h.cpu().numpy() for h in addr_heads]  # 3 × (B, addr_dim)
    batch_addresses = []
    for b in range(B):
        sample_addrs = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
        batch_addresses.append(sample_addrs)
    # Batch read: returns (B, n_mem_slots, 512) int8 numpy
    mem_np = memory.read_memory_batch(batch_addresses)
    # Convert to float tensor on device in one shot
    mem_tensor = torch.from_numpy(
        mem_np.astype(np.float32) / 127.0
    ).to(device, non_blocking=True)  # (B, n_mem_slots, d_model)
    return mem_tensor


# ---------------------------------------------------------------------------
# Evaluation with and without memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_memory(
    model: LoopedLatentController,
    val_loader: DataLoader,
    eval_memory: MemorySystem,
    device: str,
    max_batches: int = 20,
) -> dict:
    model.eval()
    cfg = model.cfg

    total_loss_no_mem  = 0.0
    total_loss_mem     = 0.0
    total_tokens       = 0

    for batch_idx, (inp, tgt) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        inp, tgt = inp.to(device), tgt.to(device)

        try:
            # --- without memory ---
            logits_nm, _ = model(inp)
            loss_nm = F.cross_entropy(
                logits_nm.reshape(-1, logits_nm.size(-1)),
                tgt.reshape(-1),
                ignore_index=cfg.pad_id,
                reduction="sum",
            )
            total_loss_no_mem += loss_nm.item()
            del logits_nm, loss_nm

            # --- with memory (batch lookup) ---
            _, _, hid = model(inp, return_hidden=True)
            h_last = hid[:, -1, :]  # (B, d_model)
            mem_tensor = batch_read_memory(model, h_last, eval_memory, cfg, device)
            del hid, h_last

            logits_m, _ = model(inp, memory_vectors=mem_tensor)
            loss_m = F.cross_entropy(
                logits_m.reshape(-1, logits_m.size(-1)),
                tgt.reshape(-1),
                ignore_index=cfg.pad_id,
                reduction="sum",
            )
            total_loss_mem += loss_m.item()
            del logits_m, loss_m, mem_tensor

            mask = tgt.reshape(-1) != cfg.pad_id
            total_tokens += mask.sum().item()

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    denom = max(1, total_tokens)
    nm_loss = total_loss_no_mem / denom
    m_loss  = total_loss_mem  / denom
    model.train()
    return {
        "loss_no_mem":  nm_loss,
        "ppl_no_mem":   math.exp(min(nm_loss, 100)),
        "loss_mem":     m_loss,
        "ppl_mem":      math.exp(min(m_loss, 100)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(checkpoint_dir: str, data_dir: str, resume: bool = False):
    hw   = detect_hardware()
    print_hardware_report(hw)
    ov   = hw['overrides']

    cfg   = ModelConfig()
    pcfg  = Phase3Config()
    mcfg  = MemoryConfig()
    device = hw['device']

    # Hardware-aware hyperparameters
    micro_batch = ov['phase3']['micro_batch']
    grad_accum  = ov['phase3']['grad_accum']
    num_workers = ov['num_workers']
    pin_memory  = ov['pin_memory']
    use_amp     = ov['use_amp'] and device == 'cuda'
    amp_dtype   = hw['amp_dtype']
    use_scaler  = use_amp and amp_dtype == torch.float16
    use_ckpt    = ov.get('gradient_checkpointing', True)

    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    phase3_dir = os.path.join(checkpoint_dir, "phase3")
    os.makedirs(phase3_dir, exist_ok=True)

    train_mem_dir = os.path.join(phase3_dir, "train_memory")
    eval_mem_dir  = os.path.join(phase3_dir, "eval_memory")
    os.makedirs(train_mem_dir, exist_ok=True)
    os.makedirs(eval_mem_dir,  exist_ok=True)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_ds, val_ds, tokenizer = prepare_data(tok_path, data_dir, seq_len=cfg.max_seq_len)

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=use_ckpt).to(device)

    # Load Phase 1 best
    best_p1 = os.path.join(phase1_dir, "best.pt")
    if os.path.exists(best_p1):
        load_checkpoint(model, None, best_p1, device)
    else:
        print("WARNING: No Phase 1 checkpoint — random weights")
    addr_heads_path = os.path.join(phase2_dir, "addr_heads.pt")
    if os.path.exists(addr_heads_path):
        ckpt = torch.load(addr_heads_path, map_location=device, weights_only=False)
        for i, sd in enumerate(ckpt["addr_heads"]):
            model.addr_heads[i].load_state_dict(sd)
        print("Loaded Phase 2 address heads")
    else:
        print("WARNING: No Phase 2 address heads found")

    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------ auto-calibrate batch size
    target_effective = micro_batch * grad_accum
    if device == 'cuda':
        trial = build_trial_fn(model, cfg, device, use_amp, amp_dtype,
                               has_memory=True, act_steps=1)
        micro_batch, grad_accum = auto_calibrate_batch_size(
            trial, device, micro_batch,
            target_effective=target_effective,
            target_vram_frac=0.90,
        )

    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    # Eval uses a smaller batch to avoid OOM (3 forward passes per batch)
    eval_batch = max(1, micro_batch // 4)
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # Memory systems
    train_memory = MemorySystem(train_mem_dir, mcfg)
    eval_memory  = MemorySystem(eval_mem_dir,  mcfg)

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

    # Compute total steps (needed before resume for tokens_seen estimate)
    tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step

    if resume:
        latest_path = os.path.join(phase3_dir, "latest.pt")
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
    amp_label = f"{'BF16' if amp_dtype == torch.bfloat16 else 'FP16'} (autocast + GradScaler)" if use_amp else "FP32"

    print("=" * 64)
    print("  PHASE 3: Memory Integration Training")
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
    accum_loss = 0.0

    try:
        while step < total_steps:
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            for micro_step in range(grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp = inp.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                try:
                    # Batch memory read: one forward pass for addresses, vectorized lookup
                    with torch.no_grad():
                        _, _, hid_nm = model(inp, return_hidden=True)
                        h_last = hid_nm[:, -1, :]  # (B, d_model)
                        mem_tensor = batch_read_memory(
                            model, h_last, train_memory, cfg, device)

                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        logits, _, hidden = model(inp, memory_vectors=mem_tensor, return_hidden=True)

                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            tgt.reshape(-1),
                            ignore_index=cfg.pad_id,
                        )
                        loss = loss / grad_accum

                    scaler.scale(loss).backward()
                    accum_loss += loss.item()
                    tokens_seen += inp.numel()

                except torch.cuda.OutOfMemoryError:
                    micro_batch, grad_accum, rebuild = handle_oom(
                        micro_batch, grad_accum, target_effective)
                    if rebuild:
                        train_loader = DataLoader(
                            train_ds, batch_size=micro_batch, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            drop_last=True, persistent_workers=num_workers > 0,
                            prefetch_factor=4 if num_workers > 0 else None)
                        loader_iter = iter(train_loader)
                        tokens_per_step = micro_batch * grad_accum * cfg.max_seq_len
                        total_steps = pcfg.total_tokens // tokens_per_step
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    break

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # Write to memory every N steps (batch vectorized)
            if step % pcfg.write_every_n_steps == 0:
                with torch.no_grad():
                    text_start = cfg.n_mem_positions
                    text_len = hidden.shape[1] - text_start
                    mid_text_pos = text_start + text_len // 2
                    h_mid = hidden[:, mid_text_pos, :]  # (B, d_model)
                    vecs_np = h_mid.detach().float().cpu().numpy()  # (B, d_model)
                    addr_heads = model.compute_addresses_batch(h_mid.detach())
                    addr_cpu = [h.cpu().numpy() for h in addr_heads]
                    for b in range(hidden.shape[0]):
                        addrs_b = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
                        train_memory.write_memory(addrs_b, vecs_np[b])

            if step % 500 == 0 or step == 1:
                log_interval = 1 if step == 1 else 500
                elapsed = timer.elapsed()
                pct = 100.0 * step / total_steps
                eta_secs = (total_steps - step) * (elapsed / max(step, 1))
                avg_loss = accum_loss / log_interval
                tok_per_sec = tokens_seen / max(elapsed, 1e-6)
                mem_entries = train_memory.total_entries() if hasattr(train_memory, 'total_entries') else -1
                print(
                    f"[Phase 3] Step {step}/{total_steps} ({pct:.1f}%)\n"
                    f"  Loss: {avg_loss:.4f} | LR: {lr:.2e}\n"
                    f"  VRAM: {vram_report()} | Peak: {peak_vram():.1f} GB\n"
                    f"  Tokens: {tokens_seen/1e9:.3f}B/{total_tokens_fmt} ({tok_per_sec:.0f} tok/s)\n"
                    f"  Memory entries: {mem_entries} | Batch: {micro_batch}×{grad_accum}\n"
                    f"  ETA: {format_eta(eta_secs)} | Elapsed: {format_time(elapsed)}",
                    flush=True,
                )
                accum_loss = 0.0

            if step % pcfg.eval_interval == 0:
                torch.cuda.empty_cache()
                # Refresh eval memory snapshot
                if step % pcfg.eval_memory_refresh_interval == 0:
                    train_memory.flush_to_disk()
                    shutil.rmtree(eval_mem_dir, ignore_errors=True)
                    shutil.copytree(train_mem_dir, eval_mem_dir)
                    eval_memory = MemorySystem(eval_mem_dir, mcfg)

                metrics = evaluate_with_memory(model, val_loader, eval_memory, device)
                print(
                    f"  [eval @ step {step}]\n"
                    f"    No-mem  loss={metrics['loss_no_mem']:.4f}  ppl={metrics['ppl_no_mem']:.2f}\n"
                    f"    W/mem   loss={metrics['loss_mem']:.4f}  ppl={metrics['ppl_mem']:.2f}\n"
                    f"    Δ ppl: {metrics['ppl_no_mem'] - metrics['ppl_mem']:+.2f} (positive = memory helps)",
                    flush=True,
                )

                if metrics["loss_mem"] < best_loss:
                    best_loss = metrics["loss_mem"]
                    save_checkpoint(
                        model, optimizer, step, best_loss,
                        os.path.join(phase3_dir, "best.pt"),
                        extra={
                            "tokens_seen": tokens_seen,
                            "best_loss": best_loss,
                            "scaler": scaler.state_dict() if use_scaler else None,
                        },
                    )
                model.train()

            if step % pcfg.save_interval == 0:
                train_memory.flush_to_disk()
                save_checkpoint(
                    model, optimizer, step, accum_loss,
                    os.path.join(phase3_dir, "latest.pt"),
                    extra={
                        "tokens_seen": tokens_seen,
                        "best_loss": best_loss,
                        "scaler": scaler.state_dict() if use_scaler else None,
                    },
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving…")
        train_memory.flush_to_disk()
        save_checkpoint(
            model, optimizer, step, accum_loss,
            os.path.join(phase3_dir, "latest.pt"),
            extra={
                "tokens_seen": tokens_seen,
                "best_loss": best_loss,
                "scaler": scaler.state_dict() if use_scaler else None,
            },
        )

    train_memory.flush_to_disk()
    wall_time = timer.elapsed()
    print("=" * 64)
    print("  PHASE 3 COMPLETE")
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
