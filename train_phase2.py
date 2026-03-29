"""
Phase 2 — Address head contrastive pretraining.

Freeze the Phase 1 model; train only the 3 address heads.
"""

import os
import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, Phase2Config
from hardware import detect_hardware, print_hardware_report, handle_oom
from model import LoopedLatentController
from dataset import prepare_data
from utils import load_checkpoint, save_checkpoint, Timer, vram_report, peak_vram, format_eta, format_time


# ---------------------------------------------------------------------------
# Collect hidden states
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_hidden_states(
    model: LoopedLatentController,
    train_loader: DataLoader,
    n_sequences: int,
    positions_per_seq: int,
    device: str,
) -> torch.Tensor:
    """
    Returns (N, d_model) float32 tensor of hidden states.
    N ≤ n_sequences × positions_per_seq.
    """
    model.eval()
    collected = []
    seq_count = 0

    for inp, _ in train_loader:
        if seq_count >= n_sequences:
            break
        inp = inp.to(device, non_blocking=True)
        _, _, hidden = model(inp, return_hidden=True)
        # hidden: (B, T, d_model) — take text positions only
        # (no memory in phase 1, so all positions are text)
        B, T, D = hidden.shape
        stride = max(1, T // positions_per_seq)
        idxs = list(range(0, T, stride))[:positions_per_seq]
        for b in range(B):
            for i in idxs:
                collected.append(hidden[b, i].cpu())
        seq_count += B
        if len(collected) % 50000 < B * len(idxs):
            print(f"  Collected {len(collected):,} hidden states…")

    model.train()
    return torch.stack(collected[:n_sequences * positions_per_seq])


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(
    addr_heads,
    hidden_batch: torch.Tensor,
    cos_sims: torch.Tensor,
    pos_threshold: float,
    neg_threshold: float,
    margin: float,
    entropy_weight: float,
    target_dim_std: float,
    device: str,
) -> torch.Tensor:
    """
    hidden_batch : (B, d_model)
    cos_sims     : (B, B) pairwise cosine similarities of hidden states
    """
    total_loss = torch.tensor(0.0, device=device)

    for head in addr_heads:
        raw = head(hidden_batch)                            # (B, addr_dim)

        # L2-normalised address vectors for distance computation
        addr_norm = F.normalize(raw.float(), dim=-1)
        addr_dist = torch.cdist(addr_norm, addr_norm, p=2) # (B, B)

        pos_mask = (cos_sims > pos_threshold) & (cos_sims < 1.0 - 1e-6)
        neg_mask = (cos_sims < neg_threshold)

        # Pull positives together
        if pos_mask.sum() > 0:
            total_loss += addr_dist[pos_mask].mean()

        # Push negatives apart
        if neg_mask.sum() > 0:
            push = F.relu(margin - addr_dist[neg_mask])
            total_loss += push.mean()

        # Entropy regularisation: encourage spread across dimensions
        dim_std = raw.std(dim=0)                            # (addr_dim,)
        entropy_reg = F.relu(target_dim_std - dim_std).mean()
        total_loss += entropy_weight * entropy_reg

    return total_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(checkpoint_dir: str, data_dir: str, resume: bool = False):
    hw  = detect_hardware()
    print_hardware_report(hw)
    ov  = hw['overrides']

    cfg  = ModelConfig()
    pcfg = Phase2Config()
    device = hw['device']

    # Hardware-aware batch size
    batch_size  = ov['phase2']['batch_size']
    num_workers = ov['num_workers']
    pin_memory  = ov['pin_memory']

    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    print("=" * 64)
    print("  PHASE 2: Address Head Contrastive Pretraining")
    print("=" * 64)
    print(f"  Steps:           {pcfg.steps:,}")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Hidden States:   {pcfg.n_hidden_states:,} from {pcfg.n_sequences:,} sequences")
    if hw['gpu_name']:
        print(f"  GPU:             {hw['gpu_name']} ({hw['gpu_vram_gb']:.1f} GB VRAM)")
    print("=" * 64)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_ds, _, _ = prepare_data(tok_path, data_dir, seq_len=cfg.max_seq_len)
    train_loader   = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)

    # Load Phase 1 best checkpoint
    best_p1 = os.path.join(phase1_dir, "best.pt")
    if os.path.exists(best_p1):
        load_checkpoint(model, None, best_p1, device)
    else:
        print("WARNING: No Phase 1 checkpoint found — using random weights")
    for p in model.parameters():
        p.requires_grad_(False)
    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for head in model.addr_heads for p in head.parameters()],
        lr=pcfg.lr,
    )

    # Collect hidden states
    print(f"Collecting {pcfg.n_hidden_states:,} hidden states from {pcfg.n_sequences:,} sequences…")
    hiddens = collect_hidden_states(
        model, train_loader,
        n_sequences=pcfg.n_sequences,
        positions_per_seq=pcfg.positions_per_seq,
        device=device,
    )
    print(f"Collected {hiddens.shape[0]:,} × {hiddens.shape[1]}-d hidden states")

    # Pre-normalize on CPU (cheap); cosine sims computed on-the-fly per batch
    hiddens_norm = F.normalize(hiddens.float(), dim=-1)  # (N, d_model) on CPU
    hiddens = hiddens.to(device)

    # Auto-calibrate batch_size for contrastive training
    if device == 'cuda':
        from hardware import auto_calibrate_batch_size

        def _trial_p2(bs):
            idx_t = torch.randperm(N, device=device)[:bs]
            b = hiddens[idx_t]
            bn = hiddens_norm[idx_t.cpu()].to(device)
            sim = torch.mm(bn, bn.T)
            loss = contrastive_loss(
                model.addr_heads, b, sim,
                pcfg.pos_threshold, pcfg.neg_threshold, pcfg.margin,
                pcfg.entropy_weight, pcfg.target_dim_std, device,
            )
            loss.backward()
            model.zero_grad(set_to_none=True)

        batch_size, _ = auto_calibrate_batch_size(
            _trial_p2, device, batch_size,
            target_effective=batch_size,  # no grad_accum in P2
            target_vram_frac=0.90,
        )
        print(f"  Phase 2 calibrated batch_size: {batch_size}")

    # Training loop
    N = hiddens.shape[0]
    model.train()
    timer = Timer()
    start_step = 0

    # Resume from a partially-completed Phase 2 run if available
    latest_p2 = os.path.join(phase2_dir, "latest.pt")
    if resume and os.path.exists(latest_p2):
        ckpt = load_checkpoint(model, optimizer, latest_p2, device)
        start_step = ckpt.get("step", 0)
        print(f"Resuming Phase 2 from step {start_step}")

    for step in range(start_step + 1, pcfg.steps + 1):
        # Random mini-batch
        idx = torch.randperm(N, device=device)[:batch_size]
        batch = hiddens[idx]

        # Compute cosine similarity on-the-fly for this mini-batch only: (B, B)
        batch_norm = hiddens_norm[idx.cpu()].to(device)  # (B, d_model)
        sim_batch = torch.mm(batch_norm, batch_norm.T)   # (B, B) — fits in VRAM

        try:
            loss = contrastive_loss(
                model.addr_heads,
                batch,
                sim_batch,
                pcfg.pos_threshold,
                pcfg.neg_threshold,
                pcfg.margin,
                pcfg.entropy_weight,
                pcfg.target_dim_std,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(256, batch_size // 2)
            print(f"  ⚠ OOM! Reduced batch_size to {batch_size}")
            optimizer.zero_grad(set_to_none=True)
            continue

        if step % 100 == 0:
            elapsed = timer.elapsed()
            pct = 100.0 * step / pcfg.steps
            eta_secs = (pcfg.steps - step) * (elapsed / max(step - start_step, 1))
            print(
                f"[Phase 2] Step {step}/{pcfg.steps} ({pct:.1f}%) | "
                f"Loss: {loss.item():.4f} | "
                f"VRAM: {vram_report()} | ETA: {format_eta(eta_secs)} | "
                f"Elapsed: {format_time(elapsed)}"
            )

        # Periodic checkpoint for resume support
        if step % 1000 == 0:
            save_checkpoint(
                model, optimizer, step, loss.item(),
                latest_p2,
                extra={"addr_heads": [h.state_dict() for h in model.addr_heads]},
            )

    wall_time = timer.elapsed()
    # Save address heads only
    save_path = os.path.join(phase2_dir, "addr_heads.pt")
    torch.save(
        {
            "addr_heads": [h.state_dict() for h in model.addr_heads],
            "step": pcfg.steps,
        },
        save_path,
    )

    print("=" * 64)
    print("  PHASE 2 COMPLETE")
    print("=" * 64)
    print(f"  Total Steps:     {pcfg.steps:,}")
    print(f"  Wall Time:       {format_time(wall_time)}")
    print(f"  Peak VRAM:       {peak_vram():.1f} GB")
    print(f"  Address heads saved → {save_path}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir, args.resume)
