"""
Phase 2 — Address head contrastive pretraining.

Freeze the Phase 1 model; train only the 3 address heads.
"""

import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, Phase2Config
from model import LoopedLatentController
from dataset import prepare_data
from utils import load_checkpoint


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
        inp = inp.to(device)
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

def train(checkpoint_dir: str, data_dir: str):
    cfg  = ModelConfig()
    pcfg = Phase2Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 2 training — device: {device}")

    phase1_dir = os.path.join(checkpoint_dir, "phase1")
    phase2_dir = os.path.join(checkpoint_dir, "phase2")
    os.makedirs(phase2_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    train_ds, _, _ = prepare_data(tok_path, data_dir, seq_len=cfg.max_seq_len)
    train_loader   = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    # ---------------------------------------------------------------- model
    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)

    # Load Phase 1 best checkpoint
    best_p1 = os.path.join(phase1_dir, "best.pt")
    if os.path.exists(best_p1):
        load_checkpoint(model, None, best_p1, device)
    else:
        print("WARNING: No Phase 1 checkpoint found — using random weights")

    # Freeze everything except address heads
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

    # Precompute pairwise cosine similarities (done in batches to save memory)
    print("Precomputing pairwise cosine similarities…")
    hiddens_norm = F.normalize(hiddens.float(), dim=-1)
    cos_sims = torch.mm(hiddens_norm, hiddens_norm.T)  # (N, N)
    # Move to device
    hiddens = hiddens.to(device)
    cos_sims = cos_sims.to(device)

    # Training loop
    N = hiddens.shape[0]
    model.train()

    for step in range(1, pcfg.steps + 1):
        # Random mini-batch
        idx = torch.randperm(N, device=device)[:pcfg.batch_size]
        batch = hiddens[idx]
        sim_batch = cos_sims[idx][:, idx]

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  step {step}/{pcfg.steps}  loss={loss.item():.4f}")

    # Save address heads only
    save_path = os.path.join(phase2_dir, "addr_heads.pt")
    torch.save(
        {
            "addr_heads": [h.state_dict() for h in model.addr_heads],
            "step": pcfg.steps,
        },
        save_path,
    )
    print(f"Phase 2 complete.  Address heads saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir)
