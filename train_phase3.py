"""
Phase 3 — Memory integration training.

Start from Phase 1 best + frozen Phase 2 address heads.
Memory builds live during training.
"""

import os
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, Phase3Config, MemoryConfig
from model import LoopedLatentController
from memory import MemorySystem
from dataset import prepare_data
from utils import get_lr, save_checkpoint, load_checkpoint, evaluate, generate_sample, Timer


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


# ---------------------------------------------------------------------------
# Evaluation with and without memory
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_memory(
    model: LoopedLatentController,
    val_loader: DataLoader,
    eval_memory: MemorySystem,
    device: str,
    max_batches: int = 50,
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

        # --- without memory ---
        logits_nm, _ = model(inp)
        loss_nm = F.cross_entropy(
            logits_nm.reshape(-1, logits_nm.size(-1)),
            tgt.reshape(-1),
            ignore_index=cfg.pad_id,
            reduction="sum",
        )
        total_loss_no_mem += loss_nm.item()

        # --- with memory (use last token's hidden from no-mem pass) ---
        _, _, hid = model(inp, return_hidden=True)
        # Take [0, -1, :] as representative hidden state for address computation
        h = hid[0, -1, :]  # (d_model,)
        addrs = model.compute_addresses(h)
        addr_bs = [addr_bytes(a) for a in addrs]
        mem_vecs = eval_memory.read_memory(addr_bs)
        mem_tensor = memory_vecs_to_tensor(mem_vecs, cfg.d_model, device)

        logits_m, _ = model(inp, memory_vectors=mem_tensor.expand(inp.size(0), -1, -1))
        loss_m = F.cross_entropy(
            logits_m.reshape(-1, logits_m.size(-1)),
            tgt.reshape(-1),
            ignore_index=cfg.pad_id,
            reduction="sum",
        )
        total_loss_mem += loss_m.item()

        mask = tgt.reshape(-1) != cfg.pad_id
        total_tokens += mask.sum().item()

    denom = max(1, total_tokens)
    import math
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
    cfg   = ModelConfig()
    pcfg  = Phase3Config()
    mcfg  = MemoryConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 3 training — device: {device}")

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

    # Load Phase 1 best
    best_p1 = os.path.join(phase1_dir, "best.pt")
    if os.path.exists(best_p1):
        load_checkpoint(model, None, best_p1, device)
    else:
        print("WARNING: No Phase 1 checkpoint — random weights")

    # Load + freeze Phase 2 address heads
    addr_heads_path = os.path.join(phase2_dir, "addr_heads.pt")
    if os.path.exists(addr_heads_path):
        ckpt = torch.load(addr_heads_path, map_location=device)
        for i, sd in enumerate(ckpt["addr_heads"]):
            model.addr_heads[i].load_state_dict(sd)
        print("Loaded Phase 2 address heads")
    else:
        print("WARNING: No Phase 2 address heads found")

    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(False)

    # Memory systems
    train_memory = MemorySystem(train_mem_dir, mcfg)
    eval_memory  = MemorySystem(eval_mem_dir,  mcfg)

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
        latest_path = os.path.join(phase3_dir, "latest.pt")
        if os.path.exists(latest_path):
            step = load_checkpoint(model, optimizer, latest_path, device)

    tokens_per_step = pcfg.micro_batch * pcfg.grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,}")

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad()
    accum_loss = 0.0

    try:
        while step < total_steps:
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            for micro_step in range(pcfg.grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp, tgt = inp.to(device), tgt.to(device)

                # Read memory for this batch (use zeros-based lookup the first pass)
                with torch.no_grad():
                    _, _, hid_nm = model(inp, return_hidden=True)
                    h0 = hid_nm[0, -1, :]
                    addrs = model.compute_addresses(h0)
                    addr_bs = [addr_bytes(a) for a in addrs]
                    mem_vecs = train_memory.read_memory(addr_bs)
                    mem_tensor = memory_vecs_to_tensor(mem_vecs, cfg.d_model, device)
                    # Expand to batch size
                    mem_tensor = mem_tensor.expand(inp.size(0), -1, -1)

                logits, _, hidden = model(inp, memory_vectors=mem_tensor, return_hidden=True)

                # LM loss on text positions only (memory positions excluded)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=cfg.pad_id,
                )
                (loss / pcfg.grad_accum).backward()
                accum_loss += loss.item() / pcfg.grad_accum
                tokens_seen += inp.numel()

            torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # Write to memory every N steps
            if step % pcfg.write_every_n_steps == 0:
                with torch.no_grad():
                    # Use a position from the middle of the text sequence as a
                    # representative hidden state (avoids edge effects at BOS/EOS).
                    mid_text_pos = cfg.n_mem_positions + hidden.shape[1] // 2
                    for b in range(hidden.shape[0]):
                        h = hidden[b, mid_text_pos, :]
                        vec_np = h.detach().float().cpu().numpy()
                        addrs_b = model.compute_addresses(h.detach())
                        addr_bs_b = [addr_bytes(a) for a in addrs_b]
                        train_memory.write_memory(addr_bs_b, vec_np)

            if step % 500 == 0:
                dt = timer.lap()
                print(
                    f"step {step:6d}/{total_steps}  loss={accum_loss:.4f}  "
                    f"lr={lr:.2e}  tokens={tokens_seen/1e9:.3f}B"
                )
                accum_loss = 0.0

            if step % pcfg.eval_interval == 0:
                # Refresh eval memory snapshot
                if step % pcfg.eval_memory_refresh_interval == 0:
                    shutil.rmtree(eval_mem_dir, ignore_errors=True)
                    shutil.copytree(train_mem_dir, eval_mem_dir)
                    eval_memory = MemorySystem(eval_mem_dir, mcfg)

                metrics = evaluate_with_memory(model, val_loader, eval_memory, device)
                print(
                    f"  [eval] no_mem ppl={metrics['ppl_no_mem']:.2f}  "
                    f"mem ppl={metrics['ppl_mem']:.2f}"
                )

                if metrics["loss_mem"] < best_loss:
                    best_loss = metrics["loss_mem"]
                    save_checkpoint(
                        model, optimizer, step, best_loss,
                        os.path.join(phase3_dir, "best.pt"),
                    )
                model.train()

            if step % pcfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, accum_loss,
                    os.path.join(phase3_dir, "latest.pt"),
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving…")
        save_checkpoint(
            model, optimizer, step, accum_loss,
            os.path.join(phase3_dir, "latest.pt"),
        )

    print(f"Phase 3 complete.  Steps: {step}  Tokens: {tokens_seen/1e9:.2f}B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir, args.resume)
