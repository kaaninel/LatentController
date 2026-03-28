"""
Phase 1 — Baseline language model training on TinyStories.

No memory, no ACT.  Pure next-token prediction.
"""

import os
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, Phase1Config
from model import LoopedLatentController
from dataset import prepare_data, train_tokenizer_from_tinystories
from utils import get_lr, save_checkpoint, load_checkpoint, evaluate, generate_sample, Timer


def train(checkpoint_dir: str, data_dir: str, resume: bool = False):
    cfg  = ModelConfig()
    pcfg = Phase1Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Phase 1 training — device: {device}")

    save_dir = os.path.join(checkpoint_dir, "phase1")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        train_tokenizer_from_tinystories(tok_path, vocab_size=16384)

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
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pcfg.lr,
        betas=pcfg.betas,
        weight_decay=pcfg.weight_decay,
    )

    # ---------------------------------------------------------------- state
    step = 0
    best_loss = float("inf")
    tokens_seen = 0

    if resume:
        latest_path = os.path.join(save_dir, "latest.pt")
        if os.path.exists(latest_path):
            step = load_checkpoint(model, optimizer, latest_path, device)

    # Compute total steps
    tokens_per_step = pcfg.micro_batch * pcfg.grad_accum * cfg.max_seq_len
    total_steps = pcfg.total_tokens // tokens_per_step
    print(f"Total steps: {total_steps:,}  tokens/step: {tokens_per_step:,}")

    model.train()
    timer = Timer()
    loader_iter = iter(train_loader)
    optimizer.zero_grad()
    accum_loss = 0.0

    try:
        while step < total_steps:
            # Update LR
            lr = get_lr(step, pcfg.warmup_steps, total_steps, pcfg.lr, pcfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            for micro_step in range(pcfg.grad_accum):
                try:
                    inp, tgt = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_loader)
                    inp, tgt = next(loader_iter)

                inp, tgt = inp.to(device), tgt.to(device)
                logits, _ = model(inp)
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

            # Logging
            if step % pcfg.log_interval == 0:
                dt = timer.lap()
                tps = tokens_per_step * pcfg.log_interval / max(dt, 1e-6)
                print(
                    f"step {step:6d}/{total_steps}  loss={accum_loss:.4f}  "
                    f"lr={lr:.2e}  {tps/1e3:.1f}k tok/s  "
                    f"tokens={tokens_seen/1e9:.3f}B"
                )
                accum_loss = 0.0

            # Evaluation
            if step % pcfg.eval_interval == 0:
                metrics = evaluate(model, val_loader, device)
                print(f"  [eval] val_loss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.2f}")

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

    print(f"Phase 1 complete.  Steps: {step}  Tokens: {tokens_seen/1e9:.2f}B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir",       default="./data")
    parser.add_argument("--resume",         action="store_true")
    args = parser.parse_args()
    train(args.checkpoint_dir, args.data_dir, args.resume)
