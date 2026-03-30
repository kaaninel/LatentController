"""
Prepare Gen 1 Memory QA dataset: bAbI + SQuAD + TinyStories replay.

Produces a pre-tokenized cache file compatible with train_phase5.py.
The cache file contains NOOP targets for context (absorb silently) and
real next-token targets for response (emit answer).

Format:
  bAbI:         context = passage + question,  text = answer
  SQuAD:        context = passage + question,  text = answer
  TinyStories:  context = "" (empty),           text = full story

Usage:
    python prepare_gen1_data.py
    python prepare_gen1_data.py --babi_samples 18000 --squad_samples 40000 --tinystories_samples 100000
"""

import os
import argparse
import random

import torch
from datasets import load_dataset

from tokenizer_utils import load_tokenizer, encode


# ---------------------------------------------------------------------------
# Dataset loaders — each returns list of (context: str, text: str) pairs
# ---------------------------------------------------------------------------

def load_babi(max_samples: int) -> list:
    """Load bAbI tasks (Muennighoff/babi) → (context, text) pairs."""
    print(f"  Loading bAbI (max {max_samples:,})...")
    ds = load_dataset("Muennighoff/babi", split="train", streaming=True)
    pairs = []
    task_counts = {}
    for item in ds:
        if len(pairs) >= max_samples:
            break
        passage = item["passage"].strip()
        question = item["question"].strip()
        answer = item["answer"].strip()
        if not passage or not question or not answer:
            continue
        context = f"{passage}\n{question}"
        pairs.append((context, answer))
        t = item.get("task", 0)
        task_counts[t] = task_counts.get(t, 0) + 1

    print(f"    {len(pairs):,} pairs across {len(task_counts)} tasks")
    for t in sorted(task_counts.keys()):
        print(f"      Task {t}: {task_counts[t]:,}")
    return pairs


def load_squad(max_samples: int) -> list:
    """Load SQuAD v1 → (context, text) pairs."""
    print(f"  Loading SQuAD (max {max_samples:,})...")
    ds = load_dataset("rajpurkar/squad", split="train", streaming=True)
    pairs = []
    for item in ds:
        if len(pairs) >= max_samples:
            break
        ctx = item["context"].strip()
        question = item["question"].strip()
        answers = item["answers"]["text"]
        if not ctx or not question or not answers:
            continue
        answer = answers[0].strip()
        context = f"{ctx}\n{question}"
        pairs.append((context, answer))

    print(f"    {len(pairs):,} pairs")
    return pairs


def load_tinystories_replay(max_samples: int) -> list:
    """Load TinyStories → (context="", text=story) pairs for replay."""
    print(f"  Loading TinyStories replay (max {max_samples:,})...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    pairs = []
    for item in ds:
        if len(pairs) >= max_samples:
            break
        text = item["text"].strip()
        if text:
            pairs.append(("", text))

    print(f"    {len(pairs):,} stories")
    return pairs


# ---------------------------------------------------------------------------
# Build NOOP-target sequences (same logic as dataset._build_noop_sequences)
# ---------------------------------------------------------------------------

def build_noop_sequences(pairs, tokenizer, seq_len, pad_id, bos_id, eos_id, noop_id):
    """
    Build input + target tensors from (context, text) pairs.

    Context tokens get NOOP targets (model absorbs silently).
    Response tokens get real next-token targets (model predicts/emits).
    """
    N = len(pairs)
    data = torch.full((N, seq_len), pad_id, dtype=torch.int32)
    targets = torch.full((N, seq_len), pad_id, dtype=torch.int32)
    actual_n = 0
    skipped = 0
    log_every = max(1, N // 20)

    for idx, (ctx_text, resp_text) in enumerate(pairs):
        ctx_ids = encode(tokenizer, ctx_text, max_len=seq_len - 4) if ctx_text else []
        max_resp = seq_len - 2 - len(ctx_ids)
        if max_resp < 2:
            skipped += 1
            continue
        resp_ids = encode(tokenizer, resp_text, max_len=max_resp) if resp_text else []
        if not resp_ids:
            skipped += 1
            continue

        # Input: [BOS, ctx_tokens..., resp_tokens..., EOS, PAD...]
        row = [bos_id] + ctx_ids + resp_ids + [eos_id]
        n = len(row)
        if n > seq_len:
            row = row[:seq_len]
            n = seq_len
        data[actual_n, :n] = torch.tensor(row, dtype=torch.int32)

        # Targets: NOOP for context positions, real tokens for response positions
        tgt = [pad_id] * seq_len
        ctx_end = 1 + len(ctx_ids)  # position after last context token
        for p in range(n - 1):
            if p < ctx_end:
                tgt[p] = noop_id
            else:
                tgt[p] = row[p + 1]
        targets[actual_n, :] = torch.tensor(tgt, dtype=torch.int32)
        actual_n += 1

        if idx % log_every == 0:
            print(f"    Tokenizing: {idx:,}/{N:,} ({100*idx/max(N,1):.0f}%)", flush=True)

    if skipped:
        print(f"    Skipped {skipped:,} sequences (too long or empty)")

    return data[:actual_n].clone(), targets[:actual_n].clone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Gen 1 Memory QA dataset (bAbI + SQuAD + TinyStories)")
    parser.add_argument("--data_dir", default="./data_cache",
                        help="Directory containing tokenizer.json and where cache is saved")
    parser.add_argument("--babi_samples", type=int, default=18_000,
                        help="Max bAbI training examples (dataset has ~18K total)")
    parser.add_argument("--squad_samples", type=int, default=40_000,
                        help="Max SQuAD training examples (dataset has ~87K total)")
    parser.add_argument("--tinystories_samples", type=int, default=100_000,
                        help="Max TinyStories replay examples")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load tokenizer
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"✗ Tokenizer not found at {tok_path}")
        print(f"  Run training Phase 1 first, or copy tokenizer.json to {args.data_dir}/")
        return
    tokenizer = load_tokenizer(tok_path)

    pad_id = tokenizer.token_to_id("<pad>") or 0
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1
    noop_id = 6

    # Load all datasets
    print("=" * 64)
    print("  Loading datasets")
    print("=" * 64)
    babi = load_babi(args.babi_samples)
    squad = load_squad(args.squad_samples)
    tiny = load_tinystories_replay(args.tinystories_samples)

    # Mix and shuffle
    all_pairs = babi + squad + tiny
    random.shuffle(all_pairs)

    n_babi = len(babi)
    n_squad = len(squad)
    n_tiny = len(tiny)
    n_total = len(all_pairs)

    print()
    print("=" * 64)
    print("  Dataset Mix")
    print("=" * 64)
    print(f"  bAbI:         {n_babi:>7,} ({100*n_babi/n_total:.1f}%)")
    print(f"  SQuAD:        {n_squad:>7,} ({100*n_squad/n_total:.1f}%)")
    print(f"  TinyStories:  {n_tiny:>7,} ({100*n_tiny/n_total:.1f}%)")
    print(f"  Total:        {n_total:>7,}")

    # Build tokenized sequences
    print()
    print("Building NOOP-target sequences...")
    data, targets = build_noop_sequences(
        all_pairs, tokenizer, args.seq_len,
        pad_id, bos_id, eos_id, noop_id,
    )

    # Save as cache compatible with prepare_data()
    # Cache key: gen1_memory_qa_context_text_tokens_v3.pt
    cache_path = os.path.join(args.data_dir, "gen1_memory_qa_context_text_tokens_v3.pt")
    torch.save({"data": data, "targets": targets, "version": 3}, cache_path)

    # Statistics
    noop_count = (targets == noop_id).sum().item()
    real_count = ((targets != pad_id) & (targets != noop_id)).sum().item()
    total_toks = noop_count + real_count
    tokens_per_epoch = data.shape[0] * args.seq_len

    print()
    print("=" * 64)
    print("  Gen 1 Dataset Ready")
    print("=" * 64)
    print(f"  Sequences:    {data.shape[0]:,}")
    print(f"  Shape:        {tuple(data.shape)}")
    print(f"  File:         {cache_path}")
    print(f"  Size:         {os.path.getsize(cache_path) / 1e6:.1f} MB")
    print(f"  NOOP tokens:  {noop_count:,} ({100*noop_count/max(total_toks,1):.1f}%)")
    print(f"  Real tokens:  {real_count:,} ({100*real_count/max(total_toks,1):.1f}%)")
    print(f"  Tokens/epoch: ~{tokens_per_epoch/1e6:.0f}M")
    print()
    print("  Next: python train_gen1.py [--resume]")
    print("=" * 64)


if __name__ == "__main__":
    main()
