"""
Dataset utilities — configurable for any HuggingFace dataset.
"""

import os
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from tokenizer_utils import train_tokenizer, load_tokenizer, encode


# ---------------------------------------------------------------------------
# Tokenizer bootstrap from TinyStories
# ---------------------------------------------------------------------------

def train_tokenizer_from_tinystories(
    save_path: str,
    vocab_size: int = 16384,
    max_stories: int = 500_000,
):
    """Train a BPE tokenizer on the first `max_stories` stories."""
    print(f"Loading TinyStories for tokenizer training (max {max_stories} stories)…")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def text_iter():
        count = 0
        for item in ds:
            yield item["text"]
            count += 1
            if count >= max_stories:
                break

    tokenizer = train_tokenizer(text_iter(), save_path, vocab_size=vocab_size)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer


# ---------------------------------------------------------------------------
# Dataset (tensor-native, zero-copy __getitem__)
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """
    Wraps pre-padded int32 tensors.

    When targets tensor is provided (Phase 5 with NOOP), uses explicit targets.
    Otherwise, standard next-token prediction: input=row[:-1], target=row[1:].
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor = None):
        self.data = data        # (N, seq_len) int32
        self.targets = targets  # (N, seq_len) int32 or None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]
        if self.targets is not None:
            return seq[:-1].long(), self.targets[idx][:-1].long()
        return seq[:-1].long(), seq[1:].long()


# Backward compat alias
TinyStoriesDataset = TokenizedDataset


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_tensor(seqs_list, seq_len: int, pad_id: int, bos_id: int, eos_id: int) -> torch.Tensor:
    """Convert a list-of-int-lists to a (N, seq_len) int32 tensor."""
    N = len(seqs_list)
    data = torch.full((N, seq_len), pad_id, dtype=torch.int32)
    for i, raw in enumerate(seqs_list):
        # Truncate so that BOS + raw + EOS fits in seq_len
        raw = raw[: seq_len - 2]
        n = len(raw)
        data[i, 0] = bos_id
        if n:
            data[i, 1: n + 1] = torch.tensor(raw, dtype=torch.int32)
        data[i, n + 1] = eos_id
    return data


def prepare_data(
    tokenizer_path: str,
    cache_dir: str,
    seq_len: int = 512,
    val_split: float = 0.1,
    dataset_name: str = "roneneldan/TinyStories",
    dataset_split: str = "train",
    text_column: str = "text",
    context_column: Optional[str] = None,
    max_samples: Optional[int] = None,
    noop_id: int = 6,
) -> Tuple['TokenizedDataset', 'TokenizedDataset', object]:
    """
    Returns (train_dataset, val_dataset, tokenizer).

    When context_column is provided (e.g. QA datasets with passage + answer),
    builds two-phase sequences: context tokens have NOOP targets, response
    tokens have standard next-token targets. This trains the unified streaming
    loop where the model learns when to absorb vs emit.

    Otherwise, builds standard next-token prediction sequences (backward compat).

    Tokenized data is cached to `cache_dir/<dataset_hash>_tokens_v3.pt`.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path)

    pad_id = tokenizer.token_to_id("<pad>") or 0
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1

    # Backward compatible cache key
    safe_name = dataset_name.replace("/", "_").replace("\\", "_")
    if context_column:
        cache_name = f"{safe_name}_{context_column}_{text_column}_tokens_v3.pt"
    elif dataset_name == "roneneldan/TinyStories":
        cache_name = "tinystories_tokens_v2.pt"
    else:
        cache_name = f"{safe_name}_tokens_v3.pt"

    cache_path = os.path.join(cache_dir, cache_name)
    old_path = os.path.join(cache_dir, "tinystories_tokens.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached tokenized data from {cache_path}…")
        cache = torch.load(cache_path, weights_only=False)
        data = cache["data"]
        targets = cache.get("targets")  # None for v2 (standard LM)

    elif dataset_name == "roneneldan/TinyStories" and os.path.exists(old_path):
        print(f"Converting old cache {old_path} to v2 tensor format…")
        old = torch.load(old_path, weights_only=False)
        data = _build_tensor(old, seq_len, pad_id, bos_id, eos_id)
        torch.save({"data": data, "version": 2}, cache_path)
        os.remove(old_path)
        print(f"Converted {data.shape[0]:,} stories → {cache_path}")
        targets = None

    else:
        print(f"Tokenizing {dataset_name} (split={dataset_split})…")
        ds = load_dataset(dataset_name, split=dataset_split)
        total = min(len(ds), max_samples) if max_samples else len(ds)

        if context_column:
            data, targets = _build_noop_sequences(
                ds, tokenizer, seq_len, pad_id, bos_id, eos_id, noop_id,
                context_column, text_column, total,
            )
            torch.save({"data": data, "targets": targets, "version": 3}, cache_path)
        else:
            data = torch.full((total, seq_len), pad_id, dtype=torch.int32)
            actual_n = 0
            log_every = max(1, total // 20)
            for idx, item in enumerate(ds):
                if idx >= total:
                    break
                ids = encode(tokenizer, item[text_column], max_len=seq_len - 2)
                if not ids:
                    continue
                n = len(ids)
                data[actual_n, 0] = bos_id
                data[actual_n, 1: n + 1] = torch.tensor(ids, dtype=torch.int32)
                data[actual_n, n + 1] = eos_id
                actual_n += 1
                if idx % log_every == 0:
                    print(f"  Tokenizing: {idx:,}/{total:,} ({100.0*idx/max(total,1):.0f}%)", flush=True)
            data = data[:actual_n].clone()
            targets = None
            torch.save({"data": data, "version": 2}, cache_path)

        print(f"Tokenized {data.shape[0]:,} sequences → {cache_path}")

    n_val   = max(1, int(data.shape[0] * val_split))
    n_train = data.shape[0] - n_val

    if targets is not None:
        train_ds = TokenizedDataset(data[:n_train], targets[:n_train])
        val_ds   = TokenizedDataset(data[n_train:], targets[n_train:])
    else:
        train_ds = TokenizedDataset(data[:n_train])
        val_ds   = TokenizedDataset(data[n_train:])

    return train_ds, val_ds, tokenizer


def _build_noop_sequences(
    ds, tokenizer, seq_len, pad_id, bos_id, eos_id, noop_id,
    context_column, text_column, total,
):
    """Build sequences with NOOP targets for context and real targets for response."""
    data = torch.full((total, seq_len), pad_id, dtype=torch.int32)
    targets = torch.full((total, seq_len), pad_id, dtype=torch.int32)
    actual_n = 0
    log_every = max(1, total // 20)

    for idx, item in enumerate(ds):
        if idx >= total:
            break

        ctx_text = item.get(context_column, "")
        resp_text = item.get(text_column, "")
        if not ctx_text and not resp_text:
            continue

        ctx_ids = encode(tokenizer, ctx_text, max_len=seq_len - 4) if ctx_text else []
        max_resp = seq_len - 2 - len(ctx_ids)
        resp_ids = encode(tokenizer, resp_text, max_len=max_resp) if resp_text else []
        if not resp_ids:
            continue

        # Build input: [BOS, ctx_tokens..., resp_tokens..., EOS, PAD...]
        row = [bos_id] + ctx_ids + resp_ids + [eos_id]
        n = len(row)
        if n > seq_len:
            row = row[:seq_len]
            n = seq_len
        data[actual_n, :n] = torch.tensor(row, dtype=torch.int32)

        # Build targets: NOOP during context, real tokens during response
        # Position 0 (BOS) → NOOP (absorbing)
        # Context positions → NOOP
        # Response positions → real next tokens
        tgt = [pad_id] * seq_len
        ctx_end = 1 + len(ctx_ids)  # position after last context token
        for p in range(n - 1):
            if p < ctx_end:
                tgt[p] = noop_id  # absorb context silently
            else:
                tgt[p] = row[p + 1]  # predict next real token
        if n - 1 < seq_len:
            tgt[n - 1] = pad_id
        targets[actual_n, :] = torch.tensor(tgt, dtype=torch.int32)
        actual_n += 1

        if idx % log_every == 0:
            print(f"  Tokenizing: {idx:,}/{total:,} ({100.0*idx/max(total,1):.0f}%)", flush=True)

    data = data[:actual_n].clone()
    targets = targets[:actual_n].clone()
    return data, targets

