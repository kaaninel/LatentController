"""
TinyStories dataset utilities.
"""

import os
from typing import Tuple

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

class TinyStoriesDataset(Dataset):
    """
    Wraps a pre-padded int32 tensor of shape (N, seq_len).

    Each row is stored as:
        [BOS, tok_0, …, tok_{k-1}, EOS, PAD, …, PAD]

    __getitem__ returns zero-copy tensor views:
        input  = row[:-1]  — length seq_len-1
        target = row[1:]   — length seq_len-1
    """

    def __init__(self, data: torch.Tensor):
        self.data = data  # (N, seq_len) int32

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1].long(), seq[1:].long()


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
) -> Tuple[TinyStoriesDataset, TinyStoriesDataset, object]:
    """
    Returns (train_dataset, val_dataset, tokenizer).

    Tokenized data is cached to `cache_dir/tinystories_tokens_v2.pt`
    as a dict `{'data': tensor, 'version': 2}` where tensor has shape
    (N, seq_len) dtype int32 with rows [BOS, tokens…, EOS, PAD…].

    If the old `tinystories_tokens.pt` (list-of-lists) is found it is
    auto-converted to v2 format and the old file is deleted.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path)

    pad_id = tokenizer.token_to_id("<pad>") or 0
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1

    v2_path  = os.path.join(cache_dir, "tinystories_tokens_v2.pt")
    old_path = os.path.join(cache_dir, "tinystories_tokens.pt")

    if os.path.exists(v2_path):
        print(f"Loading cached tokenized data from {v2_path}…")
        cache = torch.load(v2_path, weights_only=False)
        data = cache["data"]

    elif os.path.exists(old_path):
        print(f"Converting old cache {old_path} to v2 tensor format…")
        old = torch.load(old_path, weights_only=False)
        data = _build_tensor(old, seq_len, pad_id, bos_id, eos_id)
        torch.save({"data": data, "version": 2}, v2_path)
        os.remove(old_path)
        print(f"Converted {data.shape[0]:,} stories → {v2_path}")

    else:
        print("Tokenizing TinyStories…")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        total = len(ds)
        data = torch.full((total, seq_len), pad_id, dtype=torch.int32)
        actual_n = 0
        log_every = max(1, total // 20)
        for idx, item in enumerate(ds):
            ids = encode(tokenizer, item["text"], max_len=seq_len - 2)
            if not ids:
                continue
            n = len(ids)
            data[actual_n, 0] = bos_id
            data[actual_n, 1: n + 1] = torch.tensor(ids, dtype=torch.int32)
            data[actual_n, n + 1] = eos_id
            actual_n += 1
            if idx % log_every == 0:
                print(f"  Tokenizing: {idx:,}/{total:,} stories ({100.0*idx/max(total,1):.0f}%)", flush=True)
        data = data[:actual_n].clone()
        torch.save({"data": data, "version": 2}, v2_path)
        print(f"Tokenized {actual_n:,} stories → {v2_path}")

    n_val   = max(1, int(data.shape[0] * val_split))
    n_train = data.shape[0] - n_val

    train_ds = TinyStoriesDataset(data[:n_train])
    val_ds   = TinyStoriesDataset(data[n_train:])

    return train_ds, val_ds, tokenizer

