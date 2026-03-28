"""
TinyStories dataset utilities.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset, random_split
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
# Dataset
# ---------------------------------------------------------------------------

class TinyStoriesDataset(Dataset):
    """
    Each item is (input_ids, targets) where:
      input_ids = [BOS, tok_0, …, tok_{n-1}]   length seq_len
      targets   = [tok_0, …, tok_{n-1}, EOS]    length seq_len
    Both are padded to seq_len with pad_id.
    """

    def __init__(
        self,
        token_sequences,        # list of int-lists (raw token IDs, no BOS/EOS)
        seq_len: int,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 1,
    ):
        self.seqs = token_sequences
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        raw = self.seqs[idx]
        # Truncate so that with BOS it fits in seq_len
        raw = raw[: self.seq_len - 1]

        inp = [self.bos_id] + raw
        tgt = raw + [self.eos_id]

        # Pad to seq_len
        pad_len = self.seq_len - len(inp)
        inp = inp + [self.pad_id] * pad_len
        tgt = tgt + [self.pad_id] * pad_len

        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    tokenizer_path: str,
    cache_dir: str,
    seq_len: int = 512,
    val_split: float = 0.1,
) -> Tuple[TinyStoriesDataset, TinyStoriesDataset, object]:
    """
    Returns (train_dataset, val_dataset, tokenizer).

    Tokenized data is cached to `cache_dir/tinystories_tokens.pt`.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "tinystories_tokens.pt")
    tokenizer = load_tokenizer(tokenizer_path)

    if os.path.exists(cache_path):
        print(f"Loading cached tokenized data from {cache_path}…")
        all_seqs = torch.load(cache_path)
    else:
        print("Tokenizing TinyStories…")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        all_seqs = []
        for item in ds:
            ids = encode(tokenizer, item["text"], max_len=seq_len - 1)
            if ids:
                all_seqs.append(ids)
        torch.save(all_seqs, cache_path)
        print(f"Tokenized {len(all_seqs)} stories → {cache_path}")

    n_val = max(1, int(len(all_seqs) * val_split))
    n_train = len(all_seqs) - n_val

    pad_id = tokenizer.token_to_id("<pad>") or 0
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1

    train_seqs, val_seqs = all_seqs[:n_train], all_seqs[n_train:]

    train_ds = TinyStoriesDataset(train_seqs, seq_len, pad_id, bos_id, eos_id)
    val_ds   = TinyStoriesDataset(val_seqs,   seq_len, pad_id, bos_id, eos_id)

    return train_ds, val_ds, tokenizer
