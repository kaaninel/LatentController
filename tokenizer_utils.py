"""
BPE Tokenizer utilities using the HuggingFace `tokenizers` library.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# Special tokens in ID order (must match ModelConfig special-token IDs)
_SPECIAL_TOKENS = (
    ["<pad>", "<eos>", "<bos>", "<unk>", "<MEM>", "</MEM>", "<NOOP>"]
    + [f"<RESERVED_{i}>" for i in range(7, 128)]
)


def train_tokenizer(texts_iterator, save_path: str, vocab_size: int = 16384):
    """
    Train a BPE tokenizer on an iterator of plain-text strings.

    The final vocabulary is `vocab_size + len(_SPECIAL_TOKENS)`, but the BPE
    model itself is asked to learn `vocab_size` merge rules on top of the
    256 byte-level alphabet.  Special tokens are injected separately so they
    never get split by merge rules.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=_SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts_iterator, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str, max_len: int = 512):
    """
    Encode text to a list of token IDs, truncated to max_len.
    Does NOT add BOS/EOS — callers do that themselves.
    """
    enc = tokenizer.encode(text)
    return enc.ids[:max_len]
