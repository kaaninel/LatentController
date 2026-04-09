from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    ANT — 937K parameter byte-level transformer with persistent hierarchical memory.

    Pure 256-byte vocabulary (token ID = raw byte value).
    Special tokens mapped to ASCII control characters:
        PAD=NUL(0x00)  BOS=STX(0x02)  EOS=ETX(0x03)  ANS=ENQ(0x05)
        MEM_START=SOH(0x01)  MEM_END=EOT(0x04)  NOOP=ACK(0x06)  UNK=SUB(0x1A)
    """
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    head_dim: int = 32
    ffn_dim: int = 256
    n_layers: int = 4
    max_seq_len: int = 192
    dropout: float = 0.0
    rope_theta: float = 10000.0
    pad_id: int = 0x00
    eos_id: int = 0x03
    bos_id: int = 0x02
    unk_id: int = 0x1A
    mem_start_id: int = 0x01
    mem_end_id: int = 0x04
    noop_id: int = 0x06
    ans_id: int = 0x05
    # Cross-attention memory
    memory_topk: int = 0              # 0=softmax, >0=top-k sparse attention
    # AddrNet: 3 separate co-processors generating hierarchical addresses
    n_addr_nets: int = 3
    addr_hidden_dim: int = 16
    addr_n_bins: int = 256
    addr_depth: int = 8
    # Tag system: persistent context register
    use_tag_system: bool = True
    # Memory cross-attention slot count
    n_mem_slots: int = 25


@dataclass
class MemoryConfig:
    """Rust-backed hierarchical trie memory (ant_memory crate)."""
    data_path: str = "data_cache/memory"
    ema_alpha_base: float = 0.1
    ema_alpha_min: float = 0.001
    depth_cap: int = 8
    d_model: int = 128
    flush_interval: int = 1000

