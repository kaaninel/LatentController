from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    828K parameter looping transformer with persistent external memory.

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
    n_mem_slots: int = 9
    n_mem_positions: int = 11
    n_text_positions: int = 181       # max_seq_len - n_mem_positions
    n_addr_heads: int = 3
    addr_dim: int = 8
    pad_id: int = 0x00                # NUL
    eos_id: int = 0x03                # ETX
    bos_id: int = 0x02                # STX
    unk_id: int = 0x1A                # SUB
    mem_start_id: int = 0x01          # SOH
    mem_end_id: int = 0x04            # EOT
    noop_id: int = 0x06               # ACK
    # Cross-attention memory
    use_memory_cross_attention: bool = True
    memory_topk: int = 0              # 0=softmax, >0=top-k sparse attention
    memory_hops: int = 1              # 1=standard, 2=multi-hop
    # Streaming chunk encoding
    chunk_size: int = 8
    slots_per_chunk: int = 2
    max_temporal_chunks: int = 32


@dataclass
class MemoryConfig:
    alpha_base: float = 0.1
    max_write_count: int = 65535
    write_count_decay_cap: int = 1000
    write_count_decay_rate: float = 0.01
    neighbor_k: int = 2
    coarse_dims: int = 4
    fine_dims: int = 4
    n_mem_slots: int = 9
