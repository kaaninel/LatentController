from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 16_512          # 16384 BPE + 128 reserved
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    ffn_dim: int = 2048
    n_layers: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0
    n_mem_slots: int = 9
    n_mem_positions: int = 11         # <MEM> + 9 vecs + </MEM>
    n_text_positions: int = 501
    n_addr_heads: int = 3
    addr_dim: int = 8
    pad_id: int = 0
    eos_id: int = 1
    bos_id: int = 2
    unk_id: int = 3
    mem_start_id: int = 4
    mem_end_id: int = 5
    noop_id: int = 6


@dataclass
class Phase1Config:
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    micro_batch: int = 4
    grad_accum: int = 8
    total_tokens: int = 1_500_000_000
    max_grad_norm: float = 1.0
    eval_interval: int = 5000
    save_interval: int = 10000
    log_interval: int = 100


@dataclass
class Phase2Config:
    lr: float = 1e-3
    n_hidden_states: int = 800_000
    n_sequences: int = 50_000
    positions_per_seq: int = 16
    batch_size: int = 3000
    steps: int = 10_000
    margin: float = 4.0
    entropy_weight: float = 0.1
    pos_threshold: float = 0.8
    neg_threshold: float = 0.2
    target_dim_std: float = 15.0


@dataclass
class Phase3Config:
    lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    micro_batch: int = 4
    grad_accum: int = 8
    total_tokens: int = 2_000_000_000
    max_grad_norm: float = 1.0
    eval_interval: int = 5000
    save_interval: int = 10000
    write_every_n_steps: int = 5
    eval_memory_refresh_interval: int = 20000


@dataclass
class Phase4Config:
    lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    micro_batch: int = 2
    grad_accum: int = 16
    total_tokens: int = 1_000_000_000
    max_grad_norm: float = 1.0
    eval_interval: int = 5000
    save_interval: int = 10000
    ponder_curriculum: list = field(default_factory=lambda: [
        (0,      2, 0.0,    1.0),
        (10000,  4, 0.0,    1.0),
        (25000,  4, 0.0005, 1.0),
        (50000,  6, 0.002,  1.0),
        (80000,  6, 0.005,  1.0),
        (100000, 6, 0.005,  0.5),
        (120000, 6, 0.005,  0.1),
    ])


@dataclass
class MemoryConfig:
    alpha_base: float = 0.1
    max_write_count: int = 65535
    write_count_decay_cap: int = 1000
    write_count_decay_rate: float = 0.01
    neighbor_k: int = 2
    coarse_dims: int = 4
    fine_dims: int = 4
    n_mem_slots: int = 9           # must match ModelConfig.n_mem_slots
