/// ANT model and training configuration.
///
/// Single source of truth for all dimensions — mirrors Python config.py.
/// Everything reads from these constants; no hardcoded dims in other modules.

// ---------------------------------------------------------------------------
// Byte vocabulary
// ---------------------------------------------------------------------------

pub const VOCAB_SIZE: usize = 256;
pub const PAD_ID: u32 = 0x00;
pub const MEM_START_ID: u32 = 0x01;
pub const BOS_ID: u32 = 0x02;
pub const EOS_ID: u32 = 0x03;
pub const MEM_END_ID: u32 = 0x04;
pub const ANS_ID: u32 = 0x05;
pub const NOOP_ID: u32 = 0x06;
pub const UNK_ID: u32 = 0x1A;

// ---------------------------------------------------------------------------
// Model architecture
// ---------------------------------------------------------------------------

pub const D_MODEL: usize = 128;
pub const N_HEADS: usize = 4;
pub const HEAD_DIM: usize = 32;
pub const FFN_DIM: usize = 256;
pub const N_LAYERS: usize = 4;
pub const MAX_SEQ_LEN: usize = 192;
pub const DROPOUT: f64 = 0.0;
pub const ROPE_THETA: f32 = 10000.0;

// AddrNet co-processors
pub const N_ADDR_NETS: usize = 3;
pub const ADDR_HIDDEN_DIM: usize = 16;
pub const ADDR_N_BINS: usize = 256;
pub const ADDR_DEPTH: usize = 8;

// Memory cross-attention slots
pub const N_MEM_SLOTS: usize = 25;

// ---------------------------------------------------------------------------
// Memory (trie) configuration
// ---------------------------------------------------------------------------

pub const MEM_EMA_ALPHA_BASE: f32 = 0.1;
pub const MEM_EMA_ALPHA_MIN: f32 = 0.001;
pub const MEM_DEPTH_CAP: usize = 8;
pub const MEM_FLUSH_INTERVAL: u64 = 1000;

// ---------------------------------------------------------------------------
// Training inner-loop configuration
// ---------------------------------------------------------------------------

/// Number of gradient steps per trie read/write cycle (Phases B and C).
///
/// One trie read is shared across INNER_STEPS mini-batches, then one trie write.
/// This amortises trie I/O cost: GPU runs INNER_STEPS forward+backward passes
/// while the trie thread only does 1 read and 1 write per cycle.
///
/// Effective batch size = batch_b * INNER_STEPS  (e.g. 512 * 8 = 4096)
/// Learning rate should scale as lr / sqrt(INNER_STEPS) relative to 1-step baseline.
pub const INNER_STEPS: usize = 8;
