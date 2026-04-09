# ANT — Architecture

## Core Principle

ANT is a sub-1M parameter byte-level transformer where the model weights
contain NO knowledge. ALL knowledge lives in persistent hierarchical trie
memory. The weights learn HOW to read and write memory. The memory stores
WHAT was learned.

Every forward pass reads from the trie. Every encoding writes to the trie.
Memory is not a feature — it IS the system.

## Overview

```
  937K params │ 4 layers │ d_model=128 │ 4 heads │ 256 vocab (raw bytes)

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Token ──► Embed ──► 4× TransformerBlock ──► Norm      │
  │                      │ Self-Attn                       │
  │                      │ Tag-Attn (context register)     │
  │                      │ Mem-Attn ◄── Trie READ          │
  │                      │ FFN                              │
  │                      ▼                                 │
  │              Hidden state ──► AddrNets ──► Trie WRITE   │
  │              Hidden state ──► V_proj ──┘                │
  │              Hidden state ──► LM Head (logits)         │
  │              Hidden state ──► Halt Head (continue/halt) │
  │                                                        │
  └────────────────────────────────────────────────────────┘
            ▲                                ▼
            │    ┌──────────────────────┐     │
            └────│  Hierarchical Trie   │◄────┘
                 │  (persistent, on disk)│
                 │  EMA value vectors    │
                 │  256 bins × 8 levels  │
                 │  3 address paths      │
                 └──────────────────────┘
```

## Memory Is The Architecture

The trie is not an add-on. It is how the model knows anything.

```
  Traditional LM:   weights = knowledge + computation
  ANT:              weights = computation only
                    trie    = all knowledge

  Without memory, ANT outputs random bytes.
  With memory, ANT recalls facts, answers questions, holds conversations.
```

The 937K parameters learn ONLY:
- How to generate addresses (where to read/write)
- How to project values (what to store)
- How to attend to retrieved vectors (how to use recalled knowledge)
- How to predict the next byte (language modeling)

## Transformer Block

Each of 4 layers follows: Self-Attn → Tag-Attn → Mem-Attn → FFN

```
  Input x [B, T, 128]
    │
    ├──► RMSNorm → Self-Attention (causal, RoPE) → + residual
    │    4 heads × 32 dim, KV cache for generation
    │
    ├──► RMSNorm → Tag Cross-Attention → + residual
    │    GRU-style gated update from persistent tag_register
    │    tag_head(128→128) + tag_gate(128→1)
    │    Tracks current speaker/context/mode
    │
    ├──► RMSNorm → Memory Cross-Attention → + residual
    │    Q,K,V,O: each 128→128 (4 heads × 32 dim)
    │    Learned inv_temp per head (sharper attention)
    │    Attends to up to 25 trie-retrieved vectors
    │    No causal mask (every token sees all memory)
    │    Optional top-k sparse attention with STE
    │
    └──► RMSNorm → SiLU FFN (128→256→128) → + residual
```

## AddrNet — Address Generation

3 separate MLP co-processors, each generating an 8-level hierarchical
address into the trie. ~10.5K params each, 31.5K total.

```
  hidden_state (128-dim)
       │
       ▼
  proj_in: Linear(128 → 16)
       │
       ▼ ─── repeat 8 times (one per trie level) ──────┐
       │                                                │
       ├──► out: Linear(16 → 256) → logits              │
       │         │                                      │
       │    Training: Gumbel-softmax(τ, hard=True)      │
       │    Inference: argmax                            │
       │         │                                      │
       │         ▼                                      │
       │    bin_idx ──► bin_embed: Embed(256, 16)       │
       │         │                                      │
       │         ▼                                      │
       └──► h = h + embed(bin_idx)                      │
            h = SiLU(mlp(h))        ◄───────────────────┘
       │
       ▼
  address: [bin_0, bin_1, ..., bin_7]  (8 × int64)
```

3 AddrNets produce 3 different address paths into the trie.
Gumbel-softmax makes address selection differentiable during training.

## Hierarchical Trie Memory

The trie stores EMA-blended value vectors at every node (root to leaf).

```
  Structure: 256-ary tree, depth cap = 8
  Nodes:     each stores a float32 vector (128-dim) + write_count

  WRITE path (per token):
    1. V_proj(hidden) → value vector (128 × float32)
    2. 3 AddrNets → 3 addresses, each 8 bins deep
    3. For each address: traverse root→leaf, create nodes as needed
    4. Leaf gets full value (EMA: α decays with write_count)
    5. Each ancestor gets decayed EMA: α × 1/√(depth_diff+1) × 1/(level+1)
       Coarser nodes decay slower → stable summaries

  READ path (per token):
    1. 3 AddrNets → 3 addresses (same nets as write)
    2. For each address: traverse root→deepest existing node
    3. Collect ALL ancestor vectors along the path
    4. 3 paths × up to 9 levels (root + 8 depth) = up to 25 unique vectors
    5. Root is shared across paths (deduplicated)
    6. Pad to 25 vectors → cross-attention keys/values

  Why full hierarchy (not just leaves)?
    - Root = global context (everything ever written)
    - Depth 1-2 = coarse category summaries
    - Depth 7-8 = specific fact vectors
    - Cross-attention K_proj learns to weight fine > coarse
    - Free: no extra lookup cost, just collect along traversal

  Storage:
    Single flat binary file: header + values + write_counts + adjacency
    Header: n_nodes(u32), d_model(u32), n_records(u32)
    Values: contiguous float32 block (mmap-friendly)
    ~500 bytes per node (128 × 4 = 512 bytes value + metadata)
```

## V_proj — Value Projection

```
  Linear(128 → 128) = 16,512 params

  Decouples what the model thinks (hidden state) from what gets stored
  (memory value). This prevents the trie from accumulating raw hidden
  states which are optimized for next-byte prediction, not storage.
```

## Tag System

Persistent context register tracking current speaker/mode.

```
  tag_register: (B, d_model) — persists across tokens

  At each layer:
    normed = RMSNorm(x)
    new_tag = tanh(tag_head(normed))     # candidate tag update
    gate = sigmoid(tag_gate(normed))      # how much to update
    tag_context = gate * new_tag + (1 - gate) * tag_register
    x = x + tag_context                   # residual connection

  GRU-style: model learns WHEN to update the tag register
  (e.g., speaker change, topic change) and WHAT to store.
```

## Sliding Window

Causal sliding window with multi-pass refinement for unlimited context.

```
  Window size: 8 bytes, stride: 1, passes: 4

  Pass 1:  ────[████████]────────────────  4 bytes left context
  Pass 2:  ────[████████]────────────────  8 bytes left context
  Pass 3:  ────[████████]────────────────  12 bytes left context
  Pass 4:  ────[████████]────────────────  16 bytes left context

  Each pass:
    1. Pad edges with PAD embedding
    2. Unfold into overlapping windows
    3. Process each window through all 4 transformer layers
       (self-attn + tag-attn + mem-attn + FFN)
    4. Extract center position → update hidden state

  Combined with memory cross-attention:
    Local context:  ~16 bytes (sliding window)
    Global context: unlimited (trie memory, no distance limit)
```

## Data Flow — Complete Cycle

```
  ┌─ ENCODE (every token processed) ──────────────────────────┐
  │                                                           │
  │  input byte → embed → sliding window → hidden state       │
  │                                     ↑                     │
  │                        trie READ ───┘                     │
  │                                                           │
  │  hidden → AddrNets → 3 addresses ─┐                       │
  │  hidden → V_proj → value ──────────┼──► trie WRITE        │
  │                                                           │
  │  hidden → LM head → logits (next byte prediction)        │
  │  hidden → halt head → continue/halt decision             │
  └───────────────────────────────────────────────────────────┘

  Every single token: READ from trie, process, WRITE to trie.
  The trie grows with every token seen. Knowledge accumulates.
```

## Special Tokens

```
  Hex   ASCII   Role
  ────  ─────   ──────────────────────────
  0x00  NUL     PAD (padding)
  0x01  SOH     MEM_START (memory block open)
  0x02  STX     BOS (beginning of sequence)
  0x03  ETX     EOS (end of sequence)
  0x04  EOT     MEM_END (memory block close)
  0x05  ENQ     ANS (answer marker in QA)
  0x06  ACK     NOOP (nothing to output yet)
  0x1A  SUB     UNK (unknown/fallback)

  All other byte values (0x07–0xFF) = data
```

## Parameter Breakdown

```
  Component                       Params      %
  ─────────────────────────────   ────────   ─────
  Byte Embedding (256 × 128)      32,768     3.5%
  4 × TransformerBlock:
    Self-Attention (Q,K,V,O)      65,536
    Tag head + gate + RMSNorm     16,641
    Memory Cross-Attn (Q,K,V,O)   65,540
    FFN (up + down)               65,536
    RMSNorm × 3                      384
    Subtotal per layer:          213,637
    × 4 layers =                             854,548    91.2%
  3 × AddrNet:
    proj_in + bin_embed + mlp + out
    Subtotal per net: ~10,528
    × 3 nets =                                31,584     3.4%
  V_proj (128 → 128)              16,512     1.8%
  Halt Head (128 → 2)                258     0.0%
  Final RMSNorm                      128     0.0%
  LM Head                    (tied with embed)
  ─────────────────────────────   ────────   ─────
  TOTAL                          ~937,078   100.0%
```

## Training Curriculum

### Phase A — Base Language Model (with memory)

```
  Wiki + Shell + Chat → engine.encode() → causal LM loss
  Memory: ON from step 1 (trie starts empty, grows)
  Purpose: learn language AND memory interaction simultaneously
```

### Phase B — Memory Training (frozen base)

```
  Freeze: all base model weights
  Train:  AddrNets, V_proj, tag system only
  Losses:
    - Contrastive address loss (same passage → similar addresses)
    - Quadratic depth cost (incentivize shallow addresses for common concepts)
    - Retrieval accuracy (write then read back → should match)
  Purpose: learn stable address space and value projection
```

### Phase C — End-to-End (memory always active)

```
  Unfreeze: base model
  Keep frozen: AddrNet, V_proj (stable address space)
  Losses: LM + QA + Chat (all through memory)
  Every forward pass: trie READ + WRITE
  Purpose: model learns to generate WITH memory, not without it
```

## Inference — Duplex Streaming

```
  Model reads input and generates output simultaneously.

  For each output token:
    1. Previous token's hidden → AddrNets → 3 addresses
    2. Trie READ along 3 paths → up to 25 vectors
    3. Forward pass with memory cross-attention
    4. Predict next byte (or NOOP = "still reading")
    5. Hidden → V_proj → value
    6. Trie WRITE at 3 addresses (knowledge accumulates)
    7. Halt head decides: 1–4 memory fetch cycles per output

  NOOP token (0x06 ACK):
    When model is still reading input and has nothing to say,
    it outputs NOOP. This allows asymmetric read/write speeds.

  Context window: unlimited.
    Local:  sliding window (~16 bytes)
    Global: entire trie (every fact ever stored)
```

## Spatiotemporal Tags

```
  host/agent/dataplane@ISO-timestamp: content

  Examples:
    localhost/user/chat@2026-04-08T12:00:00Z: Hello!
    localhost/ant/chat@2026-04-08T12:00:01Z: Hi there!
    shell/root@2026-04-01T14:22:33Z: ls -la /home
    wiki/article@2025-06-15T08:30:00Z: The French Revolution began in 1789.
```

## Configuration

```python
ModelConfig(
    vocab_size    = 256,       # raw bytes
    d_model       = 128,       # hidden dimension
    n_heads       = 4,         # attention heads
    head_dim      = 32,        # per-head dimension
    ffn_dim       = 256,       # FFN intermediate
    n_layers      = 4,         # transformer blocks
    max_seq_len   = 192,       # max sequence length
    n_addr_nets   = 3,         # parallel address co-processors
    addr_hidden_dim = 16,      # AddrNet internal dim
    addr_n_bins   = 256,       # bins per trie level
    addr_depth    = 8,         # max address depth (hard cap)
    use_tag_system = True,     # persistent context register
    n_mem_slots   = 25,        # max memory vectors for cross-attention
    memory_topk   = 0,         # 0=softmax, >0=top-k sparse
)

MemoryConfig(
    data_path       = "data_cache/memory",
    ema_alpha_base  = 0.1,     # base EMA momentum
    ema_alpha_min   = 0.001,   # minimum alpha
    depth_cap       = 8,       # hard cap on depth
    d_model         = 128,     # vector dimensionality
    n_bins          = 256,     # bins per level
    flush_interval  = 1000,    # flush every N writes
)
```

## Files

```
  ant_memory/          Rust crate (PyO3 extension) — arena-allocated trie
    Cargo.toml         pyo3 0.24, numpy 0.24, ndarray 0.16, memmap2 0.9
    src/lib.rs         PyO3 module entry
    src/trie.rs        HierarchicalTrie: arena nodes, EMA vectors, serialize
    src/memory.rs      MemorySystem: batch read/write, Python-facing API

  config.py           ModelConfig + MemoryConfig
  model.py            ANT transformer (AddrNet, Attention, MemoryAttention,
                      TransformerBlock, StaticKVCache, ANT)
  engine.py           Core engine: per-token READ→PROCESS→WRITE cycle
                      encode() for training, generate() for inference
  data.py             Data pipelines: tokenizer, QA/shell/wiki/chat generators
  train.py            Training: Phase A/B/C curriculum using engine
  inference.py        Terminal chat interface using engine.generate()
  train_colab.ipynb   A100 GPU training notebook
```
