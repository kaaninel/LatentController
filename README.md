# 🐜 ANT

> **937K parameters. Zero knowledge in weights. All knowledge in memory.**

ANT is a byte-level transformer where the model weights learn **how** to read
and write — but store no knowledge themselves. All facts, context, and learned
information live in a persistent hierarchical trie memory. The result: unlimited
context, persistent recall across sessions, and a model that can grow its
knowledge without growing its parameters.

```
  ┌──────────────────────────────────────────────────────────┐
  │                         🐜 ANT                            │
  │                                                          │
  │    937K params · 4 layers · 128-dim · byte-level         │
  │    Weights = computation only. Trie = all knowledge.     │
  │                                                          │
  │    Token ──► Transformer ──► Hidden ──► Trie WRITE       │
  │                   ▲                                      │
  │                   └──── Trie READ (25 vectors)           │
  │                   │                                      │
  │                   ▼                                      │
  │              Next byte prediction (unlimited output)     │
  └──────────────────────────────────────────────────────────┘
```

## How It Works

The model processes raw bytes through a sliding window transformer. At every
token, 3 learned **AddrNet** co-processors generate hierarchical addresses
into a 256-ary trie. The trie is read (up to 25 ancestor vectors via
cross-attention) and written (EMA-blended value vectors at every level).

Without memory, ANT outputs random bytes. With memory, it recalls facts,
answers questions, and holds conversations.

| Feature | ANT | Typical Small LM |
|---------|-----|-------------------|
| Parameters | 937K | 1M–10M |
| Knowledge storage | External trie (unlimited) | Model weights (fixed) |
| Context | Unlimited (trie has no distance limit) | Fixed (512–2048 tokens) |
| Tokenizer | None (raw bytes, vocab=256) | BPE/SentencePiece |
| Persistence | Trie survives restart | Weights only |
| QA Accuracy | 100% (bAbI 1/2/3-fact) | N/A at this size |

## Quick Start

```bash
pip install -r requirements.txt

# Build Rust memory extension
cd ant_memory && pip install maturin
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ..

# Train
python3 train.py 2>&1 | tee training.log

# Interactive chat
python3 inference.py
```

## Architecture

```
  937K params │ 4 layers │ d_model=128 │ 4 heads │ 256 vocab (raw bytes)
```

Each transformer block: **Self-Attn → Tag-Attn → Mem-Attn → FFN**

```
  Input
    │
    ├──► Self-Attention (causal, RoPE)       local context
    │
    ├──► Tag Cross-Attention                 speaker/mode tracking
    │         GRU-gated persistent register
    │
    ├──► Memory Cross-Attention              global recall
    │         ▲
    │         │ up to 25 trie vectors (3 paths × hierarchy)
    │
    └──► SiLU FFN (128→256→128)
```

### Memory = The Architecture

This is the key insight. In a normal transformer, the weights store both
computation patterns AND factual knowledge. ANT separates these:

- **Weights** (937K params): learn attention patterns, address generation,
  value projection, language modeling — pure computation
- **Trie** (persistent, unbounded): stores every fact ever encountered as
  EMA-blended vectors at hierarchical addresses

Every token processed: READ from trie → process → WRITE to trie.
Knowledge accumulates with every byte seen.

### AddrNet Co-processors

3 separate MLP networks (10.5K params each) that generate 8-level
hierarchical addresses into the 256-ary trie:

```
  hidden → proj_in(128→16) → 8× [out(16→256) → Gumbel-softmax → embed(256,16) → SiLU(MLP)]
                                                                    │
                                                                    ▼
                                                          address: [b0, b1, ..., b7]
```

### Hierarchical Trie

```
  256 bins per level × 8 levels deep = up to 256⁸ addresses
  Every node stores: float32 vector (128-dim) + write_count

  WRITE: V_proj(hidden) → value. EMA-blend at leaf + all ancestors.
         Coarser nodes decay slower → stable category summaries.

  READ:  3 addresses → 3 paths through trie → collect ALL ancestor vectors.
         Root = global context. Leaves = specific facts.
         Up to 25 unique vectors → cross-attention.
```

Storage: single flat binary file. ~500 bytes/node. 3ms load for 9K nodes.

### Sliding Window

Causal sliding window with multi-pass refinement:

```
  W=8, stride=1, passes=4 → ~16 bytes effective local context
  Combined with trie cross-attention → unlimited global context
```

## Training Curriculum

1. **Phase A — Base LM with Memory**: LM on wiki + shell + chat with memory active
2. **Phase B — Memory Training**: Freeze base, train AddrNet + V_proj + tags
   with QA data (write passage, answer from memory)
3. **Phase C — End-to-End**: Unfreeze base (keep AddrNet/V_proj frozen),
   every pass reads and writes the trie

## Files

```
ant_memory/         Rust trie memory (PyO3 extension, arena-allocated)
config.py           ModelConfig (937K) + MemoryConfig
model.py            ANT transformer: AddrNet, MemoryAttention, tag system
engine.py           Core engine: per-token READ→PROCESS→WRITE trie cycle
data.py             Data pipelines: tokenizer, QA/shell/wiki/chat generators
train.py            Training: Phase A/B/C curriculum using engine
inference.py        Terminal chat using engine.generate()
train_colab.ipynb   A100 GPU training notebook
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full technical details.

## License

Research prototype. Not yet licensed for production use.