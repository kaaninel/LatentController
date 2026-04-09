# Copilot Instructions — ANT

## ⚠️ MANDATORY PROCESS RULES

These rules are non-negotiable. Violating them wastes time and creates broken code.

1. **ALWAYS ask the user before making architectural decisions.** You implement,
   you do not decide. If something is unclear, ask — do not assume.

2. **Documentation first.** When changing architecture or behavior, update the
   relevant docs BEFORE writing code. Docs are the source of truth. Code follows
   docs, never the reverse.

3. **Never leave deprecated documents.** If you change something, update every
   doc that references it in the same session. Stale docs cause confusion.

4. **Memory IS the architecture.** Every forward pass MUST read from the trie.
   Every encoding MUST write to the trie. The trie is not optional, not
   secondary, not a feature. It IS the system. Without memory, ANT outputs
   random bytes. Never implement a code path that bypasses memory.

5. **No prototypes or toys.** Write production-quality code. Every implementation
   must actually work end-to-end with the memory system. A working system that
   does one thing is better than a broken system that attempts five things.

## What This Is

ANT is a 937K parameter byte-level transformer where the weights contain NO
knowledge. ALL knowledge lives in a persistent hierarchical trie memory accessed
via cross-attention. The weights learn HOW to read and write. The trie stores
WHAT was learned.

## Core Architecture

```
  937K params │ 4 layers │ d_model=128 │ 4 heads │ 256 vocab (raw bytes)

  Layer order: Self-Attn → Tag-Attn → Mem-Attn → FFN

  Components:
    Byte Embedding:     256 × 128 (weight-tied with LM head)
    TransformerBlock:   4 layers, each with self-attn + tag-attn + mem-attn + FFN
    AddrNet (×3):       MLP co-processors, 8 clock cycles each, Gumbel-softmax
    V_proj:             Linear(128→128), projects hidden → stored value
    Tag system:         tag_head + tag_gate + tag_register (GRU-style)
    Halt head:          Linear(128→2), decides continue/halt
```

### Memory Data Flow (every single token)

```
  1. READ:  3 AddrNets → 3 addresses → traverse trie → collect ancestor vectors
           → up to 25 vectors → memory cross-attention at each layer
  2. PROCESS: Self-Attn (local) + Tag-Attn (context) + Mem-Attn (global) + FFN
  3. WRITE: V_proj(hidden) → value. Write at 3 leaf addresses + EMA to ancestors.
  4. OUTPUT: LM head → next byte logits. Halt head → continue/halt.
```

This cycle happens for EVERY token. Memory is not a separate phase or mode.

### Hierarchical Trie (Rust)

```
  Rust extension (ant_memory crate) via PyO3
  Arena-allocated nodes (no Python object overhead)
  256-ary tree, 8 levels deep, float32 vectors (128-dim) at every node
  Write: EMA blend at leaf + decay-propagate to all ancestors
  Read:  Collect ALL ancestor vectors along 3 paths = up to 25 unique vectors
  Storage: single flat binary file (header + values + write_counts + adjacency)
```

### AddrNet

```
  3 separate MLP co-processors, ~10.5K params each
  proj_in(128→16) + 8× [out(16→256) → Gumbel-softmax → embed(256,16) → SiLU(MLP)]
  Training: Gumbel-softmax (hard=True, differentiable)
  Inference: argmax (deterministic)
```

## Running

```bash
source .venv/bin/activate  # Python 3.14, PyTorch, datasets, numpy

# Build Rust memory extension (first time only)
cd ant_memory && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release && cd ..

# Train
python3 train.py 2>&1 | tee training.log

# Chat interface
python3 inference.py

# Smoke test
python -c "from config import ModelConfig; from model import ANT; import torch; m = ANT(ModelConfig()); print(sum(p.numel() for p in m.parameters()))"
```

No linters, formatters, or test suites configured.

## Files

```
ant_memory/         Rust trie memory (PyO3 extension, arena-allocated)
  src/trie.rs       HierarchicalTrie: arena nodes, EMA, serialize
  src/memory.rs     MemorySystem: batch read/write, Python API
config.py           ModelConfig (937K) + MemoryConfig — single source of truth
model.py            ANT transformer: AddrNet, MemoryAttention, tag system
engine.py           Core engine: per-token READ→PROCESS→WRITE trie cycle
data.py             Data pipelines: tokenizer, QA/shell/wiki/chat generators
train.py            Training: Phase A/B/C curriculum using engine
inference.py        Terminal chat using engine.generate()
train_colab.ipynb   A100 GPU notebook
```

## Key Conventions

### Pure byte vocabulary

Token ID = raw byte value. `tokenize("hi")` = `[104, 105]`. Vocab is exactly
256. No BPE, no subword tokenizer. Special tokens use ASCII control characters:

```
PAD=0x00  MEM_START=0x01  BOS=0x02  EOS=0x03
MEM_END=0x04  ANS=0x05  NOOP=0x06  UNK=0x1A
```

Do NOT add tokens with IDs ≥ 256.

### Config is the single source of truth

`ModelConfig` in `config.py` controls all dimensions: d_model, n_heads,
vocab_size, special token IDs, memory parameters. `model.py` reads everything
from config. Never hardcode dimensions.

### Spatiotemporal tags

All data tagged as `host/agent/dataplane@ISO-timestamp: content`. The tag
system in each TransformerBlock tracks current speaker/context/mode via a
GRU-gated persistent register.

### Training curriculum

```
Phase A: Base LM with memory — learn language + memory interaction from day 1
Phase B: Freeze base, train AddrNet/V_proj/tags — learn stable address space
Phase C: Unfreeze base (keep AddrNet/V_proj frozen) — learn to use memory
```

### Inference: duplex streaming

Model reads and generates simultaneously. NOOP token (0x06) = "nothing to
say yet". Halt head decides 1–4 memory fetch cycles per output token.

## Gotchas

- `model.forward()` returns `(logits, halt_logits, ...)` — unpack or index `[0]`
- Memory cross-attention uses `memory_keys` and `memory_values` kwargs
- LM head is weight-tied: `F.linear(hidden, model.embed.weight)`
- `data_cache/` stores downloaded data — auto-regenerated if missing
- `checkpoints/` can grow large across experiments
- ANT does not stand for anything — it is just "ANT"
