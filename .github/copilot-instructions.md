# Copilot Instructions — LatentController

## What This Is

An 828K parameter looping transformer with persistent external memory. Research prototype — not a standard ML training repo. The entire training pipeline (tokenizer, datasets, encoders, training loops, evaluation) lives in a single file (`train_micro.py`, ~3000 lines).

## Running

```bash
# Activate the venv first (Python 3.14, PyTorch, datasets, numpy)
source .venv/bin/activate

# QA-only training (~10 min on MPS)
python train_micro.py --chunk_size 16

# Multi-task: LM (shell+wiki) + QA (bAbI) (~25 min)
python train_micro.py --chunk_size 16 --multitask

# Eval only (loads last checkpoint)
python train_micro.py --eval_only

# Quick smoke test — verify imports and forward pass
python -c "from config import ModelConfig; from model import LoopedLatentController; import torch; m = LoopedLatentController(ModelConfig()); print(sum(p.numel() for p in m.parameters()))"
```

There are no linters, formatters, or test suites configured.

## Architecture (4 files, ~4000 lines total)

```
config.py          ModelConfig (828K) + MemoryConfig
model.py           LoopedLatentController — the transformer itself
memory.py          TrieIndex persistent memory system (int8 vectors on disk)
train_micro.py     Everything else: tokenizer, data gen, encoders, training, eval
```

`train_micro.py` is intentionally monolithic — it's a research prototype where rapid iteration matters more than modularity. Don't refactor it into multiple files unless explicitly asked.

### Data flow

1. **Encode**: Passage → sliding window chunks → causal forward → per-token hidden states → memory vectors (quantized to int8, stored in TrieIndex by learned address)
2. **Decode**: Question → causal forward with memory cross-attention (9 retrieved vectors per read) → answer logits
3. **LM mode**: Raw text → standard causal forward (no sliding window, no memory) → next-byte prediction

### Two forward paths

The model has two distinct forward paths — this is a critical performance decision:
- **QA path**: `encode_streaming_memory()` → `sliding_lm_encode()` with memory cross-attention. Slow (~0.2 it/s if used for everything).
- **LM path**: `model(token_ids=...)` — standard causal forward, no memory. Fast (~1.5 it/s).

Multi-task training alternates between these. Never use the QA path for plain LM training.

### Transformer block structure

Each of the 4 layers: **RMSNorm → Self-Attention (causal, RoPE) → RMSNorm → Memory Cross-Attention → RMSNorm → SiLU FFN**

Memory cross-attention uses learned inverse-temperature per head and attends to external memory vectors (not the input sequence).

## Key Conventions

### Pure byte vocabulary (no tokenizer)

Token ID = raw byte value. `tokenize("hi")` returns `[104, 105]`. Vocabulary is exactly 256. There is no BPE, no subword tokenizer, no offset.

Special tokens are ASCII control characters — do not add new special tokens with IDs ≥ 256:
```
PAD=0x00(NUL)  BOS=0x02(STX)  EOS=0x03(ETX)  ANS=0x05(ENQ)
MEM_START=0x01(SOH)  MEM_END=0x04(EOT)  NOOP=0x06(ACK)  UNK=0x1A(SUB)
```

### Training curriculum matters

QA training uses a staged curriculum (D1→D2). Skipping stages or changing ratios breaks learning:
- **Phase D1** (first 30% of steps): No passage in context, frozen encoder. Forces the model to read from memory instead of copying from context.
- **Phase D2** (remaining 70%): No passage in context, differentiable encoder. End-to-end gradient flow through memory.

### Memory system is external and persistent

`MemorySystem` in `memory.py` is a trie-indexed flat file of int8 vectors, not part of the model's parameters. Three address heads (each Linear(128→8)) produce addresses for lookup. Neighbor search (±1 on each dimension) provides implicit clustering. EMA blending prevents catastrophic overwriting.

### Config is the single source of truth for model dimensions

`ModelConfig` in `config.py` controls everything: d_model, n_heads, vocab_size, special token IDs, memory parameters, and chunk encoding settings. `model.py` reads all dimensions from the config object passed to `LoopedLatentController.__init__`. Don't hardcode dimensions.

### Experiment results to preserve

Current best: **99.5% QA accuracy** (bAbI 1/2/3-fact) with simultaneous LM training (loss 2.49). Achieved with `--chunk_size 16 --multitask`. Any architecture changes should be validated against this baseline.

## Common CLI Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--chunk_size` | `None` (uses config) | Sliding window chunk size for encoder |
| `--multitask` | off | Enable LM + QA multi-task training |
| `--encoder_mode` | `sentence` | Encoder type: `sentence`, `sliding`, `sliding_lm`, `enc_dec` |
| `--num_passes` | 4 | Number of sliding window passes |
| `--n_shell` / `--n_wiki` | 5000 | Number of shell/Wikipedia training examples |
| `--lm_weight` / `--qa_weight` | 0.5 | Loss weights for multi-task |
| `--phase_d_steps` | 3000 | Main training phase steps |
| `--device` | auto-detect | Force `cpu`, `mps`, or `cuda` |

## Codebase-Specific Gotchas

- `model.forward()` returns a tuple `(logits, halt_logits, ...)` — always unpack or index `[0]` for logits.
- Memory cross-attention uses `memory_keys` and `memory_values` kwargs (not `memory_kv`).
- The LM head is weight-tied with the embedding: `F.linear(hidden, model.embed.weight)`.
- `data_cache/` stores downloaded Wikipedia sentences — auto-regenerated if missing.
- `checkpoints/` can grow large (1.5GB+) across experiments. Each run saves to a subdirectory under `--output_dir`.
