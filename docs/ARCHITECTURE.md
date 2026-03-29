# LatentController — Architecture Reference

## Overview

LatentController is a **looping transformer** that reconciles autoregressive generation with diffusion-style iterative refinement in latent space. The key insight is a clean separation:

- **Weights** = Intelligence (how to think, pattern recognition, reasoning)
- **Memory** = Knowledge (what was computed, facts, context)
- **ACT** = Effort allocation (how hard to think per token)

The model is a ~34M parameter decoder-only transformer that can think in loops before emitting tokens, read/write to a persistent external memory bank, and adaptively allocate compute time per token.

## Core Architecture

### Model Dimensions (current config)
| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `n_heads` | 8 |
| `head_dim` | 64 |
| `ffn_dim` | 2048 |
| `n_layers` | 8 (shared weights for ACT) |
| `max_seq_len` | 512 |
| `vocab_size` | 16,512 (16,384 BPE + 128 reserved) |
| `n_mem_slots` | 9 |
| Total params | ~34M |

### Forward Pass Flow

```
Input: token_ids (B, T_text) + optional memory_vectors (B, n_mem, d_model)

1. Embed tokens → (B, T_text, d_model)
2. If memory present:
   - Prepend [<MEM>, vec1..vec9, </MEM>] → (B, T_text + 11, d_model)
   - Use asymmetric mask: memory sees only memory, text sees memory + causal text
3. Apply 8 transformer layers (RMSNorm + GQA + SiLU FFN + RoPE)
4. Extract text_hidden = hidden[:, n_mem:, :]
5. logits = text_hidden @ embed.weight.T
6. halt_logits = halt_head(text_hidden) → (B, T_text, 2)
```

### Transformer Block
- **RMSNorm** (pre-norm)
- **Multi-head Self-Attention** with RoPE positional encoding
- Flash Attention via PyTorch's `scaled_dot_product_attention`
- **SiLU-gated FFN** (gate + up projections, SiLU, down projection)
- Residual connections around both attention and FFN

### Special Tokens
| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<PAD>` | Padding |
| 1 | `<EOS>` | End of sequence |
| 2 | `<BOS>` | Beginning of sequence |
| 3 | `<UNK>` | Unknown |
| 4 | `<MEM>` | Memory region start |
| 5 | `</MEM>` | Memory region end |
| 6 | `<NOOP>` | No operation (reserved, **not yet trained**) |

## Memory System

### Persistent Trie-Indexed Memory (`memory.py`)
- 3 independent address heads, each producing an 8-dimensional int8 address
- Trie-based indexing for fast exact + neighborhood lookup
- int8 quantized vector storage (512 bytes per entry)
- Adaptive EMA blending: `new_vec = alpha * incoming + (1 - alpha) * existing`
  - Alpha decreases as write_count increases (stabilize over time)
- Write count decay mechanism to prevent staleness

### Memory Read Flow
1. Model produces hidden state `h`
2. 3 address heads compute: `addr_i = round(head_i(h) * 127)` → int8 vector
3. Each head looks up trie: exact match + ±1 neighbors in coarse dims
4. Returns 9 memory vectors (3 heads × 3 results each)
5. Vectors are dequantized to float and prepended to token sequence

### Memory Write Flow
1. Extract representative hidden state from middle of text sequence
2. Compute addresses via same 3 heads
3. Write (with EMA blending) to trie storage
4. Write count incremented, decay applied periodically

## Adaptive Computation Time (ACT)

### Training (Phase 4)
- Shared-weight transformer blocks run multiple iterations
- Soft halting: weighted mixture of all pondering steps
- Ponder loss penalizes unnecessary computation
- Curriculum: gradually increase max_steps (2→4→6) and ponder weight

### Inference (Agent)
- Hard halting: `p(halt) > threshold` → stop immediately
- Variable compute per token based on difficulty
- Simple tokens ("the") → 1 step; complex tokens → up to max_steps

### Ponder Curriculum (Phase 4)
| Step | max_act | ponder_weight | temperature |
|------|---------|---------------|-------------|
| 0 | 2 | 0.0 | 1.0 |
| 10K | 4 | 0.0 | 1.0 |
| 25K | 4 | 0.0005 | 1.0 |
| 50K | 6 | 0.002 | 1.0 |
| 80K | 6 | 0.005 | 1.0 |
| 100K | 6 | 0.005 | 0.5 |
| 120K | 6 | 0.005 | 0.1 |

## Training Pipeline

### Phase 1: Baseline Language Model
- Standard causal LM training on TinyStories
- No memory, no ACT — just the transformer backbone
- Target: 1.5B tokens, LR 3e-4 with cosine schedule
- **Status: COMPLETE** (checkpoint on HuggingFace)

### Phase 2: Address Head Pretraining
- Freeze backbone, train only 3 address heads
- Collect 800K hidden states from Phase 1 model
- Contrastive loss: similar hidden states → similar addresses
- Target: 10K steps
- **Status: BLOCKED** (OOM bug — see BUGS.md)

### Phase 3: Memory Integration
- Unfreeze backbone at lower LR (1e-4)
- Live memory building: read before forward, write every 5 steps
- Train model to use memory context for better predictions
- Target: 2B tokens
- **Status: BLOCKED** (per-sample memory bug — see BUGS.md)

### Phase 4: ACT Training
- Train adaptive computation with ponder curriculum
- Gradually increase loop iterations and ponder penalty
- Model learns when to think more vs. emit quickly
- Target: 1B tokens
- **Status: BLOCKED** (depends on Phase 3)

### Phase 5: Unified Integration (designed, not implemented)
- Load all trained components, unfreeze address heads at 0.3× LR
- Train on harder dataset (beyond TinyStories)
- All systems active simultaneously
- Streaming architecture needed

## Agent & Orchestrator

### Agent (`agent.py`)
- Token stream processor with rolling context buffer (max 501 tokens)
- Calls `process_token()` for each input token
- Uses ACT hard halting during inference
- Integrates memory read/write per token
- Maintains persistent hidden state `self.h` across calls

### Orchestrator (`orchestrator.py`)
- High-level API wrapping agents
- Agent creation, feeding, text generation
- Agent piping (A→B communication)
- Document ingestion mode
- Query interface with thinking budget

## Design Philosophy

### Why Autoregressive Shell + Diffusion Core?
- **AR shell**: Native streaming, backspace support, proven at small scale
- **Diffusion core**: Latent loop = denoising, iterative refinement, warm-start from memory
- **External memory**: Infinite knowledge storage, learned addressing, multi-agent safe
- **Action interface**: Think/read/write/lock, not just text output

### Planned But Not Yet Implemented
- **Multi-token prediction** (K=4 heads) — designed but dropped for V1 (conflicts with action tokens)
- **Backspace tokens** (`<bs1>` through `<bs8>`) — designed, not trained
- **`<NOOP>` selective emission** — reserved token, never trained
- **Multi-agent collaboration** — orchestrator API exists, no training data
- **Streaming training mode** — critical gap between batch training and token-by-token inference
