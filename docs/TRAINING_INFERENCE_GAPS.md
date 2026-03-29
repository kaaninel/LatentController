# LatentController — Training↔Inference Gap Analysis

This document catalogs every identified mismatch between how the model is trained and how the agent uses it at inference time. These gaps must be addressed for the model to work correctly in production.

## Critical Gaps

### 1. Batch vs. Streaming Processing
| | Training | Inference (Agent) |
|---|---|---|
| **Input** | Full sequence `[BOS, t1, ..., t499, EOS, PAD]` | One token at a time via `process_token()` |
| **Context** | All 512 positions available at once | Growing buffer from 1 to 501 tokens |
| **Memory read** | Once per batch step | Every `process_token()` call |
| **Memory write** | Every N training steps | Every token (when enabled) |
| **Hidden state** | Fresh each batch | Persistent `self.h` across calls |
| **ACT halting** | Soft weighted average | Hard cutoff at threshold |

**Impact:** The model was never trained to handle incrementally growing context, persistent state, or per-token memory access. These are fundamental to the agent's operation.

**Mitigation:** Phase 5a streaming training — simulate the agent loop during training.

### 2. Soft vs. Hard ACT Halting
**Training (Phase 4):** `output = Σ(halt_weight_t × hidden_t)` — weighted average of all ACT steps.
**Inference (Agent):** `if p(halt) > 0.5: break` — hard cutoff, only use last step's output.

The model learns a smooth blending function but is evaluated with binary decisions. It may never learn to produce a clean halt probability because soft halting always "works" during training.

**Mitigation:** Anneal from soft→hard halting during Phase 4 curriculum. At later stages, use Gumbel-Softmax or straight-through estimator for halt decisions.

### 3. NOOP / Selective Emission
**Training:** Every position has a real token target. Model always predicts next token.
**Inference:** Agent can return `None` (stay silent). Uses NOOP concept but token ID 6 is never trained.

**Mitigation:** Create training data with `<NOOP>` targets at 5-10% of positions. Positions where the model should "think" but not "speak" get NOOP labels.

### 4. Memory Read Frequency
**Training (Phase 3-4):** Memory read once at batch start, held constant through forward pass and all ACT iterations.
**Inference:** Memory re-read after each token emission, potentially updated between reads.

**Mitigation:** During ACT iterations in training, re-read memory between steps (expensive but faithful).

## Medium Gaps

### 5. Context Truncation
**Training:** All sequences are pre-truncated to `max_seq_len=512`.
**Inference:** Agent silently truncates context at 501 tokens, dropping oldest tokens.

The model never sees the boundary effects of mid-sequence context truncation.

### 6. Memory Dependence
**Training:** No training phase creates scenarios where correct answer REQUIRES memory.
**Inference:** Memory is the primary mechanism for knowledge beyond training data.

Without forced memory dependence, the model learns to ignore memory entirely (weights are sufficient for TinyStories).

**Mitigation:** Create training scenarios where answer is only available via memory read:
- Train on facts, store in memory, mask from context, require model to recall
- Retrieval-augmented training data

### 7. Document Ingestion Mode
**Training:** All sequences are standalone stories.
**Inference:** Orchestrator has `ingest_document()` mode where agent absorbs text without responding.

The model doesn't distinguish "absorb and store to memory" from "read and respond." No training for passive ingestion.

### 8. Multi-Agent Communication
**Training:** Single-agent training only.
**Inference:** Orchestrator supports agent piping (A→B), shared memory, coherence protocol.

No training data for multi-agent scenarios.

## Minor Gaps

### 9. Rolling Context Window
The 501-token rolling window works mechanically, but the model was never exposed to sequences where early tokens were dropped mid-generation.

### 10. Token Budget / Thinking Budget
Orchestrator's `query()` accepts a thinking budget parameter. No training signal teaches the model to respect thinking limits.

### 11. Error Recovery
No training for recovering from memory corruption, address space drift, or unexpected context changes.

---

## Recommended Phase 5a: Streaming Training

To close the critical gaps, a streaming training mode should:

1. **Process tokens sequentially** with growing context (not full-sequence teacher forcing)
2. **Read memory every step** (not once per batch)
3. **Use hard ACT halting** (annealed from soft)
4. **Include NOOP targets** (5-10% of positions)
5. **Carry hidden state** across positions within a sequence (simulate agent's `self.h`)
6. **Write memory during training** after each ACT completion

This is significantly more expensive than standard training but is necessary for the agent to work correctly at inference time.
