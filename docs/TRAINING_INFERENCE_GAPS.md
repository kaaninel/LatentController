# LatentController — Training↔Inference Gap Analysis

This document catalogs every identified mismatch between how the model is trained and how the agent uses it at inference time.

**Last updated:** March 30, 2026 (after Phase 5 implementation)

## Critical Gaps

### 1. Batch vs. Streaming Processing — ✅ MOSTLY ADDRESSED (Phase 5)
| | Training (Phase 3-4) | Training (Phase 5) | Inference (Agent) |
|---|---|---|---|
| **Input** | Full sequence at once | Full sequence (causal) | One token at a time |
| **Memory read** | Once per batch step | Per-sample + re-read per ACT step | Every `process_token()` |
| **Memory write** | Every N steps | Every 64 positions + end | Every token |
| **ACT halting** | Soft weighted average | Soft (T annealed to 0.05) | Hard cutoff |

**Phase 5 closes most of the gap:** per-sample memory reads, per-ACT-step memory re-reads, multi-position memory writes. The remaining difference is full-sequence causal attention vs true token-by-token — a reasonable approximation since causal masking prevents future token leakage.

### 2. Soft vs. Hard ACT Halting — ✅ PARTIALLY ADDRESSED (Phase 5)
**Phase 4:** Soft halting with temperature annealed to T=0.1
**Phase 5:** Soft halting with temperature annealed to T=0.05 (near-hard)
**Agent:** Hard cutoff at `p(halt) > 0.5`

At T=0.05, the softmax becomes very peaked — close to hard halting. Full hard halting training (Gumbel-Softmax / straight-through) remains a future improvement.

### 3. NOOP / Selective Emission — ✅ ADDRESSED (Phase 5)
**Phase 5** supports data-driven NOOP targets when a `context_column` is provided:
- Context positions → NOOP target (model learns to stay silent)
- Response positions → real next-token target (model learns to emit)

The model learns autonomously when to absorb vs emit from the data structure.

### 4. Memory Read Frequency — ✅ ADDRESSED (Phase 5)
**Phase 5** `streaming_act_forward()` re-reads memory between ACT steps. On each step, the model computes new addresses from the updated hidden state and reads fresh memory. This matches the agent's behavior exactly.

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

## Phase 5 Implementation Summary

Phase 5 (`train_phase5.py`) closes gaps 1-4 with its unified streaming training:

| Gap | Status | How |
|-----|--------|-----|
| Batch vs Streaming | ✅ Mostly closed | Per-sample memory, multi-position writes, causal masking |
| Soft vs Hard ACT | ✅ Partially closed | Temperature annealed to T=0.05 (near-hard) |
| NOOP training | ✅ Closed | Data-driven NOOP targets from context/response structure |
| Memory read frequency | ✅ Closed | Per-ACT-step memory re-reads in `streaming_act_forward()` |
| Context truncation | ❌ Open | No mid-sequence truncation training |
| Memory dependence | ❌ Open | Needs retrieval-augmented training data |
| Document ingestion | ✅ Partially closed | NOOP targets simulate passive absorption |

### Remaining Future Work
- True token-by-token streaming (vs full-sequence causal approximation)
- Hard ACT halting via Gumbel-Softmax / straight-through estimator
- Retrieval-augmented training for forced memory dependence
- Multi-agent communication training
- Persistent hidden state training across sequence boundaries
