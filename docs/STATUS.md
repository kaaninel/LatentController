# LatentController — Project Status & Objectives

## Current State (March 29, 2026)

### What Works
- ✅ **Phase 1 COMPLETE** — Baseline LM trained on TinyStories, checkpoint on HuggingFace (`kaaninel/latentcontroller`)
- ✅ Core transformer architecture (34M params, 8-layer, 512 d_model)
- ✅ Memory system implementation (trie-indexed, int8 quantized)
- ✅ Agent + Orchestrator inference API
- ✅ Colab notebook for A100 training
- ✅ Hardware auto-detection for A100/H100/T4/CPU

### What's Broken
- ❌ **Phase 2 crashes immediately** — OOM from 800K×800K cosine similarity matrix (2.56 TB)
- ❌ **Phase 3 & 4 memory bug** — All batch samples share one memory lookup (sample 0 only)
- ❌ **Phase 3 position indexing** — Memory write uses wrong position in hidden states
- ❌ **A100 40GB VRAM overestimation** — Hardware configs don't account for ACT iteration memory growth

### What's Missing
- ❌ Streaming training (batch training ≠ token-by-token inference)
- ❌ NOOP token training (selective emission)
- ❌ Hard ACT halting training (only soft halting trained)
- ❌ Configurable dataset (hardcoded to TinyStories)
- ❌ Streaming dataset loading (full dataset loaded to RAM)
- ❌ Phase 5 (unified integration on harder data)

---

## Objectives

### Objective 1: Fix Phase 2 to Run on A100 40GB
**Priority:** P0 — Unblocks everything

1. Replace full cosine similarity precomputation with batched/on-the-fly computation
2. Two approaches:
   - **Option A**: Compute `sim_batch` on-the-fly per training step (small mini-batch against mini-batch)
   - **Option B**: Precompute in chunks to CPU, load relevant chunks per step
3. Validate: Phase 2 runs to completion on A100 40GB without OOM
4. Verify address heads learn meaningful spatial separation

### Objective 2: Fix Per-Sample Memory in Phase 3 & 4
**Priority:** P0 — Without this, memory system is useless

1. Change memory lookup from sample-0-only to per-sample in both Phase 3 and Phase 4
2. Fix `mid_text_pos` calculation in Phase 3 memory write
3. Validate: Different samples in a batch get different memory vectors
4. Profile: Per-sample lookup is slower — ensure it fits in A100 40GB budget
5. Consider: Batch memory lookup with vectorized address computation

### Objective 3: Validate Hardware VRAM Budgets
**Priority:** P1 — Prevent runtime surprises

1. Profile actual VRAM usage for each phase on A100 40GB
2. Enable gradient checkpointing for Phase 4 (ACT iterations multiply memory)
3. Right-size micro_batch values based on real measurements
4. Add VRAM monitoring/logging to training loops
5. Add safety: catch OOM, reduce batch size, retry

### Objective 4: Add Streaming Dataset Support
**Priority:** P1 — Required for scaling beyond TinyStories

1. Make `dataset.py` accept configurable dataset name, split, text column
2. Add HuggingFace streaming mode (`load_dataset(..., streaming=True)`)
3. Implement on-the-fly tokenization with batched processing
4. Support tokenizer retraining for new domains
5. Test with at least one larger dataset (OpenWebText, FineWeb, or similar)

### Objective 5: Address Training↔Inference Gap
**Priority:** P2 — Critical for model quality but not blocking

1. **Hard ACT training**: Add training mode where halt is hard-cutoff (not soft weighted)
   - Anneal from soft→hard during Phase 4 curriculum
2. **NOOP training**: Create training data with `<NOOP>` targets for selective emission
   - 5-10% of positions get NOOP labels (model learns when NOT to emit)
3. **Per-ACT-step memory re-read**: During ACT iterations, re-read memory between steps
4. **Streaming training mode** (Phase 5a): Process sequences token-by-token with growing context

### Objective 6: Complete Phase 2→4 Training Pipeline
**Priority:** P0 — Core deliverable

After bugs are fixed:
1. Run Phase 2 (address head pretraining) — 10K steps
2. Run Phase 3 (memory integration) — 2B tokens
3. Run Phase 4 (ACT training) — 1B tokens
4. Validate at each phase: eval loss, memory utilization stats, halt histograms
5. Push checkpoints to HuggingFace at each phase

### Objective 7: Design and Implement Phase 5
**Priority:** P2 — After Phase 4 works

1. Select harder training dataset (beyond TinyStories)
2. Unfreeze address heads at 0.3× base LR
3. All systems active: memory read/write + ACT + backbone
4. Combined loss: LM + ponder penalty
5. ACT curriculum: max_pondering 4→6, ponder_weight 0.002→0.005
6. Consider streaming training mode for this phase

---

## Prioritized Action Plan

### Immediate (Fix & Run)
1. **Fix BUG-001**: Phase 2 OOM — batched similarity computation
2. **Fix BUG-002**: Per-sample memory lookup in Phase 3 & 4
3. **Fix BUG-003**: Correct mid_text_pos calculation
4. **Profile VRAM**: Measure actual usage per phase, adjust configs
5. **Run Phase 2→4**: Train on A100 40GB, push to HuggingFace

### Short-term (Dataset & Config)
6. **Configurable dataset**: Remove TinyStories hardcoding
7. **Streaming dataset**: HuggingFace streaming mode
8. **VRAM safety**: Auto-reduce batch size on OOM

### Medium-term (Quality)
9. **Hard ACT training**: Anneal soft→hard halting
10. **NOOP training**: Selective emission data
11. **Phase 5 design**: Unified training on harder data

### Long-term (Scaling)
12. **Streaming training mode**: Token-by-token with state persistence
13. **Distributed training**: DDP/FSDP support
14. **Larger model configs**: 60M+ parameter options

---

## Key Design Decisions to Revisit

| Decision | Current | Question |
|----------|---------|----------|
| Multi-token K=4 | Dropped for V1 | Re-add after action tokens work? |
| Backspace tokens | Designed, not implemented | Worth the complexity for V1? |
| Memory grid size | Infinite sparse trie | Need eviction policy for long runs? |
| Vocab size | 16,384 + 128 reserved | Enough for non-TinyStories data? |
| d_model=512 | 34M params | Scale to 640/768 for harder tasks? |
| max_seq_len=512 | Current limit | Need 1024/2048 for longer documents? |

---

## Conversation History Summary

| Conversation | Key Content |
|---|---|
| `fri_mar_27_2026_building_a_looping_transformer_model` | Original architecture design: 6-stage build from baseline through multi-agent. Full feature spec including ACT, diffusion latent, multi-token, backspace, 3D memory, coherence protocol. |
| `sun_mar_29_2026_analysis_of_conversation_on_llm_architecture` | Debugging session turning into full implementation. Produced all current source files. Identified OOM issues, dataset coupling, streaming gaps. Multiple code iterations (v1→v6 for dataset). |
| `sun_mar_29_2026_critical_analysis_and_recommendations_for_ai` | Architectural critique: ACT must stay in computational graph (✅ fixed), K=4 multi-token conflicts with actions (dropped), non-differentiable memory addressing (uses round() — gradient concern), too many losses risk collapse (phased curriculum addresses this). |
| `sun_mar_29_2026_training_pipeline_adjustments_for_new_datasets` | Phase 5 design: unified training, dataset configurability, streaming architecture gap analysis. 22 identified gaps between training and inference behavior. |
