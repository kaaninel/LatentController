# ANT Project Coherency Audit — 2026-04-08

## Executive Summary

**~30% of the codebase is dead or vestigial.** The project evolved from a persistent-memory
trie-indexed architecture to a cross-attention memory-based chat model, but the old code
was never cleaned out. The current working pipeline is narrow: `train_chat()` in train_micro.py
called by train_overnight.py/train_colab.ipynb, using only `encode_sentence_*` encoders and
`ChatMemoryDataset`. Everything else is legacy.

---

## CRITICAL: Completely Dead Files

### 1. memory.py (347 lines) — DEAD
- MemorySystem / TrieIndex is **imported but never instantiated**
- Original design: persistent trie-indexed int8 vector store
- Current design: encoder-computed KV tensors → cross-attention (no persistence layer)
- MemoryConfig in config.py also dead (never instantiated)
- **Action: DELETE entirely**

### 2. chat_old.py (552 lines) — SUPERSEDED
- Completely replaced by chat.py (Terminal Canvas rewrite)
- Zero references anywhere in the codebase
- **Action: DELETE entirely**

### 3. fused_kernel.py — DEAD
- Experimental Metal/MLX kernel, zero imports anywhere
- Comments document it underperformed (~0.64x vs MLX standard)
- **Action: DELETE or move to archive/**

**Total dead files: ~900 lines**

---

## HIGH: Dead Code in train_micro.py (~1200 lines, ~30% of file)

### Dead Encoder Implementations
| Function | Lines | Why Dead |
|----------|-------|----------|
| encode_streaming_memory | ~1438-1540 | Per-token streaming encoder, never used by train_chat |
| encode_enc_dec | ~1544-1622 | Bidirectional enc-dec, never used by train_chat |
| encode_sliding_window_memory | (sliding variants) | Not used — train_chat hard-codes encode_sentence_* |
| 6 wrapper functions | ~1624-1952 | encode_*_frozen/differentiable wrappers for unused modes |

### Dead Training Functions
| Function | Lines | Why Dead |
|----------|-------|----------|
| train_phase_b | ~1174-1301 | Address head training — not in train_chat pipeline |
| train_phase_c | ~1302-1411 | ACT curriculum — not in train_chat pipeline |
| train_multitask | ~2331-2511 | Old multi-task — replaced by train_chat phases |
| train_phase_d | ~2510-2900 | Old memory QA — replaced by train_chat Phase 2 |
| train_sliding_lm | ~1057-1110 | Standalone sliding LM — folded into train_chat Phase 1 |

### Dead Dataset Classes
| Class | Lines | Why Dead |
|-------|-------|----------|
| TokenBudgetLoader | ~886-946 | 100% dead, zero references anywhere |
| ContextQADataset | ~950-995 | Only used by phases A-C (not train_chat) |
| MemoryQADataset | ~997-1050 | Only used by phase D (not train_chat) |

### Dead Data Templates (when use_hf_chat=True)
- _CHAT_GREETINGS, _CHAT_FACTUAL, _CHAT_TECHNICAL, _CHAT_YESNO,
  _CHAT_OPENENDED, _CHAT_SHELL, _CHAT_MARKDOWN (~180 lines)
- generate_chat_data() function — fallback only, never triggered with HF data

---

## MEDIUM: Vestigial Model Components

### model.py
| Component | Lines | Issue |
|-----------|-------|-------|
| halt_head | ~404-406 | Created, returned in forward(), NEVER used for loss in train_chat |
| temporal_emb | ~414-417 | nn.Embedding created but never referenced in any forward path |
| forward_memory() | ~466-499 | Dead method, never called anywhere |
| Legacy prepend mode | ~545-554 | memory_vectors prepending path, abandoned |

### config.py
| Field | Issue |
|-------|-------|
| chunk_size, slots_per_chunk, max_temporal_chunks | In ModelConfig but only used locally in train_micro.py |
| MemoryConfig class (lines 45-54) | Never instantiated, designed for deleted TrieIndex system |

---

## MEDIUM: Stale/Misleading Documentation

### docs/training_report.md
- Claims LM loss ~1.9, README says ~1.5
- Says "Window: 16 tokens" — should be "bytes"
- No mention of memory-based chat architecture
- Still frames the project as "proof of concept" when QA is at 100%

### .github/copilot-instructions.md
- Says train_micro.py is "~3000 lines" — it's ~4250 now
- Best result listed as "99.5% QA" — it's 100% now
- References --encoder_mode CLI flag — completely ignored by train_chat()

### CLI arg --encoder_mode
- Defined with choices ["enc_dec", "streaming", "sentence", "sliding", "sliding_lm"]
- train_chat() hard-codes encode_sentence_frozen — encoder_mode is IGNORED

---

## LOW: Cleanup Items

### data_cache/
- Stale wiki_sentences variants (100, 200, 500, 1000, 20000, 200000)
- Only 5000 and 50000 actively used
- hf_chat_10000_600b.txt is old format (pre-30K download)

### checkpoints/
- checkpoint_colab_best.pt — stale download copy
- "overnight backup/" — non-standard naming

### model_mlx.py
- halt_head missing bias initialization (minor)
- Generally in sync but will diverge after model.py cleanup

### requirements.txt
- Missing huggingface_hub (used by train_overnight.py)
- mlx not listed (optional, for model_mlx.py)

---

## Architecture Alignment Check

### What the code SHOULD be (current goals):
1. Sliding window causal LM (W=8, passes=4) for all text
2. Memory-based chat: user → frozen encoder → memory → cross-attention
3. QA via cross-attention memory (100% bAbI)
4. Byte-level (256 vocab), 828K params
5. Unlimited context via sliding window + memory

### What contradicts these goals:
1. memory.py describes a persistent storage system that doesn't exist in the pipeline
2. halt_head wastes 258 params on a feature never integrated into training
3. temporal_emb wastes 4,096 params on an embedding never used in forward
4. Multiple encoder options documented but only sentence encoder works
5. Old training phases (B, C, old D) clutter understanding of the actual pipeline

---

## Recommended Actions (Priority Order)

### Immediate (before next training run):
1. DELETE memory.py, chat_old.py, fused_kernel.py
2. Remove MemoryConfig from config.py
3. Update requirements.txt (add huggingface_hub)

### Soon (code health):
4. Remove dead encoder functions from train_micro.py (~400 lines)
5. Remove dead training functions (phases B, C, old D, multitask) (~800 lines)
6. Remove TokenBudgetLoader, ContextQADataset, MemoryQADataset
7. Remove --encoder_mode CLI arg or restrict to "sentence" only
8. Clean synthetic chat templates (keep generate_chat_data as HF fallback only)

### Model cleanup (requires retraining):
9. Remove halt_head from model.py (saves 258 params, breaks checkpoint compat)
10. Remove temporal_emb from model.py (saves 4,096 params, breaks checkpoint compat)
11. Remove forward_memory() dead method
12. Remove legacy prepend mode

### Documentation:
13. Update copilot-instructions.md (line count, encoder mode, best results)
14. Update training_report.md or delete (mostly obsolete)
15. Clean data_cache/ (remove unused wiki_sentences variants)
