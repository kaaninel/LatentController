# Tiled Agent Architecture — Proposal

> **Status**: Design proposal. Not yet implemented.

---

## Motivation

ANT's per-token trie I/O is a structural bottleneck: reads and writes dominate step time
even after async prefetch and CPU-resident AddrNets. For long-context ingestion the problem
compounds — a document of L tokens requires L/stride sequential outer steps, each with a
full trie read and write.

The trie is designed as *persistent cross-session memory*. Using it as the only
communication channel within a single document is wasteful — it conflates "remember this
across sessions" with "pass context to the next token." This proposal separates those two
concerns and uses them in both directions (ingestion and generation).

---

## Two-Memory Separation

| | Trie (long-term) | Cross-agent attention (short-term) |
|---|---|---|
| Scope | Across documents and sessions | Within the current tiled batch |
| Write cost | Trie I/O (~1s per step) | Zero (tensor ops) |
| Read cost | CpuAddrBank + trie traverse | Mean-pool of peer hidden states |
| Survives after batch | Yes (EMA-blended) | No |
| Implemented | Yes | Proposed |

Cross-agent attention is fed as extra slots in the **existing** `mem_attn` cross-attention
layer. No architecture change is required — `mem_attn` already handles variable-length
inputs via the attention mask.

---

## Tiling

A document is split into B non-overlapping sequential chunks of T tokens each, processed
as one batch of shape `(B, T, D)`. Agents are ordered by document position.

```
Document:  [ chunk 0 ][ chunk 1 ][ chunk 2 ] ... [ chunk B-1 ]
Agent:         0          1          2                 B-1
Context:    trie only  trie+peer0  trie+peer0,1    trie+all peers
```

### Causal Peer Summary

Agent i receives one additional cross-attention slot alongside the 25 trie slots:

```
peer_ctx[i] = mean(h_mean[0], h_mean[1], ..., h_mean[i-1])
peer_ctx[0] = zeros  (no preceding agents)
```

where `h_mean[j] = mean(hidden_p1[j], dim=T)` — the mean hidden state from agent j's
pass-1 forward.

This is a running average of all preceding chunk representations, computed once per outer
step as a cheap O(B×D) mean-pool. It collapses the preceding document context into one
vector that the current agent attends over.

---

## Tiled Ingestion

B agents read B sequential chunks of a long document in parallel on GPU. Each agent writes
its understanding to the trie. Since chunks are content-adjacent, their trie addresses will
partially overlap — EMA blending naturally strengthens recurring representations.

**Trie I/O reduction**: Instead of one write per inner step per agent, write ONE vector per
document — the mean of all agents' hidden states — as the document-level trie entry.

```
doc_summary = mean(h_mean[i] for i in 0..B)
```

For B=8, INNER_STEPS=8: this is a 64× reduction in trie writes per outer step.

### NOOP Output During Ingestion

While ingesting, agents output NOOP (byte 0x06 — already in vocabulary). The `halt_head`
decides when an agent has enough context to start producing real output. Agents 0..B-2
are in "ingestion mode" and are supervised to output NOOP. Agent B-1 (last, with full
causal context from all preceding agents) can begin generating real tokens.

---

## Tiled Generation

The same mechanism applies to generation. B agents each generate a K-token segment of the
output, where K = total_output_length / B.

```
Agent 0: generates tokens [0 : K]     — reads trie, peer_ctx = zeros
Agent 1: generates tokens [K : 2K]    — reads trie, peer_ctx = mean(h_mean[0])
...
Agent B-1: generates tokens [(B-1)K : BK] — reads trie, peer_ctx = mean(h_mean[0..B-2])
```

### Training

Ground truth is fully known during training. All B agents are trained simultaneously in
one forward pass. Each agent's loss is next-token prediction on its K-token segment.
Peer context flows through `inject_peer_context()` — same call used for ingestion.

### Inference

**Sequential chunk mode**: Agents run one at a time. Agent i generates its K tokens, then
passes `h_mean[i]` to agent i+1. The speedup comes from shorter local self-attention
contexts: O(B · K²) instead of O((B·K)²).

**Speculative parallel mode** (future): All agents generate simultaneously using provisional
peer context. After each chunk, peer context is updated and tokens near chunk boundaries
may be revised. Analogous to speculative decoding at chunk granularity.

---

## Full Duplex: Concurrent Ingestion + Generation

The most general mode tiles a context into B agents where some ingest and others generate:

```
Agents 0..B/2-1 :  ingest a long prompt (output NOOP)
Agents B/2..B-1 :  generate the continuation (real tokens)
```

Both groups share:
- The same trie (long-term knowledge)
- Cross-agent peer context (short-term, within this batch)

The ingesting agents' final `h_mean` is passed as peer context to the first generating
agent. The model learns to produce useful context summaries through the peer cross-attention
training signal.

This directly implements the "duplex streaming" design intent: the model reads and generates
simultaneously, coordinated via peer attention instead of requiring sequential processing.

---

## Expected Impact

| Metric | Current | With tiling |
|---|---|---|
| Trie writes per document | B × T × INNER_STEPS | 1 |
| Trie reads per document | per outer step | 1 |
| Cross-agent context cost | 0 | O(B×D) mean-pool (negligible) |
| Context span per outer step | T tokens | B × T tokens |
| Inference time for L tokens | O(L²) (full self-attn) | O(B · (L/B)²) = O(L²/B) |

---

## Implementation Notes

When this is implemented, the required changes are:

| File | Change |
|---|---|
| `config.rs` | `N_CHUNK_AGENTS`, `CHUNK_STRIDE`, `N_PEER_SLOTS`, `K_TOKENS_PER_AGENT`, `TRIE_WRITE_GRANULARITY`, `NOOP_LOSS_WEIGHT` |
| `data.rs` | `sliding_window_chunks()`, `tile_sequence_for_generation()` |
| `engine.rs` | `inject_peer_context()`, `write_doc_summary()` |
| `train.rs` | Phase C tiled paths (ingestion, generation, mixed), NOOP supervision |
| `inference.rs` | `generate_chunked()` for sequential chunk generation |

No changes to `model.rs` or `trie.rs`. The existing `mem_attn` and `MemorySystem`
interfaces are sufficient.
