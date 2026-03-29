# LatentController — Known Bugs & Issues

## Critical Bugs (Training Broken)

### BUG-001: Phase 2 OOM — Full Cosine Similarity Matrix ✅ FIXED
**File:** `train_phase2.py:185`
**Status:** Fixed — cosine similarity now computed on-the-fly per mini-batch (B×B instead of N×N)

```python
cos_sims = torch.mm(hiddens_norm, hiddens_norm.T)  # (N, N) where N=800,000
```

With N=800,000 hidden states, this creates an 800K × 800K matrix = **2.56 TB of memory**. Immediate OOM on any hardware.

The comment on line 182 says "done in batches to save memory" but the code does NOT batch — it computes the full matrix in one shot.

**Fix:** Compute similarities in mini-batches:
```python
# Only compute sim_batch for the current mini-batch indices
idx = torch.randperm(N, device=device)[:batch_size]
batch = hiddens[idx]
batch_norm = F.normalize(batch.float(), dim=-1)
all_norm = F.normalize(hiddens.float(), dim=-1)
sim_batch = torch.mm(batch_norm, all_norm[idx].T)  # (batch_size, batch_size) — fits in memory
```
Or precompute in chunks to CPU and index into it.

---

### BUG-002: Phase 3 & 4 — Shared Memory for Entire Batch ✅ FIXED
**File:** `train_phase3.py:270-277`, `train_phase4.py:246-253`
**Status:** Fixed — per-sample memory lookup now loops over all batch samples

```python
h0 = hid_nm[0, -1, :]                              # Only sample 0's hidden state
addrs = model.compute_addresses(h0)                  # One address for entire batch
mem_tensor = mem_tensor.expand(inp.size(0), -1, -1)  # Same memory for all B samples
```

Every sample in the batch receives **identical memory vectors** based solely on the first sample's last-token hidden state. The model cannot learn per-sample memory associations.

**Fix:** Per-sample memory lookup:
```python
mem_list = []
for b in range(inp.size(0)):
    h_b = hid_nm[b, -1, :]
    addrs_b = model.compute_addresses(h_b)
    addr_bs_b = [addr_bytes(a) for a in addrs_b]
    vecs_b = train_memory.read_memory(addr_bs_b)
    mem_list.append(memory_vecs_to_tensor(vecs_b, cfg.d_model, device))
mem_tensor = torch.stack(mem_list, dim=0)  # (B, n_mem, d_model)
```

---

### BUG-003: Phase 3 — Wrong Position Index for Memory Write ✅ FIXED
**File:** `train_phase3.py:306`
**Status:** Fixed — now correctly calculates `text_start + text_len // 2`

```python
mid_text_pos = cfg.n_mem_positions + hidden.shape[1] // 2
```

When memory is present, `hidden.shape[1]` = 512 (11 mem + 501 text). So `mid_text_pos = 11 + 256 = 267`. This points to position 267 in the combined sequence, which is within the text region — but `hidden.shape[1] // 2` is wrong because it includes memory positions in the division.

**Correct calculation:**
```python
text_start = cfg.n_mem_positions  # 11
text_len = hidden.shape[1] - text_start  # 501
mid_text_pos = text_start + text_len // 2  # 11 + 250 = 261
```

The difference is 6 positions (267 vs 261), which means the model writes hidden states from slightly later in the sequence than intended. Not catastrophic but incorrect.

---

## Hardware / VRAM Issues

### BUG-004: Phase 2 Hardware Config — Batch Size Meaningless ✅ FIXED
**File:** `hardware.py:98`
**Status:** Fixed — batch sizes reduced to reasonable values, on-the-fly similarity eliminates precompute OOM

```python
'phase2': {'batch_size': 16000},  # A100 40GB
```

This batch_size controls mini-batch during contrastive training, which is fine. But the actual OOM happens in the **precomputation** step (BUG-001) which ignores this setting entirely. Even with batch_size=16000, the 800K×800K matrix is always computed in full.

---

### BUG-005: A100 40GB Gradient Checkpointing Disabled ✅ FIXED
**File:** `hardware.py:105`
**Status:** Fixed — gradient checkpointing now enabled for all GPU configs

```python
'gradient_checkpointing': False,  # A100 40GB config
```

Phase 1 with micro_batch=160 and no gradient checkpointing uses ~9GB, well within limits. But Phase 4 with ACT iterations (up to 6 forward passes per step) could spike memory. The config should enable checkpointing for Phase 4 at minimum.

---

### BUG-006: Hardware Detection Fragile Without psutil
**File:** `hardware.py:12-14`
**Severity:** LOW — Falls back to 0 RAM

```python
try:
    import psutil as _psutil
except ImportError:
    _psutil = None
```

If psutil is not installed, `info['ram_gb']` becomes 0, which may cause incorrect hardware tier selection.

---

## Architectural Gaps (Training ↔ Inference Mismatch)

### GAP-001: No Streaming Training — Critical
**Severity:** CRITICAL for real-world use

**Training**: Processes full sequences `[BOS, tok1, ..., tok499, EOS, PAD]` at once.
**Inference**: Agent calls `process_token()` one token at a time with growing context.

The model is **never trained** in the mode it actually runs during inference. This creates mismatches in:
- Memory read frequency (once per batch vs. every token)
- ACT behavior (soft weighted vs. hard cutoff)
- Hidden state persistence (fresh each batch vs. carried across calls)
- Context growth (full sequence vs. incrementally growing)

### GAP-002: NOOP Token Never Trained
**Severity:** HIGH

`noop_id=6` is reserved in config but never appears in training data. The agent supports selective emission (return None when uncertain), but the model was never taught when to stay silent.

### GAP-003: Soft vs Hard ACT Halting
**Severity:** HIGH

Phase 4 trains with **soft halting** (weighted average of all loop steps). The agent uses **hard halting** (`p(halt) > 0.5` → stop immediately). The model never learns the hard cutoff behavior it will use at inference time.

### GAP-004: Memory Not Per-ACT-Step
**Severity:** MEDIUM

The agent re-reads memory every call, but during ACT iterations within Phase 4 training, memory is read once at the beginning and held constant through all ACT steps. The model doesn't learn to re-consult memory mid-thought.

### GAP-005: No Training for Agent State Persistence
**Severity:** MEDIUM

The agent maintains `self.h` (persistent hidden state) across `process_token()` calls. Training always starts fresh from full context — there's no training for carrying state across inference calls.

### GAP-006: Dataset Hardcoded to TinyStories
**Severity:** HIGH for scaling

`dataset.py` hardcodes `load_dataset("roneneldan/TinyStories", split="train")`. No parameter to switch datasets. Tokenizer is trained only on 500K TinyStories samples with `vocab_size=16384`.

### GAP-007: No Streaming Dataset Loading
**Severity:** MEDIUM for large datasets

The entire dataset is tokenized and loaded into RAM as a single tensor. For TinyStories (~2.1M stories) this is ~4.3GB, acceptable. For larger datasets (OpenWebText, RedPajama), this approach won't scale.

---

## Code Quality Issues

### ISSUE-001: Agent Assumes batch_size=1
**File:** `agent.py:99`

```python
self.h = hidden[0, -1, :].detach()
```

Works because agent always processes single tokens, but fragile if anyone tries batch inference.

### ISSUE-002: Silent Context Truncation
**File:** `agent.py:57-58`

```python
if len(self.context_buffer) > self.cfg.n_text_positions:
    self.context_buffer = self.context_buffer[-self.cfg.n_text_positions:]
```

Context truncation happens silently with no logging. Model may behave unexpectedly when early context is dropped.

### ISSUE-003: Memory Grows Unbounded
**File:** `memory.py`

No eviction policy. Trie grows forever on disk. No deduplication for identical vectors.

### ISSUE-004: No Distributed Training Support
No DDP/FSDP anywhere. Single-GPU only.
