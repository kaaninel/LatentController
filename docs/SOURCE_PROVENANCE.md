# Source Provenance Tagging — Design Report

## Problem

The model memorizes facts but has no concept of *who said it* or *when*. When
contradictory information arrives from different sources or different times, the
memory system treats all facts equally. We need every piece of data tagged with:

- **Dataplane** — where the data came from (reddit, wikipedia, shell, etc.)
- **Author** — who created it (username, article ID, system)
- **Timestamp** — when it was created (ISO 8601 full precision)

The model should memorize all versions and let the query context decide which to trust.

## Tag Format

```
[dataplane/author@2026-03-06T12:32:17Z]
```

Examples:
```
[reddit/u_testuser@2026-03-06T12:32:17Z] John is in the kitchen.
[wikipedia/Paris@2026-01-15T08:00:00Z] The population of Paris is 2.1 million.
[reuters/editor7@2026-03-07T09:15:00Z] The population of Paris is 2.2 million.
[shell/root@2026-04-01T14:22:33Z] ls -la /home/user
```

### Byte Budget

```
Component         Bytes    Example
─────────────     ─────    ───────────────────────────────
Opening bracket   1        [
Dataplane         4-12     reddit, wikipedia, shell, news
Slash             1        /
Author            3-20     u_bob, editor7, article_12345
At-sign           1        @
ISO 8601 full     20       2026-03-06T12:32:17Z
Closing bracket   1        ]
Space             1        (separator)
─────────────     ─────
TOTAL             32-57    typical ~40 bytes
```

## Capacity Impact & Mitigation

Current limits and impact of a ~40-byte prefix:

```
┌─────────────────────┬───────┬──────────┬──────────┬─────────────┐
│ Dataset             │ Limit │ Usable   │ Capacity │ Severity    │
│                     │       │ w/ tag   │ Loss     │             │
├─────────────────────┼───────┼──────────┼──────────┼─────────────┤
│ QA passage (memory) │ 128   │ ~88      │ 31%      │ HIGH        │
│ LM text (context)   │ 192   │ ~152     │ 21%      │ MODERATE    │
│ Shell commands      │ ~100  │ ~60      │ 40%      │ HIGH        │
│ Wiki sentences      │ 30-300│ varies   │ varies   │ LOW-MOD     │
└─────────────────────┴───────┴──────────┴──────────┴─────────────┘
```

### Recommended: Increase max_passage_len to 192

The 128 limit is arbitrary. Increasing to 192 (matching max_seq_len) gives
152 bytes for content after a 40-byte tag — enough for all current bAbI
patterns (typically 40-100 bytes) and Wikipedia sentences.

This costs ~50% more memory encoding time per example but the encoder is
fast (streaming chunks). The alternative — short tags — loses expressiveness.

```
max_passage_len: 128 → 192    (config change)
max_seq_len:     192 → 256    (optional, for LM headroom)
```

## Implementation Plan

### Phase 1: Tag Infrastructure

Add a `SourceTag` dataclass and tag-generation utilities:

```python
@dataclass
class SourceTag:
    dataplane: str     # "reddit", "wikipedia", "shell", "news"
    author: str        # "u_testuser", "editor7", "article_12345"
    timestamp: str     # "2026-03-06T12:32:17Z"

    def __str__(self):
        return f"[{self.dataplane}/{self.author}@{self.timestamp}]"
```

This gets prepended to text *before* tokenization. Since our tokenizer is
identity (byte = token ID), the tag becomes the first ~40 tokens in the
byte stream. The model sees:

```
[0x5B] [0x72 0x65 0x64 ...] [0x2F] [0x75 0x5F ...] [0x40] [0x32 0x30 ...] [0x5D] [0x20] ...content...
  [         reddit              /      u_bob          @      2026...          ]     SP
```

### Phase 2: Mock Data with Contradictions

Current datasets have no contradictions. We need synthetic multi-source
data where the same entity has different attributes:

**Contradiction Types:**

1. **Location conflicts** (builds on bAbI patterns):
   ```
   [reddit/u_alice@2026-03-06T10:00:00Z] John is in the kitchen.
   [reddit/u_bob@2026-03-06T10:05:00Z] John is in the garden.
   Question: Where is John?
   ```
   The model should store BOTH in memory. Answer depends on which source
   the question trusts (or recency, or both).

2. **Fact updates over time** (same source, different dates):
   ```
   [wikipedia/Paris@2025-01-01T00:00:00Z] The capital of France has 2.1M people.
   [wikipedia/Paris@2026-01-01T00:00:00Z] The capital of France has 2.2M people.
   ```

3. **Cross-source disagreement** (different dataplanes):
   ```
   [news/reuters@2026-03-06T12:00:00Z] The event starts at 3pm.
   [reddit/u_organizer@2026-03-06T14:00:00Z] The event starts at 4pm.
   ```

**New Generator: `generate_sourced_facts()`**

Creates QAExample tuples where:
- Multiple sources report on same entities
- Some agree, some contradict
- Questions may specify source context: "According to u_alice, where is John?"
- Questions may ask for latest: "Where is John?" (no source hint — any valid answer accepted)

### Phase 3: Dataset Integration

Each data source gets tagged at generation time:

```
bAbI generators:
  generate_single_fact()  → random SourceTag per fact sentence
  generate_two_facts()    → each fact gets its own tag
  generate_three_facts()  → each fact gets its own tag

Shell generators:
  _gen_simple_command()   → [shell/root@timestamp]
  _gen_pipe_chain()       → [shell/admin@timestamp]

Wikipedia loader:
  Each sentence            → [wikipedia/article_title@article_date]

Contradiction generator (NEW):
  generate_contradiction() → multiple tags, same entity, different values
```

### Phase 4: Training Curriculum Adjustment

The model must learn three things:
1. Parse the `[dataplane/author@timestamp]` format (structural)
2. Associate tagged content with the tag in memory (binding)
3. Retrieve the right fact when query specifies a source (selective recall)

**Suggested curriculum:**

```
Stage 1 (warmup):      Tagged facts, no contradictions
                        Model learns tag structure is part of content
                        Same curriculum as today but with tags prepended

Stage 2 (binding):     Tagged facts, questions include source hints
                        "According to [reddit/u_alice], where is John?"
                        Forces model to bind tag → content in memory

Stage 3 (conflict):    Contradicting facts from different sources
                        Questions with source hints → specific answer
                        Questions without hints → any valid answer accepted
                        (multi-label: both "kitchen" and "garden" are correct)

Stage 4 (temporal):    Same source, different timestamps
                        "Where was John on March 6?" vs "Where is John now?"
                        Tests whether model learns to use timestamps
```

## Data Flow Diagram

```
  Raw text generation
  ┌─────────────────────────────────────────────────┐
  │  generate_single_fact()                          │
  │    name = "John", loc = "kitchen"                │
  │    tag = SourceTag("reddit","u_alice","2026...") │
  │    passage = f"{tag} {name} is in the {loc}."    │
  │                                                  │
  │  Output: "[reddit/u_alice@2026-03-06T10:00:00Z]  │
  │           John is in the kitchen."                │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  Tokenization (identity: byte = token)
  ┌─────────────────────────────────────────────────┐
  │  [0x5B, 0x72, 0x65, 0x64, 0x64, 0x69, 0x74,    │
  │   0x2F, 0x75, 0x5F, 0x61, 0x6C, 0x69, 0x63,    │
  │   0x65, 0x40, 0x32, 0x30, 0x32, 0x36, ...]      │
  │   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      │
  │   tag bytes (first ~45 tokens)                    │
  │   [...0x4A, 0x6F, 0x68, 0x6E, 0x20, 0x69,       │
  │    0x73, 0x20, 0x69, 0x6E, ...]                   │
  │   ^^^^^^^^^^^^^^^^^^^^^^^^^^^                      │
  │   content bytes                                    │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  Sliding Window Encoder (chunk_size=8)
  ┌─────────────────────────────────────────────────┐
  │  chunk 0: [reddit/u  (8 bytes of tag)            │
  │  chunk 1: _alice@20  (8 bytes of tag)            │
  │  chunk 2: 26-03-06T  (8 bytes of tag)            │
  │  chunk 3: 10:00:00Z  (8 bytes of tag)            │
  │  chunk 4: ] John is  (bracket + content start)   │
  │  chunk 5:  in the k  (content)                   │
  │  chunk 6: itchen.    (content + padding)         │
  │                                                  │
  │  Each chunk → 2 memory vectors (mean + last)     │
  │  = 14 memory vectors total                       │
  │  Tag chunks get their own memory representations │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
  Memory Storage (TrieIndex)
  ┌─────────────────────────────────────────────────┐
  │  addr_head(chunk_0_hidden) → address A0          │
  │  addr_head(chunk_1_hidden) → address A1          │
  │  ...                                             │
  │  Tag chunks and content chunks get DIFFERENT     │
  │  addresses (different hidden states).             │
  │                                                  │
  │  Cross-attention retrieves both tag and content  │
  │  vectors — model learns to correlate them.       │
  └─────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why text prefix (not structured embedding)?

1. **No architecture changes** — prefix is just more bytes, no new embedding layers
2. **Self-supervised** — model learns tag structure from data, generalizable
3. **Composable** — tag format can evolve without retraining embeddings
4. **Inspectable** — can literally read what's in memory (raw bytes)
5. **Shell-native** — works at the byte level, consistent with our UTF-8 design

### Why store all versions (not pick one)?

1. **Truth is context-dependent** — "Where is John?" depends on whose perspective
2. **Temporal reasoning** — newer isn't always better (historical queries)
3. **Source reliability** — let downstream queries judge credibility
4. **Richer memory** — contradictions contain more information than consensus

### Multi-label answers for conflicting facts

When facts contradict and the question doesn't specify a source, multiple
answers are correct. Training options:

```
Option A: Accept any matching answer (easiest)
  If "kitchen" OR "garden" is correct, loss = min(loss_kitchen, loss_garden)

Option B: Enumerate all answers
  Answer: "kitchen (u_alice) or garden (u_bob)"
  More informative but harder to evaluate

Option C: Default to latest timestamp
  Simplest heuristic, but loses the "memorize all" property

Recommendation: Option A for initial training, Option B as stretch goal
```

## Files to Modify

```
train_micro.py:
  + SourceTag dataclass and tag generators
  + generate_sourced_facts() — contradiction generator
  + generate_sourced_single/two/three_facts() — tagged versions of existing
  + Modify generate_shell_texts() — add shell source tags
  + Modify load_wikipedia_sentences() — add wiki source tags
  + Modify MemoryQADataset — increase max_passage_len
  + Modify train_multitask() — new curriculum stages
  + New eval: contradiction resolution accuracy

config.py:
  + max_passage_len field (currently hardcoded in train_micro.py)
  + Potentially increase max_seq_len to 256

No changes needed to:
  model.py — architecture unchanged (prefix is just more input bytes)
  memory.py — memory system unchanged (stores whatever encoder produces)
```

## Risk Assessment

```
Risk                          Impact    Mitigation
────────────────────────────  ────────  ──────────────────────────
Tag bytes waste capacity      MEDIUM    Increase max_passage_len
Model ignores tags            MEDIUM    Source-hint questions force binding
Contradictions confuse model  LOW       Staged curriculum (no conflicts first)
Tag parsing hurts LM loss     LOW       Tags are structured text, easy patterns
Chunk boundaries split tags   NONE      Encoder handles arbitrary byte boundaries
```

## Success Criteria

1. Model achieves ≥95% on standard (non-conflicting) tagged QA
2. Model achieves ≥80% on source-hinted QA ("According to X, where is Y?")
3. Model stores both contradicting facts in memory (verified by probing)
4. LM loss not significantly degraded vs untagged baseline
5. Model can distinguish temporal versions of same fact
