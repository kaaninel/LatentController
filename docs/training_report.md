# ANT — Markdown + Shell Training: Viability Report

## Current Approach

### Data Pipeline
- **Shell commands**: 5,000 synthetic examples generated from template grammar
  - Simple commands, pipes, redirects, loops, conditionals, one-liners
  - ~8 pattern types with weighted random selection
- **Wikipedia**: 5,000 sentences extracted from HuggingFace datasets
  - Filtered to 30–300 chars, alphabetic content, no headers
  - Falls back to synthetic diverse text if download fails
- **QA (bAbI)**: Synthetic 1/2/3-fact location tracking examples

### Source Tagging
All text tagged with provenance headers during training:
```
shell/root@2026-04-01T14:22:33Z: ls -la /home/user
wiki/article@2025-06-15T08:30:00Z: The French Revolution began in 1789.
social/alice@2026-03-06T10:05:00Z: Hello, how are you?
```

### Training Config
- **Model**: 828K params, byte-level (vocab=256+4 special), d_model=128, 4 layers
- **Window**: 16 tokens sliding, 4 passes (bidirectional per window)
- **LM loss**: ~1.9 (byte-level cross-entropy after convergence)
- **QA accuracy**: 100% on bAbI with memory

---

## Honest Assessment

### What Works Well
1. **QA memory system** — 100% accuracy proves cross-attention memory retrieval works
2. **Source tagging** — model learns to generate tagged output, can distinguish dataplanes
3. **Shell commands** — templated grammar has clear patterns the model can learn

### What's Fundamentally Hard

#### 1. LM Loss of 1.9 Is High for Coherent Text
- For byte-level English, loss ~1.0 = reasonable character prediction
- Loss 1.9 = uncertainty of ~6.7 bits/byte = model predicts ~1/100 chance for correct next byte
- At this level, output is semi-coherent babble: recognizable words mixed with nonsense
- **Root cause**: 828K params simply cannot model English well at byte level

#### 2. Byte-Level Tokenization Is 4–6× Less Efficient
| Tokenizer | Tokens per word | Effective context at 192 tokens |
|-----------|----------------|-------------------------------|
| BPE (GPT-2) | ~1.3 | ~148 words |
| Byte-level | ~5.5 | ~35 words |

The model sees ~35 words of context. For markdown/shell:
- A shell pipeline like `cat file.txt | grep pattern | sort -u | head -n 10` = ~53 bytes = uses 28% of the context window
- A markdown heading + paragraph = 100-200 bytes = entire window

#### 3. Synthetic Shell Data Is Shallow
The 8 template types cover syntax but not semantics:
- No `git` workflows, `docker`, `kubectl`, `make`, `npm` commands
- No error handling patterns, no real script structure
- No interactive patterns (prompts, menus, REPL sessions)
- ~5,000 examples ≈ 200KB of text — tiny by any standard

#### 4. Wikipedia Sentences Lack Markdown Structure
Current extraction splits on `. ` — produces plain sentences, not markdown:
- No headings (`#`, `##`), no lists (`-`, `*`, `1.`), no code blocks, no links
- No tables, no bold/italic, no nested structures
- The model never sees actual markdown syntax during training

---

## Improvement Strategy

### Tier 1: Quick Wins (No Architecture Changes)

#### A. Real Markdown Training Data
```python
# Instead of extracting sentences, keep markdown structure:
# - Download raw wikitext (already in markdown-like format)
# - Include README files from GitHub (real markdown)
# - Include man pages (structured text)
```
- Use `wikitext-103-raw-v1` dataset — already has `== Headings ==`, lists, structure
- Add a markdown converter to produce `#` headings, `**bold**`, `- lists` etc.
- **Impact**: Model actually sees markdown syntax → can learn to generate it

#### B. Real Shell Training Data
```python
# Instead of templates, use real shell history/scripts:
# - Aggregate anonymized shell commands from public datasets
# - Parse and include bash scripts from GitHub
# - Include dotfiles (.bashrc, .zshrc patterns)
```
- Use `the-stack` dataset filtered to shell/bash files
- Include real Dockerfiles, Makefiles, CI configs
- **Impact**: Model sees actual usage patterns, not toy grammar

#### C. Increase Data Volume
Current: 10K examples (~500KB text)  
Recommended: 100K+ examples (~5MB text)  
Colab A100 target: 500K+ examples (~25MB text)

For 828K params, overfitting risk is low with 5MB+ of diverse text.

#### D. Longer Context Windows
Current: max_seq_len=192 bytes (~35 words)  
Recommended: max_seq_len=512 bytes (~93 words)  

Requires RoPE base adjustment but no architecture change.  
Lets the model see full markdown paragraphs and multi-line scripts.

### Tier 2: Architecture Considerations

#### E. Character-Aware Embeddings
Instead of raw byte embeddings (128-dim per byte), add local convolutions:
```
bytes → embed(128) → Conv1D(kernel=3) → Conv1D(kernel=5) → transformer
```
This gives the model sub-word features without a tokenizer. Estimated +50K params.

#### F. Larger Model Variants
| Variant | Params | d_model | Layers | Expected LM Loss |
|---------|--------|---------|--------|-------------------|
| Current | 828K   | 128     | 4      | ~1.9             |
| Medium  | 2.5M   | 256     | 4      | ~1.4             |
| Large   | 10M    | 384     | 6      | ~1.0             |

At 2.5M params, the model should produce mostly coherent English.
At 10M params, it should handle markdown formatting reliably.

**Recommendation**: Keep 828K for memory/QA research, add 2.5M variant for language fluency.

### Tier 3: Curriculum & Training

#### G. Progressive Data Mixing
```
Phase 1 (30%): Pure English text (Wikipedia/books) → learn language
Phase 2 (30%): Mixed text + shell commands → learn both domains  
Phase 3 (40%): Tagged multitask (LM + QA + markdown + shell) → final model
```

#### H. Domain-Specific Evaluation
Current eval: only QA accuracy + LM loss (aggregate).  
Add domain-specific metrics:
- **Shell validity**: % of generated commands that parse (`bash -n`)
- **Markdown validity**: % of output with correct markdown syntax
- **Perplexity by dataplane**: separate LM loss for shell vs wiki vs social

---

## Realistic Expectations by Model Size

| Size | Shell Commands | Markdown | Conversational English |
|------|---------------|----------|----------------------|
| 828K | Recognizable syntax, frequent errors | Heading-level only | Semi-coherent babble |
| 2.5M | Common commands correct | Basic formatting works | Short coherent sentences |
| 10M | Pipelines, loops correct | Full markdown with lists/code | Paragraph-level coherence |
| 50M+ | Complex scripts | Nested structures | Fluent conversation |

---

## Conclusion

The current 828K model is a valid **proof of concept** for the memory architecture (100% QA proves the mechanism works), but it is **undersized for fluent markdown/shell generation**. The byte-level tokenization means the model needs ~4× more capacity than a BPE model for equivalent language quality.

**Recommended path:**
1. Improve data quality immediately (real markdown, real shell) — free improvement
2. Scale data volume 10–50× — fits easily on A100
3. Test a 2.5M variant for language tasks while keeping 828K for memory research
4. Add domain-specific evaluation to track progress per skill
