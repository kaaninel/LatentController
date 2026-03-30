#!/usr/bin/env python3
"""
Benchmark suite for LatentController.

Measures: perplexity, memory utility, ACT behavior, generation quality,
memory recall, throughput, and VRAM usage.

Usage:
    python benchmark.py --checkpoint_dir ./checkpoints --data_dir ./data_cache
    python benchmark.py --checkpoint_dir ./checkpoints --data_dir ./data_cache --quick
"""

import argparse
import math
import os
import random
import shutil
import time
from collections import Counter
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, MemoryConfig, Phase3Config
from model import LoopedLatentController
from memory import MemorySystem
from agent import Agent
from orchestrator import Orchestrator
from dataset import build_datasets
from tokenizer_utils import load_tokenizer, encode, decode
from utils import load_checkpoint, vram_report, peak_vram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(checkpoint_dir, data_dir, device):
    """Load the best available checkpoint + tokenizer."""
    cfg = ModelConfig()
    tok_path = os.path.join(data_dir, "tokenizer.json")
    tokenizer = load_tokenizer(tok_path)
    model = LoopedLatentController(cfg).to(device)

    # Try phase checkpoints in reverse order (latest phase = most capable)
    for phase in [5, 4, 3, 1]:
        ckpt = os.path.join(checkpoint_dir, f"phase{phase}", "best.pt")
        if os.path.exists(ckpt):
            load_checkpoint(model, None, ckpt, device)
            print(f"  Loaded checkpoint: phase{phase}/best.pt")
            # Load address heads if phase >= 3
            if phase >= 3:
                p2_ckpt = os.path.join(checkpoint_dir, "phase2", "best.pt")
                if os.path.exists(p2_ckpt):
                    p2_data = torch.load(p2_ckpt, map_location=device, weights_only=False)
                    if "addr_heads" in p2_data:
                        for i, h in enumerate(model.addr_heads):
                            h.load_state_dict(p2_data["addr_heads"][i])
                        print("  Loaded Phase 2 address heads")
            return model, tokenizer, phase

    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


# ---------------------------------------------------------------------------
# Benchmark 1: Perplexity (baseline LM quality)
# ---------------------------------------------------------------------------

def bench_perplexity(model, tokenizer, data_dir, device, max_batches=100):
    """Validation perplexity on held-out data."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 1: Perplexity")
    print("=" * 60)

    cfg = ModelConfig()
    _, val_ds = build_datasets(data_dir, cfg, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp_dtype = torch.bfloat16 if device == "cuda" else None

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(val_loader):
            if i >= max_batches:
                break
            inp, tgt = inp.to(device), tgt.to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, _ = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1), ignore_index=0, reduction="sum",
            )
            mask = tgt.reshape(-1) != 0
            total_loss += loss.item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 100))
    print(f"  Val Loss:       {avg_loss:.4f}")
    print(f"  Val Perplexity: {ppl:.2f}")
    print(f"  Tokens scored:  {total_tokens:,}")
    model.train()
    return {"val_loss": avg_loss, "val_ppl": ppl}


# ---------------------------------------------------------------------------
# Benchmark 2: Memory Utility (does memory help?)
# ---------------------------------------------------------------------------

def bench_memory_utility(model, tokenizer, data_dir, checkpoint_dir, device, max_batches=50):
    """Compare perplexity WITH and WITHOUT memory."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 2: Memory Utility")
    print("=" * 60)

    cfg = ModelConfig()
    mcfg = MemoryConfig()
    _, val_ds = build_datasets(data_dir, cfg, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=True)

    # Check for trained memory
    mem_dir = os.path.join(checkpoint_dir, "phase3", "memory_train")
    if not os.path.exists(mem_dir):
        print("  ⚠ No Phase 3 memory found. Skipping memory utility benchmark.")
        return {"memory_available": False}

    memory = MemorySystem(mem_dir, mcfg)
    print(f"  Memory entries: {memory.total_entries():,}")

    model.eval()
    loss_no_mem = 0.0
    loss_with_mem = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(val_loader):
            if i >= max_batches:
                break
            inp, tgt = inp.to(device), tgt.to(device)

            try:
                # Without memory
                logits_nm, _ = model(inp)
                l_nm = F.cross_entropy(
                    logits_nm.reshape(-1, logits_nm.size(-1)),
                    tgt.reshape(-1), ignore_index=0, reduction="sum",
                )
                loss_no_mem += l_nm.item()
                del logits_nm, l_nm

                # With memory
                _, _, hid = model(inp, return_hidden=True)
                h_last = hid[:, -1, :]
                addrs = model.compute_addresses_batch(h_last)
                addr_cpu = [a.cpu().numpy() for a in addrs]
                B = inp.shape[0]
                batch_addrs = []
                for b in range(B):
                    sample_addrs = [addr_cpu[h][b].tobytes() for h in range(len(addrs))]
                    batch_addrs.append(sample_addrs)
                mem_np = memory.read_memory_batch(batch_addrs)
                mem_t = torch.from_numpy(mem_np).float().to(device) / 127.0
                del hid, h_last

                logits_m, _ = model(inp, memory_vectors=mem_t)
                l_m = F.cross_entropy(
                    logits_m.reshape(-1, logits_m.size(-1)),
                    tgt.reshape(-1), ignore_index=0, reduction="sum",
                )
                loss_with_mem += l_m.item()
                del logits_m, l_m, mem_t

                mask = tgt.reshape(-1) != 0
                total_tokens += mask.sum().item()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

    denom = max(1, total_tokens)
    nm = loss_no_mem / denom
    wm = loss_with_mem / denom
    ppl_nm = math.exp(min(nm, 100))
    ppl_wm = math.exp(min(wm, 100))
    delta = ppl_nm - ppl_wm

    print(f"  No-mem  PPL: {ppl_nm:.2f} (loss {nm:.4f})")
    print(f"  W/mem   PPL: {ppl_wm:.2f} (loss {wm:.4f})")
    print(f"  Δ PPL:       {delta:+.3f} (positive = memory helps)")
    print(f"  Memory entries: {memory.total_entries():,}")
    model.train()
    return {
        "ppl_no_mem": ppl_nm, "ppl_with_mem": ppl_wm,
        "delta_ppl": delta, "memory_entries": memory.total_entries(),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Generation Quality (qualitative + stats)
# ---------------------------------------------------------------------------

PROMPTS = [
    "Once upon a time",
    "The little cat",
    "One day, a brave knight",
    "There was a big",
    "Lily wanted to",
    "The sun was shining and",
    "A small bird sat on",
    "Mom said to the boy",
]

def bench_generation(model, tokenizer, device, n_prompts=8, max_tokens=150):
    """Generate text from prompts and report statistics."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 3: Generation Quality")
    print("=" * 60)

    cfg = ModelConfig()
    mcfg = MemoryConfig()
    mem_dir = "/tmp/bench_gen_mem"
    os.makedirs(mem_dir, exist_ok=True)
    memory = MemorySystem(mem_dir, mcfg)

    orch = Orchestrator(model, tokenizer, memory, cfg)
    prompts = PROMPTS[:n_prompts]

    results = []
    for prompt in prompts:
        agent = orch.create_agent(max_act_steps=4, emit_threshold=0.3)
        orch.feed(agent, prompt)
        output = orch.generate(agent, max_tokens=max_tokens)
        results.append({"prompt": prompt, "output": output, "length": len(output.split())})
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: {output[:200]}{'...' if len(output) > 200 else ''}")
        print(f"  Words:  {len(output.split())}")

    avg_len = sum(r["length"] for r in results) / len(results)
    # Check for degenerate outputs
    empty = sum(1 for r in results if r["length"] < 3)
    repetitive = sum(1 for r in results if _is_repetitive(r["output"]))

    print(f"\n  --- Generation Stats ---")
    print(f"  Avg words:     {avg_len:.0f}")
    print(f"  Empty (<3w):   {empty}/{len(results)}")
    print(f"  Repetitive:    {repetitive}/{len(results)}")

    shutil.rmtree(mem_dir, ignore_errors=True)
    return {"avg_words": avg_len, "empty": empty, "repetitive": repetitive}


def _is_repetitive(text, threshold=0.5):
    """Check if text has too many repeated n-grams."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    if not trigrams:
        return False
    most_common_freq = counts.most_common(1)[0][1]
    return most_common_freq / len(trigrams) > threshold


# ---------------------------------------------------------------------------
# Benchmark 4: Memory Recall (can agent retrieve stored facts?)
# ---------------------------------------------------------------------------

RECALL_FACTS = [
    ("The wizard's name is Glorpax.", "What is the wizard's name?", "Glorpax"),
    ("The red ball is under the table.", "Where is the red ball?", "table"),
    ("The princess lives in a tall tower.", "Where does the princess live?", "tower"),
    ("The dog's favorite toy is a bone.", "What is the dog's favorite toy?", "bone"),
    ("The magic word is abracadabra.", "What is the magic word?", "abracadabra"),
]

def bench_memory_recall(model, tokenizer, device, max_output=50):
    """Test whether the agent can recall facts stored in memory."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 4: Memory Recall")
    print("=" * 60)

    cfg = ModelConfig()
    mcfg = MemoryConfig()
    mem_dir = "/tmp/bench_recall_mem"
    os.makedirs(mem_dir, exist_ok=True)
    memory = MemorySystem(mem_dir, mcfg)

    orch = Orchestrator(model, tokenizer, memory, cfg)
    correct = 0

    for fact, question, expected in RECALL_FACTS:
        agent = orch.create_agent(max_act_steps=4, emit_threshold=0.3)
        # Ingest fact
        orch.feed(agent, fact)
        # Feed some filler to push fact out of context window
        orch.feed(agent, " The sky is blue. Birds can fly. Water is wet. " * 3)
        # Ask question
        orch.feed(agent, question)
        output = orch.generate(agent, max_tokens=max_output)

        found = expected.lower() in output.lower()
        if found:
            correct += 1
        status = "✅" if found else "❌"
        print(f"  {status} Q: {question}")
        print(f"      A: {output[:100]}{'...' if len(output) > 100 else ''}")
        print(f"      Expected: '{expected}' → {'Found' if found else 'NOT found'}")

    accuracy = correct / len(RECALL_FACTS) * 100
    print(f"\n  Recall Accuracy: {correct}/{len(RECALL_FACTS)} ({accuracy:.0f}%)")

    shutil.rmtree(mem_dir, ignore_errors=True)
    return {"recall_accuracy": accuracy, "correct": correct, "total": len(RECALL_FACTS)}


# ---------------------------------------------------------------------------
# Benchmark 5: ACT Behavior (does halting depth vary?)
# ---------------------------------------------------------------------------

def bench_act_behavior(model, tokenizer, device):
    """Measure ACT halting depth distribution across different inputs."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 5: ACT Halting Behavior")
    print("=" * 60)

    cfg = ModelConfig()
    mcfg = MemoryConfig()
    mem_dir = "/tmp/bench_act_mem"
    os.makedirs(mem_dir, exist_ok=True)
    memory = MemorySystem(mem_dir, mcfg)

    # Check if model has trained halt head (Phase 4+)
    test_input = torch.zeros(1, 10, dtype=torch.long, device=device)
    with torch.no_grad():
        _, halt_logits = model(test_input)
    halt_range = halt_logits.max().item() - halt_logits.min().item()
    if halt_range < 0.01:
        print("  ⚠ Halt head appears untrained (uniform logits). Skipping.")
        shutil.rmtree(mem_dir, ignore_errors=True)
        return {"halt_trained": False}

    orch = Orchestrator(model, tokenizer, memory, cfg)
    texts = [
        "The cat sat on the mat.",
        "Once upon a time in a land far far away there lived a wise old wizard.",
        "One two three four five.",
        "The complex relationship between quantum mechanics and general relativity.",
    ]

    halt_counts = Counter()
    total_tokens = 0

    for text in texts:
        agent = orch.create_agent(max_act_steps=6, emit_threshold=0.3)
        ids = encode(tokenizer, text)
        for tid in ids:
            # Track ACT steps manually
            agent.context_buffer.append(tid)
            if len(agent.context_buffer) > cfg.n_text_positions:
                agent.context_buffer = agent.context_buffer[-cfg.n_text_positions:]
            inp = agent._build_input()
            mem_vecs = agent._read_memory()

            steps_taken = 0
            for act_step in range(agent.max_act_steps):
                mem_tensor = agent._build_mem_tensor(mem_vecs)
                with torch.no_grad():
                    _, halt_logits, hidden = agent.model(
                        inp, memory_vectors=mem_tensor, return_hidden=True
                    )
                agent.h = hidden[0, -1, :].detach()
                halt_prob = F.softmax(halt_logits[0, -1, :], dim=-1)[1].item()
                steps_taken = act_step + 1
                if halt_prob > 0.5:
                    break
                mem_vecs = agent._read_memory()

            halt_counts[steps_taken] += 1
            total_tokens += 1
            agent._write_memory()

    print(f"  Halt step distribution ({total_tokens} tokens):")
    for step in sorted(halt_counts.keys()):
        count = halt_counts[step]
        pct = count / total_tokens * 100
        bar = "█" * int(pct / 2)
        print(f"    Step {step}: {count:4d} ({pct:5.1f}%) {bar}")

    steps_list = []
    for s, c in halt_counts.items():
        steps_list.extend([s] * c)
    avg = sum(steps_list) / len(steps_list)
    std = (sum((s - avg) ** 2 for s in steps_list) / len(steps_list)) ** 0.5

    print(f"  Avg halt depth: {avg:.2f} ± {std:.2f}")
    print(f"  Variance > 1.0: {'✅ Yes' if std > 1.0 else '❌ No (halting not diverse)'}")

    shutil.rmtree(mem_dir, ignore_errors=True)
    return {"avg_halt": avg, "std_halt": std, "halt_dist": dict(halt_counts)}


# ---------------------------------------------------------------------------
# Benchmark 6: Throughput & VRAM
# ---------------------------------------------------------------------------

def bench_throughput(model, tokenizer, data_dir, device, n_steps=50):
    """Measure tokens/sec and VRAM for forward + backward pass."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 6: Throughput & VRAM")
    print("=" * 60)

    cfg = ModelConfig()
    _, val_ds = build_datasets(data_dir, cfg, tokenizer)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True)
    amp_dtype = torch.bfloat16 if device == "cuda" else None

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    start = time.perf_counter()
    total_tokens = 0

    for i, (inp, tgt) in enumerate(loader):
        if i >= n_steps:
            break
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, _ = model(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1), ignore_index=0,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_tokens += (tgt != 0).sum().item()

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tok_per_sec = total_tokens / elapsed

    print(f"  Steps:     {min(n_steps, i + 1)}")
    print(f"  Tokens:    {total_tokens:,}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"  Throughput: {tok_per_sec:.0f} tok/s ({tok_per_sec/1e3:.1f}k tok/s)")
    if device == "cuda":
        print(f"  VRAM:      {vram_report()}")
        print(f"  Peak VRAM: {peak_vram():.1f} GB")

    model.eval()
    return {"tok_per_sec": tok_per_sec, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Benchmark 7: Address Space Quality
# ---------------------------------------------------------------------------

def bench_address_space(model, tokenizer, data_dir, device, n_samples=1000):
    """Analyze address head quality: spread, collision rate, clustering."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 7: Address Space Quality")
    print("=" * 60)

    cfg = ModelConfig()
    _, val_ds = build_datasets(data_dir, cfg, tokenizer)
    loader = DataLoader(val_ds, batch_size=64, shuffle=True, drop_last=True)

    model.eval()
    all_addrs = [[] for _ in range(cfg.n_addr_heads)]  # 3 heads

    with torch.no_grad():
        collected = 0
        for inp, _ in loader:
            if collected >= n_samples:
                break
            inp = inp.to(device)
            _, _, hid = model(inp, return_hidden=True)
            h_last = hid[:, -1, :]
            addrs = model.compute_addresses_batch(h_last)
            for h_idx, a in enumerate(addrs):
                all_addrs[h_idx].append(a.cpu())
            collected += inp.shape[0]

    for h_idx in range(cfg.n_addr_heads):
        addr_tensor = torch.cat(all_addrs[h_idx], dim=0)[:n_samples]  # (N, 8)
        n = addr_tensor.shape[0]

        # Unique addresses
        addr_tuples = set(tuple(a.numpy().tolist()) for a in addr_tensor)
        unique_ratio = len(addr_tuples) / n

        # Per-dimension spread
        spreads = []
        for d in range(cfg.addr_dim):
            vals = addr_tensor[:, d].float()
            spreads.append(f"{vals.std():.1f}")

        # Collision rate (exact matches)
        from collections import Counter
        counts = Counter(tuple(a.numpy().tolist()) for a in addr_tensor)
        collisions = sum(1 for c in counts.values() if c > 1)
        collision_rate = collisions / len(counts) if counts else 0

        print(f"\n  Head {h_idx}:")
        print(f"    Unique addresses: {len(addr_tuples)}/{n} ({unique_ratio:.1%})")
        print(f"    Collision rate:   {collision_rate:.1%}")
        print(f"    Per-dim spread:   [{', '.join(spreads)}]")

    model.train()
    return {"n_samples": n_samples}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark LatentController")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Run quick subset of benchmarks")
    args = parser.parse_args()

    print("=" * 60)
    print("  LATENTCONTROLLER BENCHMARK SUITE")
    print("=" * 60)
    print(f"  Device: {args.device}")
    if args.device == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer, phase = _load_model_and_tokenizer(
        args.checkpoint_dir, args.data_dir, args.device
    )
    print(f"  Model:  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"  Phase:  {phase}")

    results = {}

    # Always run
    results["perplexity"] = bench_perplexity(model, tokenizer, args.data_dir, args.device)
    results["generation"] = bench_generation(model, tokenizer, args.device)

    if not args.quick:
        results["memory"] = bench_memory_utility(
            model, tokenizer, args.data_dir, args.checkpoint_dir, args.device
        )
        results["recall"] = bench_memory_recall(model, tokenizer, args.device)
        results["act"] = bench_act_behavior(model, tokenizer, args.device)
        results["throughput"] = bench_throughput(model, tokenizer, args.data_dir, args.device)
        results["address_space"] = bench_address_space(
            model, tokenizer, args.data_dir, args.device
        )

    # Summary
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    if "perplexity" in results:
        print(f"  Val PPL:           {results['perplexity']['val_ppl']:.2f}")
    if "memory" in results and "delta_ppl" in results["memory"]:
        d = results["memory"]["delta_ppl"]
        print(f"  Memory Δ PPL:      {d:+.3f} ({'✅ helps' if d > 0 else '❌ hurts'})")
    if "recall" in results:
        print(f"  Memory Recall:     {results['recall']['recall_accuracy']:.0f}%")
    if "act" in results and "avg_halt" in results["act"]:
        print(f"  ACT Avg Depth:     {results['act']['avg_halt']:.2f} ± {results['act']['std_halt']:.2f}")
    if "generation" in results:
        g = results["generation"]
        print(f"  Generation:        {g['avg_words']:.0f} avg words, {g['empty']} empty, {g['repetitive']} repetitive")
    if "throughput" in results:
        print(f"  Throughput:        {results['throughput']['tok_per_sec']/1e3:.1f}k tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
