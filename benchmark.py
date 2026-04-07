#!/usr/bin/env python3
"""Performance benchmarks for LatentController (828K param looping transformer).

Measures tokens/second for:
  1. LM forward (causal, no memory) — batch 1 and 16
  2. LM autoregressive generation — batch 1 and 16
  3. Memory encoding (passage → memory vectors) — batch 1 and 16
  4. QA inference (encode + sliding window decode) — batch 1 and 16

Usage:
    python benchmark.py [--device mps|cpu|cuda] [--warmup 5] [--iters 50]
"""

import argparse
import time
import statistics
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import LoopedLatentController
from train_micro import (
    VOCAB, VOCAB_SIZE, tokenize, detokenize,
    encode_sentence_frozen, sliding_lm_encode,
    generate_dataset, tag_passage,
)


def timer(fn, warmup=5, iters=50):
    """Run fn, return list of per-call durations in seconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def fmt(tok_per_sec):
    if tok_per_sec >= 1_000_000:
        return f"{tok_per_sec/1e6:.1f}M"
    elif tok_per_sec >= 1_000:
        return f"{tok_per_sec/1e3:.1f}K"
    return f"{tok_per_sec:.0f}"


def bench_lm_forward(model, device, seq_len, batch_size, warmup, iters):
    """Forward pass throughput (teacher-forced, no generation loop)."""
    inp = torch.randint(32, 128, (batch_size, seq_len), device=device)

    def fn():
        with torch.no_grad():
            model(inp)

    times = timer(fn, warmup, iters)
    total_tokens = batch_size * seq_len
    tps = [total_tokens / t for t in times]
    return tps


def bench_lm_generate(model, device, prompt_len, gen_len, batch_size, warmup, iters):
    """Autoregressive generation with KV cache."""
    prompt = torch.randint(32, 128, (batch_size, prompt_len), device=device)

    def fn():
        with torch.no_grad():
            # Prefill
            logits, _, kv = model(prompt, kv_cache=[], cache_position=0)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            pos = prompt_len
            for _ in range(gen_len):
                logits, _, kv = model(next_tok, kv_cache=kv, cache_position=pos)
                next_tok = logits[:, -1:, :].argmax(dim=-1)
                pos += 1

    times = timer(fn, warmup, iters)
    total_tokens = batch_size * gen_len
    tps = [total_tokens / t for t in times]
    return tps


def bench_memory_encode(model, device, passage_len, batch_size, warmup, iters):
    """Sentence-boundary memory encoding throughput."""
    pad = VOCAB["<pad>"]
    passages = torch.full((batch_size, passage_len), pad, dtype=torch.long, device=device)
    # Fill with realistic content (words separated by period-space)
    for b in range(batch_size):
        example_text = "mary went to the garden . john moved to the kitchen . then mary ran to the office ."
        ids = tokenize(example_text)
        plen = min(len(ids), passage_len)
        passages[b, :plen] = torch.tensor(ids[:plen], dtype=torch.long)

    def fn():
        with torch.no_grad():
            encode_sentence_frozen(model, passages, device)

    times = timer(fn, warmup, iters)
    total_tokens = batch_size * passage_len
    tps = [total_tokens / t for t in times]
    return tps


def bench_qa_inference(model, device, batch_size, warmup, iters, window_size=16, num_passes=4):
    """Full QA pipeline: encode passage → sliding window decode."""
    pad = VOCAB["<pad>"]
    bos = VOCAB["<bos>"]
    ans_marker = VOCAB["<ans>"]
    passage_len = 128

    examples = generate_dataset(max(batch_size * 2, 20), seed=99)

    # Pre-build batches
    all_passages = []
    all_inputs = []
    for ex in examples[:batch_size]:
        p_ids = tokenize(ex.passage)
        p_ids = p_ids[:passage_len]
        while len(p_ids) < passage_len:
            p_ids.append(pad)
        all_passages.append(p_ids)

        q_ids = tokenize(ex.question)
        a_ids = tokenize(ex.answer)
        inp = [bos, ans_marker] + q_ids + a_ids
        all_inputs.append(inp)

    # Pad inputs to same length
    max_inp_len = max(len(x) for x in all_inputs)
    for i in range(len(all_inputs)):
        while len(all_inputs[i]) < max_inp_len:
            all_inputs[i].append(pad)

    p_tensor = torch.tensor(all_passages, dtype=torch.long, device=device)
    i_tensor = torch.tensor(all_inputs, dtype=torch.long, device=device)

    def fn():
        with torch.no_grad():
            mk, mv, mm = encode_sentence_frozen(model, p_tensor, device)
            hidden = sliding_lm_encode(model, i_tensor, window_size, num_passes,
                                       mem_keys=mk, mem_vals=mv, mem_mask=mm)
            F.linear(hidden, model.embed.weight)

    times = timer(fn, warmup, iters)
    # Count: passage tokens encoded + question tokens decoded
    total_tokens = batch_size * (passage_len + max_inp_len)
    tps = [total_tokens / t for t in times]
    return tps


def print_result(label, tps_list, batch_size):
    med = statistics.median(tps_list)
    p10 = sorted(tps_list)[len(tps_list) // 10]
    p90 = sorted(tps_list)[len(tps_list) * 9 // 10]
    per_user = med / batch_size
    print(f"  {label:<40} {fmt(med):>8} tok/s  "
          f"(p10={fmt(p10)}, p90={fmt(p90)})  "
          f"| {fmt(per_user)}/user")


def main():
    parser = argparse.ArgumentParser(description="LatentController Performance Benchmarks")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg = ModelConfig()
    cfg.vocab_size = VOCAB_SIZE
    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 78)
    print("  LatentController Performance Benchmark")
    print("=" * 78)
    print(f"  Model:   {n_params:,} params ({n_params/1e6:.2f}M)")
    print(f"  Device:  {device}")
    print(f"  Config:  d={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads}")
    print(f"  Warmup:  {args.warmup} iters")
    print(f"  Measure: {args.iters} iters (median ± p10/p90)")
    print()

    W = args.warmup
    I = args.iters

    # ── 1. LM Forward (teacher-forced) ──
    print("─" * 78)
    print("  1. LM Forward Pass (causal, no memory, teacher-forced)")
    print("     Input: random tokens, measure full forward throughput")
    print("─" * 78)
    for bs in [1, 16]:
        for seq_len in [64, 128, 192]:
            tps = bench_lm_forward(model, device, seq_len, bs, W, I)
            print_result(f"B={bs:>2}  T={seq_len}", tps, bs)
    print()

    # ── 2. LM Autoregressive Generation ──
    print("─" * 78)
    print("  2. LM Autoregressive Generation (KV cache)")
    print("     Prompt=16 tokens, generate 64/128 tokens")
    print("─" * 78)
    for bs in [1, 16]:
        for gen_len in [64, 128]:
            tps = bench_lm_generate(model, device, 16, gen_len, bs, W, I)
            print_result(f"B={bs:>2}  gen={gen_len}", tps, bs)
    print()

    # ── 3. Memory Encoding ──
    print("─" * 78)
    print("  3. Memory Encoding (sentence-boundary → memory vectors)")
    print("     Passage with 3 sentences, frozen encoder")
    print("─" * 78)
    for bs in [1, 16]:
        for p_len in [128, 256]:
            tps = bench_memory_encode(model, device, p_len, bs, W, I)
            print_result(f"B={bs:>2}  passage={p_len}", tps, bs)
    print()

    # ── 4. QA Inference (encode + sliding decode) ──
    print("─" * 78)
    print("  4. Full QA Inference (encode passage + sliding window decode)")
    print("     W=16, P=4 passes, passage=128 + question/answer")
    print("─" * 78)
    for bs in [1, 16]:
        tps = bench_qa_inference(model, device, bs, W, I, window_size=16, num_passes=4)
        print_result(f"B={bs:>2}  W=16 P=4", tps, bs)
    # Also test with fewer passes
    for bs in [1, 16]:
        tps = bench_qa_inference(model, device, bs, W, I, window_size=16, num_passes=2)
        print_result(f"B={bs:>2}  W=16 P=2", tps, bs)
    print()

    # ── 5. Latency per request ──
    print("─" * 78)
    print("  5. Per-Request Latency (single-item, end-to-end)")
    print("─" * 78)

    # LM generation latency
    prompt = torch.randint(32, 128, (1, 16), device=device)
    def gen_64():
        with torch.no_grad():
            logits, _, kv = model(prompt, kv_cache=[], cache_position=0)
            tok = logits[:, -1:, :].argmax(dim=-1)
            pos = 16
            for _ in range(64):
                logits, _, kv = model(tok, kv_cache=kv, cache_position=pos)
                tok = logits[:, -1:, :].argmax(dim=-1)
                pos += 1

    lat_times = timer(gen_64, W, I)
    med_lat = statistics.median(lat_times) * 1000
    print(f"  LM generate 64 tokens:     {med_lat:>7.1f} ms")

    # QA inference latency
    pad = VOCAB["<pad>"]
    bos = VOCAB["<bos>"]
    ans_marker = VOCAB["<ans>"]
    ex = generate_dataset(1, seed=99)[0]
    p_ids = tokenize(ex.passage)[:128]
    while len(p_ids) < 128:
        p_ids.append(pad)
    p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
    q_ids = tokenize(ex.question)
    a_ids = tokenize(ex.answer)
    i_ids = [bos, ans_marker] + q_ids + a_ids
    i_t = torch.tensor([i_ids], dtype=torch.long, device=device)

    def qa_one():
        with torch.no_grad():
            mk, mv, mm = encode_sentence_frozen(model, p_t, device)
            h = sliding_lm_encode(model, i_t, 16, 4, mem_keys=mk, mem_vals=mv, mem_mask=mm)
            F.linear(h, model.embed.weight)

    qa_times = timer(qa_one, W, I)
    med_qa = statistics.median(qa_times) * 1000
    print(f"  QA (encode+decode) W=16 P=4: {med_qa:>6.1f} ms")

    # Memory encode latency
    def enc_one():
        with torch.no_grad():
            encode_sentence_frozen(model, p_t, device)

    enc_times = timer(enc_one, W, I)
    med_enc = statistics.median(enc_times) * 1000
    print(f"  Memory encode (128 tokens):  {med_enc:>6.1f} ms")

    print()
    print("=" * 78)
    print("  Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
