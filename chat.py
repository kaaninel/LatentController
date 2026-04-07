#!/usr/bin/env python3
"""
ANT (Addressable Neural Transformer) — Interactive Chat CLI

Loads a trained checkpoint and provides an interactive terminal interface
for testing the model with autoregressive byte-level generation.

Supports:
  - Free-form text generation (LM mode)
  - Memory-backed QA (passage → memory → question → answer)
  - Configurable sampling (temperature, top-k, top-p)
  - Conversation history with source tagging

Usage:
    python chat.py                                    # auto-detect checkpoint
    python chat.py --checkpoint path/to/best.pt       # specific checkpoint
    python chat.py --device cuda                      # force device
    python chat.py --temperature 0.8 --top_k 50       # sampling params
"""

import argparse
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

from config import ModelConfig, MemoryConfig
from model import ANT, StaticKVCache

# Import training utilities for tokenization and encoding
from train_micro import (
    VOCAB, VOCAB_SIZE, ID2WORD, PAD_ID, BOS_ID, EOS_ID, ANS_ID,
    tokenize, detokenize,
    encode_sentence_frozen, sliding_lm_encode,
    tag_text, tag_passage, _TAG_REGISTRY, _DOMAIN_MAP,
)

# Matches source provenance tags: host/user/path@timestamp: content
_TAG_RE = re.compile(
    r'^([^/]+)/([^/]+)/([^@]*)@(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z): '
)

# ANSI color codes per host
_HOST_COLORS = {
    "localhost": "\033[35m",  # magenta
    "server1":  "\033[36m",  # cyan
    "wiki":     "\033[32m",  # green
    "news":     "\033[33m",  # yellow
    "cam1":     "\033[34m",  # blue
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def format_tagged_line(line: str) -> str:
    """Parse a source tag and return a color-formatted line.

    Input:  'localhost/alice/chat@2026-04-07T17:04:13Z: Hello world'
    Output: '\033[1m\033[35m[localhost/alice]\033[0m \033[2m~/chat\033[0m Hello world'
    """
    m = _TAG_RE.match(line)
    if not m:
        return line
    host, user, path, timestamp = m.group(1), m.group(2), m.group(3), m.group(4)
    content = line[m.end():]
    color = _HOST_COLORS.get(host, "\033[37m")
    return f"{_BOLD}{color}[{host}/{user}]{_RESET} {_DIM}{path}{_RESET} {content}"


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ModelConfig()
    cfg.vocab_size = VOCAB_SIZE

    if "window_size" in ckpt:
        cfg.chunk_size = ckpt["window_size"]
    if "config" in ckpt:
        for k, v in ckpt["config"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = ANT(cfg, use_checkpoint=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", "?")
    acc = ckpt.get("accuracy", 0)
    mode = ckpt.get("mode", "unknown")
    window_size = ckpt.get("window_size", cfg.chunk_size)
    num_passes = ckpt.get("num_passes", 4)

    print(f"  Model loaded: step={step}, accuracy={acc:.1%}, mode={mode}")
    print(f"  Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"W={window_size}, P={num_passes}")

    return model, cfg, ckpt


@torch.no_grad()
def generate_autoregressive(model, cfg, prompt_ids, device,
                            max_new_tokens=200, temperature=0.8,
                            top_k=50, top_p=0.9, repetition_penalty=1.2,
                            mem_keys=None, mem_vals=None, mem_mask=None,
                            use_sliding=False, window_size=8, num_passes=4,
                            stop_on_eos=True, stream=True, strip_output_tags=False):
    """Generate tokens autoregressively with configurable sampling."""
    model.eval()

    if use_sliding:
        return _generate_sliding(
            model, cfg, prompt_ids, device, max_new_tokens, temperature,
            top_k, top_p, repetition_penalty,
            mem_keys, mem_vals, mem_mask,
            window_size, num_passes, stop_on_eos, stream,
            strip_output_tags)

    # Standard causal generation with KV cache
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    B, T = input_ids.shape

    cache = model.make_cache(B, max_seq=T + max_new_tokens, device=device)

    # Prefill
    logits, _, cache = model(input_ids, kv_cache=cache, cache_position=0,
                             memory_keys=mem_keys, memory_values=mem_vals,
                             memory_mask=mem_mask)

    generated = list(prompt_ids)
    pos = T
    line_buf = []  # Buffer for tag-aware line output

    for _ in range(max_new_tokens):
        next_logits = logits[0, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for prev_id in set(generated[-50:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= repetition_penalty
                else:
                    next_logits[prev_id] *= repetition_penalty

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-K filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        # Sample
        if temperature > 0:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = next_logits.argmax().item()

        generated.append(next_token)

        if stop_on_eos and next_token == EOS_ID:
            break

        # Stream output with tag-aware line buffering
        if stream:
            if next_token == ord('\n'):
                line = bytes(line_buf).decode('utf-8', errors='replace')
                if strip_output_tags:
                    line = format_tagged_line(line)
                print(line, flush=True)
                line_buf = []
            else:
                line_buf.append(next_token)
                # Buffer until tag window passes (~42 chars) to parse tags
                if not strip_output_tags or len(line_buf) > 45:
                    chunk = bytes(line_buf).decode('utf-8', errors='replace')
                    if strip_output_tags:
                        chunk = format_tagged_line(chunk)
                    print(chunk, end='', flush=True)
                    line_buf = []

        # Next step
        next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _, cache = model(next_input, kv_cache=cache, cache_position=pos,
                                 memory_keys=mem_keys, memory_values=mem_vals,
                                 memory_mask=mem_mask)
        pos += 1

    # Flush remaining buffer
    if stream and line_buf:
        line = bytes(line_buf).decode('utf-8', errors='replace')
        if strip_output_tags:
            line = format_tagged_line(line)
        print(line)
    elif stream:
        print()

    return generated[len(prompt_ids):]


@torch.no_grad()
def _generate_sliding(model, cfg, prompt_ids, device,
                      max_new_tokens, temperature, top_k, top_p,
                      repetition_penalty,
                      mem_keys, mem_vals, mem_mask,
                      window_size, num_passes, stop_on_eos, stream,
                      strip_output_tags=False):
    """Generate using sliding window encoder (bidirectional per window)."""
    generated = list(prompt_ids)
    line_buf = []

    for _ in range(max_new_tokens):
        input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        hidden = sliding_lm_encode(
            model, input_tensor, window_size, num_passes,
            mem_keys=mem_keys, mem_vals=mem_vals, mem_mask=mem_mask)
        logits = F.linear(hidden, model.embed.weight)
        next_logits = logits[0, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for prev_id in set(generated[-50:]):
                if next_logits[prev_id] > 0:
                    next_logits[prev_id] /= repetition_penalty
                else:
                    next_logits[prev_id] *= repetition_penalty

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
            next_logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

        if temperature > 0:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = next_logits.argmax().item()

        generated.append(next_token)

        if stop_on_eos and next_token == EOS_ID:
            break

        if stream:
            if next_token == ord('\n'):
                line = bytes(line_buf).decode('utf-8', errors='replace')
                if strip_output_tags:
                    line = format_tagged_line(line)
                print(line, flush=True)
                line_buf = []
            else:
                line_buf.append(next_token)
                if not strip_output_tags or len(line_buf) > 45:
                    chunk = bytes(line_buf).decode('utf-8', errors='replace')
                    if strip_output_tags:
                        chunk = format_tagged_line(chunk)
                    print(chunk, end='', flush=True)
                    line_buf = []

    if stream and line_buf:
        line = bytes(line_buf).decode('utf-8', errors='replace')
        if strip_output_tags:
            line = format_tagged_line(line)
        print(line)
    elif stream:
        print()

    return generated[len(prompt_ids):]


def encode_passage_to_memory(model, passage_text, device, tagged=False):
    """Encode a passage into memory vectors for QA."""
    if tagged:
        passage_text = tag_passage(passage_text)
    passage_ids = tokenize(passage_text)
    p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)
    mem_keys, mem_vals, mem_mask = encode_sentence_frozen(model, p_tensor, device)
    return mem_keys, mem_vals, mem_mask


HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║                       ANT Chat CLI                           ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    /help          - Show this help                           ║
║    /quit          - Exit                                     ║
║    /mode lm       - Free-form text generation (default)      ║
║    /mode qa       - Memory-backed QA mode                    ║
║    /mode sliding   - Sliding window generation               ║
║    /temp <float>  - Set temperature (0=greedy, default=0.8)  ║
║    /topk <int>    - Set top-k (default=50, 0=disabled)       ║
║    /topp <float>  - Set top-p nucleus (default=0.9)          ║
║    /maxlen <int>  - Set max generation length (default=200)  ║
║    /rep <float>   - Set repetition penalty (default=1.2)     ║
║    /remember <text> - Store text in memory (QA mode)         ║
║    /forget         - Clear stored memory                     ║
║    /memory         - Show stored passages                    ║
║    /ask <question> - Ask about stored passages (QA mode)     ║
║    /config         - Show current settings                   ║
║                                                              ║
║  In LM mode: just type text and press Enter to generate      ║
║  In QA mode: use /remember and /ask                          ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="ANT Chat CLI")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--device", default=None,
                        help="Force device (cpu/mps/cuda)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--tagged", action="store_true",
                        help="Use source provenance tags")
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Auto-detect checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            "checkpoints/local_test/multitask/best_multitask.pt",
            "checkpoints/micro/multitask/best_multitask.pt",
            "checkpoints/micro/sliding_lm/best_model.pt",
            "checkpoints/micro/memory/best_model.pt",
            "checkpoints/micro/best_model.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            print("ERROR: No checkpoint found. Train a model first or specify --checkpoint.")
            sys.exit(1)

    print("=" * 62)
    print("  ANT — Interactive Chat CLI")
    print("=" * 62)

    model, cfg, ckpt = load_model(ckpt_path, device)
    window_size = ckpt.get("window_size", cfg.chunk_size)
    num_passes = ckpt.get("num_passes", 4)

    # State
    mode = "lm"
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens
    rep_penalty = args.repetition_penalty
    tagged = args.tagged
    stored_passages = []
    mem_keys, mem_vals, mem_mask = None, None, None

    print(f"\n  Device: {device} | Mode: {mode} | Temp: {temperature}")
    print(f"  Type /help for commands\n")

    while True:
        try:
            user_input = input("\033[1;36myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit" or cmd == "/exit":
                print("  Goodbye!")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/mode":
                if arg in ("lm", "qa", "sliding"):
                    mode = arg
                    print(f"  Mode → {mode}")
                else:
                    print("  Usage: /mode lm|qa|sliding")
            elif cmd == "/temp":
                try:
                    temperature = float(arg)
                    print(f"  Temperature → {temperature}")
                except ValueError:
                    print("  Usage: /temp <float>")
            elif cmd == "/topk":
                try:
                    top_k = int(arg)
                    print(f"  Top-K → {top_k}")
                except ValueError:
                    print("  Usage: /topk <int>")
            elif cmd == "/topp":
                try:
                    top_p = float(arg)
                    print(f"  Top-P → {top_p}")
                except ValueError:
                    print("  Usage: /topp <float>")
            elif cmd == "/maxlen":
                try:
                    max_tokens = int(arg)
                    print(f"  Max tokens → {max_tokens}")
                except ValueError:
                    print("  Usage: /maxlen <int>")
            elif cmd == "/rep":
                try:
                    rep_penalty = float(arg)
                    print(f"  Repetition penalty → {rep_penalty}")
                except ValueError:
                    print("  Usage: /rep <float>")
            elif cmd == "/remember":
                if not arg:
                    print("  Usage: /remember <passage text>")
                else:
                    stored_passages.append(arg)
                    # Re-encode all passages
                    full_passage = " . ".join(stored_passages) + " ."
                    mem_keys, mem_vals, mem_mask = encode_passage_to_memory(
                        model, full_passage, device, tagged=tagged)
                    n_valid = mem_mask[0].sum().item() if mem_mask is not None else 0
                    print(f"  ✓ Stored. {len(stored_passages)} passage(s), "
                          f"{n_valid} memory slots active.")
            elif cmd == "/forget":
                stored_passages = []
                mem_keys, mem_vals, mem_mask = None, None, None
                print("  ✓ Memory cleared.")
            elif cmd == "/memory":
                if not stored_passages:
                    print("  No passages stored. Use /remember <text>")
                else:
                    for i, p in enumerate(stored_passages, 1):
                        print(f"  [{i}] {p}")
                    n_valid = mem_mask[0].sum().item() if mem_mask is not None else 0
                    print(f"  → {n_valid} memory slots active")
            elif cmd == "/ask":
                if not arg:
                    print("  Usage: /ask <question>")
                elif mem_keys is None:
                    print("  No passages stored. Use /remember first.")
                else:
                    question = arg
                    if not question.endswith("?"):
                        question += " ?"
                    prompt = [BOS_ID, ANS_ID] + tokenize(question)
                    print(f"\033[1;33mbot>\033[0m ", end='', flush=True)
                    t0 = time.time()
                    tokens = generate_autoregressive(
                        model, cfg, prompt, device,
                        max_new_tokens=max_tokens, temperature=temperature,
                        top_k=top_k, top_p=top_p, repetition_penalty=rep_penalty,
                        mem_keys=mem_keys, mem_vals=mem_vals, mem_mask=mem_mask,
                        use_sliding=(mode == "sliding"),
                        window_size=window_size, num_passes=num_passes,
                        stream=True, strip_output_tags=tagged)
                    elapsed = time.time() - t0
                    n_gen = len(tokens)
                    print(f"  \033[2m({n_gen} tokens, {elapsed:.2f}s, "
                          f"{n_gen/max(elapsed,0.001):.0f} tok/s)\033[0m")
            elif cmd == "/config":
                print(f"  Mode:        {mode}")
                print(f"  Temperature: {temperature}")
                print(f"  Top-K:       {top_k}")
                print(f"  Top-P:       {top_p}")
                print(f"  Max tokens:  {max_tokens}")
                print(f"  Rep penalty: {rep_penalty}")
                print(f"  Tagged:      {tagged}")
                print(f"  Window:      {window_size}")
                print(f"  Passes:      {num_passes}")
                print(f"  Device:      {device}")
                print(f"  Checkpoint:  {ckpt_path}")
                print(f"  Passages:    {len(stored_passages)}")
            else:
                print(f"  Unknown command: {cmd}. Type /help for commands.")
            continue

        # Text generation
        if tagged:
            prompt_text = tag_text(user_input, domain="social")
        else:
            prompt_text = user_input

        prompt_ids = [BOS_ID] + tokenize(prompt_text)

        use_mem = (mode == "qa" and mem_keys is not None)

        print(f"\033[1;33mbot>\033[0m ", end='', flush=True)
        t0 = time.time()
        tokens = generate_autoregressive(
            model, cfg, prompt_ids, device,
            max_new_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=rep_penalty,
            mem_keys=mem_keys if use_mem else None,
            mem_vals=mem_vals if use_mem else None,
            mem_mask=mem_mask if use_mem else None,
            use_sliding=(mode == "sliding"),
            window_size=window_size, num_passes=num_passes,
            stream=True, strip_output_tags=tagged)
        elapsed = time.time() - t0
        n_gen = len(tokens)
        print(f"  \033[2m({n_gen} tokens, {elapsed:.2f}s, "
              f"{n_gen/max(elapsed,0.001):.0f} tok/s)\033[0m")


if __name__ == "__main__":
    main()
