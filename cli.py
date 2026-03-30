#!/usr/bin/env python3
"""
Interactive CLI for LoopedLatentController.

Usage:
    # Download checkpoints first (one-time):
    pip install huggingface_hub
    python cli.py --download

    # Interactive chat:
    python cli.py

    # Single prompt:
    python cli.py -p "Once upon a time"

    # Ingest a document into memory, then chat:
    python cli.py --ingest story.txt

    # Adjust generation:
    python cli.py --max-tokens 200 --act-steps 4 --emit-threshold 0.2
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

from model import LoopedLatentController
from config import ModelConfig, MemoryConfig
from memory import MemorySystem
from orchestrator import Orchestrator
from tokenizer_utils import load_tokenizer
from utils import load_checkpoint


# ─── Device detection ─────────────────────────────────────────────────

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"  Device: CUDA — {name}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: Apple MPS (Metal)")
        return torch.device("mps")
    print("  Device: CPU")
    return torch.device("cpu")


# ─── Model loading ────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, data_dir: str, device: torch.device):
    cfg = ModelConfig()

    # Tokenizer
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"  ✗ Tokenizer not found at {tok_path}")
        print(f"    Run: python cli.py --download")
        sys.exit(1)
    tokenizer = load_tokenizer(tok_path)

    # Model
    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    # Find best checkpoint (prefer inference-only, fall back to full)
    loaded_phase = None
    for phase in [5, 4, 3, 1]:
        infer_ckpt = os.path.join(checkpoint_dir, f"phase{phase}", "best.inference.pt")
        full_ckpt = os.path.join(checkpoint_dir, f"phase{phase}", "best.pt")
        ckpt = infer_ckpt if os.path.exists(infer_ckpt) else full_ckpt
        if os.path.exists(ckpt):
            ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
            # Load model weights
            state_dict = ckpt_data["model"]
            cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
            # Load address heads from checkpoint or Phase 2
            if "addr_heads" in ckpt_data:
                for i, h in enumerate(model.addr_heads):
                    h.load_state_dict(ckpt_data["addr_heads"][i])
            elif phase >= 3:
                for ext in ["best.inference.pt", "best.pt"]:
                    p2 = os.path.join(checkpoint_dir, "phase2", ext)
                    if os.path.exists(p2):
                        p2_data = torch.load(p2, map_location=device, weights_only=False)
                        if "addr_heads" in p2_data:
                            for i, h in enumerate(model.addr_heads):
                                h.load_state_dict(p2_data["addr_heads"][i])
                        break
            loaded_phase = phase
            break

    if loaded_phase is None:
        print(f"  ✗ No checkpoint found in {checkpoint_dir}/")
        print(f"    Run: python cli.py --download")
        sys.exit(1)

    model.eval()
    print(f"  Model: {param_count/1e6:.1f}M params, Phase {loaded_phase} checkpoint")
    return model, tokenizer, cfg


# ─── Download from HuggingFace ────────────────────────────────────────

def download_checkpoints():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub")
        sys.exit(1)

    print("Downloading checkpoints from kaaninel/latentcontroller...")
    snapshot_download(
        repo_id="kaaninel/latentcontroller",
        local_dir="./",
        allow_patterns=[
            "checkpoints/**/*.inference.pt",
            "checkpoints/**/best.pt",
            "data_cache/tokenizer.json",
        ],
    )
    print("✓ Done")


# ─── Interactive CLI ──────────────────────────────────────────────────

def run_interactive(orch: Orchestrator, args):
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║       LoopedLatentController — Interactive       ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Commands:                                       ║")
    print("║    /quit          — exit                         ║")
    print("║    /clear         — reset memory + agent state   ║")
    print("║    /ingest <file> — load document into memory    ║")
    print("║    /memory        — show memory stats            ║")
    print("║    /config        — show current settings        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            rest = user_input[len(cmd):].strip()

            if cmd == "/quit":
                print("Goodbye!")
                break

            elif cmd == "/clear":
                orch.memory = MemorySystem(
                    orch.memory.data_path, orch.memory.cfg
                )
                print("  ✓ Memory cleared\n")
                continue

            elif cmd == "/ingest":
                if not rest or not os.path.exists(rest):
                    print(f"  ✗ File not found: {rest}\n")
                    continue
                with open(rest) as f:
                    text = f.read()
                t0 = time.perf_counter()
                orch.ingest_document(text)
                dt = time.perf_counter() - t0
                entries = orch.memory.total_entries()
                print(f"  ✓ Ingested {len(text)} chars in {dt:.1f}s ({entries} memory entries)\n")
                continue

            elif cmd == "/memory":
                entries = orch.memory.total_entries()
                data_mb = entries * 512 / 1e6
                print(f"  Memory entries: {entries:,}")
                print(f"  Memory size:    {data_mb:.1f} MB")
                print()
                continue

            elif cmd == "/config":
                print(f"  ACT steps:       {args.act_steps}")
                print(f"  Emit threshold:  {args.emit_threshold}")
                print(f"  Max tokens:      {args.max_tokens}")
                print(f"  Device:          {orch.device}")
                print()
                continue

            else:
                print(f"  Unknown command: {cmd}\n")
                continue

        # ── Generation ──
        t0 = time.perf_counter()
        agent = orch.create_agent(
            max_act_steps=args.act_steps,
            emit_threshold=args.emit_threshold,
        )
        orch.feed(agent, user_input)
        response = orch.generate(agent, max_tokens=args.max_tokens)
        dt = time.perf_counter() - t0

        tok_count = len(orch.tokenizer.encode(response).ids) if response else 0
        tok_s = tok_count / dt if dt > 0 else 0

        print(f"\033[1;33mAgent:\033[0m {response}")
        print(f"\033[2m  [{tok_count} tokens, {dt:.2f}s, {tok_s:.0f} tok/s]\033[0m\n")


# ─── Single prompt mode ──────────────────────────────────────────────

def run_prompt(orch: Orchestrator, prompt: str, args):
    t0 = time.perf_counter()
    response = orch.query(
        prompt,
        think_budget=args.act_steps,
        max_output=args.max_tokens,
    )
    dt = time.perf_counter() - t0
    print(response)
    tok_count = len(orch.tokenizer.encode(response).ids) if response else 0
    print(f"\n\033[2m[{tok_count} tokens, {dt:.2f}s, {tok_count/dt:.0f} tok/s]\033[0m",
          file=sys.stderr)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoopedLatentController — Interactive CLI"
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--data-dir", default="./data_cache")
    parser.add_argument("--memory-dir", default="./memory_store")
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="Single prompt (non-interactive)")
    parser.add_argument("--ingest", type=str, default=None,
                        help="Ingest file into memory before starting")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--act-steps", type=int, default=4,
                        help="ACT halting steps (1-6)")
    parser.add_argument("--emit-threshold", type=float, default=0.3,
                        help="Token emission confidence threshold")
    parser.add_argument("--download", action="store_true",
                        help="Download checkpoints from HuggingFace and exit")
    args = parser.parse_args()

    if args.download:
        download_checkpoints()
        return

    print("╔══════════════════════════════════════════════════╗")
    print("║          LoopedLatentController v0.1             ║")
    print("╚══════════════════════════════════════════════════╝")

    device = detect_device()
    model, tokenizer, cfg = load_model(args.checkpoint_dir, args.data_dir, device)

    # Memory system
    mem_cfg = MemoryConfig()
    memory = MemorySystem(args.memory_dir, mem_cfg)
    entries = memory.total_entries()
    if entries > 0:
        print(f"  Memory: {entries:,} entries loaded from {args.memory_dir}/")
    else:
        print(f"  Memory: empty (new store at {args.memory_dir}/)")

    orch = Orchestrator(model, memory, tokenizer, device)

    # Ingest file if requested
    if args.ingest:
        if os.path.exists(args.ingest):
            with open(args.ingest) as f:
                text = f.read()
            print(f"  Ingesting {args.ingest} ({len(text)} chars)...")
            orch.ingest_document(text)
            print(f"  ✓ Ingested → {memory.total_entries():,} memory entries")
        else:
            print(f"  ✗ File not found: {args.ingest}")

    # Run
    if args.prompt:
        run_prompt(orch, args.prompt, args)
    else:
        run_interactive(orch, args)


if __name__ == "__main__":
    main()
