"""
ANT — Terminal chat interface with per-token trie interaction.

Uses engine.generate() for autoregressive generation with live trie updates.
Every token: trie READ → process → trie WRITE → output.
"""

import argparse
import sys
import time

import torch

from config import ModelConfig, MemoryConfig
from model import ANT
from engine import ANTEngine
from data import tokenize, detokenize, BOS_ID, EOS_ID


def load_model(ckpt_path: str, device: str) -> ANTEngine:
    """Load model from checkpoint and wrap in engine."""
    cfg = ModelConfig()
    mem_cfg = MemoryConfig()

    model = ANT(cfg).to(device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        phase = ckpt.get("phase", "?")
        step = ckpt.get("step", 0)
        print(f"Loaded checkpoint: phase {phase}, step {step}")

    engine = ANTEngine(model, mem_cfg, device=device)
    return engine


def chat(engine: ANTEngine, max_tokens: int = 256,
         temperature: float = 0.8, top_k: int = 40):
    """Interactive chat loop."""
    print("\nANT Chat — Type your message (Ctrl+C to exit)")
    print(f"Memory: {engine.memory_stats()}")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            engine.flush()
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            engine.flush()
            break

        if user_input.lower() == "/stats":
            stats = engine.memory_stats()
            print(f"  Trie nodes: {stats['total_nodes']}")
            print(f"  Write records: {stats['total_entries']}")
            continue

        if user_input.lower() == "/reset":
            engine.reset_memory()
            print("  Memory reset.")
            continue

        if user_input.lower() == "/flush":
            engine.flush()
            print("  Memory flushed to disk.")
            continue

        # Tokenize input
        prompt_ids = [BOS_ID] + tokenize(user_input)

        # Generate response
        t0 = time.time()
        output_ids = engine.generate(
            prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        elapsed = time.time() - t0

        # Decode and display
        response = detokenize(output_ids)
        tps = len(output_ids) / elapsed if elapsed > 0 else 0

        print(f"ANT: {response}")
        print(f"  [{len(output_ids)} tokens, {tps:.1f} tok/s, "
              f"{engine.memory_stats()['total_nodes']} trie nodes]")


def main():
    parser = argparse.ArgumentParser(description="ANT Inference")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/train/checkpoint_latest.pt")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    engine = load_model(args.checkpoint, args.device)

    chat(engine, max_tokens=args.max_tokens,
         temperature=args.temperature, top_k=args.top_k)


if __name__ == "__main__":
    main()
