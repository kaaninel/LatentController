"""
Gen 1 Training: Memory-dependent QA.

Thin wrapper around Phase 5 training with Gen1-specific defaults:
  - 500M tokens (~5 epochs over ~100M tokens/epoch)
  - Slightly higher LR (5e-5) for learning new task format
  - More frequent eval/save (smaller dataset)
  - Uses pre-built gen1_memory_qa dataset (context + text with NOOP targets)

Prerequisites:
    python generate_phase5_data.py   # generates synthetic QA + knowledge base
    python preseed_memory.py         # (optional) pre-seeds memory with KB facts

Usage:
    # Phase 5a: frozen pre-seeded memory (learn to READ)
    python train_gen1.py --freeze_memory --preseed_memory_dir ./checkpoints/phase5/preseed_memory

    # Phase 5b: unfrozen memory (learn to READ + WRITE)
    python train_gen1.py --resume

    # Full training without pre-seeding
    python train_gen1.py
"""

import os
import argparse
from config import Gen1Config
from train_phase5 import train


def main():
    parser = argparse.ArgumentParser(description="Gen 1: Memory QA Training")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze_memory", action="store_true",
                        help="Phase 5a: read-only memory (learn to use pre-seeded facts)")
    parser.add_argument("--preseed_memory_dir", default=None,
                        help="Path to pre-seeded memory store")
    args = parser.parse_args()

    gen1_cfg = Gen1Config()

    mode = "Phase 5a (frozen memory)" if args.freeze_memory else "Phase 5b (active memory)"
    print("=" * 64)
    print("  GEN 1: Memory-Dependent QA Training")
    print("=" * 64)
    print(f"  Mode:            {mode}")
    print(f"  Total tokens:    {gen1_cfg.total_tokens / 1e6:.0f}M")
    print(f"  Learning rate:   {gen1_cfg.lr:.1e}")
    print(f"  Eval interval:   every {gen1_cfg.eval_interval} steps")
    print(f"  Dataset:         gen1_memory_qa (synthetic QA + TinyStories)")
    if args.preseed_memory_dir:
        print(f"  Pre-seeded mem:  {args.preseed_memory_dir}")
    print("=" * 64)

    train(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        resume=args.resume,
        dataset_name="gen1_memory_qa",
        context_column="context",
        text_column="text",
        phase_config=gen1_cfg,
        freeze_memory=args.freeze_memory,
        preseed_memory_dir=args.preseed_memory_dir,
    )


if __name__ == "__main__":
    main()
