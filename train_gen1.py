"""
Gen 1 Training: Memory-dependent QA.

Thin wrapper around Phase 5 training with Gen1-specific defaults:
  - 500M tokens (~5 epochs over ~100M tokens/epoch)
  - Slightly higher LR (5e-5) for learning new task format
  - More frequent eval/save (smaller dataset)
  - Uses pre-built gen1_memory_qa dataset (context + text with NOOP targets)

Prerequisites:
    python prepare_gen1_data.py   # builds bAbI + SQuAD + TinyStories cache

Usage:
    python train_gen1.py                    # fresh start from Phase 4 checkpoint
    python train_gen1.py --resume           # resume interrupted training
    python train_gen1.py --data_dir ./data  # custom data directory
"""

import argparse
from config import Gen1Config
from train_phase5 import train


def main():
    parser = argparse.ArgumentParser(description="Gen 1: Memory QA Training")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gen1_cfg = Gen1Config()
    print("=" * 64)
    print("  GEN 1: Memory-Dependent QA Training")
    print("=" * 64)
    print(f"  Total tokens:    {gen1_cfg.total_tokens / 1e6:.0f}M")
    print(f"  Learning rate:   {gen1_cfg.lr:.1e}")
    print(f"  Eval interval:   every {gen1_cfg.eval_interval} steps")
    print(f"  Dataset:         gen1_memory_qa (bAbI + SQuAD + TinyStories)")
    print("=" * 64)

    train(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        resume=args.resume,
        dataset_name="gen1_memory_qa",
        context_column="context",
        text_column="text",
        phase_config=gen1_cfg,
    )


if __name__ == "__main__":
    main()
