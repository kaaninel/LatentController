"""
Entry point for Google Colab / command-line training.

Usage:
    python run_colab.py --phase 1 --checkpoint_dir /content/drive/MyDrive/llc/checkpoints
    python run_colab.py --phase 2 --checkpoint_dir /content/drive/MyDrive/llc/checkpoints
    python run_colab.py --phase 3 --checkpoint_dir /content/drive/MyDrive/llc/checkpoints
    python run_colab.py --phase 4 --checkpoint_dir /content/drive/MyDrive/llc/checkpoints
"""

import argparse
import os
import sys

import torch


def print_device_info():
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"GPU: {name}  ({mem:.1f} GB)")
    else:
        print("No GPU detected — running on CPU (will be slow!)")


def ensure_tokenizer(data_dir: str, vocab_size: int = 16384):
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Tokenizer not found — training from TinyStories…")
        from dataset import train_tokenizer_from_tinystories
        train_tokenizer_from_tinystories(tok_path, vocab_size=vocab_size)
    else:
        print(f"Tokenizer found at {tok_path}")
    return tok_path


def main():
    parser = argparse.ArgumentParser(description="Looped Latent Controller trainer")
    parser.add_argument("--phase",          type=int,   required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--data_dir",       type=str,   default="./data")
    parser.add_argument("--resume",         action="store_true", help="Resume from latest.pt")
    args = parser.parse_args()

    from hardware import detect_hardware, print_hardware_report
    hw = detect_hardware()
    print_hardware_report(hw)

    print("=" * 60)
    print(f"  Looped Latent Controller — Phase {args.phase}")
    print("=" * 60)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    ensure_tokenizer(args.data_dir)

    try:
        if args.phase == 1:
            from train_phase1 import train
            train(args.checkpoint_dir, args.data_dir, resume=args.resume)

        elif args.phase == 2:
            from train_phase2 import train
            train(args.checkpoint_dir, args.data_dir)

        elif args.phase == 3:
            from train_phase3 import train
            train(args.checkpoint_dir, args.data_dir, resume=args.resume)

        elif args.phase == 4:
            from train_phase4 import train
            train(args.checkpoint_dir, args.data_dir, resume=args.resume)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
