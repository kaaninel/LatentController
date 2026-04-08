#!/usr/bin/env python3
"""
ANT — Overnight training on M4 MacBook Air (MPS).

Downloads the best checkpoint from HuggingFace, then continues training
all night with stride=1 (correct LM), larger dataset, and more steps.

Usage:
    source .venv/bin/activate
    python train_overnight.py 2>&1 | tee overnight_$(date +%Y%m%d_%H%M%S).log
"""

import os
import sys
import time

import torch

# Force line-buffered output so tee shows lines immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("  ANT overnight script starting...", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# OVERNIGHT CONFIG — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

HF_REPO = "kaaninel/ANT"
HF_CHECKPOINT = (
    "checkpoints/chat/checkpoint_latest.pt"  # best from Colab run (11% QA at step 3000)
)
OUTPUT_DIR = "checkpoints/overnight"
DEVICE = "mps"  # M4 MacBook Air

# Data
N_WIKI = 200_000
N_SHELL = 10_000
N_QA = 50_000
N_CHAT = 10_000

# Training — Phase 3 only (QA already 100%, LM at 1.53 from overnight run)
# Load the best local checkpoint and just do chat + refinement
PHASE1_STEPS = 0        # skip — LM already solid
PHASE2_STEPS = 0        # skip — QA already at 100%
PHASE3_STEPS = 10_000   # ~7h of LM+QA+Chat refinement on M4

BATCH_SIZE = 32  # safe for MPS RAM
GRAD_ACCUM = 1
WINDOW_SIZE = 8
NUM_PASSES = 4  # full quality
STRIDE = 1  # MUST be 1 for correct LM training (stride>1 skips positions)
EVAL_INTERVAL = 500

LR = 1e-4  # lower LR for refinement (was 2e-4)

# ──────────────────────────────────────────────────────────────────────────────


def download_checkpoint(hf_repo, hf_file, local_path):
    """Download a single file from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download

        print(f"  Downloading {hf_file} from {hf_repo}...")
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_file,
            local_dir=".",
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            import shutil

            shutil.copy2(downloaded, local_path)
        print(f"  ✓ Saved to {local_path}")
        return True
    except Exception as e:
        print(f"  ✗ HF download failed: {e}")
        return False


def main():
    print("=" * 70)
    print("  ANT — Overnight Training (M4 MPS)")
    print("=" * 70)
    print(f"  Device:     {DEVICE}")
    print(f"  HF source:  {HF_REPO}/{HF_CHECKPOINT}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Phases:     P1={PHASE1_STEPS} P2={PHASE2_STEPS} P3={PHASE3_STEPS}")
    print(f"  Total:      {PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS:,} steps")
    print(
        f"  Data:       wiki={N_WIKI:,} shell={N_SHELL:,} qa={N_QA:,} chat={N_CHAT:,}"
    )
    print(f"  Window:     W={WINDOW_SIZE} passes={NUM_PASSES} stride={STRIDE}")
    print(
        f"  Batch:      {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective"
    )
    total_steps = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS
    est_speed = 0.7  # it/s on M4
    print(f"  Est time:   ~{total_steps / est_speed / 3600:.1f}h at {est_speed} it/s")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from config import ModelConfig
    from model import ANT
    from train_micro import (
        VOCAB_SIZE,
        log_model_summary,
        train_chat,
    )

    cfg = ModelConfig()
    cfg.vocab_size = VOCAB_SIZE
    model = ANT(cfg, use_checkpoint=False).to(DEVICE)
    log_model_summary(model, cfg)

    # ── Load checkpoint — prefer local overnight best, then fallbacks ──
    candidates = [
        os.path.join(OUTPUT_DIR, "checkpoint_best.pt"),       # best from current/previous overnight run
        os.path.join(OUTPUT_DIR, "checkpoint_colab_best.pt"), # downloaded from Colab HF run
        "checkpoints/chat/checkpoint_best.pt",
        "checkpoints/multitask/checkpoint_best.pt",
    ]
    local_ckpt = next((p for p in candidates if os.path.exists(p)), None)

    if local_ckpt is None:
        ok = download_checkpoint(HF_REPO, HF_CHECKPOINT,
                                 os.path.join(OUTPUT_DIR, "checkpoint_colab_best.pt"))
        if ok:
            local_ckpt = os.path.join(OUTPUT_DIR, "checkpoint_colab_best.pt")
        else:
            print("  No checkpoint found — starting from scratch")

    if local_ckpt and os.path.exists(local_ckpt):
        print(f"\n  Loading weights from {local_ckpt}...")
        ckpt = torch.load(local_ckpt, map_location=DEVICE, weights_only=False)
        if "model" in ckpt:
            state = ckpt["model"]
            # Strip torch.compile wrapper prefix (_orig_mod.) if present
            if any(k.startswith("_orig_mod.") for k in state):
                state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
                print("  (stripped torch.compile prefix)")
            model.load_state_dict(state)
            step = ckpt.get("step", 0)
            acc = ckpt.get("accuracy", 0.0)
            print(f"  ✓ Loaded (step={step}, QA={acc:.1%})")
        else:
            print("  ✗ Unexpected checkpoint format — starting from scratch")
    else:
        print("  Starting from scratch (no checkpoint)")

    print()

    # ── Run overnight training ──
    best_acc = train_chat(
        model,
        cfg,
        DEVICE,
        output_dir=OUTPUT_DIR,
        steps_phase1=PHASE1_STEPS,
        steps_phase2=PHASE2_STEPS,
        steps_phase3=PHASE3_STEPS,
        lr=LR,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM,
        eval_interval=EVAL_INTERVAL,
        window_size=WINDOW_SIZE,
        num_passes=NUM_PASSES,
        stride=STRIDE,
        n_wiki=N_WIKI,
        n_shell=N_SHELL,
        n_qa=N_QA,
        n_chat=N_CHAT,
        use_bf16=False,  # MPS doesn't support BF16 well
        use_compile=False,  # torch.compile unreliable on MPS
        hf_repo=HF_REPO,
        hf_upload_interval=2000,
    )

    print(f"\n{'=' * 70}")
    print(f"  Overnight training complete.")
    print(f"  Best QA: {best_acc:.1%}")
    print(f"  Checkpoint: {OUTPUT_DIR}/checkpoint_best.pt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
