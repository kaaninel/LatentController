"""
GPU Detection and Auto-Configuration.

Detects the available hardware and returns optimal training parameters.
Includes dynamic VRAM calibration to maximize GPU utilization.
"""

import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


# ---------------------------------------------------------------------------
# Dynamic batch-size calibration
# ---------------------------------------------------------------------------

def auto_calibrate_batch_size(
    trial_fn: Callable[[int], None],
    device: str,
    initial_bs: int,
    target_effective: int = 128,
    target_vram_frac: float = 0.90,
    min_bs: int = 1,
    max_bs: int = 2048,
) -> tuple:
    """
    Find the maximum micro_batch that fits in VRAM via binary search.

    Args:
        trial_fn: Callable(batch_size) that runs one forward+backward pass
                  with the given batch size. Must raise
                  torch.cuda.OutOfMemoryError if it doesn't fit.
        device: 'cuda' or 'cpu'.
        initial_bs: Starting batch size (hardware override hint).
        target_effective: Desired effective batch (micro_batch * grad_accum).
        target_vram_frac: Target VRAM utilisation (0.0-1.0).
        min_bs: Floor for micro_batch.
        max_bs: Ceiling for micro_batch.

    Returns:
        (micro_batch, grad_accum)
    """
    if device != 'cuda':
        ga = max(1, target_effective // initial_bs)
        return initial_bs, ga

    total_vram = torch.cuda.get_device_properties(0).total_memory
    target_bytes = int(total_vram * target_vram_frac)

    print(f"  ⚡ Auto-calibrating batch size "
          f"(target: {target_vram_frac*100:.0f}% of {total_vram/1e9:.1f} GB = "
          f"{target_bytes/1e9:.1f} GB)…")

    def _fits(bs: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            trial_fn(bs)
            peak = torch.cuda.max_memory_allocated()
            torch.cuda.empty_cache()
            return peak <= target_bytes
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False

    # Quick first probe with the initial (static) hint
    if not _fits(initial_bs):
        # Static config is already too big — search [min_bs, initial_bs)
        lo, hi = min_bs, initial_bs - 1
    else:
        # Room to grow — find upper bound by doubling
        lo = initial_bs
        hi = initial_bs
        while hi < max_bs:
            nxt = min(hi * 2, max_bs)
            if _fits(nxt):
                hi = nxt
                if hi == max_bs:
                    break
            else:
                hi = nxt
                break
        # hi is either the first that didn't fit, or max_bs that fit

    # Binary search for the largest bs that fits
    best = min_bs
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    micro_batch = max(min_bs, best)
    grad_accum = max(1, target_effective // micro_batch)
    eff = micro_batch * grad_accum

    print(f"  ✓ Calibrated: micro_batch={micro_batch}, "
          f"grad_accum={grad_accum} (effective={eff})")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return micro_batch, grad_accum


def build_trial_fn(model, cfg, device, use_amp, amp_dtype,
                   has_memory=False, act_steps=1):
    """
    Build a trial function for batch-size calibration.

    The returned callable runs one forward+backward pass that mirrors
    the real training step (including memory prepend and ACT iterations).
    """
    from torch.amp import autocast

    def trial(bs):
        inp = torch.randint(0, cfg.vocab_size, (bs, cfg.max_seq_len),
                            device=device)

        mem = None
        if has_memory:
            mem = torch.zeros(bs, cfg.n_mem_slots, cfg.d_model,
                              device=device, dtype=amp_dtype if use_amp
                              else torch.float32)

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            weighted = None
            for _ in range(act_steps):
                logits, halt_logits, hidden = model(
                    inp, memory_vectors=mem, return_hidden=True)
                if weighted is None:
                    weighted = logits
                else:
                    weighted = weighted + logits

            loss = F.cross_entropy(
                weighted.reshape(-1, weighted.size(-1)),
                inp.reshape(-1),  # dummy target
                ignore_index=cfg.pad_id,
            )

        loss.backward()
        model.zero_grad(set_to_none=True)

    return trial


def handle_oom(micro_batch, grad_accum, target_effective, min_bs=1):
    """
    Called when an OOM occurs during training.
    Returns (new_micro_batch, new_grad_accum, needs_loader_rebuild).
    """
    torch.cuda.empty_cache()
    new_mb = max(min_bs, micro_batch // 2)
    new_ga = max(1, target_effective // new_mb)
    print(f"  ⚠ OOM! Reducing micro_batch {micro_batch}→{new_mb}, "
          f"grad_accum {grad_accum}→{new_ga}")
    return new_mb, new_ga, (new_mb != micro_batch)


def detect_hardware():
    """Detect GPU and return hardware info dict + optimal config overrides."""
    info = {
        'device': 'cpu',
        'gpu_name': None,
        'gpu_vram_gb': 0,
        'gpu_vram_free_gb': 0,
        'compute_capability': (0, 0),
        'cpu_count': os.cpu_count() or 2,
        'ram_gb': _psutil.virtual_memory().total / 1e9 if _psutil else 0,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'has_bf16': False,
        'has_flash_attn': False,
        'has_compile': hasattr(torch, 'compile'),
        'amp_dtype': torch.float32,  # will be overridden
    }

    if not torch.cuda.is_available():
        info['overrides'] = _cpu_config()
        return info

    info['device'] = 'cuda'
    props = torch.cuda.get_device_properties(0)
    info['gpu_name'] = props.name
    info['gpu_vram_gb'] = props.total_memory / 1e9
    info['gpu_vram_free_gb'] = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
    info['compute_capability'] = (props.major, props.minor)

    # bf16 requires compute >= 8.0 (A100, H100, L40, etc.)
    info['has_bf16'] = props.major >= 8
    # Flash attention via SDPA requires compute >= 7.0
    info['has_flash_attn'] = props.major >= 7
    # AMP dtype: bf16 if available, else fp16
    info['amp_dtype'] = torch.bfloat16 if info['has_bf16'] else torch.float16

    # Select optimal config based on GPU
    name_lower = props.name.lower()
    vram = info['gpu_vram_gb']

    if 'h100' in name_lower or 'h200' in name_lower:
        info['overrides'] = _h100_config(info)
    elif 'a100' in name_lower:
        if vram > 50:
            info['overrides'] = _a100_80gb_config(info)
        else:
            info['overrides'] = _a100_40gb_config(info)
    elif 'l40' in name_lower or 'l4' in name_lower:
        info['overrides'] = _l40_config(info)
    elif 't4' in name_lower:
        info['overrides'] = _t4_config(info)
    elif 'v100' in name_lower:
        info['overrides'] = _v100_config(info)
    elif 'rtx 4090' in name_lower or 'rtx 3090' in name_lower:
        info['overrides'] = _rtx_config(info, vram)
    else:
        # Unknown GPU: conservative based on VRAM
        info['overrides'] = _generic_config(info, vram)

    return info


def _t4_config(info):
    """Tesla T4: 16GB, fp16, 65 TFLOPS fp16"""
    return {
        'phase1': {'micro_batch': 128, 'grad_accum': 1},
        'phase2': {'batch_size': 4000},
        'phase3': {'micro_batch': 32, 'grad_accum': 4},
        'phase4': {'micro_batch': 16, 'grad_accum': 8},
        'phase5': {'micro_batch': 8, 'grad_accum': 16},
        'num_workers': min(4, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _a100_40gb_config(info):
    """A100-40GB: bf16, 312 TFLOPS bf16"""
    return {
        'phase1': {'micro_batch': 160, 'grad_accum': 1},
        'phase2': {'batch_size': 8000},
        'phase3': {'micro_batch': 64, 'grad_accum': 2},
        'phase4': {'micro_batch': 32, 'grad_accum': 4},
        'phase5': {'micro_batch': 16, 'grad_accum': 8},
        'num_workers': min(8, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _a100_80gb_config(info):
    """A100-80GB: bf16, plenty of room"""
    return {
        'phase1': {'micro_batch': 320, 'grad_accum': 1},
        'phase2': {'batch_size': 16000},
        'phase3': {'micro_batch': 128, 'grad_accum': 1},
        'phase4': {'micro_batch': 64, 'grad_accum': 2},
        'phase5': {'micro_batch': 32, 'grad_accum': 4},
        'num_workers': min(16, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _h100_config(info):
    """H100-80GB SXM: bf16, 990 TFLOPS bf16"""
    return {
        'phase1': {'micro_batch': 512, 'grad_accum': 1},
        'phase2': {'batch_size': 16000},
        'phase3': {'micro_batch': 192, 'grad_accum': 1},
        'phase4': {'micro_batch': 96, 'grad_accum': 2},
        'phase5': {'micro_batch': 48, 'grad_accum': 2},
        'num_workers': min(16, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _l40_config(info):
    """L40/L40S: 48GB bf16; L4: 24GB bf16"""
    vram = info['gpu_vram_gb']
    if vram >= 40:
        # L40/L40S (48GB)
        return {
            'phase1': {'micro_batch': 192, 'grad_accum': 1},
            'phase2': {'batch_size': 8000},
            'phase3': {'micro_batch': 96, 'grad_accum': 1},
            'phase4': {'micro_batch': 48, 'grad_accum': 2},
            'phase5': {'micro_batch': 24, 'grad_accum': 4},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': True,
        }
    else:
        # L4 (24GB)
        return {
            'phase1': {'micro_batch': 128, 'grad_accum': 1},
            'phase2': {'batch_size': 4000},
            'phase3': {'micro_batch': 48, 'grad_accum': 2},
            'phase4': {'micro_batch': 24, 'grad_accum': 4},
            'phase5': {'micro_batch': 12, 'grad_accum': 8},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': True,
        }


def _v100_config(info):
    """V100: 32GB fp16"""
    return {
        'phase1': {'micro_batch': 192, 'grad_accum': 1},
        'phase2': {'batch_size': 8000},
        'phase3': {'micro_batch': 64, 'grad_accum': 2},
        'phase4': {'micro_batch': 32, 'grad_accum': 4},
        'phase5': {'micro_batch': 16, 'grad_accum': 8},
        'num_workers': min(8, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _rtx_config(info, vram):
    if vram > 20:
        mb = 96    # RTX 4090 / 3090 (24GB)
    else:
        mb = 48    # RTX 3080/4080 (10-16GB)
    return {
        'phase1': {'micro_batch': mb, 'grad_accum': 1},
        'phase2': {'batch_size': 4000},
        'phase3': {'micro_batch': mb // 2, 'grad_accum': 2},
        'phase4': {'micro_batch': mb // 4, 'grad_accum': 4},
        'phase5': {'micro_batch': mb // 8 or 2, 'grad_accum': 8},
        'num_workers': min(8, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': True,
    }


def _generic_config(info, vram):
    if vram >= 40:
        return _a100_40gb_config(info)
    elif vram >= 16:
        return {
            'phase1': {'micro_batch': 64, 'grad_accum': 1},
            'phase2': {'batch_size': 4000},
            'phase3': {'micro_batch': 32, 'grad_accum': 2},
            'phase4': {'micro_batch': 16, 'grad_accum': 4},
            'phase5': {'micro_batch': 8, 'grad_accum': 8},
            'num_workers': min(4, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': True,
        }
    else:
        return {
            'phase1': {'micro_batch': 16, 'grad_accum': 2},
            'phase2': {'batch_size': 2000},
            'phase3': {'micro_batch': 12, 'grad_accum': 2},
            'phase4': {'micro_batch': 6, 'grad_accum': 2},
            'phase5': {'micro_batch': 2, 'grad_accum': 8},
            'num_workers': min(2, info['cpu_count']),
            'use_compile': False,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': True,
        }


def _cpu_config():
    return {
        'phase1': {'micro_batch': 2, 'grad_accum': 16},
        'phase2': {'batch_size': 500},
        'phase3': {'micro_batch': 2, 'grad_accum': 16},
        'phase4': {'micro_batch': 1, 'grad_accum': 32},
        'phase5': {'micro_batch': 1, 'grad_accum': 32},
        'num_workers': 0,
        'use_compile': False,
        'use_amp': False,
        'pin_memory': False,
        'gradient_checkpointing': True,
    }


def print_hardware_report(info):
    """Print a comprehensive hardware report."""
    print("=" * 70)
    print("  HARDWARE DETECTION REPORT")
    print("=" * 70)
    print(f"  PyTorch:         {info['torch_version']}")
    print(f"  CUDA:            {info['cuda_version'] or 'Not available'}")
    print(f"  Device:          {info['device']}")
    if info['gpu_name']:
        print(f"  GPU:             {info['gpu_name']}")
        print(f"  VRAM:            {info['gpu_vram_gb']:.1f} GB total, {info['gpu_vram_free_gb']:.1f} GB free")
        print(f"  Compute Cap:     {info['compute_capability'][0]}.{info['compute_capability'][1]}")
        print(f"  BF16 Support:    {'Yes' if info['has_bf16'] else 'No (using FP16)'}")
        print(f"  Flash Attention: {'Yes' if info['has_flash_attn'] else 'No'}")
    print(f"  torch.compile:   {'Yes' if info['has_compile'] else 'No'}")
    print(f"  CPU Cores:       {info['cpu_count']}")
    print(f"  System RAM:      {info['ram_gb']:.1f} GB")
    print("-" * 70)

    ov = info['overrides']
    print("  INITIAL SETTINGS (will be refined by auto-calibration):")
    print(f"  AMP:             {'Enabled (' + str(info['amp_dtype']).split('.')[-1] + ')' if ov.get('use_amp') else 'Disabled'}")
    print(f"  torch.compile:   {'Enabled' if ov.get('use_compile') else 'Disabled'}")
    print(f"  Grad Checkpoint: {'Enabled' if ov.get('gradient_checkpointing') else 'Disabled'}")
    print(f"  DataLoader Workers: {ov.get('num_workers', 2)}")
    for phase_name in ['phase1', 'phase2', 'phase3', 'phase4', 'phase5']:
        if phase_name in ov:
            pc = ov[phase_name]
            if 'micro_batch' in pc:
                eff = pc['micro_batch'] * pc.get('grad_accum', 1)
                print(f"  {phase_name}: micro_batch={pc['micro_batch']}, grad_accum={pc.get('grad_accum', 1)}, effective_batch={eff}")
            elif 'batch_size' in pc:
                print(f"  {phase_name}: batch_size={pc['batch_size']}")
    print("=" * 70)
