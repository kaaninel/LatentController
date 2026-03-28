"""
GPU Detection and Auto-Configuration.

Detects the available hardware and returns optimal training parameters.
"""

import os

import torch

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


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
        'phase1': {'micro_batch': 64, 'grad_accum': 1},
        'phase2': {'batch_size': 4000},
        'phase3': {'micro_batch': 48, 'grad_accum': 1},
        'phase4': {'micro_batch': 24, 'grad_accum': 1},
        'num_workers': min(4, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': False,
    }


def _a100_40gb_config(info):
    """A100-40GB: bf16, 312 TFLOPS bf16"""
    return {
        'phase1': {'micro_batch': 512, 'grad_accum': 1},
        'phase2': {'batch_size': 32000},
        'phase3': {'micro_batch': 384, 'grad_accum': 1},
        'phase4': {'micro_batch': 192, 'grad_accum': 1},
        'num_workers': min(8, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': False,
    }


def _a100_80gb_config(info):
    """A100-80GB: bf16, plenty of room"""
    return {
        'phase1': {'micro_batch': 1024, 'grad_accum': 1},
        'phase2': {'batch_size': 64000},
        'phase3': {'micro_batch': 768, 'grad_accum': 1},
        'phase4': {'micro_batch': 384, 'grad_accum': 1},
        'num_workers': min(16, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': False,
    }


def _h100_config(info):
    """H100-80GB SXM: bf16, 990 TFLOPS bf16"""
    return {
        'phase1': {'micro_batch': 1024, 'grad_accum': 1},
        'phase2': {'batch_size': 64000},
        'phase3': {'micro_batch': 768, 'grad_accum': 1},
        'phase4': {'micro_batch': 384, 'grad_accum': 1},
        'num_workers': min(16, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': False,
    }


def _l40_config(info):
    """L40/L40S: 48GB bf16; L4: 24GB bf16"""
    vram = info['gpu_vram_gb']
    if vram >= 40:
        # L40/L40S (48GB) — comparable to A100-40GB
        return {
            'phase1': {'micro_batch': 512, 'grad_accum': 1},
            'phase2': {'batch_size': 32000},
            'phase3': {'micro_batch': 384, 'grad_accum': 1},
            'phase4': {'micro_batch': 192, 'grad_accum': 1},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': False,
        }
    else:
        # L4 (24GB)
        return {
            'phase1': {'micro_batch': 256, 'grad_accum': 1},
            'phase2': {'batch_size': 16000},
            'phase3': {'micro_batch': 192, 'grad_accum': 1},
            'phase4': {'micro_batch': 96, 'grad_accum': 1},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': False,
        }


def _v100_config(info):
    """V100: 16GB or 32GB, fp16"""
    vram = info['gpu_vram_gb']
    if vram > 20:
        # V100-32GB
        return {
            'phase1': {'micro_batch': 128, 'grad_accum': 1},
            'phase2': {'batch_size': 8000},
            'phase3': {'micro_batch': 96, 'grad_accum': 1},
            'phase4': {'micro_batch': 48, 'grad_accum': 1},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': False,
        }
    else:
        # V100-16GB
        return {
            'phase1': {'micro_batch': 64, 'grad_accum': 1},
            'phase2': {'batch_size': 4000},
            'phase3': {'micro_batch': 48, 'grad_accum': 1},
            'phase4': {'micro_batch': 24, 'grad_accum': 1},
            'num_workers': min(8, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': False,
        }


def _rtx_config(info, vram):
    if vram > 20:
        mb = 256   # RTX 4090 / 3090 (24GB)
    else:
        mb = 64    # RTX 3080/4080 (10-16GB)
    return {
        'phase1': {'micro_batch': mb, 'grad_accum': 1},
        # Phase2 batch_size scales ~62.5x phase1 micro_batch (consistent with other GPU configs)
        'phase2': {'batch_size': mb * 62},
        'phase3': {'micro_batch': mb * 3 // 4, 'grad_accum': 1},
        'phase4': {'micro_batch': mb // 2, 'grad_accum': 1},
        'num_workers': min(8, info['cpu_count']),
        'use_compile': True,
        'use_amp': True,
        'pin_memory': True,
        'gradient_checkpointing': vram < 12,
    }


def _generic_config(info, vram):
    if vram >= 40:
        return _a100_40gb_config(info)
    elif vram >= 16:
        return {
            'phase1': {'micro_batch': 64, 'grad_accum': 1},
            'phase2': {'batch_size': 6000},
            'phase3': {'micro_batch': 48, 'grad_accum': 1},
            'phase4': {'micro_batch': 24, 'grad_accum': 1},
            'num_workers': min(4, info['cpu_count']),
            'use_compile': True,
            'use_amp': True,
            'pin_memory': True,
            'gradient_checkpointing': False,
        }
    else:
        return {
            'phase1': {'micro_batch': 16, 'grad_accum': 2},
            'phase2': {'batch_size': 2000},
            'phase3': {'micro_batch': 12, 'grad_accum': 2},
            'phase4': {'micro_batch': 6, 'grad_accum': 2},
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
    print("  OPTIMAL SETTINGS (auto-detected):")
    print(f"  AMP:             {'Enabled (' + str(info['amp_dtype']).split('.')[-1] + ')' if ov.get('use_amp') else 'Disabled'}")
    print(f"  torch.compile:   {'Enabled' if ov.get('use_compile') else 'Disabled'}")
    print(f"  Grad Checkpoint: {'Enabled' if ov.get('gradient_checkpointing') else 'Disabled'}")
    print(f"  DataLoader Workers: {ov.get('num_workers', 2)}")
    for phase_name in ['phase1', 'phase2', 'phase3', 'phase4']:
        if phase_name in ov:
            pc = ov[phase_name]
            if 'micro_batch' in pc:
                eff = pc['micro_batch'] * pc.get('grad_accum', 1)
                print(f"  {phase_name}: micro_batch={pc['micro_batch']}, grad_accum={pc.get('grad_accum', 1)}, effective_batch={eff}")
            elif 'batch_size' in pc:
                print(f"  {phase_name}: batch_size={pc['batch_size']}")
    print("=" * 70)
