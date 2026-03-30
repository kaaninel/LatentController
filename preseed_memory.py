"""
Pre-seed memory with knowledge base facts.

Runs each fact through the Phase 4 model to get hidden state representations,
then writes them to the memory system at the model's computed addresses.

This creates a memory store full of useful facts BEFORE training starts,
giving the model something meaningful to read during Phase 5a.

Usage:
    python preseed_memory.py                              # default paths
    python preseed_memory.py --kb_path ./data_cache/knowledge_base.jsonl
    python preseed_memory.py --output_dir ./checkpoints/phase5/preseed_memory
"""

import os
import json
import argparse
import sys

import numpy as np
import torch

from model import LoopedLatentController
from config import ModelConfig, MemoryConfig
from memory import MemorySystem
from tokenizer_utils import load_tokenizer, encode
from agent import addr_bytes


def load_model(checkpoint_dir: str, device: torch.device):
    """Load the best available checkpoint."""
    cfg = ModelConfig()
    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)

    loaded_from = None
    for phase in [5, 4, 3, 1]:
        phase_dir = os.path.join(checkpoint_dir, f"phase{phase}")
        for name in ["best.pt", "latest.pt"]:
            path = os.path.join(phase_dir, name)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=device, weights_only=False)
                state_dict = ckpt["model"]
                cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned, strict=False)
                if "addr_heads" in ckpt:
                    for i, h in enumerate(model.addr_heads):
                        h.load_state_dict(ckpt["addr_heads"][i])
                loaded_from = f"Phase {phase} ({name})"
                break
        if loaded_from:
            break

    # Try Phase 2 address heads as fallback
    if loaded_from:
        p2_path = os.path.join(checkpoint_dir, "phase2", "addr_heads.pt")
        if os.path.exists(p2_path) and "addr_heads" not in (ckpt or {}):
            p2_data = torch.load(p2_path, map_location=device, weights_only=False)
            if "addr_heads" in p2_data:
                for i, h in enumerate(model.addr_heads):
                    h.load_state_dict(p2_data["addr_heads"][i])
                print(f"  Loaded address heads from Phase 2")

    if loaded_from is None:
        print("✗ No checkpoint found")
        sys.exit(1)

    model.eval()
    print(f"  Loaded model from {loaded_from}")
    return model, cfg


@torch.no_grad()
def preseed(
    model: LoopedLatentController,
    cfg: ModelConfig,
    tokenizer,
    facts: list,
    memory: MemorySystem,
    device: torch.device,
    write_positions: str = "last",
):
    """
    Process each fact through the model and write hidden states to memory.

    write_positions:
      "last"  — write only the last token's hidden state (one entry per fact)
      "multi" — write every 32 tokens (multiple entries per fact, more coverage)
    """
    bos_id = tokenizer.token_to_id("<bos>") or 2
    eos_id = tokenizer.token_to_id("<eos>") or 1

    total_writes = 0
    log_every = max(1, len(facts) // 20)

    for idx, fact_item in enumerate(facts):
        fact_text = fact_item["fact"]

        # Tokenize
        ids = encode(tokenizer, fact_text, max_len=cfg.max_seq_len - 2)
        if not ids:
            continue
        full_ids = [bos_id] + ids + [eos_id]
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)

        # Forward pass to get hidden states
        _, _, hidden = model(inp, return_hidden=True)
        # hidden: (1, T, d_model)

        if write_positions == "multi":
            # Write at multiple positions (every 32 tokens + last)
            positions = list(range(32, len(full_ids), 32))
            if not positions or positions[-1] != len(full_ids) - 2:
                positions.append(len(full_ids) - 2)  # last content token (before EOS)
        else:
            positions = [len(full_ids) - 2]  # just last content token

        for pos in positions:
            h = hidden[0, pos, :]  # (d_model,)
            h_np = h.float().cpu().numpy()

            # Compute addresses
            addrs = model.compute_addresses(h)
            ab = [addr_bytes(a) for a in addrs]

            # Write to memory
            memory.write_memory(ab, h_np)
            total_writes += 1

        if idx % log_every == 0:
            print(f"    Processing: {idx:,}/{len(facts):,} ({100*idx/max(len(facts),1):.0f}%)", flush=True)

    return total_writes


def main():
    parser = argparse.ArgumentParser(description="Pre-seed memory with knowledge base")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--kb_path", default=None,
                        help="Path to knowledge_base.jsonl (default: data_dir/knowledge_base.jsonl)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save pre-seeded memory (default: checkpoints/phase5/preseed_memory)")
    parser.add_argument("--write_mode", default="multi", choices=["last", "multi"],
                        help="Write at last token only or at multiple positions")
    args = parser.parse_args()

    # Resolve paths
    kb_path = args.kb_path or os.path.join(args.data_dir, "knowledge_base.jsonl")
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "phase5", "preseed_memory")

    if not os.path.exists(kb_path):
        print(f"✗ Knowledge base not found at {kb_path}")
        print(f"  Run: python generate_phase5_data.py")
        return

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 64)
    print("  Pre-seeding Memory from Knowledge Base")
    print("=" * 64)
    print(f"  Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint_dir, device)

    # Load tokenizer
    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    tokenizer = load_tokenizer(tok_path)

    # Load knowledge base
    facts = []
    with open(kb_path) as f:
        for line in f:
            line = line.strip()
            if line:
                facts.append(json.loads(line))
    print(f"  Loaded {len(facts):,} knowledge base facts")

    # Create fresh memory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    mcfg = MemoryConfig()
    memory = MemorySystem(output_dir, mcfg)

    # Pre-seed
    print()
    print(f"  Writing to memory (mode: {args.write_mode})...")
    total_writes = preseed(
        model, cfg, tokenizer, facts, memory, device,
        write_positions=args.write_mode,
    )

    # Flush to disk
    memory.flush_to_disk()

    print()
    print("=" * 64)
    print("  Pre-seeding Complete")
    print("=" * 64)
    print(f"  Memory entries: {memory.total_entries():,}")
    print(f"  Total writes:   {total_writes:,}")
    print(f"  Output:         {output_dir}")
    data_size = os.path.getsize(os.path.join(output_dir, "data.bin")) / 1e6 if os.path.exists(os.path.join(output_dir, "data.bin")) else 0
    print(f"  Data size:      {data_size:.1f} MB")
    print()
    print("  Next: python train_gen1.py --preseed_memory_dir " + output_dir)
    print("=" * 64)


if __name__ == "__main__":
    main()
