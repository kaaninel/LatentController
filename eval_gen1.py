"""
Gen 1 Evaluation: Multi-turn memory recall test.

Tests whether the model can:
  1. Absorb a passage into memory (via agent.feed)
  2. Answer a question about it (via agent.generate)

Uses bAbI test set (20K examples, held out from training).
Each example: feed passage → feed question → generate → check answer.

Usage:
    python eval_gen1.py                           # full eval
    python eval_gen1.py --max_examples 500        # quick test
    python eval_gen1.py --tasks 1 2 3             # specific bAbI tasks
"""

import argparse
import os
import sys
import time

import torch
from datasets import load_dataset

from model import LoopedLatentController
from config import ModelConfig, MemoryConfig
from memory import MemorySystem
from orchestrator import Orchestrator
from tokenizer_utils import load_tokenizer


def load_model_for_eval(checkpoint_dir: str, data_dir: str, device: torch.device):
    """Load model + tokenizer (same logic as cli.py)."""
    cfg = ModelConfig()
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"✗ Tokenizer not found at {tok_path}")
        sys.exit(1)
    tokenizer = load_tokenizer(tok_path)

    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)

    loaded_phase = None
    for phase in [5, 4, 3, 1]:
        phase_dir = os.path.join(checkpoint_dir, f"phase{phase}")
        # Also check gen1 directory
        if phase == 5:
            dirs_to_check = [
                os.path.join(checkpoint_dir, "gen1"),
                phase_dir,
            ]
        else:
            dirs_to_check = [phase_dir]

        found = False
        for pdir in dirs_to_check:
            for name in ["best.pt", "latest.pt", "best.inference.pt", "latest.inference.pt"]:
                ckpt_path = os.path.join(pdir, name)
                if os.path.exists(ckpt_path):
                    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
                    state_dict = ckpt_data["model"]
                    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
                    model.load_state_dict(cleaned, strict=False)
                    if "addr_heads" in ckpt_data:
                        for i, h in enumerate(model.addr_heads):
                            h.load_state_dict(ckpt_data["addr_heads"][i])
                    loaded_phase = f"{phase} ({os.path.basename(pdir)}/{name})"
                    found = True
                    break
            if found:
                break
        if found:
            break

    # Try loading address heads from Phase 2 if not in checkpoint
    if loaded_phase and "addr_heads" not in (ckpt_data if 'ckpt_data' in dir() else {}):
        p2_path = os.path.join(checkpoint_dir, "phase2", "addr_heads.pt")
        if os.path.exists(p2_path):
            p2_data = torch.load(p2_path, map_location=device, weights_only=False)
            if "addr_heads" in p2_data:
                for i, h in enumerate(model.addr_heads):
                    h.load_state_dict(p2_data["addr_heads"][i])

    if loaded_phase is None:
        print(f"✗ No checkpoint found in {checkpoint_dir}/")
        sys.exit(1)

    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {loaded_phase}")
    return model, tokenizer, cfg


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, strip, collapse whitespace)."""
    import re
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    return s


def check_answer(generated: str, expected: str) -> bool:
    """Check if the generated text contains the expected answer."""
    gen_norm = normalize_answer(generated)
    exp_norm = normalize_answer(expected)
    if not exp_norm:
        return False
    # Exact match or containment
    return exp_norm in gen_norm


def evaluate_babi(
    orchestrator: Orchestrator,
    device: torch.device,
    max_examples: int = 0,
    tasks: list = None,
    act_steps: int = 4,
    temperature: float = 0.5,
    verbose: bool = False,
):
    """Run bAbI evaluation with multi-turn memory recall."""
    print("\n  Loading bAbI test set...")
    ds = load_dataset("Muennighoff/babi", split="test", streaming=True)

    task_correct = {}
    task_total = {}
    task_examples = {}
    total_correct = 0
    total_tested = 0
    total_time = 0.0

    for item in ds:
        task = item.get("task", 0)
        if tasks and task not in tasks:
            continue
        if max_examples and total_tested >= max_examples:
            break

        passage = item["passage"].strip()
        question = item["question"].strip()
        expected = item["answer"].strip()

        # Create fresh agent for each example (clean memory state)
        agent = orchestrator.create_agent(
            max_act_steps=act_steps, emit_threshold=0.3)

        # Step 1: Feed passage (absorbed into memory via NOOP-like processing)
        t0 = time.time()
        orchestrator.feed(agent, passage)

        # Step 2: Feed question
        orchestrator.feed(agent, f"\n{question}")

        # Step 3: Generate answer
        answer = orchestrator.generate(
            agent, max_tokens=30,
            temperature=temperature, top_k=20,
            repetition_penalty=1.3,
        )
        elapsed = time.time() - t0

        # Check correctness
        correct = check_answer(answer, expected)
        total_correct += int(correct)
        total_tested += 1
        total_time += elapsed

        task_correct[task] = task_correct.get(task, 0) + int(correct)
        task_total[task] = task_total.get(task, 0) + 1

        # Store first few examples per task for display
        if task not in task_examples:
            task_examples[task] = []
        if len(task_examples[task]) < 2:
            task_examples[task].append({
                "passage": passage[:100],
                "question": question,
                "expected": expected,
                "generated": answer.strip()[:80],
                "correct": correct,
            })

        if verbose and total_tested <= 20:
            mark = "✓" if correct else "✗"
            print(f"    [{mark}] Task {task}: Q=\"{question}\" → \"{answer.strip()[:50]}\" (expected: {expected})")

        if total_tested % 100 == 0:
            acc = 100 * total_correct / total_tested
            print(f"    Progress: {total_tested:,} examples, {acc:.1f}% accuracy", flush=True)

    return {
        "total_correct": total_correct,
        "total_tested": total_tested,
        "total_time": total_time,
        "task_correct": task_correct,
        "task_total": task_total,
        "task_examples": task_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Gen 1: Memory Recall Evaluation")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--data_dir", default="./data_cache")
    parser.add_argument("--memory_dir", default="./memory_store_eval")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Max examples to evaluate (0 = all)")
    parser.add_argument("--tasks", type=int, nargs="+", default=None,
                        help="Specific bAbI task numbers to evaluate (default: all)")
    parser.add_argument("--act_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 64)
    print("  Gen 1 Evaluation: Memory Recall (bAbI)")
    print("=" * 64)
    print(f"  Device: {device}")

    # Load model
    model, tokenizer, cfg = load_model_for_eval(
        args.checkpoint_dir, args.data_dir, device)

    # Fresh memory for eval
    mcfg = MemoryConfig()
    if os.path.exists(args.memory_dir):
        import shutil
        shutil.rmtree(args.memory_dir)
    os.makedirs(args.memory_dir, exist_ok=True)
    memory = MemorySystem(args.memory_dir, mcfg)

    orchestrator = Orchestrator(model, memory, tokenizer, device)

    # Run evaluation
    results = evaluate_babi(
        orchestrator, device,
        max_examples=args.max_examples,
        tasks=args.tasks,
        act_steps=args.act_steps,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    # Print results
    print()
    print("=" * 64)
    print("  Results")
    print("=" * 64)

    overall_acc = (100 * results["total_correct"] / max(results["total_tested"], 1))
    avg_time = results["total_time"] / max(results["total_tested"], 1)

    print(f"  Overall:  {results['total_correct']}/{results['total_tested']} ({overall_acc:.1f}%)")
    print(f"  Avg time: {avg_time:.2f}s per example")
    print()
    print("  Per-task breakdown:")
    for task in sorted(results["task_total"].keys()):
        correct = results["task_correct"].get(task, 0)
        total = results["task_total"][task]
        acc = 100 * correct / max(total, 1)
        print(f"    Task {task:2d}: {correct:4d}/{total:4d} ({acc:5.1f}%)")

    # Show example outputs
    print()
    print("  Example outputs:")
    for task in sorted(results["task_examples"].keys())[:5]:
        for ex in results["task_examples"][task][:1]:
            mark = "✓" if ex["correct"] else "✗"
            print(f"    [{mark}] Task {task}: \"{ex['question']}\"")
            print(f"        Expected:  {ex['expected']}")
            print(f"        Generated: {ex['generated']}")

    print()
    print("=" * 64)
    if overall_acc >= 60:
        print(f"  ✓ Memory recall accuracy {overall_acc:.1f}% ≥ 60% — GO for Gen 2")
    elif overall_acc >= 30:
        print(f"  ~ Memory recall accuracy {overall_acc:.1f}% — partial success, may need more training")
    else:
        print(f"  ✗ Memory recall accuracy {overall_acc:.1f}% < 30% — memory not being used effectively")
    print("=" * 64)

    # Cleanup eval memory
    import shutil
    shutil.rmtree(args.memory_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
