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
import logging
import json
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset

from model import LoopedLatentController
from config import ModelConfig, MemoryConfig
from memory import MemorySystem
from orchestrator import Orchestrator
from tokenizer_utils import load_tokenizer, encode, decode

log = logging.getLogger("eval_gen1")


def load_model_for_eval(checkpoint_dir: str, data_dir: str, device: torch.device):
    """Load model + tokenizer (same logic as cli.py)."""
    cfg = ModelConfig()
    tok_path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        log.error("Tokenizer not found at %s", tok_path)
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
        log.error("No checkpoint found in %s/", checkpoint_dir)
        sys.exit(1)

    model.eval()
    log.info("Model: %.1fM params, loaded from %s",
             sum(p.numel() for p in model.parameters()) / 1e6, loaded_phase)
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


def evaluate_babi_batched(
    model: LoopedLatentController,
    tokenizer,
    device: torch.device,
    cfg: ModelConfig,
    max_examples: int = 0,
    tasks: list = None,
    act_steps: int = 4,
    batch_size: int = 64,
    verbose: bool = False,
):
    """
    Fast batched bAbI evaluation — packs passage+question as sequences,
    runs batched forward passes, checks greedy next-token against answer.

    ~50-100× faster than token-by-token Orchestrator evaluation.
    """
    from torch.amp import autocast

    log.info("Loading bAbI test set...")
    ds = load_dataset("Muennighoff/babi", split="test", streaming=True)

    # Collect examples
    examples = []
    for item in ds:
        task = item.get("task", 0)
        if tasks and task not in tasks:
            continue
        if max_examples and len(examples) >= max_examples:
            break
        examples.append({
            "task": task,
            "passage": item["passage"].strip(),
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
        })

    log.info("Collected %d examples, evaluating in batches of %d", len(examples), batch_size)

    pad_id = cfg.pad_id
    task_correct = {}
    task_total = {}
    task_examples = {}
    total_correct = 0
    total_tested = 0
    t0 = time.time()

    # Determine if we should use AMP
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start:batch_start + batch_size]

        # Tokenize: "passage\nquestion\n"
        token_seqs = []
        for ex in batch:
            text = f"{ex['passage']}\n{ex['question']}\n"
            ids = encode(tokenizer, text)
            # Truncate to max_seq_len - 1 (leave room for prediction)
            ids = ids[:cfg.max_seq_len - 1]
            token_seqs.append(ids)

        # Pad to same length
        max_len = max(len(s) for s in token_seqs)
        padded = []
        for s in token_seqs:
            padded.append(s + [pad_id] * (max_len - len(s)))

        inp = torch.tensor(padded, device=device, dtype=torch.long)

        # Forward pass — single batch, ACT weighted
        with torch.no_grad():
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                # Run ACT loop (simplified — no memory for speed)
                remaining = torch.ones(inp.size(0), inp.size(1), device=device)
                weighted_logits = None
                for i in range(act_steps):
                    logits, halt_logits, hidden = model(
                        inp, return_hidden=True)
                    halt_prob = F.softmax(halt_logits, dim=-1)[..., 1]
                    w = remaining * halt_prob if i < act_steps - 1 else remaining
                    if weighted_logits is None:
                        weighted_logits = w.unsqueeze(-1) * logits
                    else:
                        weighted_logits = weighted_logits + w.unsqueeze(-1) * logits
                    remaining = (remaining - w).clamp(min=0.0)

        # For each example, get the predicted token at the last non-pad position
        for j, ex in enumerate(batch):
            seq_len = len(token_seqs[j])
            # Logits at last real token position → predicts next token
            last_logits = weighted_logits[j, seq_len - 1, :]  # (vocab_size,)
            pred_id = last_logits.argmax().item()
            pred_text = decode(tokenizer, [pred_id])

            # Also try greedy decode of a few more tokens
            generated_ids = [pred_id]
            cur_inp = inp[j:j+1]  # keep batch dim
            for _ in range(9):  # up to 10 tokens total
                next_inp = torch.cat([
                    cur_inp[:, 1:],
                    torch.tensor([[generated_ids[-1]]], device=device)
                ], dim=1)
                with torch.no_grad():
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        next_logits = model(next_inp)
                next_id = next_logits[0, -1, :].argmax().item()
                if next_id == (tokenizer.token_to_id("<eos>") or 1):
                    break
                generated_ids.append(next_id)
                cur_inp = next_inp

            generated_text = decode(tokenizer, generated_ids)
            correct = check_answer(generated_text, ex["answer"])

            task = ex["task"]
            total_correct += int(correct)
            total_tested += 1
            task_correct[task] = task_correct.get(task, 0) + int(correct)
            task_total[task] = task_total.get(task, 0) + 1

            if task not in task_examples:
                task_examples[task] = []
            if len(task_examples[task]) < 2:
                task_examples[task].append({
                    "passage": ex["passage"][:100],
                    "question": ex["question"],
                    "expected": ex["answer"],
                    "generated": generated_text.strip()[:80],
                    "correct": correct,
                })

            if verbose and total_tested <= 20:
                mark = "✓" if correct else "✗"
                log.info("[%s] Task %d: Q=\"%s\" → \"%s\" (expected: %s)",
                         mark, task, ex['question'], generated_text.strip()[:50], ex['answer'])

        if (batch_start // batch_size + 1) % 5 == 0:
            acc = 100 * total_correct / max(total_tested, 1)
            elapsed = time.time() - t0
            rate = total_tested / max(elapsed, 1e-6)
            log.info("Progress: %d examples, %.1f%% accuracy, %.0f ex/s", total_tested, acc, rate)

    total_time = time.time() - t0
    return {
        "total_correct": total_correct,
        "total_tested": total_tested,
        "total_time": total_time,
        "task_correct": task_correct,
        "task_total": task_total,
        "task_examples": task_examples,
    }


def evaluate_babi(
    orchestrator: Orchestrator,
    device: torch.device,
    max_examples: int = 0,
    tasks: list = None,
    act_steps: int = 4,
    temperature: float = 0.5,
    verbose: bool = False,
    use_prefill: bool = True,
):
    """Run bAbI evaluation with memory recall.
    
    use_prefill=True:  passage+question fed in one forward pass (fast, ~10-30× speedup)
    use_prefill=False: passage+question fed token-by-token (slow, writes memory per token)
    Both modes use token-by-token generation with ACT + memory."""
    log.info("Loading bAbI test set...")
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

        # Step 1+2: Feed passage + question
        t0 = time.time()
        if use_prefill:
            orchestrator.feed_prefill(agent, f"{passage}\n{question}")
        else:
            orchestrator.feed(agent, passage)
            orchestrator.feed(agent, f"\n{question}")

        # Step 3: Generate answer (token-by-token with ACT + memory)
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
            log.info("[%s] Task %d: Q=\"%s\" → \"%s\" (expected: %s)",
                     mark, task, question, answer.strip()[:50], expected)

        if total_tested % 100 == 0:
            acc = 100 * total_correct / total_tested
            rate = total_tested / max(total_time, 1e-6)
            log.info("Progress: %d examples, %.1f%% accuracy, %.0f ex/s", total_tested, acc, rate)

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
    parser.add_argument("--log_dir", default="./logs",
                        help="Directory for log files (default: ./logs)")
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Max examples to evaluate (0 = all)")
    parser.add_argument("--tasks", type=int, nargs="+", default=None,
                        help="Specific bAbI task numbers to evaluate (default: all)")
    parser.add_argument("--act_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--slow", action="store_true",
                        help="Feed passage token-by-token (slower, writes memory per token)")
    args = parser.parse_args()

    # Setup logging — console + file
    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"eval_gen1_{ts}.log")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    log.info("Logging to %s", log_file)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mode_str = "Token-by-token feed" if args.slow else "Prefill feed (fast)"
    log.info("=" * 64)
    log.info("Gen 1 Evaluation: Memory Recall (bAbI)")
    log.info("=" * 64)
    log.info("Device: %s", device)
    log.info("Mode:   %s + token-by-token generation (with memory)", mode_str)
    log.info("Args:   max_examples=%s tasks=%s act_steps=%d temperature=%.2f",
             args.max_examples or "all", args.tasks or "all", args.act_steps, args.temperature)

    # Load model
    model, tokenizer, cfg = load_model_for_eval(
        args.checkpoint_dir, args.data_dir, device)

    # Memory-based evaluation (both modes)
    mcfg = MemoryConfig()
    if os.path.exists(args.memory_dir):
        import shutil
        shutil.rmtree(args.memory_dir)
    os.makedirs(args.memory_dir, exist_ok=True)
    memory = MemorySystem(args.memory_dir, mcfg)
    orchestrator = Orchestrator(model, memory, tokenizer, device)

    results = evaluate_babi(
        orchestrator, device,
        max_examples=args.max_examples,
        tasks=args.tasks,
        act_steps=args.act_steps,
        temperature=args.temperature,
        verbose=args.verbose,
        use_prefill=not args.slow,
    )

    # Log results
    overall_acc = (100 * results["total_correct"] / max(results["total_tested"], 1))
    avg_time = results["total_time"] / max(results["total_tested"], 1)

    log.info("=" * 64)
    log.info("Results")
    log.info("=" * 64)
    log.info("Overall:  %d/%d (%.1f%%)", results['total_correct'], results['total_tested'], overall_acc)
    log.info("Avg time: %.1fms per example", avg_time * 1000)
    log.info("Total:    %.1fs", results['total_time'])

    log.info("Per-task breakdown:")
    for task in sorted(results["task_total"].keys()):
        correct = results["task_correct"].get(task, 0)
        total = results["task_total"][task]
        acc = 100 * correct / max(total, 1)
        log.info("  Task %2d: %4d/%4d (%5.1f%%)", task, correct, total, acc)

    log.info("Example outputs:")
    for task in sorted(results["task_examples"].keys())[:5]:
        for ex in results["task_examples"][task][:1]:
            mark = "✓" if ex["correct"] else "✗"
            log.info("  [%s] Task %d: \"%s\"", mark, task, ex['question'])
            log.info("      Expected:  %s", ex['expected'])
            log.info("      Generated: %s", ex['generated'])

    log.info("=" * 64)
    if overall_acc >= 60:
        log.info("✓ Memory recall accuracy %.1f%% ≥ 60%% — GO for Gen 2", overall_acc)
    elif overall_acc >= 30:
        log.warning("~ Memory recall accuracy %.1f%% — partial success, may need more training", overall_acc)
    else:
        log.warning("✗ Memory recall accuracy %.1f%% < 30%% — memory not being used effectively", overall_acc)
    log.info("=" * 64)

    # Save results as JSON for programmatic access
    results_file = os.path.join(args.log_dir, f"eval_gen1_{ts}.json")
    json_results = {
        "timestamp": ts,
        "device": str(device),
        "mode": "slow" if args.slow else "prefill",
        "max_examples": args.max_examples,
        "act_steps": args.act_steps,
        "temperature": args.temperature,
        "overall_accuracy": round(overall_acc, 2),
        "total_correct": results["total_correct"],
        "total_tested": results["total_tested"],
        "total_time_s": round(results["total_time"], 2),
        "avg_time_ms": round(avg_time * 1000, 2),
        "per_task": {
            str(t): {
                "correct": results["task_correct"].get(t, 0),
                "total": results["task_total"][t],
                "accuracy": round(100 * results["task_correct"].get(t, 0) / max(results["task_total"][t], 1), 2),
            }
            for t in sorted(results["task_total"].keys())
        },
    }
    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)
    log.info("Results saved to %s", results_file)

    # Cleanup eval memory
    import shutil
    shutil.rmtree(args.memory_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
