"""
ANT — Training script with Phase A/B/C curriculum.

Phase A: LM with memory — trie starts empty, grows. Memory from day 1.
Phase B: Freeze base, train AddrNet/V_proj/tag/mem_attn.
Phase C: Unfreeze base (keep AddrNet/V_proj frozen), end-to-end.

All phases use the engine's per-token trie READ→PROCESS→WRITE cycle.
"""

import argparse
import os
import signal
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, MemoryConfig
from model import ANT
from engine import ANTEngine
from data import (
    tokenize, BOS_ID, EOS_ID, PAD_ID, ANS_ID,
    generate_dataset, generate_shell_texts,
    load_wikipedia_sentences, generate_chat_data,
    load_hf_chat_data, tag_text,
    TextLMDataset, lm_collate_fn,
)


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, optimizer, step, phase, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "phase": phase,
    }, path)
    print(f"  Saved checkpoint: {path} (step {step}, phase {phase})")


def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        return 0, "A"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            print("  Warning: optimizer state incompatible, starting fresh")
    return ckpt.get("step", 0), ckpt.get("phase", "A")


# ---------------------------------------------------------------------------
# Phase A: Language Modeling WITH Memory
# ---------------------------------------------------------------------------

def phase_a(engine: ANTEngine, cfg: ModelConfig, device: str,
            steps: int = 5000, lr: float = 3e-4, batch_size: int = 16,
            start_step: int = 0, ckpt_dir: str = "checkpoints/train"):
    """LM training with memory active from step 1.

    Trie starts empty and grows as training data is encoded.
    Model learns language AND memory interaction simultaneously.
    """
    print(f"\n{'='*60}")
    print(f"  Phase A — LM with Memory | steps {start_step}→{steps}")
    print(f"  Trainable: {count_params(engine.model):,} params")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    wiki = load_wikipedia_sentences(5000)
    shell = generate_shell_texts(2000)
    chat = generate_chat_data(1000)
    all_texts = wiki + shell + chat
    print(f"  {len(all_texts)} texts loaded")

    dataset = TextLMDataset(all_texts, max_len=cfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lm_collate_fn, drop_last=True)

    optimizer = torch.optim.AdamW(engine.model.parameters(), lr=lr,
                                  weight_decay=0.01)

    # Load checkpoint
    if start_step > 0:
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_latest.pt")
        loaded_step, loaded_phase = load_checkpoint(
            engine.model, optimizer, ckpt_path, device)
        if loaded_step > 0:
            print(f"  Resumed from step {loaded_step} (phase {loaded_phase})")
            start_step = loaded_step

    engine.model.train()
    step = start_step
    running_loss = 0.0
    log_interval = 50
    save_interval = 500
    t0 = time.time()

    while step < steps:
        for inp, tgt in loader:
            if step >= steps:
                break

            inp, tgt = inp.to(device), tgt.to(device)

            result = engine.encode(inp, temperature=max(0.5, 1.0 - step / steps))

            logits = result["logits"]
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                tgt.reshape(-1),
                ignore_index=PAD_ID,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % log_interval == 0:
                avg = running_loss / log_interval
                elapsed = time.time() - t0
                it_s = log_interval / elapsed
                mem = engine.memory_stats()
                print(f"  A step {step}/{steps} | loss {avg:.4f} | "
                      f"{it_s:.1f} it/s | trie: {mem['total_nodes']} nodes, "
                      f"{mem['total_entries']} records")
                running_loss = 0.0
                t0 = time.time()

            if step % save_interval == 0:
                save_checkpoint(engine.model, optimizer, step, "A",
                                os.path.join(ckpt_dir, "checkpoint_latest.pt"))

    save_checkpoint(engine.model, optimizer, step, "A",
                    os.path.join(ckpt_dir, "checkpoint_phaseA.pt"))
    engine.flush()
    return step


# ---------------------------------------------------------------------------
# Phase B: Memory Training (frozen base)
# ---------------------------------------------------------------------------

def phase_b(engine: ANTEngine, cfg: ModelConfig, device: str,
            steps: int = 3000, lr: float = 1e-3, batch_size: int = 16,
            start_step: int = 0, ckpt_dir: str = "checkpoints/train"):
    """Freeze base, train memory subsystem (AddrNets, V_proj, tags, mem_attn).

    Uses QA data: write passage to trie, then answer questions from memory.
    Contrastive address loss ensures stable address space.
    """
    print(f"\n{'='*60}")
    print(f"  Phase B — Memory Training (frozen base)")
    print(f"{'='*60}\n")

    model = engine.model

    # Freeze base model weights
    for name, param in model.named_parameters():
        if any(k in name for k in ['addr_net', 'v_proj', 'tag_', 'mem_attn',
                                    'norm_tag', 'norm_mem', 'halt_head']):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"  Trainable: {count_params(model):,} params (memory subsystem only)")

    # Generate QA data
    print("Generating QA data...")
    qa_data = generate_dataset(5000, tagged=True)
    print(f"  {len(qa_data)} QA examples")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)

    model.train()
    step = start_step
    running_loss = 0.0
    log_interval = 50
    save_interval = 500
    t0 = time.time()

    while step < steps:
        for qa in qa_data:
            if step >= steps:
                break

            # Step 1: Write passage to trie
            passage_ids = [BOS_ID] + tokenize(qa.passage) + [EOS_ID]
            passage_ids = passage_ids[:cfg.max_seq_len]
            passage_t = torch.tensor([passage_ids], dtype=torch.long, device=device)

            engine.reset_state(1)
            with torch.no_grad():
                engine.encode(passage_t, write_to_trie=True)

            # Step 2: Answer question from memory (no passage in context)
            qa_text = f"{qa.question}{chr(ANS_ID)}{qa.answer}"
            qa_ids = tokenize(qa_text)
            inp_ids = [BOS_ID] + qa_ids
            tgt_ids = qa_ids + [EOS_ID]
            max_len = min(len(inp_ids), cfg.max_seq_len)
            inp_ids = inp_ids[:max_len]
            tgt_ids = tgt_ids[:max_len]

            inp_t = torch.tensor([inp_ids], dtype=torch.long, device=device)
            tgt_t = torch.tensor([tgt_ids], dtype=torch.long, device=device)

            # Find answer start position for focused loss
            ans_marker = qa_text.find(chr(ANS_ID))
            ans_start = ans_marker + 1 if ans_marker >= 0 else 0

            result = engine.encode(inp_t, write_to_trie=False)
            logits = result["logits"]

            # Loss only on answer portion
            loss = F.cross_entropy(
                logits[:, ans_start:, :].reshape(-1, cfg.vocab_size),
                tgt_t[:, ans_start:].reshape(-1),
                ignore_index=PAD_ID,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % log_interval == 0:
                avg = running_loss / log_interval
                elapsed = time.time() - t0
                it_s = log_interval / elapsed
                mem = engine.memory_stats()
                print(f"  B step {step}/{steps} | qa_loss {avg:.4f} | "
                      f"{it_s:.1f} it/s | trie: {mem['total_nodes']} nodes")
                running_loss = 0.0
                t0 = time.time()

            if step % save_interval == 0:
                save_checkpoint(model, optimizer, step, "B",
                                os.path.join(ckpt_dir, "checkpoint_latest.pt"))

    # Unfreeze all params for Phase C
    for param in model.parameters():
        param.requires_grad = True

    save_checkpoint(model, optimizer, step, "B",
                    os.path.join(ckpt_dir, "checkpoint_phaseB.pt"))
    engine.flush()
    return step


# ---------------------------------------------------------------------------
# Phase C: End-to-End (memory always active)
# ---------------------------------------------------------------------------

def phase_c(engine: ANTEngine, cfg: ModelConfig, device: str,
            steps: int = 5000, lr: float = 1e-4, batch_size: int = 8,
            start_step: int = 0, ckpt_dir: str = "checkpoints/train"):
    """Full end-to-end with memory. Freeze AddrNet/V_proj, train everything else.

    All data goes through trie. Model adapts to stable address space.
    Mixed training: LM + QA + chat, all through memory.
    """
    print(f"\n{'='*60}")
    print(f"  Phase C — End-to-End with Memory")
    print(f"{'='*60}\n")

    model = engine.model

    # Freeze address space (AddrNet + V_proj) for stability
    for name, param in model.named_parameters():
        if 'addr_net' in name or 'v_proj' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print(f"  Trainable: {count_params(model):,} params (addr/v_proj frozen)")

    # Load mixed data
    print("Loading mixed data...")
    wiki = load_wikipedia_sentences(3000)
    shell = generate_shell_texts(1000)
    qa_data = generate_dataset(2000, tagged=True)
    chat = generate_chat_data(1000)

    # Mix all text data
    lm_texts = wiki + shell + [c for c in chat]
    qa_passages = [(qa.passage, qa.question, qa.answer) for qa in qa_data]
    print(f"  LM: {len(lm_texts)}, QA: {len(qa_passages)}")

    lm_dataset = TextLMDataset(lm_texts, max_len=cfg.max_seq_len)
    lm_loader = DataLoader(lm_dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=lm_collate_fn, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)

    model.train()
    step = start_step
    running_lm = 0.0
    running_qa = 0.0
    log_interval = 50
    save_interval = 500
    t0 = time.time()
    qa_idx = 0

    while step < steps:
        for inp, tgt in lm_loader:
            if step >= steps:
                break

            inp, tgt = inp.to(device), tgt.to(device)

            # LM loss
            result = engine.encode(inp)
            lm_loss = F.cross_entropy(
                result["logits"].reshape(-1, cfg.vocab_size),
                tgt.reshape(-1),
                ignore_index=PAD_ID,
            )

            # QA loss (interleaved)
            qa_loss = torch.tensor(0.0, device=device)
            if qa_idx < len(qa_passages):
                passage, question, answer = qa_passages[qa_idx % len(qa_passages)]
                qa_idx += 1

                # Write passage
                p_ids = [BOS_ID] + tokenize(passage)[:cfg.max_seq_len - 2] + [EOS_ID]
                p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
                engine.reset_state(1)
                with torch.no_grad():
                    engine.encode(p_t, write_to_trie=True)

                # Answer from memory
                qa_text = f"{question}{chr(ANS_ID)}{answer}"
                q_ids = tokenize(qa_text)
                q_inp = [BOS_ID] + q_ids
                q_tgt = q_ids + [EOS_ID]
                max_len = min(len(q_inp), cfg.max_seq_len)

                q_inp_t = torch.tensor([q_inp[:max_len]], dtype=torch.long, device=device)
                q_tgt_t = torch.tensor([q_tgt[:max_len]], dtype=torch.long, device=device)

                r = engine.encode(q_inp_t, write_to_trie=False)
                ans_start = qa_text.find(chr(ANS_ID)) + 1
                if ans_start > 0 and ans_start < max_len:
                    qa_loss = F.cross_entropy(
                        r["logits"][:, ans_start:, :].reshape(-1, cfg.vocab_size),
                        q_tgt_t[:, ans_start:].reshape(-1),
                        ignore_index=PAD_ID,
                    )

            # Reset batch state for LM
            engine.reset_state(batch_size)

            # Combined loss
            loss = lm_loss + 0.5 * qa_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            step += 1
            running_lm += lm_loss.item()
            running_qa += qa_loss.item()

            if step % log_interval == 0:
                avg_lm = running_lm / log_interval
                avg_qa = running_qa / log_interval
                elapsed = time.time() - t0
                it_s = log_interval / elapsed
                mem = engine.memory_stats()
                print(f"  C step {step}/{steps} | lm {avg_lm:.4f} qa {avg_qa:.4f} | "
                      f"{it_s:.1f} it/s | trie: {mem['total_nodes']} nodes")
                running_lm = 0.0
                running_qa = 0.0
                t0 = time.time()

            if step % save_interval == 0:
                save_checkpoint(model, optimizer, step, "C",
                                os.path.join(ckpt_dir, "checkpoint_latest.pt"))

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    save_checkpoint(model, optimizer, step, "C",
                    os.path.join(ckpt_dir, "checkpoint_phaseC.pt"))
    engine.flush()
    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ANT Training")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps_a", type=int, default=5000)
    parser.add_argument("--steps_b", type=int, default=3000)
    parser.add_argument("--steps_c", type=int, default=5000)
    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_b", type=float, default=1e-3)
    parser.add_argument("--lr_c", type=float, default=1e-4)
    parser.add_argument("--batch_a", type=int, default=16)
    parser.add_argument("--batch_b", type=int, default=16)
    parser.add_argument("--batch_c", type=int, default=8)
    parser.add_argument("--ckpt_dir", default="checkpoints/train")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--skip_to", choices=["A", "B", "C"], default="A",
                        help="Skip to a specific phase")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    cfg = ModelConfig()
    mem_cfg = MemoryConfig()

    model = ANT(cfg).to(args.device)
    total = count_params(model, trainable_only=False)
    print(f"ANT model: {total:,} parameters")

    engine = ANTEngine(model, mem_cfg, device=args.device)

    # Graceful shutdown
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n  Force exit!")
            sys.exit(1)
        shutdown_requested = True
        print(f"\n  Graceful shutdown requested (signal {signum})...")
        print("  Saving checkpoint and flushing trie...")
        save_checkpoint(model, None, -1, "interrupted",
                        os.path.join(args.ckpt_dir, "checkpoint_latest.pt"))
        engine.flush()
        print("  Done. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Resume logic
    start_phase = args.skip_to
    start_step = 0
    if args.resume:
        ckpt_path = os.path.join(args.ckpt_dir, "checkpoint_latest.pt")
        if os.path.exists(ckpt_path):
            start_step, loaded_phase = load_checkpoint(
                model, None, ckpt_path, args.device)
            start_phase = loaded_phase
            print(f"Resumed from phase {loaded_phase}, step {start_step}")

    # Execute phases
    if start_phase <= "A" and args.steps_a > 0:
        step_a = start_step if start_phase == "A" else 0
        phase_a(engine, cfg, args.device, steps=args.steps_a,
                lr=args.lr_a, batch_size=args.batch_a,
                start_step=step_a, ckpt_dir=args.ckpt_dir)

    if start_phase <= "B" and args.steps_b > 0:
        step_b = start_step if start_phase == "B" else 0
        phase_b(engine, cfg, args.device, steps=args.steps_b,
                lr=args.lr_b, batch_size=args.batch_b,
                start_step=step_b, ckpt_dir=args.ckpt_dir)

    if start_phase <= "C" and args.steps_c > 0:
        step_c = start_step if start_phase == "C" else 0
        phase_c(engine, cfg, args.device, steps=args.steps_c,
                lr=args.lr_c, batch_size=args.batch_c,
                start_step=step_c, ckpt_dir=args.ckpt_dir)

    print("\n  Training complete!")
    engine.flush()


if __name__ == "__main__":
    main()
