#!/usr/bin/env python3
"""
ANT — Training pipeline.

Three-phase curriculum with trie memory wired into every forward pass:
  Phase A: Base LM (no memory) — learn language, embeddings, attention
  Phase B: Memory training (frozen base) — train AddrNet, V_proj, tags
  Phase C: End-to-end (all through memory) — full system

Usage:
    python train.py                         # default training
    python train.py --steps_a 3000          # shorter Phase A
    python train.py --device cuda --bf16    # GPU with mixed precision
"""

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import ModelConfig, MemoryConfig
from model import ANT
from memory import MemorySystem
from data import (
    VOCAB, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, ANS_ID,
    tokenize, detokenize,
    generate_dataset, generate_shell_texts, load_wikipedia_sentences,
    load_hf_chat_data, generate_chat_data, tag_passage,
    TextLMDataset, ChatMemoryDataset,
    lm_collate_fn, chat_memory_collate_fn,
)


# ============================================================================
# Training Utilities
# ============================================================================

class Tracker:
    """Running statistics for training logs."""
    def __init__(self, window: int = 50):
        self.window = window
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.timestamps: list[float] = []
        self.evals: list[tuple] = []

    def tick(self):
        self.timestamps.append(time.time())

    def add_loss(self, loss: float):
        self.losses.append(loss)

    def add_grad_norm(self, norm: float):
        self.grad_norms.append(norm)

    @property
    def avg_loss(self) -> float:
        recent = self.losses[-self.window:]
        return sum(recent) / max(len(recent), 1)

    @property
    def steps_per_sec(self) -> float:
        recent = self.timestamps[-self.window:]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def log_model_summary(model, cfg):
    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_p = sum(p.numel() for p in model.embed.parameters())
    layer_p = sum(p.numel() for p in model.layers.parameters())
    addr_p = sum(p.numel() for p in model.addr_nets.parameters())
    vproj_p = sum(p.numel() for p in model.v_proj.parameters())

    print(f"\n  ┌─── Model Architecture ───")
    print(f"  │ d_model={cfg.d_model}  n_heads={cfg.n_heads}  head_dim={cfg.head_dim}")
    print(f"  │ n_layers={cfg.n_layers}  ffn_dim={cfg.ffn_dim}  max_seq={cfg.max_seq_len}")
    print(f"  │ n_addr_nets={cfg.n_addr_nets}  addr_depth={cfg.addr_depth}")
    print(f"  │ params: {total:,} total ({total/1e6:.2f}M), {trainable:,} trainable")
    print(f"  │ breakdown: embed={embed_p:,}  layers={layer_p:,}  "
          f"addr_nets={addr_p:,}  v_proj={vproj_p:,}")
    print(f"  └────────────────────")


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return min_lr + (max_lr - min_lr) * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ============================================================================
# Sliding Window Encoder
# ============================================================================

def sliding_window_encode(model, input_ids, window_size=8, num_passes=4,
                          mem_keys=None, mem_vals=None, mem_mask=None,
                          causal=True, stride=1):
    """Multi-pass causal sliding window encoder with memory cross-attention.

    Each pass pads, unfolds into overlapping windows, processes through all
    transformer layers (self-attn + tag-attn + mem-attn + FFN), and extracts
    center positions. Multi-pass grows effective receptive field.

    Combined with trie memory cross-attention: local context (sliding window)
    + global knowledge (trie, unlimited).

    Returns: (B, T, d_model) hidden states (post-norm).
    """
    B, T = input_ids.shape
    device = input_ids.device
    half_w = window_size // 2
    pad_right = window_size - half_w - 1
    d_model = model.cfg.d_model

    pad_id = torch.tensor([PAD_ID], device=device)
    pad_emb = model.embed(pad_id).squeeze(0)
    hidden = model.embed(input_ids)

    cos = model.rope_cos[:window_size]
    sin = model.rope_sin[:window_size]

    if causal:
        mask = torch.triu(
            torch.full((window_size, window_size), float("-inf"), device=device),
            diagonal=1)
    else:
        mask = torch.zeros(window_size, window_size, device=device)

    for _ in range(num_passes):
        padded = torch.cat([
            pad_emb.expand(B, half_w, d_model),
            hidden,
            pad_emb.expand(B, pad_right, d_model),
        ], dim=1)

        windows = padded.unfold(1, window_size, stride)
        n_win = windows.shape[1]
        windows = windows.permute(0, 1, 3, 2).contiguous()
        x = windows.reshape(B * n_win, window_size, d_model)

        # Expand memory for all windows
        w_mk, w_mv, w_mm = None, None, None
        if mem_keys is not None:
            w_mk = mem_keys.unsqueeze(1).expand(-1, n_win, -1, -1).reshape(B * n_win, -1, d_model)
            w_mv = mem_vals.unsqueeze(1).expand(-1, n_win, -1, -1).reshape(B * n_win, -1, d_model)
            if mem_mask is not None:
                w_mm = mem_mask.unsqueeze(1).expand(-1, n_win, -1).reshape(B * n_win, -1)

        for layer in model.layers:
            x, _ = layer(x, mask, cos, sin,
                         mem_keys=w_mk, mem_values=w_mv, mem_mask=w_mm)

        centers = x[:, half_w, :].reshape(B, n_win, d_model)
        if stride == 1:
            hidden = centers
        else:
            indices = torch.arange(n_win, device=device) * stride
            indices = indices.clamp(max=T - 1)
            hidden[:, indices] = centers

    return model.norm(hidden)


# ============================================================================
# Trie Memory Bridge — connects torch model to numpy trie
# ============================================================================

def trie_write(model, memory: MemorySystem, hidden_states: torch.Tensor,
               temperature: float = 1.0):
    """Write hidden states to trie via AddrNets + V_proj.

    hidden_states: (B, T, d_model) — positions to write.
    Writes the LAST token of each batch element (pooled representation).
    """
    B = hidden_states.shape[0]

    # Pool: use last non-pad position per batch element
    pooled = hidden_states[:, -1, :]  # (B, d_model) — last token

    # Compute addresses and values
    addresses = model.compute_addresses(pooled, temperature)  # list of N tensors (B, depth)
    values = model.compute_value(pooled)  # (B, d_model)

    # Convert to numpy for trie
    values_np = values.detach().cpu().float().numpy()
    batch_addrs = []
    for b in range(B):
        addrs_b = [addr[b].detach().cpu().numpy().astype(np.int64) for addr in addresses]
        batch_addrs.append(addrs_b)

    memory.write_batch(batch_addrs, values_np)


def trie_write_all_tokens(model, memory: MemorySystem,
                          hidden_states: torch.Tensor,
                          token_ids: torch.Tensor,
                          temperature: float = 1.0):
    """Write every non-pad token position to trie.

    hidden_states: (B, T, d_model)
    token_ids: (B, T) — used to find non-pad positions
    """
    B, T, D = hidden_states.shape

    for b in range(B):
        valid_mask = token_ids[b] != PAD_ID
        valid_positions = valid_mask.nonzero(as_tuple=True)[0]

        for pos in valid_positions:
            h = hidden_states[b, pos, :]  # (d_model,)
            addresses = model.compute_addresses(h.unsqueeze(0), temperature)
            value = model.compute_value(h.unsqueeze(0))  # (1, d_model)

            addrs_np = [addr[0].detach().cpu().numpy().astype(np.int64) for addr in addresses]
            val_np = value[0].detach().cpu().float().numpy()
            memory.write(addrs_np, val_np)


def trie_read(model, memory: MemorySystem, hidden_states: torch.Tensor,
              device, max_vectors: int = 25,
              temperature: float = 1.0):
    """Read from trie using hidden states to generate addresses.

    hidden_states: (B, d_model) — query vectors (e.g., pooled question).
    Returns: (mem_keys, mem_vals, mem_mask) as torch tensors on device.
    """
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[:, -1, :]  # pool: last token

    B = hidden_states.shape[0]
    addresses = model.compute_addresses(hidden_states, temperature)

    batch_addrs = []
    for b in range(B):
        addrs_b = [addr[b].detach().cpu().numpy().astype(np.int64) for addr in addresses]
        batch_addrs.append(addrs_b)

    vecs_np, mask_np = memory.read_batch(batch_addrs, max_vectors)

    mem_keys = torch.from_numpy(vecs_np).to(device=device, dtype=torch.float32)
    mem_vals = mem_keys.clone()
    mem_mask = torch.from_numpy(mask_np).to(device=device)

    return mem_keys, mem_vals, mem_mask


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_qa(model, memory: MemorySystem, val_qa, device,
                window_size=8, num_passes=4, max_examples=200):
    """Evaluate QA accuracy using trie memory.

    Flow per example:
      1. Passage → sliding window → hidden → trie WRITE
      2. Question embedding → AddrNet → trie READ → memory vectors
      3. Question → sliding window + memory cross-attention → predict answer
    """
    model.eval()
    correct, total = 0, 0
    type_correct, type_total = Counter(), Counter()

    for ex in val_qa[:max_examples]:
        p_ids = tokenize(ex.passage)
        q_ids = tokenize(ex.question)
        a_ids = tokenize(ex.answer)

        # 1. Encode passage → write to trie
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
        p_hidden = sliding_window_encode(model, p_tensor, window_size, num_passes,
                                         causal=True)
        trie_write_all_tokens(model, memory, p_hidden, p_tensor, temperature=0.01)

        # 2. Get question representation → read from trie
        q_input = [BOS_ID, ANS_ID] + q_ids + a_ids
        q_tensor = torch.tensor([q_input], dtype=torch.long, device=device)
        q_embed = model.embed(q_tensor)
        q_pooled = q_embed[:, :2 + len(q_ids), :].mean(dim=1)  # pool question part
        mem_k, mem_v, mem_m = trie_read(model, memory, q_pooled, device)

        # 3. Process question with memory → check answer
        hidden = sliding_window_encode(model, q_tensor, window_size, num_passes,
                                       mem_keys=mem_k, mem_vals=mem_v,
                                       mem_mask=mem_m, causal=True)
        logits = F.linear(hidden, model.embed.weight)

        n_context = 2 + len(q_ids)
        is_correct = True
        for j, expected in enumerate(a_ids):
            pred_pos = n_context - 1 + j
            if pred_pos < logits.shape[1] and logits[0, pred_pos].argmax().item() != expected:
                is_correct = False
                break

        if not is_correct and ex.valid_answers:
            pred_bytes = [logits[0, n_context - 1 + j].argmax().item()
                          for j in range(len(a_ids)) if n_context - 1 + j < logits.shape[1]]
            for va in ex.valid_answers:
                if tokenize(va) == pred_bytes:
                    is_correct = True
                    break

        if is_correct:
            correct += 1
        total += 1
        n_facts = len(ex.facts)
        type_total[n_facts] += 1
        if is_correct:
            type_correct[n_facts] += 1

    accuracy = correct / max(total, 1)
    for n in sorted(type_total):
        c, t = type_correct[n], type_total[n]
        print(f"    {n}-fact: {c}/{t} = {c/max(t,1):.1%}")

    model.train()
    return accuracy


@torch.no_grad()
def evaluate_qa_direct(model, val_qa, device,
                       window_size=8, num_passes=4, max_examples=200,
                       encode_fn=None):
    """Evaluate QA without trie (for Phase A or direct memory encoding).

    If encode_fn provided: passage → encoder → memory vectors → cross-attention.
    Otherwise: full sequence in sliding window (passage + question).
    """
    model.eval()
    correct, total = 0, 0

    for ex in val_qa[:max_examples]:
        p_ids = tokenize(ex.passage)
        q_ids = tokenize(ex.question)
        a_ids = tokenize(ex.answer)

        mk, mv, mm = None, None, None
        if encode_fn is not None:
            p_t = torch.tensor([p_ids], dtype=torch.long, device=device)
            mk, mv, mm = encode_fn(model, p_t, device)
            input_ids = [BOS_ID, ANS_ID] + q_ids + a_ids
            n_ctx = 2 + len(q_ids)
        else:
            input_ids = [BOS_ID] + p_ids + [ANS_ID] + q_ids + a_ids
            n_ctx = 1 + len(p_ids) + 1 + len(q_ids)

        inp = torch.tensor([input_ids], dtype=torch.long, device=device)
        hidden = sliding_window_encode(model, inp, window_size, num_passes,
                                       mem_keys=mk, mem_vals=mv, mem_mask=mm,
                                       causal=True)
        logits = F.linear(hidden, model.embed.weight)

        ok = all(logits[0, n_ctx - 1 + j].argmax().item() == a_ids[j]
                 for j in range(len(a_ids)) if n_ctx - 1 + j < logits.shape[1])

        if not ok and ex.valid_answers:
            pred = [logits[0, n_ctx - 1 + j].argmax().item()
                    for j in range(len(a_ids)) if n_ctx - 1 + j < logits.shape[1]]
            ok = any(tokenize(va) == pred for va in ex.valid_answers)

        if ok:
            correct += 1
        total += 1

    model.train()
    return correct / max(total, 1)


# ============================================================================
# Text Generation
# ============================================================================

@torch.no_grad()
def generate(model, prompt_ids, max_new=512, temperature=0.8,
             top_k=40, top_p=0.9, repetition_penalty=1.0,
             window_size=8, num_passes=4,
             mem_keys=None, mem_vals=None, mem_mask=None,
             stop_token=None, stop_strings=None, callback=None):
    """Autoregressive generation with causal sliding window + memory."""
    if stop_token is None:
        stop_token = EOS_ID
    model.eval()
    device = next(model.parameters()).device

    max_ctx = max(window_size * num_passes * 4, 128)
    full = list(prompt_ids)
    ctx = list(prompt_ids)
    gen_bytes = bytearray()

    for _ in range(max_new):
        if len(ctx) > max_ctx:
            ctx = ctx[-max_ctx:]

        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        hidden = sliding_window_encode(model, inp, window_size, num_passes,
                                       mem_keys=mem_keys, mem_vals=mem_vals,
                                       mem_mask=mem_mask, causal=True)
        logits = F.linear(hidden[:, -1:, :], model.embed.weight).squeeze(0).squeeze(0)

        if repetition_penalty != 1.0:
            for tok in set(full[-50:]):
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty

        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            topk_vals, _ = logits.topk(min(top_k, logits.size(-1)))
            logits[logits < topk_vals[-1]] = float("-inf")

        if top_p < 1.0:
            sorted_l, sorted_i = torch.sort(logits, descending=True)
            cum = torch.softmax(sorted_l, dim=-1).cumsum(dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_l[remove] = float("-inf")
            logits.scatter_(0, sorted_i, sorted_l)

        if temperature == 0:
            next_tok = logits.argmax().item()
        else:
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        if next_tok == stop_token:
            break

        full.append(next_tok)
        ctx.append(next_tok)

        if stop_strings:
            gen_bytes.append(next_tok)
            tail = bytes(gen_bytes[-64:])
            for s in stop_strings:
                enc = s.encode() if isinstance(s, str) else s
                if tail.endswith(enc):
                    full = full[:-len(enc)]
                    return full

        if callback:
            callback(next_tok)

    return full


# ============================================================================
# Passage Encoder (direct tensor path — no trie, for Phase A/B training)
# ============================================================================

def encode_passage(model, passages, device, differentiable=False):
    """Encode passages into (keys, values, mask) for cross-attention.

    Uses sliding window to get hidden states, pools per-sentence (split at
    period tokens). Each sentence → one memory slot.

    When differentiable=True, values are routed through V_proj so gradients
    flow to the value projection during Phase B training.

    Returns: (keys, values, mask) as tensors.
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    window = model.cfg.max_seq_len

    ctx = nullcontext() if differentiable else torch.no_grad()
    with ctx:
        hidden = sliding_window_encode(model, passages, window_size=window,
                                       num_passes=2, causal=True)

    period_id = VOCAB["."]
    example_sents = []
    for b in range(B):
        tokens = passages[b].tolist()
        clen = sum(1 for t in tokens if t != PAD_ID)
        sents, start = [], 0
        for i in range(clen):
            if tokens[i] == period_id:
                sents.append((start, i + 1))
                start = i + 1
        if start < clen:
            sents.append((start, clen))
        example_sents.append(sents if sents else [(0, max(clen, 1))])

    max_s = max(len(s) for s in example_sents)

    # Build keys/values without in-place ops (preserves autograd graph)
    key_list, val_list, mask_list = [], [], []
    for b in range(B):
        bk, bv, bm = [], [], []
        for si in range(max_s):
            if si < len(example_sents[b]):
                s, e = example_sents[b][si]
                h = hidden[b, e - 1]
                bk.append(h)
                bv.append(model.compute_value(h) if differentiable else h)
                bm.append(True)
            else:
                bk.append(torch.zeros(d_model, device=device))
                bv.append(torch.zeros(d_model, device=device))
                bm.append(False)
        key_list.append(torch.stack(bk))
        val_list.append(torch.stack(bv))
        mask_list.append(bm)

    keys = torch.stack(key_list)      # (B, max_s, d_model)
    values = torch.stack(val_list)    # (B, max_s, d_model)
    mask = torch.tensor(mask_list, dtype=torch.bool, device=device)  # (B, max_s)

    return keys, values, mask


def encode_frozen(model, passages, device, **kw):
    return encode_passage(model, passages, device, differentiable=False)


def encode_differentiable(model, passages, device, **kw):
    return encode_passage(model, passages, device, differentiable=True)


# ============================================================================
# Training Pipeline
# ============================================================================

def train(model, cfg, device, output_dir,
          steps_a=3000, steps_b=2000, steps_c=5000,
          lr=3e-4, batch_size=32, grad_accum=1, eval_interval=250,
          window_size=8, num_passes=4, stride=1,
          n_wiki=50000, n_shell=5000, n_qa=20000, n_chat=5000,
          use_bf16=False, use_compile=False, use_hf_chat=False,
          hf_repo=None, hf_upload_interval=1000):
    """Three-phase training with trie memory integration.

    Phase A: Base LM (no memory) — all weights trainable
    Phase B: Memory training (frozen base) — AddrNet + V_proj + tags only
    Phase C: End-to-end (memory always active) — base unfrozen, AddrNet/V_proj frozen
    """
    from torch.amp import GradScaler, autocast

    total_steps = steps_a + steps_b + steps_c
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    amp_device = "cuda" if device == "cuda" else device
    scaler = GradScaler(amp_device) if use_bf16 and device == "cuda" else None

    print("\n" + "=" * 60)
    print("  ANT — Three-Phase Curriculum Training")
    print("=" * 60)
    print(f"  Window: {window_size}  Passes: {num_passes}  Stride: {stride}")
    print(f"  Batch:  {batch_size} × {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Phase A (LM):     {steps_a} steps  — base LM, no memory")
    print(f"  Phase B (memory): {steps_b} steps  — frozen base, train AddrNet/V_proj")
    print(f"  Phase C (e2e):    {steps_c} steps  — full system with trie")
    print(f"  Total:            {total_steps} steps")
    print(f"  Output:           {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──
    print("\n  Loading data...")
    wiki = load_wikipedia_sentences(n_wiki, seed=42)
    print(f"    Wikipedia: {len(wiki)} sentences")
    shell = generate_shell_texts(n_shell, seed=42)
    print(f"    Shell:     {len(shell)} commands")
    chat = load_hf_chat_data(n_chat, seed=42) if use_hf_chat else generate_chat_data(n_chat, seed=42)
    print(f"    Chat:      {len(chat)} pairs")

    train_qa = generate_dataset(n_qa, seed=42, tagged=False)
    val_qa = generate_dataset(500, seed=43, tagged=False)
    print(f"    QA train:  {len(train_qa)}  QA val: {len(val_qa)}")

    # Pre-tokenize QA
    pad, bos, eos, ans = PAD_ID, BOS_ID, EOS_ID, ANS_ID
    max_passage_len = 128
    qa_data = []
    for ex in train_qa:
        p_ids = tokenize(ex.passage)[:max_passage_len]
        p_ids += [pad] * (max_passage_len - len(p_ids))
        q_ids = tokenize(ex.question)
        a_ids = tokenize(ex.answer)
        inp = [bos, ans] + q_ids + a_ids + [eos]
        n_ctx = 2 + len(q_ids)
        tgt = [pad] * (n_ctx - 1) + a_ids + [eos]
        qa_data.append((inp[:-1], tgt, p_ids))

    lm_texts = wiki + shell
    random.shuffle(lm_texts)
    lm_ds = TextLMDataset(lm_texts, max_len=cfg.max_seq_len)
    chat_ds = ChatMemoryDataset(chat)
    print(f"    LM dataset:   {len(lm_ds)} samples")
    print(f"    Chat dataset: {len(chat_ds)} samples")

    # ── Persistent Memory ──
    mem_cfg = MemoryConfig()
    mem_cfg.data_path = os.path.join(output_dir, "memory")
    memory = MemorySystem(mem_cfg)
    print(f"    Trie memory:  {memory.total_entries()} entries, "
          f"{memory.total_nodes()} nodes")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    if use_compile and device == "cuda":
        print("  Compiling model...")
        model = torch.compile(model, mode="max-autotune")

    model.train()

    # ── Data loaders ──
    lm_loader = DataLoader(lm_ds, batch_size=batch_size, shuffle=True,
                           drop_last=True, collate_fn=lm_collate_fn)
    lm_iter = iter(lm_loader)
    chat_loader = None
    chat_iter = None
    if len(chat_ds) >= batch_size:
        chat_loader = DataLoader(chat_ds, batch_size=batch_size, shuffle=True,
                                 drop_last=True, collate_fn=chat_memory_collate_fn)
        chat_iter = iter(chat_loader)

    # ── Training state ──
    tracker = Tracker(window=50)
    best_qa = 0.0
    best_step = 0
    t_start = time.time()

    def get_phase(step):
        if step <= steps_a:
            return "A"
        elif step <= steps_a + steps_b:
            return "B"
        return "C"

    def phase_lr(step):
        p = get_phase(step)
        if p == "A":
            return get_lr(step, 200, steps_a, lr, lr * 0.01)
        elif p == "B":
            s = step - steps_a
            return get_lr(s, 100, steps_b, lr * 0.5, lr * 0.01)
        else:
            s = step - steps_a - steps_b
            return get_lr(s, 100, steps_c, lr * 0.33, lr * 0.01)

    def freeze_base(model):
        """Freeze base (embed, self-attn, ffn) but keep memory system trainable.

        Trainable: AddrNets, V_proj, tag system, mem_attn, norm_mem.
        """
        for p in model.parameters():
            p.requires_grad = False
        for net in model.addr_nets:
            for p in net.parameters():
                p.requires_grad = True
        for p in model.v_proj.parameters():
            p.requires_grad = True
        for layer in model.layers:
            if hasattr(layer, "tag_head"):
                for p in layer.tag_head.parameters():
                    p.requires_grad = True
            if hasattr(layer, "tag_gate"):
                for p in layer.tag_gate.parameters():
                    p.requires_grad = True
            if hasattr(layer, "norm_tag"):
                for p in layer.norm_tag.parameters():
                    p.requires_grad = True
            if hasattr(layer, "mem_attn"):
                for p in layer.mem_attn.parameters():
                    p.requires_grad = True
            if hasattr(layer, "norm_mem"):
                for p in layer.norm_mem.parameters():
                    p.requires_grad = True

    def freeze_addr_vproj(model):
        """Freeze AddrNet + V_proj (stable address space), unfreeze rest."""
        for p in model.parameters():
            p.requires_grad = True
        for net in model.addr_nets:
            for p in net.parameters():
                p.requires_grad = False
        for p in model.v_proj.parameters():
            p.requires_grad = False

    prev_phase = None

    print(f"\n  {'─' * 56}")
    print(f"  Training started at {time.strftime('%H:%M:%S')}")
    print(f"  {'─' * 56}")

    for step in range(1, total_steps + 1):
        tracker.tick()
        phase = get_phase(step)

        # Phase transitions
        if phase != prev_phase:
            if phase == "B":
                print(f"\n  ══ Phase A → B: Freezing base, training memory ══")
                freeze_base(model)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Trainable params: {trainable:,}")
            elif phase == "C":
                print(f"\n  ══ Phase B → C: Full end-to-end with trie ══")
                freeze_addr_vproj(model)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Trainable params: {trainable:,}")
            prev_phase = phase

        lr_now = phase_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        loss_parts = {}

        for _ in range(grad_accum):
            losses = []  # collect differentiable losses to sum
            amp_ctx = autocast(amp_device, dtype=amp_dtype) if use_bf16 else nullcontext()

            # ── LM loss (all phases) ──
            try:
                lm_inp, lm_tgt = next(lm_iter)
            except StopIteration:
                lm_iter = iter(lm_loader)
                lm_inp, lm_tgt = next(lm_iter)
            lm_inp, lm_tgt = lm_inp.to(device), lm_tgt.to(device)

            with amp_ctx:
                lm_hidden = sliding_window_encode(
                    model, lm_inp, window_size, num_passes,
                    causal=True, stride=stride)
                lm_logits = F.linear(lm_hidden, model.embed.weight)
                lm_loss = F.cross_entropy(
                    lm_logits.reshape(-1, VOCAB_SIZE),
                    lm_tgt.reshape(-1), ignore_index=pad)

            loss_parts.setdefault("lm", lm_loss.item())

            if phase == "A":
                losses.append(lm_loss)
            elif phase == "B":
                pass  # LM loss has no grad when base is frozen; monitor only
            else:
                losses.append(0.34 * lm_loss)

            # ── QA loss with memory (Phase B+) ──
            if phase in ("B", "C"):
                batch_idx = random.sample(range(len(qa_data)), min(batch_size, len(qa_data)))
                batch = [qa_data[i] for i in batch_idx]
                max_slen = max(len(b[0]) for b in batch)
                inp_b = [b[0] + [pad] * (max_slen - len(b[0])) for b in batch]
                tgt_b = [b[1] + [pad] * (max_slen - len(b[1])) for b in batch]
                p_b = [b[2] for b in batch]

                q_t = torch.tensor(inp_b, dtype=torch.long, device=device)
                tgt_t = torch.tensor(tgt_b, dtype=torch.long, device=device)
                p_t = torch.tensor(p_b, dtype=torch.long, device=device)

                with amp_ctx:
                    if phase == "C":
                        # Phase C: write passage to trie, read from trie for QA
                        with torch.no_grad():
                            p_hidden = sliding_window_encode(
                                model, p_t, window_size, num_passes, causal=True)
                            trie_write_all_tokens(model, memory, p_hidden, p_t,
                                                  temperature=0.01)

                        # Read from trie for question
                        q_embed = model.embed(q_t)
                        q_pooled = q_embed.mean(dim=1)
                        mk, mv, mm = trie_read(model, memory, q_pooled, device)

                        qa_hidden = sliding_window_encode(
                            model, q_t, window_size, num_passes,
                            mem_keys=mk, mem_vals=mv, mem_mask=mm,
                            causal=True, stride=stride)
                    else:
                        # Phase B: differentiable encoding so gradients flow
                        # through V_proj (values) and mem_attn (cross-attention)
                        mk, mv, mm = encode_differentiable(model, p_t, device)

                        qa_hidden = sliding_window_encode(
                            model, q_t, window_size, num_passes,
                            mem_keys=mk, mem_vals=mv, mem_mask=mm,
                            causal=True, stride=stride)

                    qa_logits = F.linear(qa_hidden, model.embed.weight)
                    qa_loss = F.cross_entropy(
                        qa_logits.reshape(-1, VOCAB_SIZE),
                        tgt_t.reshape(-1), ignore_index=pad)

                loss_parts.setdefault("qa", qa_loss.item())

                if phase == "B":
                    losses.append(qa_loss)  # sole gradient source in Phase B
                else:
                    losses.append(0.33 * qa_loss)

            # ── Chat loss with memory (Phase C only) ──
            if phase == "C" and chat_iter is not None:
                try:
                    ch_user, ch_inp, ch_tgt = next(chat_iter)
                except StopIteration:
                    chat_iter = iter(chat_loader)
                    ch_user, ch_inp, ch_tgt = next(chat_iter)
                ch_user = ch_user.to(device)
                ch_inp = ch_inp.to(device)
                ch_tgt = ch_tgt.to(device)

                with amp_ctx:
                    # Write user message to trie
                    with torch.no_grad():
                        u_hidden = sliding_window_encode(
                            model, ch_user, window_size, num_passes, causal=True)
                        trie_write(model, memory, u_hidden, temperature=0.01)

                    # Read from trie for agent response
                    u_pooled = u_hidden[:, -1, :]
                    mk, mv, mm = trie_read(model, memory, u_pooled, device)

                    ch_hidden = sliding_window_encode(
                        model, ch_inp, window_size, num_passes,
                        mem_keys=mk, mem_vals=mv, mem_mask=mm,
                        causal=True, stride=stride)
                    ch_logits = F.linear(ch_hidden, model.embed.weight)
                    ch_loss = F.cross_entropy(
                        ch_logits.reshape(-1, VOCAB_SIZE),
                        ch_tgt.reshape(-1), ignore_index=pad)

                loss_parts.setdefault("chat", ch_loss.item())
                losses.append(0.33 * ch_loss)

            # Backward
            if not losses:
                continue  # safety: skip if no differentiable loss
            total_loss = sum(losses)
            scaled = total_loss / grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            accum_loss += total_loss.item() / grad_accum

        # ── Optimizer step ──
        if scaler is not None:
            scaler.unscale_(optimizer)
        gn = grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tracker.add_loss(accum_loss)
        tracker.add_grad_norm(gn)

        # ── Logging ──
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t_start
            eta = elapsed / step * (total_steps - step)
            parts = " ".join(f"{k}={v:.4f}" for k, v in loss_parts.items())
            print(f"  [P{phase} {step:>5}/{total_steps}] "
                  f"loss={accum_loss:.4f} {parts} "
                  f"gnorm={gn:.2f} lr={lr_now:.1e} "
                  f"spd={tracker.steps_per_sec:.1f}it/s "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # ── Evaluation ──
        if step % eval_interval == 0 or step == total_steps:
            model.eval()

            # LM sample
            prompt = [BOS_ID] + tokenize("The ")
            gen = generate(model, prompt, max_new=80, temperature=0.7,
                           top_k=30, window_size=window_size,
                           num_passes=num_passes)
            sample = detokenize(gen)[:80]

            # QA accuracy
            qa_acc = 0.0
            if phase in ("B", "C"):
                if phase == "C" and memory.total_entries() > 0:
                    qa_acc = evaluate_qa(model, memory, val_qa[:100], device,
                                        window_size, num_passes)
                else:
                    qa_acc = evaluate_qa_direct(model, val_qa[:100], device,
                                               window_size, num_passes,
                                               encode_fn=encode_frozen)

                if qa_acc > best_qa:
                    best_qa = qa_acc
                    best_step = step

            print(f"\n  ╔══ EVAL @ step {step} (Phase {phase}) ══")
            if phase != "A":
                print(f"  ║ QA: {qa_acc:.1%}  (best: {best_qa:.1%})")
            for k, v in loss_parts.items():
                print(f"  ║ {k} loss: {v:.4f}")
            print(f"  ║ Sample: {sample}")
            print(f"  ║ Trie: {memory.total_entries()} entries, {memory.total_nodes()} nodes")
            print(f"  ╚{'═' * 40}\n")

            model.train()

            # Save checkpoint
            ckpt = {
                "model": model.state_dict(),
                "step": step, "phase": phase,
                "qa_accuracy": qa_acc, "best_qa": best_qa,
                "vocab": VOCAB,
                "config": {k: getattr(cfg, k) for k in [
                    "d_model", "n_heads", "head_dim", "ffn_dim", "n_layers",
                    "max_seq_len", "n_addr_nets", "addr_depth",
                ]},
            }
            save_path = os.path.join(output_dir, "checkpoint_latest.pt")
            torch.save(ckpt, save_path)
            if qa_acc >= best_qa and phase != "A":
                torch.save(ckpt, os.path.join(output_dir, "checkpoint_best.pt"))
                print(f"  ✓ Best checkpoint (QA: {qa_acc:.1%})")

            # Flush trie
            if memory.total_entries() > 0:
                memory.flush()

            # HF upload
            if hf_repo and (step % hf_upload_interval == 0 or step == total_steps):
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_file(
                        path_or_fileobj=save_path,
                        path_in_repo="checkpoints/latest.pt",
                        repo_id=hf_repo, repo_type="model")
                    print(f"    ↑ Uploaded to {hf_repo}")
                except Exception as e:
                    print(f"    ⚠ HF upload failed: {e}")

    elapsed = time.time() - t_start
    print(f"\n  {'═' * 56}")
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best QA: {best_qa:.1%} at step {best_step}")
    print(f"  Trie: {memory.total_entries()} entries, {memory.total_nodes()} nodes")
    print(f"  {'═' * 56}")

    memory.flush()
    return best_qa


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ANT — three-phase training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default="./checkpoints/train")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--n_wiki", type=int, default=50000)
    parser.add_argument("--n_shell", type=int, default=5000)
    parser.add_argument("--n_qa", type=int, default=20000)
    parser.add_argument("--n_chat", type=int, default=5000)
    parser.add_argument("--steps_a", type=int, default=3000)
    parser.add_argument("--steps_b", type=int, default=2000)
    parser.add_argument("--steps_c", type=int, default=5000)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--num_passes", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use_hf_chat", action="store_true")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("  ANT — Three-Phase Curriculum Training")
    print("=" * 60)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig()
    print(f"  Device: {device}  Vocab: {VOCAB_SIZE}")
    model = ANT(cfg, use_checkpoint=False).to(device)
    log_model_summary(model, cfg)

    # Load existing checkpoint if available
    ckpt_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_step = ckpt.get("step", 0)
        print(f"  Loaded checkpoint: step={start_step}, QA={ckpt.get('qa_accuracy', 0):.1%}")

    best = train(
        model, cfg, device, args.output_dir,
        steps_a=args.steps_a, steps_b=args.steps_b, steps_c=args.steps_c,
        lr=3e-4, batch_size=args.batch_size, grad_accum=args.grad_accum,
        eval_interval=args.eval_interval,
        window_size=args.window_size, num_passes=args.num_passes,
        stride=args.stride,
        n_wiki=args.n_wiki, n_shell=args.n_shell,
        n_qa=args.n_qa, n_chat=args.n_chat,
        use_bf16=args.bf16, use_compile=args.compile,
        use_hf_chat=args.use_hf_chat,
        hf_repo=args.hf_repo,
    )

    report = {
        "total_params": count_params(model),
        "best_qa": best,
        "device": device,
        "window_size": args.window_size,
        "num_passes": args.num_passes,
    }
    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
