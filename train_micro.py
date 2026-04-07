#!/usr/bin/env python3
"""
LatentController — 828K param looping transformer with persistent memory.

Self-contained training pipeline: tokenizer, datasets, encoders, training, evaluation.
Achieves 99.5% QA accuracy on bAbI memory tasks while simultaneously learning LM.

Usage:
    python train_micro.py --chunk_size 16              # QA-only training
    python train_micro.py --chunk_size 16 --multitask  # LM + QA multi-task
    python train_micro.py --eval_only                  # eval last checkpoint
    python train_micro.py --device cpu                 # force CPU
"""

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, MemoryConfig
from model import LoopedLatentController
from memory import MemorySystem


def addr_bytes(addr_tensor: torch.Tensor) -> bytes:
    """Convert int8 address tensor → bytes for TrieIndex."""
    return addr_tensor.cpu().numpy().tobytes()


def memory_vecs_to_tensor(vecs: list, d_model: int, device) -> torch.Tensor:
    """Convert list of int8 numpy arrays to float tensor (1, n, d_model)."""
    out = np.stack(vecs, axis=0).astype(np.float32) / 127.0
    return torch.from_numpy(out).unsqueeze(0).to(device)

# ============================================================================
# Verbose Logging Utilities
# ============================================================================

class TrainingTracker:
    """Tracks running statistics for verbose logging."""

    def __init__(self, window=50):
        self.window = window
        self.losses = []
        self.grad_norms = []
        self.step_times = []
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.eval_history = []  # (step, acc, breakdown)
        self._last_time = None

    def tick(self):
        now = time.time()
        if self._last_time is not None:
            self.step_times.append(now - self._last_time)
        self._last_time = now

    def add_loss(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss

    def add_grad_norm(self, norm):
        self.grad_norms.append(norm)

    def add_eval(self, step, acc, breakdown=None):
        self.eval_history.append((step, acc, breakdown))
        if acc > self.best_acc:
            self.best_acc = acc

    @property
    def avg_loss(self):
        w = self.losses[-self.window:]
        return sum(w) / len(w) if w else 0

    @property
    def avg_grad_norm(self):
        w = self.grad_norms[-self.window:]
        return sum(w) / len(w) if w else 0

    @property
    def steps_per_sec(self):
        w = self.step_times[-self.window:]
        return 1.0 / (sum(w) / len(w)) if w else 0

    @property
    def loss_trend(self):
        """Arrow showing loss direction over last window."""
        if len(self.losses) < self.window:
            return "─"
        first_half = self.losses[-self.window:-self.window // 2]
        second_half = self.losses[-self.window // 2:]
        avg1 = sum(first_half) / len(first_half)
        avg2 = sum(second_half) / len(second_half)
        diff = avg2 - avg1
        if diff < -0.01:
            return "↓"  # improving
        elif diff > 0.01:
            return "↑"  # worsening
        return "→"  # flat


def log_model_summary(model, cfg, label="Model"):
    """Print detailed model configuration and parameter counts."""
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ┌─── {label} Architecture ───")
    print(f"  │ d_model={cfg.d_model}  n_heads={cfg.n_heads}  head_dim={cfg.head_dim}")
    print(f"  │ n_layers={cfg.n_layers}  ffn_dim={cfg.ffn_dim}  max_seq={cfg.max_seq_len}")
    max_slots = getattr(cfg, "max_memory_slots", cfg.n_mem_slots)
    topk = getattr(cfg, "memory_topk", 0)
    print(f"  │ mem_slots={cfg.n_mem_slots}  max_memory_slots={max_slots}  addr_heads={cfg.n_addr_heads}  addr_dim={cfg.addr_dim}")
    if topk > 0:
        print(f"  │ memory_topk={topk} (sparse cross-attention with STE)")
    hops = getattr(cfg, "memory_hops", 1)
    if hops > 1:
        print(f"  │ memory_hops={hops} (multi-hop: entity→attribute retrieval)")
    print(f"  │ params: {n_total:,} total ({n_total/1e6:.2f}M), {n_train:,} trainable")
    # Parameter breakdown by module
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n or "tok_emb" in n)
    layers = sum(p.numel() for n, p in model.named_parameters() if "layers." in n)
    heads = sum(p.numel() for n, p in model.named_parameters() if "addr_" in n or "halt_" in n)
    lm_head = sum(p.numel() for n, p in model.named_parameters() if "lm_head" in n or "output" in n)
    print(f"  │ breakdown: embed={embed:,}  layers={layers:,}  heads={heads:,}  lm_head={lm_head:,}")
    print(f"  └────────────────────")


def log_memory_diagnostics(mem_vecs, label="Memory"):
    """Print memory vector statistics."""
    B, S, D = mem_vecs.shape
    norms = mem_vecs.norm(dim=-1)  # (B, S)
    # Pairwise cosine similarity between slots
    normed = F.normalize(mem_vecs, dim=-1)
    sims = torch.bmm(normed, normed.transpose(1, 2))  # (B, S, S)
    # Mask diagonal
    mask = ~torch.eye(S, device=sims.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
    off_diag_sim = sims[mask].reshape(B, -1)

    print(f"  {label}: shape=({B},{S},{D})")
    print(f"    norms:  mean={norms.mean():.2f}  std={norms.std():.2f}  "
          f"min={norms.min():.2f}  max={norms.max():.2f}")
    print(f"    cosine: mean={off_diag_sim.mean():.3f}  std={off_diag_sim.std():.3f}  "
          f"max={off_diag_sim.max():.3f}")
    # Variance across slots (high = diverse, low = redundant)
    slot_var = mem_vecs.var(dim=1).mean()  # average variance across slots
    print(f"    slot_var={slot_var:.4f} (higher=more diverse)")


def log_gradient_stats(model):
    """Compute and return gradient norm, also log per-module stats."""
    total_norm = 0.0
    module_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            pn = p.grad.data.norm(2).item() ** 2
            total_norm += pn
            # Group by first two path components
            parts = name.split(".")
            module = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            module_norms[module] = module_norms.get(module, 0) + pn
    total_norm = total_norm ** 0.5
    return total_norm, {k: v ** 0.5 for k, v in module_norms.items()}

# ============================================================================
# Byte-Level Tokenizer — pure UTF-8, vocab = 256 (raw bytes)
# Control characters serve as special tokens — no offset, no extra IDs.
# ============================================================================

# ASCII control characters as special tokens (identity-mapped)
PAD_ID  = 0x00  # NUL — padding
SOH_ID  = 0x01  # SOH — memory section start
BOS_ID  = 0x02  # STX — start of text
EOS_ID  = 0x03  # ETX — end of text
EOT_ID  = 0x04  # EOT — memory section end
ANS_ID  = 0x05  # ENQ — answer/query marker
NOOP_ID = 0x06  # ACK — no-op
UNK_ID  = 0x1A  # SUB — substitute/unknown

VOCAB_SIZE = 256  # pure byte space, no extras

# Backward-compatible VOCAB dict for code that does VOCAB["<pad>"] etc.
SPECIAL_TOKENS = {
    "<pad>": PAD_ID, "<eos>": EOS_ID, "<bos>": BOS_ID, "<unk>": UNK_ID,
    "<mem_start>": SOH_ID, "<mem_end>": EOT_ID, "<noop>": NOOP_ID, "<ans>": ANS_ID,
}
VOCAB = dict(SPECIAL_TOKENS)
for _b in range(256):
    _ch = chr(_b)
    if _ch not in VOCAB:
        VOCAB[_ch] = _b

# Reverse mapping for display
ID2WORD = {}
for _name, _tid in SPECIAL_TOKENS.items():
    ID2WORD[_tid] = _name
for _b in range(256):
    if _b not in ID2WORD:
        ID2WORD[_b] = chr(_b) if 32 <= _b < 127 else f'\\x{_b:02x}'

# bAbI word lists — still needed for data generation
NAMES = [
    "mary", "john", "daniel", "sandra", "fred",
    "bill", "julie", "emma", "bob", "alice",
]
LOCATIONS = [
    "garden", "kitchen", "office", "bedroom", "bathroom",
    "hallway", "park", "school", "cinema", "library",
]
VERBS = ["went", "moved", "journeyed", "travelled", "ran", "walked"]
PREPOSITIONS = ["to", "the", "in", "is", "a"]
QUESTION_WORDS = ["where", "?"]
PUNCTUATION = ["."]
CONNECTORS = ["then", "after", "that"]


def tokenize(text: str) -> list[int]:
    """Encode text as raw UTF-8 bytes. Token ID = byte value (no offset)."""
    return list(text.encode('utf-8'))


def detokenize(ids: list[int]) -> str:
    """Decode token IDs back to text. Skips control chars used as special tokens."""
    skip = {PAD_ID, BOS_ID, EOS_ID, SOH_ID, EOT_ID, ANS_ID, NOOP_ID}
    raw_bytes = bytes(b for b in ids if b not in skip and 0 <= b < 256)
    return raw_bytes.decode('utf-8', errors='replace')


# ============================================================================
# Dataset Generator — bAbI-style extractive QA
# ============================================================================

@dataclass
class QAExample:
    """A single memory-recall QA example."""
    passage: str       # "Mary went to the garden . John went to the kitchen ."
    question: str      # "Where is Mary ?"
    answer: str        # "garden"
    answer_entity: str # the name being asked about
    facts: dict        # {"mary": "garden", "john": "kitchen"}


def generate_single_fact() -> QAExample:
    """One person, one location."""
    name = random.choice(NAMES)
    loc = random.choice(LOCATIONS)
    verb = random.choice(VERBS)
    passage = f"{name} {verb} to the {loc} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc, name, {name: loc})


def generate_two_facts() -> QAExample:
    """Two people, ask about one."""
    names = random.sample(NAMES, 2)
    locs = random.sample(LOCATIONS, 2)
    verb1, verb2 = random.choice(VERBS), random.choice(VERBS)
    passage = f"{names[0]} {verb1} to the {locs[0]} . {names[1]} {verb2} to the {locs[1]} ."
    target = random.randint(0, 1)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_three_facts() -> QAExample:
    """Three people, ask about one."""
    names = random.sample(NAMES, 3)
    locs = random.sample(LOCATIONS, 3)
    parts = []
    for n, l in zip(names, locs):
        v = random.choice(VERBS)
        parts.append(f"{n} {v} to the {l} .")
    passage = " ".join(parts)
    target = random.randint(0, 2)
    question = f"where is {names[target]} ?"
    return QAExample(passage, question, locs[target], names[target],
                     dict(zip(names, locs)))


def generate_temporal() -> QAExample:
    """Person moves twice — answer is LAST location."""
    name = random.choice(NAMES)
    loc1, loc2 = random.sample(LOCATIONS, 2)
    v1, v2 = random.sample(VERBS, 2)
    passage = f"{name} {v1} to the {loc1} . then {name} {v2} to the {loc2} ."
    question = f"where is {name} ?"
    return QAExample(passage, question, loc2, name, {name: loc2})


def generate_distractor() -> QAExample:
    """Two people, one moves twice. Ask about either."""
    n1, n2 = random.sample(NAMES, 2)
    l1, l2, l3 = random.sample(LOCATIONS, 3)
    v1, v2, v3 = random.choices(VERBS, k=3)
    passage = (f"{n1} {v1} to the {l1} . {n2} {v2} to the {l2} . "
               f"then {n1} {v3} to the {l3} .")
    facts = {n1: l3, n2: l2}
    target = random.choice([n1, n2])
    question = f"where is {target} ?"
    return QAExample(passage, question, facts[target], target, facts)


GENERATORS = [
    (generate_single_fact, 0.15),
    (generate_two_facts, 0.30),
    (generate_three_facts, 0.20),
    (generate_temporal, 0.20),
    (generate_distractor, 0.15),
]


def generate_dataset(n: int, seed: int = 42) -> list[QAExample]:
    """Generate n QA examples with weighted type distribution."""
    random.seed(seed)
    gens, weights = zip(*GENERATORS)
    examples = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        examples.append(gen())
    return examples


# ============================================================================
# Shell/CLI Data Generator — synthetic command examples
# ============================================================================

_SHELL_COMMANDS = [
    "ls", "cd", "pwd", "cat", "echo", "grep", "find", "sed", "awk", "sort",
    "uniq", "wc", "head", "tail", "cut", "tr", "xargs", "tee", "mkdir",
    "rmdir", "rm", "cp", "mv", "touch", "chmod", "chown", "ln", "diff",
    "tar", "gzip", "gunzip", "zip", "unzip", "curl", "wget", "ssh", "scp",
    "rsync", "ps", "top", "kill", "df", "du", "mount", "umount", "ping",
    "ifconfig", "ip", "netstat", "dig", "nslookup", "git", "docker",
    "python", "node", "npm", "pip", "make", "gcc", "go", "cargo", "man",
]
_SHELL_FLAGS = {
    "ls": ["-l", "-a", "-la", "-lh", "-R", "-t", "-S", "--color"],
    "grep": ["-r", "-i", "-n", "-v", "-c", "-l", "--include", "-E", "-P"],
    "find": ["-name", "-type f", "-type d", "-mtime", "-size", "-exec"],
    "ps": ["-ef", "-aux", "-u", "--sort"],
    "chmod": ["755", "644", "+x", "-R", "u+w", "go-r"],
    "tar": ["-xzf", "-czf", "-xjf", "-tf", "--list"],
    "git": ["status", "log", "diff", "add", "commit", "push", "pull",
            "branch", "checkout", "merge", "rebase", "stash", "clone"],
    "docker": ["ps", "images", "run", "build", "exec", "stop", "rm",
               "compose up", "compose down", "logs"],
    "curl": ["-s", "-o", "-X POST", "-H", "-d", "-L", "-I", "-k"],
}
_SHELL_PATHS = [
    "/home/user", "/tmp", "/var/log", "/etc", "/usr/bin", "/opt",
    "~/Documents", "~/projects", "./src", "../lib", "/dev/null",
    ".", "..", "~", "/proc", "/sys",
]
_SHELL_FILES = [
    "README.md", "Makefile", "config.yaml", "main.py", "index.js",
    "app.go", "Cargo.toml", "package.json", "Dockerfile", ".gitignore",
    "requirements.txt", "setup.py", "test.sh", "data.csv", "output.log",
    "server.conf", ".env", "docker-compose.yml", "CMakeLists.txt",
]
_SHELL_PATTERNS = [
    "*.py", "*.js", "*.go", "*.rs", "*.c", "*.h", "*.txt", "*.log",
    "*.json", "*.yaml", "*.toml", "*.md", "*.sh", "*.sql",
]
_SHELL_VARS = [
    "$HOME", "$PATH", "$USER", "$PWD", "$SHELL", "$EDITOR", "$TERM",
    "$?", "$$", "$!", "$#", "$0", "$1", "$@",
]
_PIPE_COMMANDS = ["grep", "sort", "uniq", "wc", "head", "tail", "cut",
                  "tr", "awk", "sed", "tee", "xargs"]


def _gen_simple_command() -> str:
    cmd = random.choice(_SHELL_COMMANDS)
    flags = _SHELL_FLAGS.get(cmd, [])
    parts = [cmd]
    if flags and random.random() < 0.7:
        parts.append(random.choice(flags))
    if random.random() < 0.5:
        parts.append(random.choice(_SHELL_FILES + _SHELL_PATHS))
    return " ".join(parts)


def _gen_pipe_chain() -> str:
    n_pipes = random.randint(2, 4)
    parts = [_gen_simple_command()]
    for _ in range(n_pipes - 1):
        cmd = random.choice(_PIPE_COMMANDS)
        flags = _SHELL_FLAGS.get(cmd, [])
        seg = cmd
        if flags and random.random() < 0.5:
            seg += " " + random.choice(flags)
        if cmd in ("grep", "sed", "awk") and random.random() < 0.6:
            seg += " '" + random.choice(["error", "warning", "TODO",
                                          "import", "def ", "class ",
                                          "^#", "\\d+", "[0-9]"]) + "'"
        parts.append(seg)
    return " | ".join(parts)


def _gen_redirect() -> str:
    cmd = _gen_simple_command()
    redir = random.choice([">", ">>", "2>", "2>&1", "&>", "< "])
    target = random.choice(_SHELL_FILES + ["/dev/null"])
    return f"{cmd} {redir} {target}"


def _gen_conditional() -> str:
    c1 = _gen_simple_command()
    c2 = _gen_simple_command()
    op = random.choice(["&&", "||", ";"])
    return f"{c1} {op} {c2}"


def _gen_subshell() -> str:
    inner = _gen_simple_command()
    outer = random.choice(["echo", "export VAR=", "cd"])
    if outer == "echo":
        return f'echo "$({inner})"'
    elif outer.startswith("export"):
        return f'export VAR="$({inner})"'
    return f"cd $({inner})"


def _gen_for_loop() -> str:
    var = random.choice(["f", "i", "file", "dir", "x"])
    iterable = random.choice([
        f"*.{random.choice(['py', 'js', 'go', 'txt'])}",
        "$(seq 1 10)",
        f"$(find . -name '{random.choice(_SHELL_PATTERNS)}')",
        "$@",
    ])
    body = random.choice([
        f"echo ${var}",
        f"cat ${var}",
        f"wc -l ${var}",
        f"cp ${var} /tmp/",
    ])
    return f"for {var} in {iterable}; do {body}; done"


def _gen_if_statement() -> str:
    cond = random.choice([
        f'[ -f "{random.choice(_SHELL_FILES)}" ]',
        f'[ -d "{random.choice(_SHELL_PATHS)}" ]',
        "[ $? -eq 0 ]",
        f'[ -z "${{VAR}}" ]',
        f'[ "$1" = "{random.choice(["start", "stop", "restart"])}" ]',
    ])
    then_cmd = _gen_simple_command()
    return f"if {cond}; then {then_cmd}; fi"


def _gen_one_liner() -> str:
    templates = [
        lambda: f"while read line; do echo $line; done < {random.choice(_SHELL_FILES)}",
        lambda: f"test -f {random.choice(_SHELL_FILES)} && echo exists || echo missing",
        lambda: f"[[ {random.choice(_SHELL_VARS)} == *{random.choice(['bin', 'usr', 'home'])}* ]] && echo yes",
        lambda: f"alias ll='ls -la'",
        lambda: f"export PATH={random.choice(_SHELL_PATHS)}:$PATH",
        lambda: f"trap 'rm -f /tmp/lock' EXIT",
        lambda: f"nohup {_gen_simple_command()} &",
    ]
    return random.choice(templates)()


_SHELL_GENERATORS = [
    (_gen_simple_command, 0.25),
    (_gen_pipe_chain, 0.20),
    (_gen_redirect, 0.10),
    (_gen_conditional, 0.10),
    (_gen_subshell, 0.08),
    (_gen_for_loop, 0.10),
    (_gen_if_statement, 0.09),
    (_gen_one_liner, 0.08),
]


def generate_shell_texts(n: int, seed: int = 42) -> list[str]:
    """Generate n synthetic shell command examples."""
    random.seed(seed)
    gens, weights = zip(*_SHELL_GENERATORS)
    texts = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        text = gen()
        # Occasionally add a comment
        if random.random() < 0.15:
            comment = random.choice([
                "# list files", "# search logs", "# backup", "# deploy",
                "# cleanup", "# check status", "# build project",
                "# install deps", "# run tests", "# debug",
            ])
            text = comment + "\n" + text
        texts.append(text)
    return texts


# ============================================================================
# Wikipedia Sentence Loader — diverse factual text
# ============================================================================

def load_wikipedia_sentences(n: int = 10000, min_len: int = 30,
                             max_len: int = 300, seed: int = 42,
                             cache_dir: str = "data_cache") -> list[str]:
    """
    Load sentences from Wikipedia using HuggingFace datasets.
    Tries wikimedia/wikipedia (English), falls back to synthetic text.
    Caches processed sentences locally.
    """
    cache_path = os.path.join(cache_dir, f"wiki_sentences_{n}.txt")
    if os.path.exists(cache_path):
        print(f"  Loading cached Wikipedia sentences from {cache_path}")
        with open(cache_path, "r") as f:
            sentences = [line.strip() for line in f if line.strip()]
        if len(sentences) >= n:
            return sentences[:n]

    print(f"  Downloading Wikipedia for {n} sentences...")
    os.makedirs(cache_dir, exist_ok=True)

    loaders = [
        ("wikimedia/wikipedia", "20231101.en"),
        ("omarkamali/wikipedia-monthly", "latest.en"),
    ]

    ds = None
    for repo, config in loaders:
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load(repo, config, split="train", streaming=True)
            print(f"  Using {repo} ({config})")
            break
        except Exception as e:
            print(f"  {repo} failed: {e}")
            continue

    if ds is None:
        print("  All Wikipedia sources failed, using fallback diverse text")
        return _generate_fallback_diverse_text(n, seed)

    random.seed(seed)
    sentences = []
    for article in ds:
        text = article.get("text", "")
        for sent in text.split(". "):
            sent = sent.strip()
            if min_len <= len(sent) <= max_len and not sent.startswith("="):
                if any(c.isalpha() for c in sent):
                    sentences.append(sent + "." if not sent.endswith(".") else sent)
                    if len(sentences) >= n * 2:
                        break
        if len(sentences) >= n * 2:
            break

    random.shuffle(sentences)
    sentences = sentences[:n]

    with open(cache_path, "w") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"  Cached {len(sentences)} sentences to {cache_path}")
    return sentences


def _generate_fallback_diverse_text(n: int, seed: int = 42) -> list[str]:
    """Fallback if Wikipedia download fails — synthetic diverse text."""
    random.seed(seed)
    subjects = ["the cat", "a scientist", "the river", "an artist", "the machine",
                "a child", "the city", "a bird", "the forest", "a teacher",
                "the ocean", "a farmer", "the mountain", "a doctor", "the bridge"]
    verbs = ["discovered", "created", "moved toward", "studied", "built",
             "observed", "explored", "designed", "protected", "transformed"]
    objects = ["a new path", "the ancient ruins", "a hidden garden", "the tall tower",
               "a small village", "the dark cave", "a bright star", "the old library",
               "a deep well", "the wooden bridge", "the crystal lake", "a vast desert"]
    adverbs = ["quickly", "carefully", "silently", "eagerly", "slowly",
               "boldly", "gently", "suddenly", "deliberately", "gracefully"]
    texts = []
    for _ in range(n):
        subj = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        adv = random.choice(adverbs)
        pattern = random.choice([
            f"{subj} {adv} {verb} {obj}.",
            f"{adv}, {subj} {verb} {obj}.",
            f"{subj} {verb} {obj} and then paused.",
            f"after a long journey, {subj} {verb} {obj}.",
            f"{subj} {verb} {obj}, which surprised everyone.",
        ])
        texts.append(pattern)
    return texts


# ============================================================================
# TextLMDataset — raw text for autoregressive LM training
# ============================================================================

class TextLMDataset(Dataset):
    """
    Autoregressive LM dataset from raw text strings.
    Each sample: <bos> text_bytes <eos> (padded to max_len).
    Shifted targets: loss computed on all non-padding positions.
    """
    def __init__(self, texts: list[str], max_len: int = 192):
        self.samples = []
        pad = PAD_ID
        bos = BOS_ID
        eos = EOS_ID

        for text in texts:
            text_ids = tokenize(text)
            full_seq = [bos] + text_ids + [eos]
            if len(full_seq) > max_len + 1:
                full_seq = full_seq[:max_len + 1]

            inp_ids = full_seq[:-1]
            tgt_ids = full_seq[1:]

            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# PyTorch Datasets
# ============================================================================

class ContextQADataset(Dataset):
    """
    For warmup/LM training: passage + question + answer all in context.
    Shifted autoregressive format:
      full = <bos> passage <ans> question answer <eos>
      inp  = full[:-1]
      tgt  = [PAD...context...] answer <eos>  (PAD=ignored in loss)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 192):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos] + passage_ids + [ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            # Only compute loss on answer tokens — use pad_id for context so they're ignored
            n_context = 1 + len(passage_ids) + 1 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids), f"{len(inp_ids)} != {len(tgt_ids)}"

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MemoryQADataset(Dataset):
    """
    For memory training: passage is ONLY in memory, not in context.
    Shifted autoregressive format:
      full    = <bos> <ans> question answer <eos>
      inp     = full[:-1]
      tgt     = [PAD...context...] answer <eos>  (PAD=ignored in loss)
      passage = padded passage tokens (for memory feeding)
    """
    def __init__(self, examples: list[QAExample], max_len: int = 128,
                 max_passage_len: int = 128):
        self.samples = []
        pad = VOCAB["<pad>"]
        bos = VOCAB["<bos>"]
        eos = VOCAB["<eos>"]
        ans_marker = VOCAB["<ans>"]

        for ex in examples:
            passage_ids = tokenize(ex.passage)
            question_ids = tokenize(ex.question)
            answer_ids = tokenize(ex.answer)

            full_seq = [bos, ans_marker] + question_ids + answer_ids + [eos]
            inp_ids = full_seq[:-1]
            n_context = 2 + len(question_ids)
            tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]

            assert len(inp_ids) == len(tgt_ids)

            if len(inp_ids) > max_len:
                inp_ids = inp_ids[:max_len]
                tgt_ids = tgt_ids[:max_len]
            while len(inp_ids) < max_len:
                inp_ids.append(pad)
                tgt_ids.append(pad)

            # Pad passage to fixed length for batching
            p_ids = passage_ids[:max_passage_len]
            while len(p_ids) < max_passage_len:
                p_ids.append(pad)

            self.samples.append((
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
                torch.tensor(p_ids, dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# Helpers
# ============================================================================

def hidden_to_int8(hidden: torch.Tensor) -> np.ndarray:
    arr = hidden.detach().float().cpu().numpy()
    scale = np.abs(arr).max()
    if scale < 1e-6:
        return np.zeros(arr.shape, dtype=np.int8)
    return np.clip(np.round(arr / scale * 127.0), -128, 127).astype(np.int8)


def batch_read_memory(model, hidden_states, memory, device):
    """Read memory for a batch using hidden states to compute addresses."""
    B = hidden_states.size(0)
    d = model.cfg.d_model
    addr_heads = model.compute_addresses_batch(hidden_states)
    addr_cpu = [h.cpu().numpy() for h in addr_heads]
    batch_addresses = []
    for b in range(B):
        sample_addrs = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
        batch_addresses.append(sample_addrs)
    mem_np = memory.read_memory_batch(batch_addresses)
    mem_tensor = torch.from_numpy(
        mem_np.astype(np.float32) / 127.0
    ).to(device, non_blocking=True)
    return mem_tensor


def write_memory_batch(model, hidden, memory, positions=None):
    """Write hidden states at specified positions to memory."""
    B, T, D = hidden.shape
    if positions is None:
        positions = [T - 1]  # default: write last position only

    n_writes = 0
    for pos in positions:
        h_batch = hidden[:, pos, :].detach()
        vecs_np = h_batch.float().cpu().numpy()
        addr_heads = model.compute_addresses_batch(h_batch)
        addr_cpu = [h.cpu().numpy() for h in addr_heads]
        for b in range(B):
            ab = [addr_cpu[h][b].tobytes() for h in range(len(addr_heads))]
            memory.write_memory(ab, vecs_np[b])
            n_writes += 1
    return n_writes


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Phase A: Warmup LM on QA patterns (passage in context)
# ============================================================================

def train_phase_a(model, cfg, device, examples, steps=500, lr=3e-4,
                  batch_size=64, max_len=128):
    """Train LM on QA format: passage + question → answer. No memory."""
    print("\n" + "=" * 60)
    print("  Phase A: Warmup LM (QA patterns, no memory)")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)

    t0 = time.time()
    tracker = TrainingTracker()
    for step in range(1, steps + 1):
        tracker.tick()
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        logits, halt_logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm, _ = log_gradient_stats(model)
        tracker.add_loss(loss.item())
        tracker.add_grad_norm(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [A {step:>4}/{steps}] loss={loss.item():.4f} "
                  f"avg={tracker.avg_loss:.4f}{tracker.loss_trend} "
                  f"gnorm={grad_norm:.2f} lr={lr_now:.1e} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase A done in {elapsed:.0f}s, final loss={loss.item():.4f}")
    return loss.item()


# ============================================================================
# Phase B: Address head contrastive training
# ============================================================================

def train_phase_b(model, cfg, device, examples, steps=300, lr=1e-3,
                  batch_size=128, max_len=128):
    """
    Train address heads to produce distinct addresses for different entities.
    Same entity → similar address, different entity → different address.
    """
    print("\n" + "=" * 60)
    print("  Phase B: Address Head Contrastive Training")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")
    print(f"  Only training: addr_heads (backbone frozen)")

    # Generate hidden states for various entities
    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Only train address heads
    for p in model.parameters():
        p.requires_grad_(False)
    for head in model.addr_heads:
        for p in head.parameters():
            p.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for head in model.addr_heads for p in head.parameters()],
        lr=lr, weight_decay=0.01,
    )

    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp = inp.to(device)

        with torch.no_grad():
            _, _, hidden = model(inp, return_hidden=True)

        # Use last position hidden states
        h = hidden[:, -1, :]  # (B, d_model)
        B = h.size(0)

        # Compute addresses from all heads
        loss = torch.tensor(0.0, device=device)
        for head in model.addr_heads:
            raw = head(h)  # (B, addr_dim)
            # Contrastive: pull same-batch neighbors apart (diversity),
            # but encourage spread across address space
            # Use pairwise distance — maximize average distance
            dists = torch.cdist(raw.unsqueeze(0), raw.unsqueeze(0)).squeeze(0)  # (B, B)
            # Encourage large pairwise distances (negative = push apart)
            margin = 4.0
            spread_loss = F.relu(margin - dists).mean()

            # Entropy: encourage each dimension to use full range
            dim_std = raw.std(dim=0)
            target_std = 15.0
            entropy_loss = F.mse_loss(dim_std, torch.full_like(dim_std, target_std))

            loss = loss + spread_loss + 0.1 * entropy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for head in model.addr_heads for p in head.parameters()], 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            print(f"  [B {step}/{steps}] loss={loss.item():.4f} ({elapsed:.0f}s)")

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad_(True)

    elapsed = time.time() - t0
    print(f"  Phase B done in {elapsed:.0f}s")


# ============================================================================
# Phase C: ACT Curriculum (learn when to think more)
# ============================================================================

def streaming_act_forward(model, inp, memory_system, mem_tensor,
                          max_steps, temperature, device):
    """ACT forward with soft halting and optional memory re-read."""
    B, T = inp.shape
    HALT = 1

    remaining = torch.ones(B, T, device=device)
    weighted_logits = None
    expected_steps = torch.zeros(B, T, device=device)
    halt_counts = Counter()

    for i in range(max_steps):
        logits, halt_logits, hidden = model(
            inp, memory_vectors=mem_tensor, return_hidden=True
        )
        halt_prob = F.softmax(halt_logits / max(temperature, 1e-6), dim=-1)[..., HALT]

        if i < max_steps - 1:
            w = remaining * halt_prob
        else:
            w = remaining

        if weighted_logits is None:
            weighted_logits = w.unsqueeze(-1) * logits
        else:
            weighted_logits = weighted_logits + w.unsqueeze(-1) * logits

        expected_steps = expected_steps + (i + 1) * w
        remaining = (remaining - w).clamp(min=0.0)
        halt_counts[i + 1] += (halt_prob > 0.5).sum().item()

        # Re-read memory between ACT steps
        if i < max_steps - 1 and memory_system is not None:
            h_last = hidden[:, -1, :].detach()
            mem_tensor = batch_read_memory(model, h_last, memory_system, device)

    return weighted_logits, expected_steps, halt_counts, hidden


def train_phase_c(model, cfg, device, examples, steps=500, lr=1e-4,
                  batch_size=64, max_len=128):
    """
    ACT curriculum: ramp max_steps and ponder cost.
    Uses context QA (passage in context) so model can learn to halt properly.
    """
    print("\n" + "=" * 60)
    print("  Phase C: ACT Curriculum")
    print("=" * 60)
    print(f"  Steps: {steps}  LR: {lr}  Batch: {batch_size}")
    print(f"  Curriculum: ramp max_steps 2→4, ponder 0→0.005")

    ds = ContextQADataset(examples, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()
    loader_iter = iter(loader)
    t0 = time.time()

    # Curriculum: (step_threshold, max_act, ponder_weight, temperature)
    curriculum = [
        (0,   2, 0.0,   1.0),
        (100, 2, 0.001, 1.0),
        (200, 4, 0.002, 0.5),
        (350, 4, 0.005, 0.1),
    ]

    def get_curriculum_params(step):
        max_act, pw, temp = 2, 0.0, 1.0
        for thresh, ma, p, t in curriculum:
            if step >= thresh:
                max_act, pw, temp = ma, p, t
        return max_act, pw, temp

    for step in range(1, steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp, tgt = inp.to(device), tgt.to(device)
        max_act, ponder_w, temperature = get_curriculum_params(step)

        lr_now = get_lr(step, 50, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        weighted_logits, expected_steps, halt_counts, _ = streaming_act_forward(
            model, inp, None, None, max_act, temperature, device
        )

        lm_loss = F.cross_entropy(
            weighted_logits.reshape(-1, weighted_logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )
        ponder_loss = ponder_w * expected_steps.mean() if ponder_w > 0 else 0.0
        loss = lm_loss + ponder_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t0
            total_halt = sum(halt_counts.values())
            avg_halt = sum(k * v for k, v in halt_counts.items()) / max(total_halt, 1)
            print(f"  [C {step}/{steps}] loss={loss.item():.4f} "
                  f"act={max_act} halt={avg_halt:.1f} pw={ponder_w:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Phase C done in {elapsed:.0f}s")


# ============================================================================
# Phase D: Memory QA — streaming video-frame encoder
# ============================================================================

def _find_entity_positions(token_ids, pad_id):
    """
    Parse passage tokens to find per-entity key positions (byte-level).
    Returns list of (name_start, loc_start, period_pos) tuples, one per entity.
    Positions are offsets into the token_ids list (0-indexed).
    """
    period_id = VOCAB["."]
    ids = [t for t in token_ids if t != pad_id]

    # Precompute byte sequences for names/locations
    name_seqs = {n: tokenize(n) for n in NAMES}
    loc_seqs = {l: tokenize(l) for l in LOCATIONS}

    def find_word_at(pos, word_seqs):
        """Check if any word sequence starts at pos in ids."""
        for word, seq in word_seqs.items():
            if ids[pos:pos+len(seq)] == seq:
                return word, len(seq)
        return None, 0

    entities = []
    cur_name_pos = None
    cur_loc_pos = None
    i = 0
    while i < len(ids):
        if ids[i] == period_id and cur_name_pos is not None:
            entities.append((
                cur_name_pos,
                cur_loc_pos if cur_loc_pos is not None else i,
                i,
            ))
            cur_name_pos = None
            cur_loc_pos = None
            i += 1
            continue

        word, wlen = find_word_at(i, name_seqs)
        if word is not None:
            cur_name_pos = i
            cur_loc_pos = None
            i += wlen
            continue

        word, wlen = find_word_at(i, loc_seqs)
        if word is not None and cur_name_pos is not None:
            cur_loc_pos = i
            i += wlen
            continue

        i += 1

    return entities



def encode_streaming_memory(model, passages, device, chunk_size=8,
                            slots_per_chunk=2, differentiable=False):
    """Video-frame streaming encoder — per-token memory.

    Processes passage chunks sequentially. Each chunk's forward pass includes
    cross-attention to ALL previously emitted token hidden states, so each
    new "frame" is encoded with full awareness of what came before.

    Every content token becomes its own memory vector (no pooling).
    This preserves per-token entity/location information that pooling destroys.

    Returns: (keys, values, mask) -- memory bank for downstream QA.
      keys:   (B, n_tokens, d_model) -- per-token hidden + temporal signal
      values: (B, n_tokens, d_model) -- per-token hidden states
      mask:   (B, n_tokens) bool     -- True for valid positions
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]

    # Per-example content lengths (exclude padding)
    content_lens = []
    for b in range(B):
        clen = sum(1 for t in passages[b].tolist() if t != pad_id)
        content_lens.append(clen)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    n_chunks = (max_clen + chunk_size - 1) // chunk_size

    # Accumulate per-token memory across chunks
    all_keys = []    # list of (B, d_model) tensors, one per token
    all_values = []
    all_mask = []    # list of (B,) bool tensors

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, max_clen)
        chunk_len = end - start

        # Build chunk input: [BOS, chunk_tokens...] per example
        inp = torch.full((B, chunk_len + 1), pad_id, dtype=torch.long, device=device)
        per_token_valid = torch.zeros(B, chunk_len, dtype=torch.bool, device=device)

        for b in range(B):
            clen = content_lens[b]
            if start >= clen:
                continue
            inp[b, 0] = bos_id
            actual_end = min(end, clen)
            actual_len = actual_end - start
            inp[b, 1:actual_len + 1] = passages[b, start:actual_end]
            per_token_valid[b, :actual_len] = True

        # Prepare accumulated memory from previous chunks
        if all_keys:
            mem_keys = torch.stack(all_keys, dim=1)
            mem_vals = torch.stack(all_values, dim=1)
            mem_mask = torch.stack(all_mask, dim=1)
        else:
            mem_keys, mem_vals, mem_mask = None, None, None

        # Forward pass — each chunk sees all previous tokens via cross-attention
        if differentiable:
            _, _, hidden = model(
                inp, memory_keys=mem_keys, memory_values=mem_vals,
                memory_mask=mem_mask, return_hidden=True)
        else:
            with torch.no_grad():
                _, _, hidden = model(
                    inp, memory_keys=mem_keys, memory_values=mem_vals,
                    memory_mask=mem_mask, return_hidden=True)

        # Skip BOS (position 0), keep per-token hidden states
        content_h = hidden[:, 1:, :]  # (B, chunk_len, d_model)

        # Temporal embedding for this chunk
        t_idx = torch.tensor(ci, dtype=torch.long, device=device)
        t_idx = t_idx.clamp_max(model.temporal_emb.num_embeddings - 1)
        t_emb = model.temporal_emb(t_idx)  # (d_model,)

        # Each token becomes its own memory entry
        for ti in range(chunk_len):
            tok_h = content_h[:, ti, :]       # (B, d_model)
            tok_valid = per_token_valid[:, ti]  # (B,)
            all_keys.append(tok_h + t_emb)
            all_values.append(tok_h)
            all_mask.append(tok_valid)

    # Stack into memory bank tensors
    keys = torch.stack(all_keys, dim=1)    # (B, n_tokens, d_model)
    values = torch.stack(all_values, dim=1)
    mask = torch.stack(all_mask, dim=1)

    return keys, values, mask


# ---------------------------------------------------------------------------
# Encoder-Decoder Memory: Multi-iteration bidirectional encoder
# ---------------------------------------------------------------------------

def encode_enc_dec(model, passages, device, n_iterations=3,
                   differentiable=False):
    """Encoder-decoder memory: multi-iteration bidirectional encoder.

    Two-phase architecture:
      ENCODER: Processes the full passage with bidirectional self-attention
               across multiple iterations. Each iteration cross-attends to
               the previous iteration's output, enabling iterative refinement.
               No token output — pure understanding.
      DECODER: (handled by normal model.forward) Cross-attends to encoder
               output via memory, produces answer tokens with causal attention.

    Key insight: memory KEYS use raw token embeddings (entity identity),
    while memory VALUES use refined contextual hidden states (understanding).
    This separates "what to attend to" (entity matching) from
    "what to retrieve" (contextual answer), solving the entity-routing problem.

    Returns: (keys, values, mask) for decoder cross-attention.
      keys:   (B, n_tokens, d_model) -- raw embeddings (identity-focused)
      values: (B, n_tokens, d_model) -- refined hidden states (context-focused)
      mask:   (B, n_tokens) bool     -- True for valid positions
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]

    # Per-example content lengths
    content_lens = []
    for b in range(B):
        clen = sum(1 for t in passages[b].tolist() if t != pad_id)
        content_lens.append(clen)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    # Build full passage input: [BOS, passage_tokens...]
    inp = torch.full((B, max_clen + 1), pad_id, dtype=torch.long, device=device)
    token_valid = torch.zeros(B, max_clen, dtype=torch.bool, device=device)
    for b in range(B):
        clen = content_lens[b]
        inp[b, 0] = bos_id
        inp[b, 1:clen + 1] = passages[b, :clen]
        token_valid[b, :clen] = True

    # Raw token embeddings for memory KEYS (entity identity, no context mixing)
    with torch.no_grad():
        raw_emb = model.embed(inp[:, 1:])  # (B, max_clen, d_model), skip BOS

    # Multi-iteration bidirectional encoding for memory VALUES
    mem_keys_iter, mem_vals_iter, mem_mask = None, None, None

    ctx = torch.enable_grad if differentiable else torch.no_grad
    with ctx():
        for iteration in range(n_iterations):
            # Full forward: bidirectional self-attn + cross-attn to prev iteration
            _, _, hidden = model(
                inp,
                memory_keys=mem_keys_iter,
                memory_values=mem_vals_iter,
                memory_mask=mem_mask,
                return_hidden=True,
                bidirectional=True,
            )

            # Skip BOS, take per-token content hidden states
            content_h = hidden[:, 1:, :]  # (B, max_clen, d_model)

            # This iteration's output becomes next iteration's cross-attn target
            mem_keys_iter = content_h
            mem_vals_iter = content_h
            mem_mask = token_valid

    # Keys = raw embeddings (identity), Values = refined hidden (context)
    return raw_emb, content_h, token_valid


def encode_kv_memory_chunked(model, passages, device, chunk_size=8,
                             slots_per_chunk=2, **kwargs):
    """Non-differentiable streaming memory encoding."""
    return encode_streaming_memory(
        model, passages, device, chunk_size, slots_per_chunk,
        differentiable=False)


def encode_kv_memory_chunked_differentiable(model, passages, device,
                                             chunk_size=8, slots_per_chunk=2,
                                             **kwargs):
    """Differentiable streaming memory encoding -- gradients flow through."""
    return encode_streaming_memory(
        model, passages, device, chunk_size, slots_per_chunk,
        differentiable=True)


# ---------------------------------------------------------------------------
# Sentence-boundary streaming encoder
# ---------------------------------------------------------------------------

def encode_sentence_memory(model, passages, device, differentiable=False):
    """Sentence-boundary streaming encoder — per-token memory.

    Like the video-frame streaming encoder, but splits at sentence boundaries
    (period tokens) instead of fixed chunk sizes. This guarantees each chunk
    contains exactly one entity-location fact, preventing the boundary-splitting
    problem that caps accuracy at 73%.

    Each chunk's forward pass includes cross-attention to ALL previously
    emitted token hidden states (streaming awareness).

    Returns: (keys, values, mask) -- memory bank for downstream QA.
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]
    period_id = VOCAB["."]

    # Per-example: find sentence boundaries and content lengths
    example_sentences = []  # list of lists of (start, end) per example
    content_lens = []
    for b in range(B):
        tokens = passages[b].tolist()
        clen = sum(1 for t in tokens if t != pad_id)
        content_lens.append(clen)

        # Split at periods
        sents = []
        sent_start = 0
        for i in range(clen):
            if tokens[i] == period_id:
                sents.append((sent_start, i + 1))  # include period
                sent_start = i + 1
        # Trailing tokens without period
        if sent_start < clen:
            sents.append((sent_start, clen))
        example_sentences.append(sents)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    max_n_sents = max(len(s) for s in example_sentences)

    # Accumulate per-token memory across sentences
    all_keys = []
    all_values = []
    all_mask = []

    for si in range(max_n_sents):
        # Find max sentence length for this sentence index
        sent_lens = []
        for b in range(B):
            if si < len(example_sentences[b]):
                s, e = example_sentences[b][si]
                sent_lens.append(e - s)
            else:
                sent_lens.append(0)
        max_slen = max(sent_lens) if sent_lens else 0
        if max_slen == 0:
            continue

        # Build sentence input: [BOS, sentence_tokens...]
        inp = torch.full((B, max_slen + 1), pad_id, dtype=torch.long, device=device)
        per_token_valid = torch.zeros(B, max_slen, dtype=torch.bool, device=device)

        for b in range(B):
            if si >= len(example_sentences[b]):
                continue
            s, e = example_sentences[b][si]
            slen = e - s
            inp[b, 0] = bos_id
            inp[b, 1:slen + 1] = passages[b, s:e]
            per_token_valid[b, :slen] = True

        # Prepare accumulated memory from previous sentences
        if all_keys:
            mem_keys = torch.stack(all_keys, dim=1)
            mem_vals = torch.stack(all_values, dim=1)
            mem_mask_prev = torch.stack(all_mask, dim=1)
        else:
            mem_keys, mem_vals, mem_mask_prev = None, None, None

        # Forward pass — causal within sentence, cross-attn to prev sentences
        if differentiable:
            _, _, hidden = model(
                inp, memory_keys=mem_keys, memory_values=mem_vals,
                memory_mask=mem_mask_prev, return_hidden=True)
        else:
            with torch.no_grad():
                _, _, hidden = model(
                    inp, memory_keys=mem_keys, memory_values=mem_vals,
                    memory_mask=mem_mask_prev, return_hidden=True)

        # Skip BOS, keep per-token hidden states
        content_h = hidden[:, 1:, :]  # (B, max_slen, d_model)

        # Temporal embedding for this sentence index
        t_idx = torch.tensor(si, dtype=torch.long, device=device)
        t_idx = t_idx.clamp_max(model.temporal_emb.num_embeddings - 1)
        t_emb = model.temporal_emb(t_idx)

        # Pool: last valid token per example as single memory slot per sentence
        pooled_h = torch.zeros(B, d_model, device=device)
        sent_valid = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            if si >= len(example_sentences[b]):
                continue
            slen = example_sentences[b][si][1] - example_sentences[b][si][0]
            pooled_h[b] = content_h[b, slen - 1, :]
            sent_valid[b] = True
        all_keys.append(pooled_h + t_emb)
        all_values.append(pooled_h)
        all_mask.append(sent_valid)

    # Stack into memory bank tensors
    keys = torch.stack(all_keys, dim=1)
    values = torch.stack(all_values, dim=1)
    mask = torch.stack(all_mask, dim=1)

    return keys, values, mask


def encode_sentence_frozen(model, passages, device, **kwargs):
    """Non-differentiable sentence-boundary memory encoding."""
    return encode_sentence_memory(model, passages, device, differentiable=False)


def encode_sentence_differentiable(model, passages, device, **kwargs):
    """Differentiable sentence-boundary memory encoding."""
    return encode_sentence_memory(model, passages, device, differentiable=True)


def encode_enc_dec_frozen(model, passages, device, **kwargs):
    """Non-differentiable encoder-decoder memory."""
    return encode_enc_dec(model, passages, device, n_iterations=3,
                          differentiable=False)


def encode_enc_dec_differentiable(model, passages, device, **kwargs):
    """Differentiable encoder-decoder memory — gradients flow through all iterations."""
    return encode_enc_dec(model, passages, device, n_iterations=3,
                          differentiable=True)


# ---------------------------------------------------------------------------
# Sliding window streaming encoder (overlapping diffusion-like context)
# ---------------------------------------------------------------------------

# Module-level stride, set from --stride CLI arg
_SLIDING_STRIDE = 1


def encode_sliding_window_memory(model, passages, device, window_size=8,
                                  stride=1, differentiable=False):
    """Sliding window streaming encoder — overlapping context windows.

    Processes passage with overlapping windows that slide by `stride` tokens.
    Each token is seen in up to `window_size // stride` windows. Like
    neighboring cells in diffusion, overlapping views allow information
    to propagate between tokens through iterative re-processing.

    Each window cross-attends to the latest accumulated hidden states of
    ALL previously seen tokens, creating iterative refinement:
      Window 0: encode tokens [0..W) with no memory
      Window 1: encode tokens [S..S+W) with cross-attn to updated [0..W)
      Window 2: encode tokens [2S..2S+W) with cross-attn to updated [0..S+W)
      ...

    Tokens in overlapping regions get refined representations that incorporate
    progressively richer context — the key diffusion-like property.

    Returns: (keys, values, mask) for downstream QA cross-attention.
    """
    B = passages.size(0)
    d_model = model.cfg.d_model
    pad_id = VOCAB["<pad>"]
    bos_id = VOCAB["<bos>"]

    content_lens = []
    for b in range(B):
        clen = sum(1 for t in passages[b].tolist() if t != pad_id)
        content_lens.append(clen)

    max_clen = max(content_lens) if content_lens else 0
    if max_clen == 0:
        empty = torch.zeros(B, 1, d_model, device=device)
        empty_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        return empty, empty.clone(), empty_mask

    token_valid = torch.zeros(B, max_clen, dtype=torch.bool, device=device)
    for b in range(B):
        token_valid[b, :content_lens[b]] = True

    # Collect hidden states per token from all window views
    per_token_views = [[] for _ in range(max_clen)]

    # Running memory: latest hidden state per token for cross-attention
    # in subsequent windows (detached — only forward information flow)
    running_h = torch.zeros(B, max_clen, d_model, device=device)
    running_seen = torch.zeros(B, max_clen, dtype=torch.bool, device=device)

    # Window start positions
    win_starts = list(range(0, max(max_clen - window_size + 1, 1), stride))
    # Ensure last window covers end of passage
    if not win_starts or win_starts[-1] + window_size < max_clen:
        last = max(max_clen - window_size, 0)
        if not win_starts or last != win_starts[-1]:
            win_starts.append(last)

    max_temporal = model.temporal_emb.num_embeddings

    for wi, start in enumerate(win_starts):
        end = min(start + window_size, max_clen)
        win_len = end - start

        # Build window input: [BOS, tokens[start:end]]
        inp = torch.full((B, win_len + 1), pad_id, dtype=torch.long, device=device)
        win_valid = torch.zeros(B, win_len, dtype=torch.bool, device=device)

        for b in range(B):
            if start >= content_lens[b]:
                continue
            inp[b, 0] = bos_id
            actual_end = min(end, content_lens[b])
            actual_len = actual_end - start
            inp[b, 1:actual_len + 1] = passages[b, start:actual_end]
            win_valid[b, :actual_len] = True

        # Cross-attend to running memory (all previously seen tokens)
        # Clone to avoid in-place modification breaking autograd graph
        if running_seen.any():
            mem_keys = running_h.clone()
            mem_vals = running_h.clone()
            mem_mask = running_seen
        else:
            mem_keys, mem_vals, mem_mask = None, None, None

        # Forward pass
        if differentiable:
            _, _, hidden = model(
                inp, memory_keys=mem_keys, memory_values=mem_vals,
                memory_mask=mem_mask, return_hidden=True)
        else:
            with torch.no_grad():
                _, _, hidden = model(
                    inp, memory_keys=mem_keys, memory_values=mem_vals,
                    memory_mask=mem_mask, return_hidden=True)

        content_h = hidden[:, 1:, :]  # (B, win_len, d_model) skip BOS

        # Update running memory and collect per-token views
        for ti in range(win_len):
            gp = start + ti
            if gp >= max_clen:
                break
            h = content_h[:, ti, :]  # (B, d_model)
            per_token_views[gp].append(h)

            # Update running memory (detached for next-window cross-attn)
            with torch.no_grad():
                mask_expand = win_valid[:, ti].unsqueeze(-1)  # (B, 1)
                running_h[:, gp, :] = torch.where(
                    mask_expand, h.detach(), running_h[:, gp, :])
                running_seen[:, gp] = running_seen[:, gp] | win_valid[:, ti]

    # Average all views per token → final memory bank
    all_keys = []
    all_values = []
    for pos in range(max_clen):
        if per_token_views[pos]:
            stacked = torch.stack(per_token_views[pos], dim=0)  # (n_views, B, d)
            avg_h = stacked.mean(dim=0)  # (B, d_model)
        else:
            avg_h = torch.zeros(B, d_model, device=device)

        # Temporal embedding based on absolute position
        chunk_idx = min(pos // window_size, max_temporal - 1)
        t_idx = torch.tensor(chunk_idx, dtype=torch.long, device=device)
        t_emb = model.temporal_emb(t_idx)

        all_keys.append(avg_h + t_emb)
        all_values.append(avg_h)

    keys = torch.stack(all_keys, dim=1)    # (B, max_clen, d_model)
    values = torch.stack(all_values, dim=1)  # (B, max_clen, d_model)

    return keys, values, token_valid


def encode_sliding_frozen(model, passages, device, **kwargs):
    """Non-differentiable sliding window memory encoding."""
    window_size = getattr(model.cfg, 'chunk_size', 8)
    with torch.no_grad():
        return encode_sliding_window_memory(
            model, passages, device, window_size=window_size,
            stride=_SLIDING_STRIDE, differentiable=False)


def encode_sliding_differentiable(model, passages, device, **kwargs):
    """Differentiable sliding window memory encoding."""
    window_size = getattr(model.cfg, 'chunk_size', 8)
    return encode_sliding_window_memory(
        model, passages, device, window_size=window_size,
        stride=_SLIDING_STRIDE, differentiable=True)


# ============================================================================
# Sliding Window LM — with Memory Cross-Attention
# ============================================================================

def sliding_lm_encode(model, input_ids, window_size=5, num_passes=4,
                      mem_keys=None, mem_vals=None, mem_mask=None):
    """
    Multi-pass sliding window encoder with optional memory cross-attention.

    Each pass:
      1. Pad edges with PAD embedding (half_w each side)
      2. Create all centered windows via unfold
      3. Process each window through transformer layers
         - Self-attention within window (bidirectional)
         - Cross-attention to external memory (if provided)
      4. Update each position's hidden from its centered window output

    The memory is shared across all windows — every position can attend to it.
    This means even with W=5 and 1 pass, the model can route through memory
    to access any passage information, solving the receptive field problem.

    Args:
        model: LoopedLatentController (with cross-attention layers)
        input_ids: (B, T) token IDs
        window_size: odd integer, size of sliding window
        num_passes: number of diffusion-like refinement passes
        mem_keys: (B, S, d_model) memory keys or None
        mem_vals: (B, S, d_model) memory values or None
        mem_mask: (B, S) bool mask or None

    Returns:
        hidden: (B, T, d_model) final hidden states (norm applied)
    """
    B, T = input_ids.shape
    device = input_ids.device
    half_w = window_size // 2
    pad_right_size = window_size - half_w - 1  # handles even/odd W correctly
    d_model = model.cfg.d_model

    # PAD embedding for edge padding
    pad_id = torch.tensor([VOCAB["<pad>"]], device=device)
    pad_emb = model.embed(pad_id).squeeze(0)  # (1, d_model)

    # Initialize from token embeddings
    hidden = model.embed(input_ids)  # (B, T, d_model)

    # Precompute RoPE for window-relative positions (0..W-1)
    cos = model.rope_cos[:window_size]
    sin = model.rope_sin[:window_size]

    # Bidirectional attention mask (all-to-all within window)
    mask = torch.zeros(window_size, window_size, device=device)

    # Expand memory for windowed processing: (B, S, D) → (B*T, S, D)
    w_mem_keys, w_mem_vals, w_mem_mask = None, None, None
    if mem_keys is not None:
        w_mem_keys = mem_keys.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, d_model)
        w_mem_vals = mem_vals.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, -1, d_model)
        if mem_mask is not None:
            w_mem_mask = mem_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

    for _ in range(num_passes):
        # Pad edges with PAD embedding (asymmetric for even W)
        pad_left = pad_emb.expand(B, half_w, d_model)
        pad_right = pad_emb.expand(B, pad_right_size, d_model)
        padded = torch.cat([pad_left, hidden, pad_right], dim=1)

        # Create all windows: unfold → (B, T, d_model, W) → (B, T, W, d_model)
        windows = padded.unfold(1, window_size, 1)
        windows = windows.permute(0, 1, 3, 2).contiguous()

        # Reshape for batched layer processing: (B*T, W, d_model)
        x = windows.reshape(B * T, window_size, d_model)

        # Process through all transformer layers (self-attn + cross-attn to memory)
        for layer in model.layers:
            x, _ = layer(x, mask, cos, sin,
                         mem_keys=w_mem_keys, mem_values=w_mem_vals,
                         mem_mask=w_mem_mask)

        # Extract center position outputs → new hidden states
        centers = x[:, half_w, :]
        hidden = centers.reshape(B, T, d_model)

    return model.norm(hidden)


@torch.no_grad()
def evaluate_sliding_lm(model, cfg, examples, device, max_examples=200,
                        window_size=5, num_passes=4,
                        encode_fn=None):
    """
    Evaluate sliding window LM on bAbI QA.

    Two modes:
    - With encode_fn: passage encoded to memory, question processed via sliding window
    - Without encode_fn: full sequence (passage+question) in sliding window, no memory
    """
    model.eval()

    pad = VOCAB["<pad>"]
    bos = VOCAB["<bos>"]
    ans_marker = VOCAB["<ans>"]

    correct = 0
    total = 0
    type_correct = Counter()
    type_total = Counter()

    for ex in examples[:max_examples]:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)

        mem_keys, mem_vals, mem_mask = None, None, None
        if encode_fn is not None:
            p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)
            mem_keys, mem_vals, mem_mask = encode_fn(model, p_tensor, device)
            input_ids = [bos, ans_marker] + question_ids + answer_ids
            n_context = 2 + len(question_ids)
        else:
            input_ids = [bos] + passage_ids + [ans_marker] + question_ids + answer_ids
            n_context = 1 + len(passage_ids) + 1 + len(question_ids)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        hidden = sliding_lm_encode(
            model, input_tensor, window_size, num_passes,
            mem_keys=mem_keys, mem_vals=mem_vals, mem_mask=mem_mask)

        # Compute logits at all positions, check answer tokens (teacher-forced)
        logits = F.linear(hidden, model.embed.weight)  # (1, T, vocab_size)
        is_correct = True
        for j, expected_byte in enumerate(answer_ids):
            pred_pos = n_context - 1 + j
            if logits[0, pred_pos, :].argmax().item() != expected_byte:
                is_correct = False
                break

        if is_correct:
            correct += 1
        total += 1

        n_facts = len(ex.facts)
        type_total[n_facts] += 1
        if is_correct:
            type_correct[n_facts] += 1

    accuracy = correct / max(total, 1)
    for n_facts in sorted(type_total.keys()):
        t_corr = type_correct[n_facts]
        t_tot = type_total[n_facts]
        print(f"    {n_facts}-fact: {t_corr}/{t_tot} = {t_corr/max(t_tot,1):.1%}")

    model.train()
    breakdown = {k: (type_correct[k], type_total[k]) for k in type_total}
    return accuracy, breakdown


def train_sliding_lm(model, cfg, device, train_examples, val_examples,
                     output_dir, steps=3000, lr=1e-4, batch_size=32,
                     eval_interval=200, window_size=5, num_passes=4,
                     d1_ratio=0.3):
    """
    Train sliding window LM with memory for bAbI QA.

    Architecture:
      - Passage encoded to memory via existing encoder (enc_dec/sentence/streaming)
      - Question processed through multi-pass sliding window
      - Each window cross-attends to memory at every layer
      - Answer predicted from last position hidden state

    Training curriculum:
      D1 (frozen encoder): teach sliding window to read from memory
      D2 (differentiable): encoder + sliding window co-adapt

    This combines the encoder-only sliding window with external memory,
    giving each position access to the full passage through memory
    without needing a large receptive field.
    """
    print("\n" + "=" * 60)
    print("  Sliding Window LM + Memory Cross-Attention")
    print("=" * 60)
    print(f"  Window size:     {window_size}")
    print(f"  Num passes:      {num_passes}")
    print(f"  Steps:           {steps}")
    print(f"  Batch size:      {batch_size}")
    print(f"  LR:              {lr}")
    print(f"  D1 ratio:        {d1_ratio:.0%}")

    pad = VOCAB["<pad>"]
    bos = VOCAB["<bos>"]
    ans_marker = VOCAB["<ans>"]

    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    context_fade_start = int(steps * d1_ratio)

    # Use sentence encoder for memory (best previous results)
    encode_frozen = encode_sentence_frozen
    encode_diff = encode_sentence_differentiable
    enc_label = "sentence-boundary (bidir per sentence)"
    print(f"  Memory encoder:  {enc_label}")

    # Pre-tokenize training data (multi-token answers with teacher forcing)
    train_data = []
    eos = VOCAB["<eos>"]
    max_passage_len = 128
    for ex in train_examples:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)
        # Full sequence: <bos> <ans> question answer <eos>
        full_seq = [bos, ans_marker] + question_ids + answer_ids + [eos]
        inp_ids = full_seq[:-1]
        n_context = 2 + len(question_ids)
        tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]
        assert len(inp_ids) == len(tgt_ids)
        # Pad passage for batching
        p_ids = passage_ids[:max_passage_len]
        while len(p_ids) < max_passage_len:
            p_ids.append(pad)
        train_data.append((inp_ids, tgt_ids, p_ids))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()

    t0 = time.time()
    best_acc = 0.0
    best_step = 0
    tracker = TrainingTracker(window=50)

    # Initial memory diagnostics
    print("\n  Initial memory encoding diagnostics:")
    sample_passages = torch.tensor(
        [train_data[i][2] for i in range(min(16, len(train_data)))],
        dtype=torch.long, device=device)
    with torch.no_grad():
        model.eval()
        sample_keys, sample_vals, sample_mask = encode_frozen(
            model, sample_passages, device)
        n_valid = sample_mask.sum(dim=1).float().mean().item()
        print(f"  Avg valid slots: {n_valid:.1f} / {sample_keys.size(1)}")
        log_memory_diagnostics(sample_keys, "  Init keys")
        log_memory_diagnostics(sample_vals, "  Init vals")
        model.train()

    print(f"\n  {'─' * 56}")
    print(f"  Training started at {time.strftime('%H:%M:%S')}")
    print(f"  {'─' * 56}")

    for step in range(1, steps + 1):
        tracker.tick()

        # Sample batch
        batch_idx = random.sample(range(len(train_data)), batch_size)
        batch = [train_data[i] for i in batch_idx]

        # Collate inputs and targets (variable length, multi-token answers)
        max_seq_len = max(len(b[0]) for b in batch)
        inp_batch = []
        tgt_batch = []
        passage_batch = []

        for inp_ids, tgt_ids, p_ids in batch:
            padded_inp = inp_ids + [pad] * (max_seq_len - len(inp_ids))
            padded_tgt = tgt_ids + [pad] * (max_seq_len - len(tgt_ids))
            inp_batch.append(padded_inp)
            tgt_batch.append(padded_tgt)
            passage_batch.append(p_ids)

        q_tensor = torch.tensor(inp_batch, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
        p_tensor = torch.tensor(passage_batch, dtype=torch.long, device=device)

        # LR schedule
        lr_now = get_lr(step, 200, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Encode passage to memory
        if step >= context_fade_start:
            mem_keys, mem_vals, mem_mask = encode_diff(model, p_tensor, device)
        else:
            model.eval()
            mem_keys, mem_vals, mem_mask = encode_frozen(model, p_tensor, device)
            model.train()

        # Forward: multi-pass sliding window with memory cross-attention
        hidden = sliding_lm_encode(
            model, q_tensor, window_size, num_passes,
            mem_keys=mem_keys, mem_vals=mem_vals, mem_mask=mem_mask)

        # Compute logits at all positions, loss on answer tokens only
        logits = F.linear(hidden, model.embed.weight)  # (B, T, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            tgt_tensor.reshape(-1),
            ignore_index=pad)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm, module_norms = log_gradient_stats(model)
        tracker.add_grad_norm(grad_norm)
        tracker.add_loss(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            eta = elapsed / step * (steps - step)
            trend = tracker.loss_trend
            phase_tag = "D1" if step < context_fade_start else "D2*"
            print(f"  [{phase_tag} {step:>5}/{steps}] "
                  f"loss={loss.item():.4f} avg={tracker.avg_loss:.4f}{trend} "
                  f"gnorm={grad_norm:.2f} lr={lr_now:.1e} "
                  f"spd={tracker.steps_per_sec:.1f}it/s "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # Diagnostics every 500 steps
        if step % 500 == 0:
            print(f"\n  ┌─── Diagnostics @ step {step} ───")
            model.eval()
            with torch.no_grad():
                diag_keys, diag_vals, diag_mask = encode_frozen(
                    model, p_tensor[:4], device)
                n_valid = diag_mask[:4].sum(dim=1).float().mean().item()
                print(f"  │ Valid slots: {n_valid:.1f}/{diag_keys.size(1)}")
                log_memory_diagnostics(diag_keys, "  │ Keys")
                log_memory_diagnostics(diag_vals, "  │ Vals")
            model.train()
            top_modules = sorted(module_norms.items(), key=lambda x: -x[1])[:5]
            print(f"  │ Top grad modules: " +
                  " ".join(f"{n}={v:.3f}" for n, v in top_modules))
            print(f"  │ Loss: best={tracker.best_loss:.4f} "
                  f"avg50={tracker.avg_loss:.4f}")
            print(f"  └────────────────────\n")

        # Evaluation
        if step % eval_interval == 0 or step == steps:
            acc, breakdown = evaluate_sliding_lm(
                model, cfg, val_examples, device,
                window_size=window_size, num_passes=num_passes,
                encode_fn=encode_frozen)
            tracker.add_eval(step, acc, breakdown)

            delta = acc - tracker.eval_history[-2][1] if len(tracker.eval_history) > 1 else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1%})" if len(tracker.eval_history) > 1 else ""
            print(f"\n  ╔══ EVAL @ step {step} ══")
            print(f"  ║ Accuracy: {acc:.1%}{delta_str}  (best: {tracker.best_acc:.1%})")
            if breakdown:
                for n_facts, (corr, tot) in sorted(breakdown.items()):
                    bar = "█" * int(corr / max(tot, 1) * 20)
                    print(f"  ║   {n_facts}-fact: {corr:>3}/{tot:>3} = "
                          f"{corr/max(tot,1):.1%} {bar}")
            print(f"  ╚{'═' * 30}\n")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                save_path = os.path.join(output_dir, "best_sliding_lm.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "vocab": VOCAB,
                    "window_size": window_size,
                    "num_passes": num_passes,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

            # Early stopping: no improvement for 5 evals
            if step - best_step >= eval_interval * 5:
                print(f"\n  ⚠ Early stopping at step {step} "
                      f"(no improvement since step {best_step})")
                ckpt = torch.load(
                    os.path.join(output_dir, "best_sliding_lm.pt"),
                    map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model"])
                break

    elapsed = time.time() - t0
    print(f"  Sliding LM done in {elapsed:.0f}s, best accuracy={best_acc:.1%}")
    return best_acc


def train_phase_d(model, cfg, device, train_examples, val_examples,
                  memory_dir, steps=3000, lr=1e-4, batch_size=16,
                  max_len=128, eval_interval=200, encoder_mode='enc_dec',
                  d1_ratio=0.2):
    """
    Memory-dependent QA with encoder-decoder architecture.

    encoder_mode='enc_dec': Multi-iteration bidirectional encoder (default).
      The encoder completely consumes the passage through 3 iterations of
      bidirectional self-attention + cross-attention to previous iteration.
      No token output during encoding — pure understanding.
      The decoder then reads from encoder memory via cross-attention.

    encoder_mode='streaming': Legacy streaming chunk encoder (single-pass).

    D1: frozen encoding (teaches decoder to use memory)
    D2: differentiable encoding (encoder + decoder co-adapt)
    """
    print("\n" + "=" * 60)
    print(f"  Phase D: Memory QA (encoder={encoder_mode})")
    print("=" * 60)

    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    context_fade_start = int(steps * d1_ratio)

    # Select encoder functions based on mode
    if encoder_mode == 'enc_dec':
        encode_frozen = encode_enc_dec_frozen
        encode_diff = encode_enc_dec_differentiable
        enc_label = "enc-dec (3-iter bidir)"
    elif encoder_mode == 'sentence':
        encode_frozen = encode_sentence_frozen
        encode_diff = encode_sentence_differentiable
        enc_label = "sentence-boundary (bidir per sentence)"
    elif encoder_mode == 'sliding':
        encode_frozen = encode_sliding_frozen
        encode_diff = encode_sliding_differentiable
        enc_label = f"sliding window (win={chunk_size}, stride={_SLIDING_STRIDE})"
    else:
        encode_frozen = lambda m, p, d, **kw: encode_kv_memory_chunked(
            m, p, d, chunk_size, slots_per_chunk)
        encode_diff = lambda m, p, d, **kw: encode_kv_memory_chunked_differentiable(
            m, p, d, chunk_size, slots_per_chunk)
        enc_label = f"streaming (chunk={chunk_size})"

    print(f"  Steps:          {steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  LR:             {lr}")
    print(f"  D1 (frozen):    steps 1-{context_fade_start}")
    print(f"  D2 (diff):      steps {context_fade_start+1}-{steps}")
    print(f"  Eval every:     {eval_interval} steps")
    print(f"  Encoder:        {enc_label}")
    print(f"  d_model:        {d_model}")

    train_ds = MemoryQADataset(train_examples, max_len=max_len)
    val_ds = MemoryQADataset(val_examples, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    model.train()
    loader_iter = iter(train_loader)
    t0 = time.time()
    best_acc = 0.0
    best_step = 0
    tracker = TrainingTracker(window=50)

    # Initial memory diagnostics
    print("\n  Initial memory encoding diagnostics:")
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        model.eval()
        sample_keys, sample_vals, sample_mask = encode_frozen(
            model, sample_batch[2].to(device), device)
        n_valid = sample_mask.sum(dim=1).float().mean().item()
        print(f"  Avg valid slots: {n_valid:.1f} / {sample_keys.size(1)}")
        log_memory_diagnostics(sample_keys, "  Init keys")
        log_memory_diagnostics(sample_vals, "  Init vals")
        model.train()
    del sample_batch

    print(f"\n  {'─' * 56}")
    print(f"  Training started at {time.strftime('%H:%M:%S')}")
    print(f"  {'─' * 56}")

    for step in range(1, steps + 1):
        tracker.tick()

        try:
            inp, tgt, passages = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inp, tgt, passages = next(loader_iter)

        inp, tgt, passages = inp.to(device), tgt.to(device), passages.to(device)

        lr_now = get_lr(step, 200, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # D1: frozen encoding (teaches decoder to read from memory)
        # D2: differentiable encoding (encoder + decoder co-adapt end-to-end)
        if step >= context_fade_start:
            mem_keys, mem_vals, mem_mask = encode_diff(
                model, passages, device)
        else:
            model.eval()
            mem_keys, mem_vals, mem_mask = encode_frozen(
                model, passages, device)
            model.train()

        # Memory-only QA (no context passage in input)
        inp_d = inp
        tgt_d = tgt

        # Decoder forward
        logits, halt_logits, hidden = model(
            inp_d, memory_keys=mem_keys, memory_values=mem_vals,
            memory_mask=mem_mask, return_hidden=True
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_d.reshape(-1),
            ignore_index=VOCAB["<pad>"],
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Track gradient norms
        grad_norm, module_norms = log_gradient_stats(model)
        tracker.add_grad_norm(grad_norm)
        tracker.add_loss(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── Logging ──
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            eta = elapsed / step * (steps - step) if step > 0 else 0
            phase_tag = "D1" if step < context_fade_start else "D2*"
            trend = tracker.loss_trend

            print(f"  [{phase_tag} {step:>5}/{steps}] "
                  f"loss={loss.item():.4f} avg={tracker.avg_loss:.4f}{trend} "
                  f"gnorm={grad_norm:.2f} "
                  f"lr={lr_now:.1e} "
                  f"spd={tracker.steps_per_sec:.1f}it/s "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # Detailed diagnostics every 250 steps
        if step % 250 == 0:
            print(f"\n  ┌─── Diagnostics @ step {step} ───")
            model.eval()
            with torch.no_grad():
                diag_keys, diag_vals, diag_mask = encode_frozen(
                    model, passages[:4], device)
                n_valid = diag_mask[:4].sum(dim=1).float().mean().item()
                print(f"  │ Valid slots: {n_valid:.1f}/{diag_keys.size(1)}")
                log_memory_diagnostics(diag_keys, "  │ Keys")
                log_memory_diagnostics(diag_vals, "  │ Vals")
            model.train()
            # Gradient breakdown
            top_modules = sorted(module_norms.items(), key=lambda x: -x[1])[:5]
            print(f"  │ Top grad modules: " +
                  " ".join(f"{n}={v:.3f}" for n, v in top_modules))
            # Loss statistics
            print(f"  │ Loss: best={tracker.best_loss:.4f} "
                  f"avg50={tracker.avg_loss:.4f} "
                  f"last={loss.item():.4f}")
            if tracker.eval_history:
                last_step, last_acc, _ = tracker.eval_history[-1]
                print(f"  │ Last eval: step={last_step} acc={last_acc:.1%} "
                      f"best={tracker.best_acc:.1%}")
            print(f"  └────────────────────\n")

        # Evaluation
        if step % eval_interval == 0 or step == steps:
            acc, breakdown = evaluate_memory_qa(
                model, cfg, val_examples, device, max_examples=200,
                return_breakdown=True, encoder_mode=encoder_mode)
            tracker.add_eval(step, acc, breakdown)

            # Show detailed eval results
            delta = acc - tracker.eval_history[-2][1] if len(tracker.eval_history) > 1 else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1%})" if len(tracker.eval_history) > 1 else ""
            print(f"\n  ╔══ EVAL @ step {step} ══")
            print(f"  ║ Accuracy: {acc:.1%}{delta_str}  (best: {tracker.best_acc:.1%})")
            if breakdown:
                for n_facts, (corr, tot) in sorted(breakdown.items()):
                    bar = "█" * int(corr / max(tot, 1) * 20)
                    print(f"  ║   {n_facts}-fact: {corr:>3}/{tot:>3} = "
                          f"{corr/max(tot,1):.1%} {bar}")
            print(f"  ╚{'═' * 30}\n")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                save_path = os.path.join(memory_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "vocab": VOCAB,
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

            # Early stopping: no improvement for 5 evals (1000 steps)
            if step > context_fade_start and step - best_step >= eval_interval * 5:
                print(f"\n  ⚠ Early stopping at step {step} (no improvement since step {best_step})")
                # Reload best model
                ckpt = torch.load(os.path.join(memory_dir, "best_model.pt"),
                                  map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model"])
                break

    elapsed = time.time() - t0
    print(f"  Phase D done in {elapsed:.0f}s, best accuracy={best_acc:.1%}")
    return best_acc


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_memory_qa(model, cfg, examples, device, max_examples=200,
                       return_breakdown=False, encoder_mode='enc_dec'):
    """
    Evaluate memory recall with specified encoder.
    encoder_mode='enc_dec': multi-iteration bidirectional encoder
    encoder_mode='streaming': legacy chunked streaming encoder
    """
    model.eval()
    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    pad_id = VOCAB["<pad>"]

    correct = 0
    total = 0
    type_correct = Counter()
    type_total = Counter()

    for i, ex in enumerate(examples[:max_examples]):
        passage_ids = tokenize(ex.passage)

        p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)

        if encoder_mode == 'enc_dec':
            mem_keys, mem_vals, mem_mask = encode_enc_dec_frozen(
                model, p_tensor, device)
        elif encoder_mode == 'sentence':
            mem_keys, mem_vals, mem_mask = encode_sentence_frozen(
                model, p_tensor, device)
        elif encoder_mode == 'sliding':
            mem_keys, mem_vals, mem_mask = encode_sliding_frozen(
                model, p_tensor, device)
        else:
            mem_keys, mem_vals, mem_mask = encode_kv_memory_chunked(
                model, p_tensor, device, chunk_size, slots_per_chunk)

        # Build question input with answer tokens (teacher-forced evaluation)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids + answer_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        logits, _, hidden = model(inp, memory_keys=mem_keys,
                                   memory_values=mem_vals,
                                   memory_mask=mem_mask,
                                   return_hidden=True)

        # Check predictions at each answer position (teacher-forced)
        n_context = 2 + len(question_ids)
        is_correct = True
        for j, expected_byte in enumerate(answer_ids):
            pred_pos = n_context - 1 + j
            if logits[0, pred_pos, :].argmax().item() != expected_byte:
                is_correct = False
                break

        if is_correct:
            correct += 1
        total += 1

        n_facts = len(ex.facts)
        type_total[n_facts] += 1
        if is_correct:
            type_correct[n_facts] += 1

    accuracy = correct / max(total, 1)

    for n_facts in sorted(type_total.keys()):
        t_corr = type_correct[n_facts]
        t_tot = type_total[n_facts]
        print(f"    {n_facts}-fact: {t_corr}/{t_tot} = {t_corr/max(t_tot,1):.1%}")

    model.train()

    if return_breakdown:
        breakdown = {k: (type_correct[k], type_total[k]) for k in type_total}
        return accuracy, breakdown
    return accuracy


@torch.no_grad()
def evaluate_context_qa(model, cfg, examples, device, max_examples=200):
    """Evaluate with passage in context (no memory needed). Baseline test."""
    model.eval()
    correct = 0
    total = 0

    for ex in examples[:max_examples]:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)

        bos = VOCAB["<bos>"]
        ans_marker = VOCAB["<ans>"]
        inp_ids = [bos] + passage_ids + [ans_marker] + question_ids + answer_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        logits, _ = model(inp)

        # Teacher-forced multi-token comparison
        n_context = 1 + len(passage_ids) + 1 + len(question_ids)
        is_correct = True
        for j, expected_byte in enumerate(answer_ids):
            pred_pos = n_context - 1 + j
            if logits[0, pred_pos, :].argmax().item() != expected_byte:
                is_correct = False
                break

        if is_correct:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    model.train()
    return accuracy


def detailed_eval(model, cfg, examples, device, n=10, encoder_mode='enc_dec'):
    """Print detailed examples with specified encoder."""
    model.eval()
    d_model = cfg.d_model
    chunk_size = getattr(cfg, 'chunk_size', 8)
    slots_per_chunk = getattr(cfg, 'slots_per_chunk', 2)
    print("\n" + "-" * 60)
    print("  Detailed Examples")
    print("-" * 60)

    for i, ex in enumerate(examples[:n]):
        passage_ids = tokenize(ex.passage)

        p_tensor = torch.tensor([passage_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            if encoder_mode == 'enc_dec':
                mem_keys, mem_vals, mem_mask = encode_enc_dec_frozen(
                    model, p_tensor, device)
            elif encoder_mode == 'sentence':
                mem_keys, mem_vals, mem_mask = encode_sentence_frozen(
                    model, p_tensor, device)
            elif encoder_mode == 'sliding':
                mem_keys, mem_vals, mem_mask = encode_sliding_frozen(
                    model, p_tensor, device)
            else:
                mem_keys, mem_vals, mem_mask = encode_kv_memory_chunked(
                    model, p_tensor, device, chunk_size, slots_per_chunk)

        # Question input with answer tokens (teacher-forced)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)
        inp_ids = [VOCAB["<bos>"], VOCAB["<ans>"]] + question_ids + answer_ids
        inp = torch.tensor([inp_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_mem, _, hidden = model(inp, memory_keys=mem_keys,
                                           memory_values=mem_vals,
                                           memory_mask=mem_mask,
                                           return_hidden=True)
            n_valid = mem_mask[0].sum().item()
            avg_norm = mem_keys[0, :n_valid].norm(dim=-1).mean().item() if n_valid > 0 else 0.0
            # Without memory
            logits_no_mem, _ = model(inp)

        # Decode predicted answer bytes (teacher-forced)
        n_context = 2 + len(question_ids)
        pred_mem_ids = [logits_mem[0, n_context - 1 + j, :].argmax().item()
                        for j in range(len(answer_ids))]
        pred_no_mem_ids = [logits_no_mem[0, n_context - 1 + j, :].argmax().item()
                           for j in range(len(answer_ids))]
        pred_mem_str = detokenize(pred_mem_ids)
        pred_no_mem_str = detokenize(pred_no_mem_ids)

        # Top-5 for first answer byte position
        top5_vals, top5_ids = logits_mem[0, n_context - 1, :].topk(5)
        top5_probs = F.softmax(top5_vals, dim=0)
        top5_tokens = [(ID2WORD.get(idx.item(), "?"), f"{p.item():.3f}")
                       for idx, p in zip(top5_ids, top5_probs)]

        expected = ex.answer
        mark = "✓" if pred_mem_str == expected else "✗"

        print(f"\n  {mark} Example {i+1}:")
        print(f"    Passage:  {ex.passage}")
        print(f"    Question: {ex.question}")
        print(f"    Expected: {expected}")
        print(f"    With mem: {pred_mem_str} | No mem: {pred_no_mem_str}")
        print(f"    Top-5 (1st byte): {top5_tokens}")
        print(f"    Mem: {n_valid} slots, avg_norm={avg_norm:.2f}")

    model.train()


# ============================================================================
# Multi-Task Training: LM (shell + wiki) + QA (bAbI)
# ============================================================================

def train_multitask(model, cfg, device, train_examples, val_examples,
                    output_dir, steps=5000, lr=1e-4, batch_size=32,
                    eval_interval=200, window_size=16, num_passes=4,
                    lm_weight=0.5, qa_weight=0.5,
                    n_shell=5000, n_wiki=5000):
    """
    Multi-task training combining:
      1. LM loss on diverse text (shell commands + Wikipedia) via sliding window
      2. QA loss on bAbI via sliding window + memory cross-attention

    The LM task teaches general language modeling / byte patterns.
    The QA task teaches memory retrieval and reasoning.
    Both use the same sliding window architecture.
    """
    print("\n" + "=" * 60)
    print("  Multi-Task Training: LM + Memory QA")
    print("=" * 60)
    print(f"  Window size:     {window_size}")
    print(f"  Num passes:      {num_passes}")
    print(f"  Steps:           {steps}")
    print(f"  LM weight:       {lm_weight}")
    print(f"  QA weight:       {qa_weight}")
    print(f"  Shell examples:  {n_shell}")
    print(f"  Wiki sentences:  {n_wiki}")

    pad = VOCAB["<pad>"]
    bos = VOCAB["<bos>"]
    eos = VOCAB["<eos>"]
    ans_marker = VOCAB["<ans>"]
    d_model = cfg.d_model

    # ── Load diverse text data ──
    print("\n  Loading diverse text data...")
    shell_texts = generate_shell_texts(n_shell, seed=42)
    print(f"    Shell commands: {len(shell_texts)}")
    wiki_texts = load_wikipedia_sentences(n_wiki, seed=42)
    print(f"    Wiki sentences: {len(wiki_texts)}")

    all_lm_texts = shell_texts + wiki_texts
    random.shuffle(all_lm_texts)
    lm_ds = TextLMDataset(all_lm_texts, max_len=cfg.max_seq_len)
    lm_loader = DataLoader(lm_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    lm_iter = iter(lm_loader)
    print(f"    LM dataset: {len(lm_ds)} samples")

    # ── Pre-tokenize QA data ──
    encode_frozen = encode_sentence_frozen
    encode_diff = encode_sentence_differentiable
    max_passage_len = 128
    qa_data = []
    for ex in train_examples:
        passage_ids = tokenize(ex.passage)
        question_ids = tokenize(ex.question)
        answer_ids = tokenize(ex.answer)
        full_seq = [bos, ans_marker] + question_ids + answer_ids + [eos]
        inp_ids = full_seq[:-1]
        n_context = 2 + len(question_ids)
        tgt_ids = [pad] * (n_context - 1) + answer_ids + [eos]
        assert len(inp_ids) == len(tgt_ids)
        p_ids = passage_ids[:max_passage_len]
        while len(p_ids) < max_passage_len:
            p_ids.append(pad)
        qa_data.append((inp_ids, tgt_ids, p_ids))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.train()

    t0 = time.time()
    best_acc = 0.0
    best_step = 0
    tracker = TrainingTracker(window=50)
    context_fade_start = int(steps * 0.2)

    print(f"\n  {'─' * 56}")
    print(f"  Training started at {time.strftime('%H:%M:%S')}")
    print(f"  D1 (frozen enc): steps 1-{context_fade_start}")
    print(f"  D2 (co-adapt):   steps {context_fade_start+1}-{steps}")
    print(f"  {'─' * 56}")

    for step in range(1, steps + 1):
        tracker.tick()

        lr_now = get_lr(step, 200, steps, lr, lr * 0.01)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        total_loss = torch.tensor(0.0, device=device)

        # ── LM loss on diverse text (causal forward, no sliding window) ──
        try:
            lm_inp, lm_tgt = next(lm_iter)
        except StopIteration:
            lm_iter = iter(lm_loader)
            lm_inp, lm_tgt = next(lm_iter)
        lm_inp, lm_tgt = lm_inp.to(device), lm_tgt.to(device)

        lm_logits, _ = model(token_ids=lm_inp)
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, VOCAB_SIZE),
            lm_tgt.reshape(-1),
            ignore_index=pad)
        total_loss = total_loss + lm_weight * lm_loss

        # ── QA loss with memory ──
        batch_idx = random.sample(range(len(qa_data)), batch_size)
        batch = [qa_data[i] for i in batch_idx]

        max_seq_len = max(len(b[0]) for b in batch)
        inp_batch, tgt_batch, passage_batch = [], [], []
        for inp_ids, tgt_ids, p_ids in batch:
            inp_batch.append(inp_ids + [pad] * (max_seq_len - len(inp_ids)))
            tgt_batch.append(tgt_ids + [pad] * (max_seq_len - len(tgt_ids)))
            passage_batch.append(p_ids)

        q_tensor = torch.tensor(inp_batch, dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(tgt_batch, dtype=torch.long, device=device)
        p_tensor = torch.tensor(passage_batch, dtype=torch.long, device=device)

        if step >= context_fade_start:
            mem_keys, mem_vals, mem_mask = encode_diff(model, p_tensor, device)
        else:
            model.eval()
            mem_keys, mem_vals, mem_mask = encode_frozen(model, p_tensor, device)
            model.train()

        qa_hidden = sliding_lm_encode(
            model, q_tensor, window_size, num_passes,
            mem_keys=mem_keys, mem_vals=mem_vals, mem_mask=mem_mask)
        qa_logits = F.linear(qa_hidden, model.embed.weight)
        qa_loss = F.cross_entropy(
            qa_logits.reshape(-1, VOCAB_SIZE),
            tgt_tensor.reshape(-1),
            ignore_index=pad)
        total_loss = total_loss + qa_weight * qa_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm, module_norms = log_gradient_stats(model)
        tracker.add_loss(total_loss.item())
        tracker.add_grad_norm(grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            eta = elapsed / step * (steps - step)
            phase_tag = "D1" if step < context_fade_start else "D2"
            print(f"  [{phase_tag} {step:>5}/{steps}] "
                  f"loss={total_loss.item():.4f} "
                  f"lm={lm_loss.item():.4f} qa={qa_loss.item():.4f} "
                  f"gnorm={grad_norm:.2f} lr={lr_now:.1e} "
                  f"spd={tracker.steps_per_sec:.1f}it/s "
                  f"[{elapsed:.0f}s / ETA {eta:.0f}s]")

        # Evaluation (QA accuracy)
        if step % eval_interval == 0 or step == steps:
            acc, breakdown = evaluate_sliding_lm(
                model, cfg, val_examples, device,
                window_size=window_size, num_passes=num_passes,
                encode_fn=encode_frozen)
            tracker.add_eval(step, acc, breakdown)

            delta = acc - tracker.eval_history[-2][1] if len(tracker.eval_history) > 1 else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.1%})" if len(tracker.eval_history) > 1 else ""
            print(f"\n  ╔══ EVAL @ step {step} ══")
            print(f"  ║ QA Accuracy: {acc:.1%}{delta_str}  (best: {tracker.best_acc:.1%})")
            print(f"  ║ LM loss: {lm_loss.item():.4f}  QA loss: {qa_loss.item():.4f}")
            if breakdown:
                for n_facts, (corr, tot) in sorted(breakdown.items()):
                    bar = "█" * int(corr / max(tot, 1) * 20)
                    print(f"  ║   {n_facts}-fact: {corr:>3}/{tot:>3} = "
                          f"{corr/max(tot,1):.1%} {bar}")
            print(f"  ╚{'═' * 30}\n")

            if acc > best_acc:
                best_acc = acc
                best_step = step
                save_path = os.path.join(output_dir, "best_multitask.pt")
                torch.save({
                    "model": model.state_dict(),
                    "step": step,
                    "accuracy": acc,
                    "vocab": VOCAB,
                    "window_size": window_size,
                    "num_passes": num_passes,
                    "mode": "multitask",
                }, save_path)
                print(f"  ✓ New best! Saved to {save_path}")

            if step - best_step >= eval_interval * 5:
                print(f"\n  ⚠ Early stopping at step {step} "
                      f"(no improvement since step {best_step})")
                ckpt = torch.load(
                    os.path.join(output_dir, "best_multitask.pt"),
                    map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model"])
                break

    elapsed = time.time() - t0
    print(f"  Multi-task done in {elapsed:.0f}s, best QA accuracy={best_acc:.1%}")
    return best_acc


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Micro Prototype Training")
    parser.add_argument("--device", default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument("--output_dir", default="./checkpoints/micro",
                        help="Where to save checkpoints")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation")
    parser.add_argument("--n_train", type=int, default=20000,
                        help="Number of training examples")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation examples")
    parser.add_argument("--phase_a_steps", type=int, default=500)
    parser.add_argument("--phase_b_steps", type=int, default=300)
    parser.add_argument("--phase_c_steps", type=int, default=500)
    parser.add_argument("--phase_d_steps", type=int, default=3000)
    parser.add_argument("--encoder_mode", default="sentence",
                        choices=["enc_dec", "streaming", "sentence", "sliding", "sliding_lm"],
                        help="Memory encoder, or 'sliding_lm' for encoder-only (no decoder/memory)")
    parser.add_argument("--num_passes", type=int, default=4,
                        help="Number of diffusion passes for sliding_lm mode")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Override chunk_size / window_size for encoder")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for sliding window encoder (1=token-by-token)")
    parser.add_argument("--memory_topk", type=int, default=0,
                        help="Top-K sparse cross-attention (0=softmax, >0=top-k with STE)")
    parser.add_argument("--memory_hops", type=int, default=1,
                        help="Cross-attention hops (1=standard, 2=multi-hop entity→attribute)")
    parser.add_argument("--d1_ratio", type=float, default=0.2,
                        help="Fraction of Phase D steps for D1 (frozen encoder)")
    parser.add_argument("--multitask", action="store_true",
                        help="Multi-task training: LM (shell+wiki) + QA (bAbI)")
    parser.add_argument("--n_shell", type=int, default=5000,
                        help="Number of shell command examples for multi-task LM")
    parser.add_argument("--n_wiki", type=int, default=5000,
                        help="Number of Wikipedia sentences for multi-task LM")
    parser.add_argument("--lm_weight", type=float, default=0.5,
                        help="Weight for LM loss in multi-task training")
    parser.add_argument("--qa_weight", type=float, default=0.5,
                        help="Weight for QA loss in multi-task training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("  MICRO PROTOTYPE — Memory Recall Experiment")
    print("=" * 60)

    # Set sliding window stride from CLI
    global _SLIDING_STRIDE
    _SLIDING_STRIDE = args.stride

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig()
    cfg.vocab_size = VOCAB_SIZE  # override with actual vocab size
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.memory_topk > 0:
        cfg.memory_topk = args.memory_topk
    if args.memory_hops > 1:
        cfg.memory_hops = args.memory_hops
    print(f"  Device:     {device}")
    print(f"  Vocab:      {VOCAB_SIZE} tokens (byte-level UTF-8)")

    model = LoopedLatentController(cfg, use_checkpoint=False).to(device)
    n_params = count_params(model)
    log_model_summary(model, cfg)

    # Generate data
    print(f"\n  Generating {args.n_train} train + {args.n_val} val examples...")
    train_examples = generate_dataset(args.n_train, seed=args.seed)
    val_examples = generate_dataset(args.n_val, seed=args.seed + 1)

    # Show data stats
    type_counts = Counter()
    for ex in train_examples[:1000]:
        type_counts[len(ex.facts)] += 1
    print(f"  Type distribution (first 1000): {dict(type_counts)}")
    print(f"  Example: {train_examples[0].passage}")
    print(f"           {train_examples[0].question} → {train_examples[0].answer}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save vocab
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(VOCAB, f, indent=2)

    if args.eval_only:
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  ERROR: No checkpoint at {ckpt_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint (step={ckpt['step']}, acc={ckpt['accuracy']:.1%})")

        print("\n  Context QA (baseline, no memory needed):")
        ctx_acc = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc:.1%}")

        print("\n  Memory QA (passage from memory only):")
        mem_acc = evaluate_memory_qa(model, cfg, val_examples, device)
        print(f"  → {mem_acc:.1%}")

        detailed_eval(model, cfg, val_examples, device, n=10)
        return

    # ===== Full Training Pipeline =====
    t_start = time.time()

    if args.multitask:
        # ── Multi-task: LM (shell + wiki) + QA (bAbI) ──
        window_size = cfg.chunk_size if args.chunk_size is not None else 16
        num_passes = args.num_passes

        print("\n" + "─" * 60)
        print(f"  Pipeline: A({args.phase_a_steps}) → MultiTask({args.phase_d_steps})")
        print("─" * 60)

        # Phase A: LM warmup
        loss_a = train_phase_a(model, cfg, device, train_examples,
                               steps=args.phase_a_steps, batch_size=64)

        print("\n  Context QA after Phase A:")
        ctx_acc_a = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc_a:.1%}")

        # Multi-task training
        mt_dir = os.path.join(args.output_dir, "multitask")
        os.makedirs(mt_dir, exist_ok=True)
        best_acc = train_multitask(
            model, cfg, device, train_examples, val_examples, mt_dir,
            steps=args.phase_d_steps, lr=1e-4, batch_size=32,
            eval_interval=200,
            window_size=window_size, num_passes=num_passes,
            lm_weight=args.lm_weight, qa_weight=args.qa_weight,
            n_shell=args.n_shell, n_wiki=args.n_wiki,
        )

        total_time = time.time() - t_start
        print("\n" + "=" * 60)
        print("  FINAL REPORT — Multi-Task LM + Memory QA")
        print("=" * 60)
        print(f"  Total time:      {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Parameters:      {n_params:,}")
        print(f"  Window/passes:   W={window_size} P={num_passes}")
        print(f"  Weights:         LM={args.lm_weight} QA={args.qa_weight}")
        print(f"  Data:            {args.n_shell} shell + {args.n_wiki} wiki")
        print(f"  Context QA:      {ctx_acc_a:.1%} (after Phase A)")
        print(f"  Multi-task QA:   {best_acc:.1%}")

        report = {
            "total_time_s": total_time,
            "n_params": n_params,
            "mode": "multitask",
            "window_size": window_size,
            "num_passes": num_passes,
            "lm_weight": args.lm_weight,
            "qa_weight": args.qa_weight,
            "n_shell": args.n_shell,
            "n_wiki": args.n_wiki,
            "context_qa_acc": ctx_acc_a,
            "multitask_qa_acc": best_acc,
            "config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "vocab_size": VOCAB_SIZE,
            },
            "device": device,
        }
    elif args.encoder_mode == 'sliding_lm':
        # ── Sliding window LM with memory cross-attention ──
        window_size = cfg.chunk_size if args.chunk_size is not None else 5
        num_passes = args.num_passes

        print("\n" + "─" * 60)
        print(f"  Pipeline: A({args.phase_a_steps}) → SlidingLM({args.phase_d_steps})")
        print("─" * 60)

        # Phase A: LM warmup (teaches embeddings + basic attention)
        loss_a = train_phase_a(model, cfg, device, train_examples,
                               steps=args.phase_a_steps, batch_size=64)

        print("\n  Context QA after Phase A:")
        ctx_acc_a = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc_a:.1%}")

        # Sliding LM training with memory (replaces B/C/D)
        slm_dir = os.path.join(args.output_dir, "sliding_lm")
        os.makedirs(slm_dir, exist_ok=True)
        best_acc = train_sliding_lm(
            model, cfg, device, train_examples, val_examples, slm_dir,
            steps=args.phase_d_steps, lr=1e-4, batch_size=32,
            eval_interval=200,
            window_size=window_size, num_passes=num_passes,
            d1_ratio=args.d1_ratio,
        )

        # Final report
        total_time = time.time() - t_start
        print("\n" + "=" * 60)
        print("  FINAL REPORT — Sliding Window LM + Memory")
        print("=" * 60)
        print(f"  Total time:      {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Parameters:      {n_params:,}")
        print(f"  Window size:     {window_size}")
        print(f"  Num passes:      {num_passes}")
        print(f"  Context QA:      {ctx_acc_a:.1%} (after Phase A)")
        print(f"  Sliding LM QA:   {best_acc:.1%}")
        print(f"  Target:          >80% QA accuracy")

        report = {
            "total_time_s": total_time,
            "n_params": n_params,
            "mode": "sliding_lm",
            "window_size": window_size,
            "num_passes": num_passes,
            "context_qa_acc": ctx_acc_a,
            "sliding_lm_acc": best_acc,
            "config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "vocab_size": VOCAB_SIZE,
            },
            "device": device,
        }
    else:
        # ── Standard encoder-decoder pipeline ──
        print("\n" + "─" * 60)
        print(f"  Pipeline: A({args.phase_a_steps}) → B({args.phase_b_steps}) "
              f"→ C({args.phase_c_steps}) → D({args.phase_d_steps})")
        print("─" * 60)

        # Phase A: Warmup LM
        loss_a = train_phase_a(model, cfg, device, train_examples,
                               steps=args.phase_a_steps, batch_size=64)

        # Quick context QA check
        print("\n  Context QA after Phase A:")
        ctx_acc_a = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc_a:.1%}")

        # Phase B: Address heads
        train_phase_b(model, cfg, device, train_examples,
                      steps=args.phase_b_steps, batch_size=128)

        # Phase C: ACT curriculum
        train_phase_c(model, cfg, device, train_examples,
                      steps=args.phase_c_steps, batch_size=64)

        # Quick context QA check
        print("\n  Context QA after Phase C:")
        ctx_acc_c = evaluate_context_qa(model, cfg, val_examples, device)
        print(f"  → {ctx_acc_c:.1%}")

        # Memory QA baseline (before Phase D)
        print("\n  Memory QA BEFORE Phase D (random baseline):")
        pre_d_acc = evaluate_memory_qa(model, cfg, val_examples, device,
                                       max_examples=200,
                                       encoder_mode=args.encoder_mode)
        print(f"  → {pre_d_acc:.1%}")

        # Phase D: Memory QA (the real test)
        mem_dir = os.path.join(args.output_dir, "memory")
        os.makedirs(mem_dir, exist_ok=True)
        best_acc = train_phase_d(
            model, cfg, device, train_examples, val_examples, mem_dir,
            steps=args.phase_d_steps, lr=1e-4, batch_size=32,
            eval_interval=200, encoder_mode=args.encoder_mode,
            d1_ratio=args.d1_ratio,
        )

        # Final report
        total_time = time.time() - t_start
        print("\n" + "=" * 60)
        print("  FINAL REPORT")
        print("=" * 60)
        print(f"  Total time:      {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Parameters:      {n_params:,}")
        print(f"  Context QA:      {ctx_acc_c:.1%} (passage in context)")
        print(f"  Memory QA:       {best_acc:.1%} (passage from memory)")
        print(f"  Target:          >80% memory QA")
        print(f"  ACT fix:         halt bias [0,0] (50/50 init)")

        # Detailed eval
        detailed_eval(model, cfg, val_examples, device, n=10,
                      encoder_mode=args.encoder_mode)

        report = {
            "total_time_s": total_time,
            "n_params": n_params,
            "context_qa_acc": ctx_acc_c,
            "memory_qa_acc": best_acc,
            "phases": {
                "A": {"steps": args.phase_a_steps, "final_loss": loss_a},
                "B": {"steps": args.phase_b_steps},
                "C": {"steps": args.phase_c_steps, "context_qa": ctx_acc_c},
                "D": {"steps": args.phase_d_steps, "best_acc": best_acc},
            },
            "config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "vocab_size": VOCAB_SIZE,
            },
            "device": device,
        }

    report_path = os.path.join(args.output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
