"""
ANT — Data loading, tokenization, and dataset generation.

Byte tokenizer, bAbI QA, shell commands, Wikipedia, HF chat,
source provenance tags, and torch Dataset/collate utilities.
"""

import os
import random
import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


# ============================================================================
# Byte Tokenizer — token ID = raw byte value
# ============================================================================

VOCAB = {
    "<pad>": 0x00, "<soh>": 0x01, "<bos>": 0x02, "<eos>": 0x03,
    "<eot>": 0x04, "<ans>": 0x05, "<noop>": 0x06, "<unk>": 0x1A,
    ".": ord("."), " ": ord(" "),
}
for i in range(256):
    c = chr(i) if 32 <= i < 127 else f"0x{i:02x}"
    VOCAB.setdefault(c, i)

VOCAB_SIZE = 256
ID2WORD = {v: k for k, v in VOCAB.items()}
PAD_ID = VOCAB["<pad>"]
BOS_ID = VOCAB["<bos>"]
EOS_ID = VOCAB["<eos>"]
ANS_ID = VOCAB["<ans>"]
SOH_ID = VOCAB["<soh>"]
EOT_ID = VOCAB["<eot>"]
NOOP_ID = VOCAB["<noop>"]


def tokenize(text: str) -> list[int]:
    """Encode text as raw UTF-8 bytes. Token ID = byte value."""
    return list(text.encode("utf-8"))


def detokenize(ids: list[int]) -> str:
    """Decode token IDs back to text. Skips special control tokens."""
    skip = {PAD_ID, BOS_ID, EOS_ID, SOH_ID, EOT_ID, ANS_ID, NOOP_ID}
    raw = bytes(b for b in ids if b not in skip and 0 <= b < 256)
    return raw.decode("utf-8", errors="replace")


# ============================================================================
# Source Provenance Tags
# ============================================================================

_TAG_REGISTRY = {
    "localhost": {
        "users": ["alice", "bob", "charlie", "diana", "frank",
                  "root", "admin", "deploy", "cron", "user1",
                  "syslog", "app", "auth", "kern"],
        "paths": ["~", "chat", "home/user", "home/project",
                  "tmp", "var/run", "opt/app", "var/log"],
    },
    "server1": {
        "users": ["root", "admin", "deploy", "cron", "app",
                  "syslog", "auth", "kern"],
        "paths": ["home/deploy", "var/www", "opt/service",
                  "etc/nginx", "var/log"],
    },
    "wiki": {
        "users": ["article", "editor1", "editor2", "bot"],
        "paths": ["history", "science", "geography", "people", "culture",
                  "technology", "nature", "politics", "mathematics", "art"],
    },
    "news": {
        "users": ["reuters", "ap", "bbc", "cnn"],
        "paths": ["world", "tech", "science", "politics", "sports"],
    },
    "cam1": {
        "users": ["sensor", "feed"],
        "paths": ["entrance", "lobby", "parking", "hallway"],
    },
}

_DOMAIN_MAP = {
    "shell": {"hosts": ["localhost", "server1"],
              "users": ["root", "admin", "deploy", "cron", "user1"]},
    "wiki":  {"hosts": ["wiki"],
              "users": ["article", "editor1", "editor2", "bot"]},
    "news":  {"hosts": ["news"],
              "users": ["reuters", "ap", "bbc", "cnn"]},
    "social": {"hosts": ["localhost"],
               "users": ["alice", "bob", "charlie", "diana", "frank"]},
    "observer": {"hosts": ["cam1"],
                 "users": ["sensor", "feed"]},
    "log":   {"hosts": ["localhost", "server1"],
              "users": ["syslog", "app", "auth", "kern"]},
}


def _random_timestamp() -> str:
    y = random.choice([2025, 2026])
    return (f"{y}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            f"T{random.randint(0,23):02d}:{random.randint(0,59):02d}"
            f":{random.randint(0,59):02d}Z")


def _random_tag(domain: str = None, path: str = None) -> str:
    if domain and domain in _DOMAIN_MAP:
        info = _DOMAIN_MAP[domain]
        host = random.choice(info["hosts"])
        user = random.choice(info["users"])
    else:
        host = random.choice(list(_TAG_REGISTRY.keys()))
        user = random.choice(_TAG_REGISTRY[host]["users"])
    if path is None:
        host_paths = _TAG_REGISTRY.get(host, {}).get("paths", ["~"])
        path = random.choice(host_paths)
    return f"{host}/{user}/{path}@{_random_timestamp()}"


def tag_passage(passage: str, domain: str = None) -> str:
    parts = [p.strip() for p in passage.split(".") if p.strip()]
    return "\n".join(f"{_random_tag(domain)}: {p} ." for p in parts)


def tag_text(text: str, domain: str = None, path: str = None) -> str:
    return f"{_random_tag(domain, path)}: {text}"


# ============================================================================
# bAbI-style QA Generator
# ============================================================================

NAMES = ["mary", "john", "sandra", "daniel", "emma", "oliver", "sophia",
         "james", "lucas", "mia", "noah", "lily", "liam", "ava", "ethan"]
LOCATIONS = ["garden", "kitchen", "bedroom", "bathroom", "hallway",
             "office", "park", "school", "library", "store"]
VERBS = ["went", "moved", "travelled", "ran", "walked"]


@dataclass
class QAExample:
    passage: str
    question: str
    answer: str
    answer_entity: str
    facts: dict
    valid_answers: list = None


def generate_single_fact() -> QAExample:
    n, l, v = random.choice(NAMES), random.choice(LOCATIONS), random.choice(VERBS)
    return QAExample(f"{n} {v} to the {l} .", f"where is {n} ?", l, n, {n: l})


def generate_two_facts() -> QAExample:
    ns = random.sample(NAMES, 2)
    ls = random.sample(LOCATIONS, 2)
    v1, v2 = random.choice(VERBS), random.choice(VERBS)
    passage = f"{ns[0]} {v1} to the {ls[0]} . {ns[1]} {v2} to the {ls[1]} ."
    t = random.randint(0, 1)
    return QAExample(passage, f"where is {ns[t]} ?", ls[t], ns[t], dict(zip(ns, ls)))


def generate_three_facts() -> QAExample:
    ns = random.sample(NAMES, 3)
    ls = random.sample(LOCATIONS, 3)
    parts = [f"{n} {random.choice(VERBS)} to the {l} ." for n, l in zip(ns, ls)]
    t = random.randint(0, 2)
    return QAExample(" ".join(parts), f"where is {ns[t]} ?", ls[t], ns[t],
                     dict(zip(ns, ls)))


def generate_temporal() -> QAExample:
    n = random.choice(NAMES)
    l1, l2 = random.sample(LOCATIONS, 2)
    v1, v2 = random.sample(VERBS, 2)
    passage = f"{n} {v1} to the {l1} . then {n} {v2} to the {l2} ."
    return QAExample(passage, f"where is {n} ?", l2, n, {n: l2})


def generate_distractor() -> QAExample:
    n1, n2 = random.sample(NAMES, 2)
    l1, l2, l3 = random.sample(LOCATIONS, 3)
    v1, v2, v3 = random.choices(VERBS, k=3)
    passage = (f"{n1} {v1} to the {l1} . {n2} {v2} to the {l2} . "
               f"then {n1} {v3} to the {l3} .")
    facts = {n1: l3, n2: l2}
    target = random.choice([n1, n2])
    return QAExample(passage, f"where is {target} ?", facts[target], target, facts)


def generate_source_conflict() -> QAExample:
    n = random.choice(NAMES)
    l1, l2 = random.sample(LOCATIONS, 2)
    v1, v2 = random.choice(VERBS), random.choice(VERBS)
    passage = (f"{_random_tag()}: {n} {v1} to the {l1} .\n"
               f"{_random_tag()}: {n} {v2} to the {l2} .")
    return QAExample(passage, f"where is {n} ?", l2, n,
                     {n: l2}, valid_answers=[l1, l2])


_QA_GENERATORS = [
    (generate_single_fact, 0.15),
    (generate_two_facts, 0.30),
    (generate_three_facts, 0.20),
    (generate_temporal, 0.20),
    (generate_distractor, 0.15),
]


def generate_dataset(n: int, seed: int = 42, tagged: bool = False,
                     include_conflicts: bool = False) -> list[QAExample]:
    random.seed(seed)
    gens, weights = zip(*_QA_GENERATORS)
    examples = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        ex = gen()
        if tagged:
            ex = QAExample(tag_passage(ex.passage), ex.question, ex.answer,
                           ex.answer_entity, ex.facts, ex.valid_answers)
        examples.append(ex)
    if include_conflicts:
        for _ in range(int(n * 0.1)):
            examples.append(generate_source_conflict())
        random.shuffle(examples)
    return examples


# ============================================================================
# Shell Command Generator
# ============================================================================

_CMDS = ["ls", "cd", "pwd", "cat", "echo", "grep", "find", "sed", "awk",
         "sort", "uniq", "wc", "head", "tail", "cut", "tr", "xargs", "tee",
         "mkdir", "rmdir", "rm", "cp", "mv", "touch", "chmod", "chown", "ln",
         "diff", "tar", "gzip", "curl", "wget", "ssh", "scp", "rsync", "ps",
         "top", "kill", "df", "du", "ping", "git", "docker", "python", "node",
         "npm", "pip", "make", "gcc", "go", "cargo"]

_FLAGS = {
    "ls": ["-l", "-a", "-la", "-lh", "-R"],
    "grep": ["-r", "-i", "-n", "-v", "-c", "-l", "-E"],
    "find": ["-name", "-type f", "-type d", "-mtime", "-size"],
    "ps": ["-ef", "-aux", "-u"],
    "chmod": ["755", "644", "+x", "-R"],
    "tar": ["-xzf", "-czf", "-tf"],
    "git": ["status", "log", "diff", "add", "commit", "push", "pull",
            "branch", "checkout", "merge", "rebase", "stash"],
    "docker": ["ps", "images", "run", "build", "exec", "stop", "rm", "logs"],
    "curl": ["-s", "-o", "-X POST", "-H", "-d", "-L"],
}

_PATHS = ["/home/user", "/tmp", "/var/log", "/etc", "/usr/bin", "~/Documents",
          "~/projects", "./src", "../lib", "/dev/null", ".", "~"]

_FILES = ["README.md", "Makefile", "config.yaml", "main.py", "index.js",
          "app.go", "Cargo.toml", "package.json", "Dockerfile", ".gitignore",
          "requirements.txt", "setup.py", "test.sh", "data.csv", "output.log"]

_PATTERNS = ["*.py", "*.js", "*.go", "*.rs", "*.c", "*.txt", "*.log",
             "*.json", "*.yaml", "*.md"]

_PIPE_CMDS = ["grep", "sort", "uniq", "wc", "head", "tail", "cut", "tr",
              "awk", "sed", "tee", "xargs"]


def _gen_simple_cmd() -> str:
    cmd = random.choice(_CMDS)
    parts = [cmd]
    flags = _FLAGS.get(cmd, [])
    if flags and random.random() < 0.7:
        parts.append(random.choice(flags))
    if random.random() < 0.5:
        parts.append(random.choice(_FILES + _PATHS))
    return " ".join(parts)


def _gen_pipe() -> str:
    parts = [_gen_simple_cmd()]
    for _ in range(random.randint(1, 3)):
        cmd = random.choice(_PIPE_CMDS)
        seg = cmd
        flags = _FLAGS.get(cmd, [])
        if flags and random.random() < 0.5:
            seg += " " + random.choice(flags)
        if cmd in ("grep", "sed", "awk") and random.random() < 0.6:
            seg += " '" + random.choice(["error", "warning", "TODO",
                                         "import", "def ", "^#"]) + "'"
        parts.append(seg)
    return " | ".join(parts)


def _gen_redirect() -> str:
    redir = random.choice([">", ">>", "2>", "2>&1", "&>"])
    return f"{_gen_simple_cmd()} {redir} {random.choice(_FILES)}"


def _gen_conditional() -> str:
    op = random.choice(["&&", "||", ";"])
    return f"{_gen_simple_cmd()} {op} {_gen_simple_cmd()}"


def _gen_for_loop() -> str:
    var = random.choice(["f", "i", "file", "dir", "x"])
    iterable = random.choice([
        f"*.{random.choice(['py', 'js', 'go', 'txt'])}",
        "$(seq 1 10)",
        f"$(find . -name '{random.choice(_PATTERNS)}')",
        "$@",
    ])
    body = random.choice([f"echo ${var}", f"cat ${var}",
                          f"wc -l ${var}", f"cp ${var} /tmp/"])
    return f"for {var} in {iterable}; do {body}; done"


def _gen_if_stmt() -> str:
    cond = random.choice([
        f'[ -f "{random.choice(_FILES)}" ]',
        f'[ -d "{random.choice(_PATHS)}" ]',
        "[ $? -eq 0 ]",
        '[ -z "${VAR}" ]',
    ])
    return f"if {cond}; then {_gen_simple_cmd()}; fi"


def _gen_one_liner() -> str:
    templates = [
        lambda: f"while read line; do echo $line; done < {random.choice(_FILES)}",
        lambda: f"test -f {random.choice(_FILES)} && echo exists || echo missing",
        lambda: "alias ll='ls -la'",
        lambda: f"export PATH={random.choice(_PATHS)}:$PATH",
        lambda: f"nohup {_gen_simple_cmd()} &",
    ]
    return random.choice(templates)()


_SHELL_GENS = [
    (_gen_simple_cmd, 0.25), (_gen_pipe, 0.20), (_gen_redirect, 0.10),
    (_gen_conditional, 0.10), (_gen_for_loop, 0.12), (_gen_if_stmt, 0.10),
    (_gen_one_liner, 0.13),
]


def generate_shell_texts(n: int, seed: int = 42) -> list[str]:
    random.seed(seed)
    gens, weights = zip(*_SHELL_GENS)
    texts = []
    for _ in range(n):
        gen = random.choices(gens, weights=weights, k=1)[0]
        text = gen()
        if random.random() < 0.15:
            text = random.choice(["# list files", "# search logs", "# backup",
                                  "# deploy", "# cleanup", "# check status",
                                  "# build project"]) + "\n" + text
        texts.append(text)
    return texts


# ============================================================================
# Wikipedia Sentence Loader
# ============================================================================

def load_wikipedia_sentences(n: int = 10000, min_len: int = 30,
                             max_len: int = 300, seed: int = 42,
                             cache_dir: str = "data_cache") -> list[str]:
    cache_path = os.path.join(cache_dir, f"wiki_sentences_{n}.txt")
    if os.path.exists(cache_path):
        print(f"  Loading cached Wikipedia sentences from {cache_path}")
        with open(cache_path) as f:
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

    if ds is None:
        print("  All Wikipedia sources failed, using fallback")
        return _generate_fallback_text(n, seed)

    random.seed(seed)
    sentences = []
    for article in ds:
        for sent in article.get("text", "").split(". "):
            sent = sent.strip()
            if min_len <= len(sent) <= max_len and not sent.startswith("="):
                if any(c.isalpha() for c in sent):
                    sentences.append(sent + ("" if sent.endswith(".") else "."))
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


def _generate_fallback_text(n: int, seed: int = 42) -> list[str]:
    random.seed(seed)
    subjects = ["the cat", "a scientist", "the river", "an artist", "the machine",
                "a child", "the city", "a bird", "the forest", "a teacher"]
    verbs = ["discovered", "created", "moved toward", "studied", "built",
             "observed", "explored", "designed", "protected", "transformed"]
    objects = ["a new path", "the ancient ruins", "a hidden garden",
               "the tall tower", "a small village", "the dark cave"]
    adverbs = ["quickly", "carefully", "silently", "eagerly", "slowly"]
    texts = []
    for _ in range(n):
        s, v, o, a = (random.choice(subjects), random.choice(verbs),
                      random.choice(objects), random.choice(adverbs))
        texts.append(random.choice([
            f"{s} {a} {v} {o}.", f"{a}, {s} {v} {o}.",
            f"{s} {v} {o} and then paused.",
        ]))
    return texts


# ============================================================================
# Chat Data — Synthetic + HuggingFace
# ============================================================================

_CHAT_PAIRS = {
    "greetings": [
        ("Hello!", "Hello! How can I help you?"),
        ("Hi there", "Hi! What can I do for you?"),
        ("Good morning", "Good morning! How are you today?"),
        ("Hey", "Hey! What's up?"),
        ("Thanks!", "You're welcome!"),
        ("Goodbye", "Goodbye! Have a great day!"),
        ("What's your name?", "I'm ANT, a compact language model with persistent memory."),
    ],
    "factual": [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
        ("What is the capital of Germany?", "Berlin is the capital of Germany."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
        ("What is the speed of light?", "The speed of light is about 300,000 km per second."),
        ("How many planets are in our solar system?", "There are eight planets in our solar system."),
        ("What is the chemical formula for water?", "The formula for water is H2O."),
    ],
    "technical": [
        ("What is a variable?", "A variable stores a value that can change during execution."),
        ("What is a function?", "A function is a reusable block of code that performs a task."),
        ("What is recursion?", "Recursion is when a function calls itself."),
        ("What is an API?", "An API is an interface for programs to communicate."),
        ("What is Python?", "Python is a high-level interpreted programming language."),
        ("What is Git?", "Git is a distributed version control system."),
        ("What is machine learning?", "Machine learning trains algorithms to learn from data."),
    ],
    "shell": [
        ("How do I list files?", "Use `ls -la` to list all files with details."),
        ("How to find a file?", "Use `find . -name 'filename'` to search for files."),
        ("How to search in files?", "Use `grep -r 'pattern' .` to search recursively."),
        ("How to check disk usage?", "Use `df -h` for disk space or `du -sh *` for sizes."),
        ("How to check git status?", "Use `git status` to see modified and staged files."),
    ],
    "yesno": [
        ("Is the Earth flat?", "No, the Earth is roughly spherical."),
        ("Is the sun a star?", "Yes, the sun is a star."),
        ("Do plants need water?", "Yes, plants need water to survive."),
        ("Can fish breathe underwater?", "Yes, fish breathe using gills underwater."),
    ],
    "open": [
        ("Tell me about the moon.", "The moon is Earth's only natural satellite. It orbits Earth every 27 days."),
        ("How do computers work?", "Computers process data using a CPU that executes instructions in binary."),
        ("Why is the sky blue?", "The sky appears blue because the atmosphere scatters shorter blue wavelengths."),
        ("What causes earthquakes?", "Earthquakes occur when tectonic plates suddenly shift along faults."),
    ],
}


def generate_chat_data(n: int = 5000, seed: int = 42) -> list[str]:
    random.seed(seed)
    all_pairs = []
    for pairs in _CHAT_PAIRS.values():
        all_pairs.extend(pairs)

    results = []
    for _ in range(n):
        q, a = random.choice(all_pairs)
        ts = _random_timestamp()
        text = f"localhost/user/chat@{ts}: {q}\nlocalhost/ant/chat@{ts}: {a}"
        if len(text.encode("utf-8")) > 188:
            overhead = len(f"localhost/user/chat@{ts}: {q}\nlocalhost/ant/chat@{ts}: ".encode("utf-8"))
            max_a = 188 - overhead
            if max_a > 10:
                a = a.encode("utf-8")[:max_a].decode("utf-8", errors="ignore").rsplit(" ", 1)[0]
                text = f"localhost/user/chat@{ts}: {q}\nlocalhost/ant/chat@{ts}: {a}"
            else:
                continue
        results.append(text)
    return results


def load_hf_chat_data(n: int = 20000, seed: int = 42,
                      cache_dir: str = "data_cache",
                      max_seq_bytes: int = 600) -> list[str]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"hf_chat_{n}_{max_seq_bytes}b.txt")

    if os.path.exists(cache_path):
        print(f"  Loading cached HF chat data from {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            raw_lines = [l.rstrip("\n") for l in f]
        pairs, current = [], []
        for line in raw_lines:
            if line.startswith("localhost/user/") and current:
                pairs.append("\n".join(current))
                current = [line]
            elif line.strip():
                current.append(line)
        if current:
            pairs.append("\n".join(current))
        return pairs

    print(f"  Downloading HF chat data (target {n} pairs)...")

    _cap_word = re.compile(r"\b[A-Z][a-z]{2,}\b")
    _year_num = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
    TAG_OVERHEAD = 90

    def is_fact_heavy(text):
        return len(_cap_word.findall(text)) > 8 or len(_year_num.findall(text)) > 3

    def is_ascii_dominant(text):
        return sum(1 for c in text if ord(c) < 128) / max(len(text), 1) > 0.85

    def extract_pairs(messages):
        out = []
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                out.append((messages[i]["content"].strip(),
                            messages[i + 1]["content"].strip()))
        return out

    random.seed(seed)
    results = []
    max_content = max_seq_bytes - TAG_OVERHEAD

    sources = [
        ("HuggingFaceH4/ultrachat_200k", {"split": "train_sft"}),
        ("HuggingFaceTB/smoltalk", {"name": "all", "split": "train"}),
    ]

    try:
        from datasets import load_dataset
        for ds_name, ds_kwargs in sources:
            if len(results) >= n:
                break
            print(f"    Streaming {ds_name}...")
            try:
                ds = load_dataset(ds_name, **ds_kwargs, streaming=True)
                for ex in ds:
                    if len(results) >= n:
                        break
                    for q, a in extract_pairs(ex.get("messages", [])):
                        if len(results) >= n:
                            break
                        q_b, a_b = len(q.encode("utf-8")), len(a.encode("utf-8"))
                        if a_b < 20 or q_b + a_b > max_content:
                            continue
                        if not is_ascii_dominant(q) or not is_ascii_dominant(a):
                            continue
                        if is_fact_heavy(a):
                            continue
                        results.append((q, a))
            except Exception as e:
                print(f"    Warning: failed {ds_name}: {e}")
    except ImportError:
        print("  datasets not installed, using synthetic")

    if len(results) < max(10, n // 4):
        print(f"  HF data insufficient ({len(results)}), using synthetic fallback")
        return generate_chat_data(n, seed)

    print(f"  Collected {len(results)} chat pairs from HF")

    tagged = []
    for q, a in results:
        ts = _random_timestamp()
        tagged.append(f"localhost/user/chat@{ts}: {q}\nlocalhost/ant/chat@{ts}: {a}")
    random.shuffle(tagged)

    with open(cache_path, "w", encoding="utf-8") as f:
        for t in tagged:
            f.write(t + "\n")
    print(f"  Cached {len(tagged)} tagged pairs to {cache_path}")
    return tagged


# ============================================================================
# Torch Datasets
# ============================================================================

class TextLMDataset(Dataset):
    """Autoregressive LM dataset from raw text strings.
    Each sample: <bos> text_bytes <eos>, truncated to max_len.
    """
    def __init__(self, texts: list[str], max_len: int | None = None):
        self.samples = []
        for text in texts:
            full = [BOS_ID] + tokenize(text) + [EOS_ID]
            if max_len and len(full) > max_len + 1:
                full = full[:max_len + 1]
            self.samples.append((
                torch.tensor(full[:-1], dtype=torch.long),
                torch.tensor(full[1:], dtype=torch.long),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ChatMemoryDataset(Dataset):
    """Chat pairs for memory-based training.
    Splits tagged exchanges into (user_content_ids, agent_line_ids).
    """
    def __init__(self, chat_texts, max_user_tokens=190):
        self.user_ids = []
        self.agent_ids = []
        for text in chat_texts:
            parts = text.split("\n", 1)
            if len(parts) < 2:
                continue
            colon = parts[0].find(": ")
            user_content = parts[0][colon + 2:] if colon >= 0 else parts[0]
            agent_line = parts[1]
            if len(user_content) < 2 or len(agent_line) < 10:
                continue
            self.user_ids.append(torch.tensor(tokenize(user_content)[:max_user_tokens],
                                              dtype=torch.long))
            self.agent_ids.append(torch.tensor(tokenize(agent_line), dtype=torch.long))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.agent_ids[idx]


def lm_collate_fn(batch, pad_id: int = PAD_ID):
    max_len = max(inp.size(0) for inp, _ in batch)
    inp_b, tgt_b = [], []
    for inp, tgt in batch:
        n = inp.size(0)
        pad_n = max_len - n
        inp_b.append(torch.cat([inp, inp.new_full((pad_n,), pad_id)]))
        tgt_b.append(torch.cat([tgt, tgt.new_full((pad_n,), pad_id)]))
    return torch.stack(inp_b), torch.stack(tgt_b)


def chat_memory_collate_fn(batch):
    user_list, agent_list = zip(*batch)
    max_u = max(u.size(0) for u in user_list)
    user_padded = torch.full((len(batch), max_u), PAD_ID, dtype=torch.long)
    for i, u in enumerate(user_list):
        user_padded[i, :u.size(0)] = u

    max_a = max(a.size(0) for a in agent_list)
    agent_inp = torch.full((len(batch), max_a + 1), PAD_ID, dtype=torch.long)
    agent_tgt = torch.full((len(batch), max_a + 1), PAD_ID, dtype=torch.long)
    for i, a in enumerate(agent_list):
        agent_inp[i, 0] = BOS_ID
        agent_inp[i, 1:1 + a.size(0)] = a
        agent_tgt[i, :a.size(0)] = a
        agent_tgt[i, a.size(0)] = EOS_ID

    return user_padded, agent_inp, agent_tgt
