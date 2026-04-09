/// Data loading and batch generation for ANT training.

use std::fs;
use std::path::Path;
use rand::Rng;

use crate::config::*;

// ---------------------------------------------------------------------------
// Text loading
// ---------------------------------------------------------------------------

pub fn load_text_file(path: &str) -> Vec<String> {
    match fs::read_to_string(path) {
        Ok(s) => s.lines().map(|l| l.to_string()).filter(|l| !l.is_empty()).collect(),
        Err(_) => vec![],
    }
}

/// Load all available data files from data_cache directory, returning a flat Vec<String>.
pub fn load_all_texts(data_dir: &str) -> Vec<String> {
    let dir = Path::new(data_dir);
    let mut texts = Vec::new();

    // Prefer mid-size files that fit in RAM
    let preferred = [
        "wiki_sentences_50000.txt",
        "wiki_sentences_20000.txt",
        "wiki_sentences_5000.txt",
        "hf_chat_10000_600b.txt",
    ];
    for fname in &preferred {
        let path = dir.join(fname);
        if path.exists() {
            let mut lines = load_text_file(path.to_str().unwrap());
            texts.append(&mut lines);
        }
    }

    if texts.is_empty() {
        // Fallback: any .txt
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map(|e| e == "txt").unwrap_or(false) {
                    let mut lines = load_text_file(entry.path().to_str().unwrap());
                    texts.append(&mut lines);
                }
            }
        }
    }

    texts
}

/// Shell command templates for Phase A/B training.
pub fn shell_texts() -> Vec<String> {
    let templates = vec![
        "user@host:~$ ls -la\ntotal 32\ndrwxr-xr-x 4 user user 4096 Jan 01 00:00 .\ndrwxr-xr-x 3 user user 4096 Jan 01 00:00 ..\n-rw-r--r-- 1 user user  220 Jan 01 00:00 .bash_profile",
        "user@host:~$ git status\nOn branch main\nnothing to commit, working tree clean",
        "user@host:~$ cat /proc/meminfo | head -5\nMemTotal:       16384000 kB\nMemFree:         8192000 kB\nMemAvailable:   12288000 kB",
        "user@host:~$ ps aux | grep python\nuser       123 0.1  1.2 python3 train.py",
        "user@host:~$ find . -name '*.rs' | wc -l\n8",
        "user@host:~$ cargo build --release\n   Compiling ant_train v0.1.0\n    Finished release [optimized] target(s) in 12.34s",
        "user@host:~$ echo $PATH\n/usr/local/bin:/usr/bin:/bin",
        "user@host:~$ df -h\nFilesystem  Size  Used Avail Use% Mounted on\n/dev/sda1    50G   20G   28G  42% /",
    ];
    templates.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Byte-level tokenization
// ---------------------------------------------------------------------------

pub fn tokenize(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

pub fn detokenize(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

/// Build a training sample: BOS + bytes + EOS, truncated to max_len.
pub fn make_sample(text: &str, max_len: usize) -> Vec<u8> {
    let mut sample = vec![BOS_ID as u8];
    sample.extend_from_slice(&text.as_bytes()[..text.len().min(max_len - 2)]);
    sample.push(EOS_ID as u8);
    sample
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

pub struct TextDataset {
    pub texts: Vec<String>,
    pub max_len: usize,
}

impl TextDataset {
    pub fn new(texts: Vec<String>, max_len: usize) -> Self {
        Self { texts, max_len }
    }

    /// Random sample crop: uniform text, random start offset.
    pub fn random_sample(&self, rng: &mut impl Rng) -> Vec<u8> {
        if self.texts.is_empty() {
            return vec![BOS_ID as u8, EOS_ID as u8];
        }
        let text = &self.texts[rng.gen_range(0..self.texts.len())];
        let bytes = text.as_bytes();
        let max_start = if bytes.len() > self.max_len { bytes.len() - self.max_len } else { 0 };
        let start = if max_start > 0 { rng.gen_range(0..max_start) } else { 0 };
        let end = (start + self.max_len - 2).min(bytes.len());
        make_sample(&text[start..end], self.max_len)
    }
}

// ---------------------------------------------------------------------------
// Batch collation
// ---------------------------------------------------------------------------

/// Collate a batch of variable-length byte sequences into a (B, T) matrix.
/// Sequences are right-padded with PAD_ID.
pub fn collate_batch(samples: &[Vec<u8>], pad_id: u8) -> (Vec<u32>, usize, usize) {
    let b = samples.len();
    let t = samples.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut out = vec![pad_id as u32; b * t];
    for (bi, sample) in samples.iter().enumerate() {
        for (ti, &byte) in sample.iter().enumerate() {
            out[bi * t + ti] = byte as u32;
        }
    }
    (out, b, t)
}

/// Sample a batch of random texts.
pub fn random_batch(dataset: &TextDataset, batch_size: usize, rng: &mut impl Rng)
    -> (Vec<u32>, usize, usize)
{
    let samples: Vec<Vec<u8>> = (0..batch_size)
        .map(|_| dataset.random_sample(rng))
        .collect();
    collate_batch(&samples, PAD_ID as u8)
}

// ---------------------------------------------------------------------------
// Chat data helpers
// ---------------------------------------------------------------------------

/// Parse chat format: "localhost/user/chat@...: <content>" lines.
pub fn parse_chat_line(line: &str) -> Option<(String, String)> {
    let at = line.find('@')?;
    let colon = line[at..].find(':').map(|i| at + i)?;
    let tag = line[..colon].to_string();
    let content = line[colon + 2..].trim().to_string();
    if content.is_empty() { return None; }
    Some((tag, content))
}

// ---------------------------------------------------------------------------
// Data split helpers
// ---------------------------------------------------------------------------

pub struct DataSplit {
    pub train: Vec<String>,
    pub val: Vec<String>,
}

impl DataSplit {
    pub fn from_texts(mut texts: Vec<String>, val_frac: f32) -> Self {
        let n = texts.len();
        let n_val = ((n as f32 * val_frac) as usize).max(1).min(n / 2);
        let val = texts.split_off(n - n_val);
        Self { train: texts, val }
    }
}
