/// Hierarchical trie with EMA value vectors at every node.
///
/// Self-contained copy (no PyO3) from ant_memory/src/trie.rs.
/// Arena-allocated, 256-ary, depth_cap levels, EMA write propagation.

use std::collections::HashMap;

pub struct TrieNode {
    pub children: HashMap<u8, u32>,
    pub value: Option<Vec<f32>>,
    pub write_count: u32,
}

impl TrieNode {
    pub fn new() -> Self {
        Self { children: HashMap::new(), value: None, write_count: 0 }
    }

    pub fn with_value(d_model: usize) -> Self {
        Self {
            children: HashMap::new(),
            value: Some(vec![0.0f32; d_model]),
            write_count: 0,
        }
    }
}

pub struct HierarchicalTrie {
    pub d_model: usize,
    pub depth_cap: usize,
    pub nodes: Vec<TrieNode>,
    pub n_records: u64,
}

impl HierarchicalTrie {
    pub fn new(d_model: usize, depth_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(1024);
        nodes.push(TrieNode::with_value(d_model));
        Self { d_model, depth_cap, nodes, n_records: 0 }
    }

    pub fn write(&mut self, address: &[u8], value: &[f32], alpha_base: f32, alpha_min: f32) {
        let depth = address.len().min(self.depth_cap);
        if depth == 0 || value.len() != self.d_model { return; }

        let mut path_indices: Vec<u32> = Vec::with_capacity(depth + 1);
        path_indices.push(0);
        let mut current = 0u32;
        for &bin in &address[..depth] {
            let next = if let Some(&ci) = self.nodes[current as usize].children.get(&bin) {
                ci
            } else {
                let ni = self.nodes.len() as u32;
                self.nodes.push(TrieNode::with_value(self.d_model));
                self.nodes[current as usize].children.insert(bin, ni);
                ni
            };
            path_indices.push(next);
            current = next;
        }

        let leaf_idx = *path_indices.last().unwrap() as usize;
        let leaf = &mut self.nodes[leaf_idx];
        leaf.write_count += 1;
        let alpha_leaf = (alpha_base / (1.0 + 0.01 * leaf.write_count as f32)).max(alpha_min);
        if leaf.write_count == 1 {
            if let Some(ref mut v) = leaf.value {
                v.copy_from_slice(value);
            }
        } else if let Some(ref mut v) = leaf.value {
            let inv = 1.0 - alpha_leaf;
            for (d, &s) in v.iter_mut().zip(value.iter()) { *d = inv * *d + alpha_leaf * s; }
        }

        let total_depth = depth;
        for (i, &ni) in path_indices[..path_indices.len() - 1].iter().enumerate() {
            let depth_diff = total_depth - i;
            let decay = 1.0 / ((depth_diff as f32 + 1.0).sqrt());
            let coarse = 1.0 / (i as f32 + 1.0);
            let alpha = (alpha_base * decay * coarse).max(alpha_min);
            let node = &mut self.nodes[ni as usize];
            node.write_count += 1;
            if let Some(ref mut v) = node.value {
                let inv = 1.0 - alpha;
                for (d, &s) in v.iter_mut().zip(value.iter()) { *d = inv * *d + alpha * s; }
            }
        }
        self.n_records += 1;
    }

    /// Returns (node_index, value_slice) from root to deepest matching node.
    pub fn read_path(&self, address: &[u8]) -> Vec<(u32, &[f32])> {
        let mut results = Vec::with_capacity(self.depth_cap + 1);
        if let Some(ref v) = self.nodes[0].value { results.push((0u32, v.as_slice())); }
        let mut current = 0u32;
        for &bin in address.iter().take(self.depth_cap) {
            if let Some(&ci) = self.nodes[current as usize].children.get(&bin) {
                current = ci;
                if let Some(ref v) = self.nodes[current as usize].value {
                    results.push((current, v.as_slice()));
                }
            } else { break; }
        }
        results
    }

    pub fn total_nodes(&self) -> usize { self.nodes.len() }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(TrieNode::with_value(self.d_model));
        self.n_records = 0;
    }

    pub fn serialize(&self) -> Vec<u8> {
        let n = self.nodes.len();
        let d = self.d_model;
        let mut buf = Vec::with_capacity(20 + n * d * 4 + n * 4 + n * 10);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(d as u32).to_le_bytes());
        buf.extend_from_slice(&(self.depth_cap as u32).to_le_bytes());
        buf.extend_from_slice(&self.n_records.to_le_bytes());
        for node in &self.nodes {
            if let Some(ref v) = node.value {
                for &f in v { buf.extend_from_slice(&f.to_le_bytes()); }
            } else {
                for _ in 0..d { buf.extend_from_slice(&0.0f32.to_le_bytes()); }
            }
        }
        for node in &self.nodes { buf.extend_from_slice(&node.write_count.to_le_bytes()); }
        for node in &self.nodes {
            buf.extend_from_slice(&(node.children.len() as u16).to_le_bytes());
            let mut entries: Vec<_> = node.children.iter().collect();
            entries.sort_by_key(|(&k, _)| k);
            for (&bin_id, &ci) in entries {
                buf.push(bin_id);
                buf.extend_from_slice(&ci.to_le_bytes());
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 20 { return None; }
        let n = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let d = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let depth_cap = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
        let n_records = u64::from_le_bytes(data[12..20].try_into().ok()?);
        if n == 0 || d == 0 { return None; }
        let values_end = 20 + n * d * 4;
        let wc_end = values_end + n * 4;
        if data.len() < wc_end { return None; }
        let mut nodes: Vec<TrieNode> = Vec::with_capacity(n);
        for i in 0..n {
            let off = 20 + i * d * 4;
            let mut value = vec![0.0f32; d];
            for j in 0..d {
                let s = off + j * 4;
                value[j] = f32::from_le_bytes(data[s..s+4].try_into().ok()?);
            }
            let wc = u32::from_le_bytes(data[values_end + i*4..values_end + i*4 + 4].try_into().ok()?);
            nodes.push(TrieNode { children: HashMap::new(), value: Some(value), write_count: wc });
        }
        let mut pos = wc_end;
        for i in 0..n {
            if pos + 2 > data.len() { break; }
            let nc = u16::from_le_bytes(data[pos..pos+2].try_into().ok()?) as usize;
            pos += 2;
            for _ in 0..nc {
                if pos + 5 > data.len() { break; }
                let bin_id = data[pos];
                let ci = u32::from_le_bytes(data[pos+1..pos+5].try_into().ok()?);
                nodes[i].children.insert(bin_id, ci);
                pos += 5;
            }
        }
        Some(Self { d_model: d, depth_cap, nodes, n_records })
    }
}

// ---------------------------------------------------------------------------
// Memory system wrapper (no PyO3)
// ---------------------------------------------------------------------------

use std::sync::Mutex;
use std::path::PathBuf;
use std::fs;

pub struct MemorySystem {
    pub trie: Mutex<HierarchicalTrie>,
    bin_path: PathBuf,
    pub d_model: usize,
    pub alpha_base: f32,
    pub alpha_min: f32,
    pub flush_interval: u64,
    write_count: Mutex<u64>,
}

impl MemorySystem {
    pub fn new(data_path: &str, d_model: usize, depth_cap: usize,
               alpha_base: f32, alpha_min: f32, flush_interval: u64) -> Self {
        fs::create_dir_all(data_path).ok();
        let bin_path = PathBuf::from(data_path).join("memory.bin");
        let trie = if bin_path.exists() {
            fs::read(&bin_path)
                .ok()
                .and_then(|d| HierarchicalTrie::deserialize(&d))
                .unwrap_or_else(|| HierarchicalTrie::new(d_model, depth_cap))
        } else {
            HierarchicalTrie::new(d_model, depth_cap)
        };
        Self {
            trie: Mutex::new(trie),
            bin_path,
            d_model,
            alpha_base,
            alpha_min,
            flush_interval,
            write_count: Mutex::new(0),
        }
    }

    /// Batch write: batch_addresses[b][n] = address for sample b, addr_net n
    pub fn write_batch(&self, batch_addresses: &Vec<Vec<Vec<u8>>>, batch_values: &[&[f32]]) {
        let mut trie = self.trie.lock().unwrap();
        for (b, addrs) in batch_addresses.iter().enumerate() {
            let val = batch_values[b];
            for addr in addrs {
                trie.write(addr, val, self.alpha_base, self.alpha_min);
            }
        }
        let mut wc = self.write_count.lock().unwrap();
        *wc += batch_addresses.len() as u64;
        if *wc % self.flush_interval == 0 {
            let data = trie.serialize();
            drop(trie);
            fs::write(&self.bin_path, data).ok();
        }
    }

    /// Batch read: returns (vectors, mask) as flat f32 vecs.
    /// vectors: [B, max_vectors, d_model], mask: [B, max_vectors]
    pub fn read_batch(&self, batch_addresses: &Vec<Vec<Vec<u8>>>, max_vectors: usize)
        -> (Vec<f32>, Vec<bool>)
    {
        let trie = self.trie.lock().unwrap();
        let b_size = batch_addresses.len();
        let d = self.d_model;
        let mut vec_data = vec![0.0f32; b_size * max_vectors * d];
        let mut mask_data = vec![false; b_size * max_vectors];
        for (b, addrs) in batch_addresses.iter().enumerate() {
            let mut all_vectors: Vec<&[f32]> = Vec::new();
            let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for addr in addrs {
                for (ni, val) in trie.read_path(addr) {
                    if seen.insert(ni) { all_vectors.push(val); }
                }
            }
            let n = all_vectors.len().min(max_vectors);
            for i in 0..n {
                let off = b * max_vectors * d + i * d;
                vec_data[off..off+d].copy_from_slice(all_vectors[i]);
                mask_data[b * max_vectors + i] = true;
            }
        }
        (vec_data, mask_data)
    }

    pub fn flush(&self) {
        let trie = self.trie.lock().unwrap();
        let data = trie.serialize();
        drop(trie);
        fs::write(&self.bin_path, data).ok();
    }

    pub fn reset(&self) {
        let mut trie = self.trie.lock().unwrap();
        trie.reset();
        *self.write_count.lock().unwrap() = 0;
    }

    pub fn total_nodes(&self) -> usize { self.trie.lock().unwrap().total_nodes() }
    pub fn total_entries(&self) -> u64 { self.trie.lock().unwrap().n_records }
}
