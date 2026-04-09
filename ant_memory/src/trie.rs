/// Hierarchical trie with EMA value vectors at every node.
///
/// Architecture:
///   - Arena-allocated: all nodes in a single Vec<TrieNode>
///   - 256 bins per level, up to depth_cap levels (default 8)
///   - Each node stores an EMA-blended f32 vector (d_model dim)
///   - Writes propagate EMA to all ancestors with depth-dependent decay
///   - Reads collect the full ancestor path (root→leaf) = up to depth+1 vectors
///
/// Memory: ~(d_model×4 + 16) bytes per node. At d_model=128: ~528 bytes/node.
/// 1M nodes ≈ 528MB. No Python object overhead.

use std::collections::HashMap;

/// Single trie node — index-based children for zero-copy arena allocation.
pub struct TrieNode {
    /// Map from bin index (0..255) to child node index in the arena.
    pub children: HashMap<u8, u32>,
    /// EMA-blended value vector (d_model floats). None until first write.
    pub value: Option<Vec<f32>>,
    /// Number of writes through this node.
    pub write_count: u32,
}

impl TrieNode {
    pub fn new() -> Self {
        Self {
            children: HashMap::new(),
            value: None,
            write_count: 0,
        }
    }

    pub fn with_value(d_model: usize) -> Self {
        Self {
            children: HashMap::new(),
            value: Some(vec![0.0f32; d_model]),
            write_count: 0,
        }
    }
}

/// Arena-allocated hierarchical trie.
pub struct HierarchicalTrie {
    pub d_model: usize,
    pub depth_cap: usize,
    /// All nodes stored contiguously. Index 0 is always root.
    pub nodes: Vec<TrieNode>,
    /// Total write operations (each write = 1 record).
    pub n_records: u64,
}

impl HierarchicalTrie {
    pub fn new(d_model: usize, depth_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(1024);
        nodes.push(TrieNode::with_value(d_model)); // root at index 0
        Self {
            d_model,
            depth_cap,
            nodes,
            n_records: 0,
        }
    }

    /// Write a value at the given address, propagating EMA to ancestors.
    ///
    /// address: slice of bin indices [0..255], length up to depth_cap
    /// value: slice of d_model floats (already projected by V_proj)
    pub fn write(&mut self, address: &[u8], value: &[f32],
                 alpha_base: f32, alpha_min: f32) {
        let depth = address.len().min(self.depth_cap);
        if depth == 0 || value.len() != self.d_model {
            return;
        }

        // Collect path indices from root to leaf, creating nodes as needed
        let mut path_indices: Vec<u32> = Vec::with_capacity(depth + 1);
        path_indices.push(0); // root

        let mut current = 0u32;
        for &bin in &address[..depth] {
            let next = if let Some(&child_idx) = self.nodes[current as usize].children.get(&bin) {
                child_idx
            } else {
                let new_idx = self.nodes.len() as u32;
                self.nodes.push(TrieNode::with_value(self.d_model));
                self.nodes[current as usize].children.insert(bin, new_idx);
                new_idx
            };
            path_indices.push(next);
            current = next;
        }

        // Write leaf with full weight
        let leaf_idx = *path_indices.last().unwrap() as usize;
        let leaf = &mut self.nodes[leaf_idx];
        leaf.write_count += 1;
        let alpha_leaf = (alpha_base / (1.0 + 0.01 * leaf.write_count as f32)).max(alpha_min);

        if leaf.write_count == 1 {
            // First write: copy value directly
            if let Some(ref mut v) = leaf.value {
                v.copy_from_slice(value);
            }
        } else {
            // EMA blend
            if let Some(ref mut v) = leaf.value {
                let inv = 1.0 - alpha_leaf;
                for (dst, &src) in v.iter_mut().zip(value.iter()) {
                    *dst = inv * *dst + alpha_leaf * src;
                }
            }
        }

        // Propagate EMA to ancestors with decay 1/√(depth_diff+1)
        // path_indices[0] = root, path_indices[D] = leaf at depth D
        let total_depth = depth;
        for (i, &node_idx) in path_indices[..path_indices.len() - 1].iter().enumerate() {
            let depth_diff = total_depth - i;
            let decay = 1.0 / ((depth_diff as f32 + 1.0).sqrt());
            let coarse_factor = 1.0 / (i as f32 + 1.0);
            let alpha = (alpha_base * decay * coarse_factor).max(alpha_min);

            let node = &mut self.nodes[node_idx as usize];
            node.write_count += 1;
            if let Some(ref mut v) = node.value {
                let inv = 1.0 - alpha;
                for (dst, &src) in v.iter_mut().zip(value.iter()) {
                    *dst = inv * *dst + alpha * src;
                }
            }
        }

        self.n_records += 1;
    }

    /// Read the full ancestor path for an address.
    ///
    /// Returns vectors from root to deepest matching node.
    /// If exact address doesn't exist, returns path to deepest existing prefix.
    pub fn read_path(&self, address: &[u8]) -> Vec<&[f32]> {
        let mut vectors = Vec::with_capacity(self.depth_cap + 1);

        if let Some(ref v) = self.nodes[0].value {
            vectors.push(v.as_slice());
        }

        let mut current = 0u32;
        for &bin in address.iter().take(self.depth_cap) {
            if let Some(&child_idx) = self.nodes[current as usize].children.get(&bin) {
                current = child_idx;
                if let Some(ref v) = self.nodes[current as usize].value {
                    vectors.push(v.as_slice());
                }
            } else {
                break;
            }
        }

        vectors
    }

    /// Total number of nodes in the trie.
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Reset trie to empty state (just root).
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(TrieNode::with_value(self.d_model));
        self.n_records = 0;
    }

    /// Serialize trie to bytes.
    ///
    /// Format: header(12B) | values(N×D×4B) | write_counts(N×4B) | adjacency
    /// Header: n_nodes(u32), d_model(u32), n_records(u32)
    /// Adjacency per node: n_children(u16) + n_children × (bin_id(u8) + child_idx(u32))
    pub fn serialize(&self) -> Vec<u8> {
        let n = self.nodes.len();
        let d = self.d_model;
        // Estimate size: header + values + write_counts + adjacency
        let est = 12 + n * d * 4 + n * 4 + n * 10;
        let mut buf = Vec::with_capacity(est);

        // Header
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&(d as u32).to_le_bytes());
        buf.extend_from_slice(&(self.n_records as u32).to_le_bytes());

        // Values: contiguous f32 block
        for node in &self.nodes {
            if let Some(ref v) = node.value {
                for &f in v {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            } else {
                for _ in 0..d {
                    buf.extend_from_slice(&0.0f32.to_le_bytes());
                }
            }
        }

        // Write counts
        for node in &self.nodes {
            buf.extend_from_slice(&node.write_count.to_le_bytes());
        }

        // Adjacency
        for node in &self.nodes {
            let nc = node.children.len() as u16;
            buf.extend_from_slice(&nc.to_le_bytes());
            // Sort by bin for deterministic output
            let mut entries: Vec<_> = node.children.iter().collect();
            entries.sort_by_key(|(&k, _)| k);
            for (&bin_id, &child_idx) in entries {
                buf.push(bin_id);
                buf.extend_from_slice(&child_idx.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize trie from bytes.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }

        let n = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let d = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let n_records = u32::from_le_bytes(data[8..12].try_into().ok()?) as u64;

        if n == 0 || d == 0 {
            return None;
        }

        let values_start = 12;
        let values_end = values_start + n * d * 4;
        let wc_end = values_end + n * 4;

        if data.len() < wc_end {
            return None;
        }

        // Read values
        let mut nodes: Vec<TrieNode> = Vec::with_capacity(n);
        for i in 0..n {
            let offset = values_start + i * d * 4;
            let mut value = vec![0.0f32; d];
            for j in 0..d {
                let start = offset + j * 4;
                value[j] = f32::from_le_bytes(data[start..start + 4].try_into().ok()?);
            }
            let wc_offset = values_end + i * 4;
            let write_count = u32::from_le_bytes(data[wc_offset..wc_offset + 4].try_into().ok()?);

            nodes.push(TrieNode {
                children: HashMap::new(),
                value: Some(value),
                write_count,
            });
        }

        // Read adjacency
        let mut pos = wc_end;
        for i in 0..n {
            if pos + 2 > data.len() {
                break;
            }
            let nc = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?) as usize;
            pos += 2;
            for _ in 0..nc {
                if pos + 5 > data.len() {
                    break;
                }
                let bin_id = data[pos];
                let child_idx = u32::from_le_bytes(data[pos + 1..pos + 5].try_into().ok()?);
                nodes[i].children.insert(bin_id, child_idx);
                pos += 5;
            }
        }

        Some(Self {
            d_model: d,
            depth_cap: 8,
            nodes,
            n_records,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_basic() {
        let mut trie = HierarchicalTrie::new(4, 8);
        let addr = vec![1u8, 2, 3];
        let value = vec![1.0f32, 2.0, 3.0, 4.0];
        trie.write(&addr, &value, 0.1, 0.001);

        let path = trie.read_path(&addr);
        assert_eq!(path.len(), 4); // root + 3 levels
        // Leaf should be exact value on first write
        assert_eq!(path[3], &value[..]);
    }

    #[test]
    fn test_partial_read() {
        let mut trie = HierarchicalTrie::new(4, 8);
        let addr = vec![1u8, 2, 3];
        let value = vec![1.0f32, 2.0, 3.0, 4.0];
        trie.write(&addr, &value, 0.1, 0.001);

        // Read with different suffix — should get root + first match
        let partial = vec![1u8, 2, 99];
        let path = trie.read_path(&partial);
        assert_eq!(path.len(), 3); // root + bin1 + bin2, stops at 99
    }

    #[test]
    fn test_serialize_roundtrip() {
        let mut trie = HierarchicalTrie::new(4, 8);
        trie.write(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0], 0.1, 0.001);
        trie.write(&[1, 2, 4], &[5.0, 6.0, 7.0, 8.0], 0.1, 0.001);

        let data = trie.serialize();
        let trie2 = HierarchicalTrie::deserialize(&data).unwrap();

        assert_eq!(trie2.total_nodes(), trie.total_nodes());
        assert_eq!(trie2.n_records, trie.n_records);

        let path1 = trie.read_path(&[1, 2, 3]);
        let path2 = trie2.read_path(&[1, 2, 3]);
        assert_eq!(path1.len(), path2.len());
        assert_eq!(path1[3], path2[3]);
    }

    #[test]
    fn test_reset() {
        let mut trie = HierarchicalTrie::new(4, 8);
        trie.write(&[1, 2], &[1.0, 2.0, 3.0, 4.0], 0.1, 0.001);
        assert!(trie.total_nodes() > 1);

        trie.reset();
        assert_eq!(trie.total_nodes(), 1);
        assert_eq!(trie.n_records, 0);
    }

    #[test]
    fn test_ema_blending() {
        let mut trie = HierarchicalTrie::new(2, 8);
        let addr = vec![1u8];

        trie.write(&addr, &[10.0, 0.0], 0.5, 0.001);
        let v1 = trie.read_path(&addr)[1].to_vec();
        assert_eq!(v1, vec![10.0, 0.0]); // first write = exact copy

        trie.write(&addr, &[0.0, 10.0], 0.5, 0.001);
        let v2 = trie.read_path(&addr)[1].to_vec();
        // EMA: alpha ~= 0.5 / (1 + 0.01*2) ≈ 0.49
        // new = (1-0.49)*10 + 0.49*0 ≈ 5.1, (1-0.49)*0 + 0.49*10 ≈ 4.9
        assert!(v2[0] > 4.0 && v2[0] < 6.0);
        assert!(v2[1] > 4.0 && v2[1] < 6.0);
    }
}
