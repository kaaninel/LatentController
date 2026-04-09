/// Python-facing memory system wrapping the hierarchical trie.
///
/// Provides batch read/write operations that interface with PyTorch tensors
/// via numpy arrays. Thread-safe via internal mutex.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::trie::HierarchicalTrie;

/// Python-visible memory system.
#[pyclass]
pub struct MemorySystem {
    trie: Mutex<HierarchicalTrie>,
    data_path: PathBuf,
    bin_path: PathBuf,
    d_model: usize,
    depth_cap: usize,
    alpha_base: f32,
    alpha_min: f32,
    flush_interval: u64,
    write_count: Mutex<u64>,
}

#[pymethods]
impl MemorySystem {
    /// Create a new memory system.
    ///
    /// Args:
    ///     data_path: Directory for persistent storage
    ///     d_model: Dimension of value vectors (default 128)
    ///     depth_cap: Maximum trie depth (default 8)
    ///     alpha_base: Base EMA momentum (default 0.1)
    ///     alpha_min: Minimum EMA alpha (default 0.001)
    ///     flush_interval: Auto-flush every N writes (default 1000)
    #[new]
    #[pyo3(signature = (data_path, d_model=128, depth_cap=8, alpha_base=0.1, alpha_min=0.001, flush_interval=1000))]
    fn new(
        data_path: &str,
        d_model: usize,
        depth_cap: usize,
        alpha_base: f32,
        alpha_min: f32,
        flush_interval: u64,
    ) -> PyResult<Self> {
        fs::create_dir_all(data_path)?;
        let bin_path = PathBuf::from(data_path).join("memory.bin");

        let trie = if bin_path.exists() {
            let data = fs::read(&bin_path)?;
            HierarchicalTrie::deserialize(&data)
                .unwrap_or_else(|| HierarchicalTrie::new(d_model, depth_cap))
        } else {
            HierarchicalTrie::new(d_model, depth_cap)
        };

        Ok(Self {
            trie: Mutex::new(trie),
            data_path: PathBuf::from(data_path),
            bin_path,
            d_model,
            depth_cap,
            alpha_base,
            alpha_min,
            flush_interval,
            write_count: Mutex::new(0),
        })
    }

    /// Write a value to multiple address paths.
    ///
    /// Args:
    ///     addresses: list of N numpy arrays, each shape (depth,) uint8
    ///     value: numpy array shape (d_model,) float32
    fn write(&self, addresses: Vec<PyReadonlyArray1<u8>>, value: PyReadonlyArray1<f32>) -> PyResult<()> {
        let val = value.as_slice()?;
        let mut trie = self.trie.lock().unwrap();

        for addr in &addresses {
            let a = addr.as_slice()?;
            trie.write(a, val, self.alpha_base, self.alpha_min);
        }

        let mut wc = self.write_count.lock().unwrap();
        *wc += 1;
        if *wc % self.flush_interval == 0 {
            let data = trie.serialize();
            drop(trie); // release trie lock before file I/O
            fs::write(&self.bin_path, data)?;
        }

        Ok(())
    }

    /// Batch write: B items, each with N address paths.
    ///
    /// Args:
    ///     batch_addresses: list of B lists, each containing N numpy arrays of shape (depth,) uint8
    ///     batch_values: numpy array shape (B, d_model) float32
    fn write_batch(
        &self,
        batch_addresses: Vec<Vec<PyReadonlyArray1<u8>>>,
        batch_values: PyReadonlyArray2<f32>,
    ) -> PyResult<()> {
        let values = batch_values.as_array();
        let mut trie = self.trie.lock().unwrap();

        for (b, addrs) in batch_addresses.iter().enumerate() {
            let val = values.row(b);
            let val_slice = val.as_slice().unwrap();
            for addr in addrs {
                let a = addr.as_slice()?;
                trie.write(a, val_slice, self.alpha_base, self.alpha_min);
            }
        }

        let mut wc = self.write_count.lock().unwrap();
        *wc += batch_addresses.len() as u64;
        if *wc % self.flush_interval == 0 {
            let data = trie.serialize();
            drop(trie);
            fs::write(&self.bin_path, data)?;
        }

        Ok(())
    }

    /// Read memory for multiple address paths.
    ///
    /// Collects full ancestor path from each address, deduplicates root,
    /// pads/truncates to max_vectors.
    ///
    /// Args:
    ///     addresses: list of N numpy arrays, each shape (depth,) uint8
    ///     max_vectors: maximum number of vectors to return (default 25)
    ///
    /// Returns: (vectors, mask)
    ///     vectors: numpy array (max_vectors, d_model) float32
    ///     mask: numpy array (max_vectors,) bool
    #[pyo3(signature = (addresses, max_vectors=25))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        addresses: Vec<PyReadonlyArray1<u8>>,
        max_vectors: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>)> {
        let trie = self.trie.lock().unwrap();
        let d = self.d_model;

        let mut all_vectors: Vec<&[f32]> = Vec::new();
        let mut root_added = false;

        for addr in &addresses {
            let a = addr.as_slice()?;
            let path = trie.read_path(a);

            if !root_added && !path.is_empty() {
                all_vectors.extend_from_slice(&path);
                root_added = true;
            } else if path.len() > 1 {
                all_vectors.extend_from_slice(&path[1..]);
            }
        }

        // Build output arrays
        let n = all_vectors.len().min(max_vectors);
        let mut vec_data = vec![0.0f32; max_vectors * d];
        let mut mask_data = vec![false; max_vectors];

        for i in 0..n {
            vec_data[i * d..(i + 1) * d].copy_from_slice(all_vectors[i]);
            mask_data[i] = true;
        }

        let vectors = PyArray2::from_owned_array(py, Array2::from_shape_vec((max_vectors, d), vec_data).unwrap());
        let mask = PyArray1::from_owned_array(py, Array1::from_vec(mask_data));

        Ok((vectors, mask))
    }

    /// Batch read: B items.
    ///
    /// Args:
    ///     batch_addresses: list of B lists, each containing N numpy arrays of shape (depth,) uint8
    ///     max_vectors: maximum number of vectors per item (default 25)
    ///
    /// Returns: (vectors, mask)
    ///     vectors: numpy array (B, max_vectors, d_model) float32
    ///     mask: numpy array (B, max_vectors) bool
    #[pyo3(signature = (batch_addresses, max_vectors=25))]
    fn read_batch<'py>(
        &self,
        py: Python<'py>,
        batch_addresses: Vec<Vec<PyReadonlyArray1<u8>>>,
        max_vectors: usize,
    ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<bool>>)> {
        let trie = self.trie.lock().unwrap();
        let b_size = batch_addresses.len();
        let d = self.d_model;

        let mut vec_data = vec![0.0f32; b_size * max_vectors * d];
        let mut mask_data = vec![false; b_size * max_vectors];

        for (b, addrs) in batch_addresses.iter().enumerate() {
            let mut all_vectors: Vec<&[f32]> = Vec::new();
            let mut root_added = false;

            for addr in addrs {
                let a = addr.as_slice()?;
                let path = trie.read_path(a);

                if !root_added && !path.is_empty() {
                    all_vectors.extend_from_slice(&path);
                    root_added = true;
                } else if path.len() > 1 {
                    all_vectors.extend_from_slice(&path[1..]);
                }
            }

            let n = all_vectors.len().min(max_vectors);
            for i in 0..n {
                let offset = b * max_vectors * d + i * d;
                vec_data[offset..offset + d].copy_from_slice(all_vectors[i]);
                mask_data[b * max_vectors + i] = true;
            }
        }

        let vectors = PyArray3::from_owned_array(py, Array3::from_shape_vec((b_size, max_vectors, d), vec_data).unwrap());
        let mask = PyArray2::from_owned_array(py, Array2::from_shape_vec((b_size, max_vectors), mask_data).unwrap());

        Ok((vectors, mask))
    }

    /// Flush trie to disk.
    fn flush(&self) -> PyResult<()> {
        let trie = self.trie.lock().unwrap();
        let data = trie.serialize();
        drop(trie);
        fs::write(&self.bin_path, data)?;
        Ok(())
    }

    /// Reset all memory.
    fn reset(&self) {
        let mut trie = self.trie.lock().unwrap();
        trie.reset();
        *self.write_count.lock().unwrap() = 0;
    }

    /// Total write records.
    fn total_entries(&self) -> u64 {
        self.trie.lock().unwrap().n_records
    }

    /// Total nodes in trie.
    fn total_nodes(&self) -> usize {
        self.trie.lock().unwrap().total_nodes()
    }

    /// Get d_model dimension.
    fn d_model(&self) -> usize {
        self.d_model
    }

    /// String representation for debugging.
    fn __repr__(&self) -> String {
        let trie = self.trie.lock().unwrap();
        format!(
            "MemorySystem(d_model={}, nodes={}, records={}, path={:?})",
            self.d_model,
            trie.total_nodes(),
            trie.n_records,
            self.data_path,
        )
    }
}
