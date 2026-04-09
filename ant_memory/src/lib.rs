mod trie;
mod memory;

use pyo3::prelude::*;

/// ANT persistent memory system — Rust extension.
///
/// Hierarchical trie with EMA value vectors for the ANT byte-level transformer.
/// Arena-allocated, zero Python object overhead, mmap-friendly serialization.
#[pymodule]
fn ant_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<memory::MemorySystem>()?;
    Ok(())
}
