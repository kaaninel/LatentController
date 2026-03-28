"""
Trie-based persistent memory system.

Layout on disk:
  data_path/data.bin  — flat binary, 512 bytes per record (int8)
  data_path/meta.bin  — per-record write_count (uint16, 2 bytes each)

Three TrieIndex objects correspond to the three address heads.
"""

import os
import struct
import threading
from typing import List, Optional, Tuple

import numpy as np

from config import MemoryConfig


# ---------------------------------------------------------------------------
# TrieNode
# ---------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "record_number", "write_count", "flags")

    def __init__(self):
        self.children: dict = {}
        self.record_number: int = -1
        self.write_count: int = 0
        self.flags: int = 0


# ---------------------------------------------------------------------------
# TrieIndex
# ---------------------------------------------------------------------------

class TrieIndex:
    """Trie indexed over 8 signed bytes (int8 range −128…127, stored as uint8)."""

    def __init__(self):
        self.root = TrieNode()

    def _key(self, address_bytes: bytes) -> bytes:
        """Normalise signed int8 address to uint8 key bytes."""
        return bytes(b & 0xFF for b in address_bytes)

    def insert(self, address_bytes: bytes, record_number: int, write_count: int):
        key = self._key(address_bytes)
        node = self.root
        for byte in key:
            node = node.children.setdefault(byte, TrieNode())
        node.record_number = record_number
        node.write_count = write_count

    def lookup(self, address_bytes: bytes) -> Optional[TrieNode]:
        key = self._key(address_bytes)
        node = self.root
        for byte in key:
            node = node.children.get(byte)
            if node is None:
                return None
        return node if node.record_number >= 0 else None

    def find_nearest(
        self,
        address_bytes: bytes,
        k: int = 2,
        coarse_dims: int = 4,
    ) -> List[Tuple[int, int, int]]:
        """
        Exact match on dimensions 0..coarse_dims-1.
        ±1 search on dimensions coarse_dims..7.
        Returns list of (hamming_dist, record_number, write_count) sorted by dist.
        """
        key = list(self._key(address_bytes))
        results: List[Tuple[int, int, int]] = []

        def _search(node: TrieNode, depth: int, dist: int):
            if depth == 8:
                if node.record_number >= 0:
                    results.append((dist, node.record_number, node.write_count))
                return
            byte = key[depth]
            if depth < coarse_dims:
                # exact only
                child = node.children.get(byte)
                if child is not None:
                    _search(child, depth + 1, dist)
            else:
                # ±1 neighbourhood
                for delta in (-1, 0, 1):
                    candidate = (byte + delta) & 0xFF
                    child = node.children.get(candidate)
                    if child is not None:
                        _search(child, depth + 1, dist + abs(delta))

        _search(self.root, 0, 0)
        results.sort(key=lambda t: t[0])
        return results[:k]


# ---------------------------------------------------------------------------
# MemorySystem
# ---------------------------------------------------------------------------

RECORD_SIZE = 512   # bytes (int8)


class MemorySystem:
    def __init__(self, data_path: str, cfg: MemoryConfig):
        self.data_path = data_path
        self.cfg = cfg
        os.makedirs(data_path, exist_ok=True)

        self._data_file = os.path.join(data_path, "data.bin")
        self._meta_file = os.path.join(data_path, "meta.bin")
        self._lock = threading.Lock()

        # Three trie indexes (one per address head)
        self.indexes: List[TrieIndex] = [TrieIndex() for _ in range(3)]

        # Count of stored records
        self._n_records = self._count_records()

        # Rebuild trie indexes from disk is not done here to keep startup fast.
        # Callers should call rebuild_indexes() after loading a checkpoint if
        # they need consistent lookup from a previous session.

    # ------------------------------------------------------------------
    # Disk helpers
    # ------------------------------------------------------------------

    def _count_records(self) -> int:
        if not os.path.exists(self._data_file):
            return 0
        size = os.path.getsize(self._data_file)
        return size // RECORD_SIZE

    def _read_record(self, record_number: int) -> np.ndarray:
        offset = record_number * RECORD_SIZE
        with open(self._data_file, "rb") as f:
            f.seek(offset)
            data = f.read(RECORD_SIZE)
        return np.frombuffer(data, dtype=np.int8).copy()

    def _write_record(self, record_number: int, vec: np.ndarray):
        offset = record_number * RECORD_SIZE
        with open(self._data_file, "r+b") as f:
            f.seek(offset)
            f.write(vec.astype(np.int8).tobytes())

    def _append_record(self, vec: np.ndarray) -> int:
        """Append a new record, return its record number."""
        with self._lock:
            rec_num = self._n_records
            with open(self._data_file, "ab") as f:
                f.write(vec.astype(np.int8).tobytes())
            # Append write_count = 1 to meta
            with open(self._meta_file, "ab") as f:
                f.write(struct.pack("<H", 1))
            self._n_records += 1
        return rec_num

    def _read_write_count(self, record_number: int) -> int:
        offset = record_number * 2
        if not os.path.exists(self._meta_file):
            return 0
        with open(self._meta_file, "rb") as f:
            f.seek(offset)
            data = f.read(2)
        if len(data) < 2:
            return 0
        return struct.unpack("<H", data)[0]

    def _write_write_count(self, record_number: int, count: int):
        offset = record_number * 2
        with open(self._meta_file, "r+b") as f:
            f.seek(offset)
            f.write(struct.pack("<H", min(count, self.cfg.max_write_count)))

    # ------------------------------------------------------------------
    # Read memory
    # ------------------------------------------------------------------

    def read_memory(
        self, addresses: List[bytes]
    ) -> List[np.ndarray]:
        """
        addresses : list of 3 byte-strings (one per address head)
        Returns up to 9 int8 numpy arrays of shape (512,).
        Probes: 3 heads × (1 exact + 2 nearest neighbours) = up to 9.
        Deduplicated by record number, padded with zeros if fewer than 9 found.
        """
        seen = set()
        vecs = []
        for head_idx, addr in enumerate(addresses):
            idx = self.indexes[head_idx]
            # exact
            node = idx.lookup(addr)
            if node is not None and node.record_number not in seen:
                seen.add(node.record_number)
                vecs.append(self._read_record(node.record_number))

            # k nearest neighbours
            for _, rec_num, _ in idx.find_nearest(
                addr, k=self.cfg.neighbor_k,
                coarse_dims=self.cfg.coarse_dims
            ):
                if rec_num not in seen:
                    seen.add(rec_num)
                    vecs.append(self._read_record(rec_num))

        # Pad to n_mem_slots
        while len(vecs) < self.cfg.n_mem_slots:
            vecs.append(np.zeros(RECORD_SIZE, dtype=np.int8))

        return vecs[:self.cfg.n_mem_slots]

    # ------------------------------------------------------------------
    # Write memory
    # ------------------------------------------------------------------

    def write_memory(self, addresses: List[bytes], vector: np.ndarray):
        """
        addresses : list of 3 byte-strings (one per address head)
        vector    : float32 numpy array of shape (512,) — will be scaled to int8
        """
        # Scale vector to int8
        scale = np.abs(vector).max()
        if scale < 1e-6:
            return
        int8_vec = np.clip(np.round(vector / scale * 127.0), -128, 127).astype(np.int8)

        # Determine if any head already has this address
        existing_rec = None
        for head_idx, addr in enumerate(addresses):
            node = self.indexes[head_idx].lookup(addr)
            if node is not None:
                existing_rec = (head_idx, node.record_number, node.write_count)
                break

        if existing_rec is None:
            # New record
            rec_num = self._append_record(int8_vec)
            for head_idx, addr in enumerate(addresses):
                self.indexes[head_idx].insert(addr, rec_num, 1)
        else:
            _, rec_num, write_count = existing_rec
            # Adaptive EMA blend
            effective_count = min(write_count, self.cfg.write_count_decay_cap)
            alpha = self.cfg.alpha_base / (1.0 + self.cfg.write_count_decay_rate * effective_count)
            alpha = max(alpha, 0.001)

            old_vec = self._read_record(rec_num)

            # int16 intermediate to avoid overflow
            blended = (
                (1.0 - alpha) * old_vec.astype(np.int16) +
                alpha * int8_vec.astype(np.int16)
            )
            new_vec = np.clip(np.round(blended), -128, 127).astype(np.int8)
            self._write_record(rec_num, new_vec)

            new_count = min(write_count + 1, self.cfg.max_write_count)
            self._write_write_count(rec_num, new_count)
            for head_idx, addr in enumerate(addresses):
                self.indexes[head_idx].insert(addr, rec_num, new_count)
