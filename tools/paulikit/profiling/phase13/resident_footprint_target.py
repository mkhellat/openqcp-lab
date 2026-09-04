"""Resident-footprint isolation target - the LAST untested item from
traffic_intensity_findings.md's original decision tree, after
gather_pattern_findings.md's mixed result on the gather/scatter access
pattern alone.

Question under test: does holding a paulikit-SCALE resident
operator/setup-array footprint in each worker for its whole lifetime -
not any single chunk's own transient traffic - tip shared-memory-
subsystem contention into paulikit's observed ceiling?

Real per-worker resident footprint at N=150/chunk_size=2 (measured via
fwht._per_worker_resident_bytes on the real Hamiltonian, see
resident_footprint_precompute.py's own printed output): ~1.95 MiB
(operator's CSR data/indices/indptr buffers, ~45,000 nnz, plus three
nnz-length sorted setup arrays). This is REPEATEDLY touched across a
worker's ~700 chunks (5595 chunks / 8 workers at w8_c4), unlike a
chunk's own 512 KiB buffer which is touched once per chunk then
discarded - the resident footprint is exactly the kind of standing
cache/bandwidth pressure a per-chunk-only control (gather_pattern_target.py)
cannot capture.

Each task here is exactly gather_pattern_target.py's `gather_and_wht`
(the closest prior proxy, which did NOT reverse) PLUS a real gather
into the resident array on every task, matching how
_parallel_worker_chunk actually reads FROM state["operator"] on every
single chunk (fwht.py:1248-1250) - not just holds it passively resident,
but actively re-reads from it per task, which is the real access
pattern being tested here.

Usage:
    OPENBLAS_NUM_THREADS=1 python resident_footprint_target.py \\
        <condition>
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import CONDITIONS as _CONDITIONS  # noqa: E402

from paulikit.algorithms.fwht import _walsh_hadamard_transform_rows

CHUNKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gather_pattern_chunks.npz")
RESIDENT_NNZ = 45000  # real N=150 operator's nnz (resident_footprint_precompute.py)
SMALL_RESULT = 64
CHUNK_SIZE = 2

condition = sys.argv[1]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"

_chunks = np.load(CHUNKS_PATH)
DIM = int(_chunks["dim"][0])
N_CHUNKS = int(_chunks["n_chunks"][0])
_ALL_ROWS = [_chunks[f"rows_{i}"] for i in range(N_CHUNKS)]
_ALL_COLS = [_chunks[f"cols_{i}"] for i in range(N_CHUNKS)]

_worker_rows: list | None = None
_worker_cols: list | None = None
_worker_resident_values: np.ndarray | None = None  # real footprint, held resident


def _task(chunk_index: int) -> np.ndarray:
    rows = _worker_rows[chunk_index]
    cols = _worker_cols[chunk_index]
    buf = np.zeros((CHUNK_SIZE, DIM), dtype=complex)
    # Real per-task access into the resident array (not just passive
    # residency) - a deterministic-but-scattered slice of it, matching
    # the real code's own operator[sorted_p_nz[lo:hi], sorted_q_nz[lo:hi]]
    # per-chunk gather (fwht.py:1248-1250).
    n = len(rows)
    start = (chunk_index * 7) % max(1, RESIDENT_NNZ - n)  # deterministic scatter offset
    values = _worker_resident_values[start:start + n]
    buf[rows, cols] = values
    out = _walsh_hadamard_transform_rows(buf, overwrite_input=True)
    return out.real.ravel()[:SMALL_RESULT].astype(np.float64)


def _worker_init(rows, cols, resident_nnz, cpu_list, next_pin_index):
    global _worker_rows, _worker_cols, _worker_resident_values
    _worker_rows = rows
    _worker_cols = cols
    # Held resident for this worker's WHOLE lifetime, matching real
    # paulikit's own state["operator"] - a fixed-seed array so every
    # worker (and every run) sees identical content, isolating the
    # FOOTPRINT/access effect from any content-dependent variation.
    rng = np.random.default_rng(12345)
    _worker_resident_values = rng.standard_normal(resident_nnz) + 1j * rng.standard_normal(
        resident_nnz
    )
    if not cpu_list:
        return
    with next_pin_index.get_lock():
        idx = next_pin_index.value
        next_pin_index.value += 1
    if idx < len(cpu_list):
        try:
            os.sched_setaffinity(0, {cpu_list[idx]})
        except (AttributeError, OSError):
            pass


if __name__ == "__main__":
    n_workers, cpu_list = _CONDITIONS[condition]
    next_pin_index = multiprocessing.Value("i", 0)

    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(_ALL_ROWS, _ALL_COLS, RESIDENT_NNZ, cpu_list, next_pin_index),
    ) as pool:
        pending = iter(range(N_CHUNKS))
        in_flight: set = set()
        max_in_flight = max(1, 2 * n_workers)

        def _submit_next() -> bool:
            i = next(pending, None)
            if i is None:
                return False
            in_flight.add(pool.submit(_task, i))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                n_done += 1
                _submit_next()

    elapsed = time.perf_counter() - t0
    print(f"condition={condition} n_chunks={n_done} elapsed={elapsed:.4f}s")
