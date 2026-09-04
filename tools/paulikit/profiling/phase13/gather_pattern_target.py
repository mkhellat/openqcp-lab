"""Gather-pattern isolation target - the fourth control from
traffic_intensity_findings.md's "Actual next isolation step" and
dag_gst_master_analysis.md section 3e.

Question under test: does reproducing paulikit's real IRREGULAR
gather/scatter access pattern (not just its dense buffer size/stage-
touch count, already refuted as sufficient by
traffic_intensity_findings.md) reproduce the 2-physical-core ceiling?

Each task here does exactly what one real _parallel_worker_chunk call
does for its gather/scatter step (fwht.py:1246-1255): zero a
(chunk_size, dim) complex128 buffer, then scatter `nnz` values at the
REAL (row_offset, column) positions that chunk actually writes to -
loaded from gather_pattern_chunks.npz (run gather_pattern_precompute.py
first). Values themselves are synthetic (fast RNG output, not real
Hamiltonian entries) - only the access PATTERN (which positions are
touched, in what order, how irregular) is under test.

Two workload variants:
  gather_only    - zero + scatter only, tiny IPC (64 floats). Isolates
                    the irregular-write cost alone.
  gather_and_wht - zero + scatter + real
                    _walsh_hadamard_transform_rows, tiny IPC (64
                    floats). Isolates gather+WHT together (closest
                    full-pipeline proxy without the real operator/
                    phase-multiply/threshold steps).

If EITHER fails to scale like paulikit (roughly flat/reversed at
w8_c4 vs w2_c1, unlike the traffic-intensity controls' 2.2-2.7x) ->
gather/irregular access is sufficient on its own. If both still scale
-> look at operator-sized resident set / per-worker footprint next
(per traffic_intensity_findings.md's own decision tree).

Usage:
    OPENBLAS_NUM_THREADS=1 python gather_pattern_target.py \\
        <condition> <workload>
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
SMALL_RESULT = 64
CHUNK_SIZE = 2  # matches gather_pattern_precompute.py's CHUNK_SIZE
WORKLOADS = ("gather_only", "gather_and_wht")

condition = sys.argv[1]
workload = sys.argv[2]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"
assert workload in WORKLOADS, f"unknown workload {workload!r}"

# Loaded once in the MAIN process, then handed to every worker
# explicitly via the pool's initializer/initargs (pickled once per
# worker at pool startup, not re-read from disk per worker) - this
# mirrors how paulikit's own _parallel_worker_init actually receives
# its operator/setup arrays (fwht.py's ProcessPoolExecutor call,
# initargs=(operator, ..., sorted_p_nz, sorted_q_nz, ...)), and avoids
# an artifact found while smoke-testing this script: this machine's
# default multiprocessing start method is "forkserver" (confirmed via
# multiprocessing.get_start_method()), NOT "fork" - workers do NOT
# inherit already-loaded module-level globals via copy-on-write under
# forkserver, so relying on module-level np.load() here would silently
# re-read the 3 MiB gather_pattern_chunks.npz file from disk fresh in
# EVERY worker, with that per-worker disk-I/O cost scaling with
# n_workers and contending under core-packing - exactly the kind of
# apples-to-oranges confound this whole investigation exists to avoid.
_chunks = np.load(CHUNKS_PATH)
DIM = int(_chunks["dim"][0])
N_CHUNKS = int(_chunks["n_chunks"][0])
_ALL_ROWS = [_chunks[f"rows_{i}"] for i in range(N_CHUNKS)]
_ALL_COLS = [_chunks[f"cols_{i}"] for i in range(N_CHUNKS)]

_worker_rows: list | None = None
_worker_cols: list | None = None


def _gather_only(chunk_index: int) -> np.ndarray:
    rows = _worker_rows[chunk_index]
    cols = _worker_cols[chunk_index]
    buf = np.zeros((CHUNK_SIZE, DIM), dtype=complex)
    rng = np.random.default_rng(chunk_index)
    values = rng.standard_normal(len(rows)) + 1j * rng.standard_normal(len(rows))
    buf[rows, cols] = values
    return buf.real.ravel()[:SMALL_RESULT].astype(np.float64)


def _gather_and_wht(chunk_index: int) -> np.ndarray:
    rows = _worker_rows[chunk_index]
    cols = _worker_cols[chunk_index]
    buf = np.zeros((CHUNK_SIZE, DIM), dtype=complex)
    rng = np.random.default_rng(chunk_index)
    values = rng.standard_normal(len(rows)) + 1j * rng.standard_normal(len(rows))
    buf[rows, cols] = values
    out = _walsh_hadamard_transform_rows(buf, overwrite_input=True)
    return out.real.ravel()[:SMALL_RESULT].astype(np.float64)


_TASK = {
    "gather_only": _gather_only,
    "gather_and_wht": _gather_and_wht,
}[workload]


def _worker_init(rows, cols, cpu_list, next_pin_index):
    global _worker_rows, _worker_cols
    _worker_rows = rows
    _worker_cols = cols
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
        initargs=(_ALL_ROWS, _ALL_COLS, cpu_list, next_pin_index),
    ) as pool:
        pending = iter(range(N_CHUNKS))
        in_flight: set = set()
        max_in_flight = max(1, 2 * n_workers)

        def _submit_next() -> bool:
            i = next(pending, None)
            if i is None:
                return False
            in_flight.add(pool.submit(_TASK, i))
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
    print(
        f"condition={condition} workload={workload} n_chunks={n_done} "
        f"elapsed={elapsed:.4f}s"
    )
