"""Traffic-intensity suspect targets - isolate whether paulikit's
per-chunk memory traffic (not MESIF coherence, not generic pool IPC)
is enough to reproduce the 2-physical-core ceiling.

Each workload uses the SAME ProcessPoolExecutor shape as
synthetic_ipc_control.py / parallel_decompose (pinned workers,
bounded in-flight, pickle-over-pipe results), with N_CHUNKS matching
the real N=150/chunk_size=2 outer DAG (5595). Only the per-task body
and return payload change:

  wht_small   - real _walsh_hadamard_transform_rows on a fresh
                (2, 16384) complex128 buffer; return 64 floats.
                Same footprint + 14 full-array stage touches as one
                paulikit chunk's WHT; IPC payload stays tiny.
  touch_small - 14 read-modify-write passes over the same (2,16384)
                buffer (no butterfly); return 64 floats.
                Isolates "touch the working set 14 times" traffic
                from WHT arithmetic structure.
  wht_large   - same WHT as wht_small, but return the FULL transformed
                (2,16384) complex array (pickle ~512 KiB/task).
                Isolates whether LARGE result IPC alone collapses
                scaling when compute traffic is already WHT-like.

Usage:
    OPENBLAS_NUM_THREADS=1 python traffic_intensity_target.py \\
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

# Real N=150 / chunk_size=2 outer DAG size (dag_gst_master_analysis.md).
N_CHUNKS = 5595
CHUNK_ROWS = 2
DIM = 16384  # 2**14
SMALL_RESULT = 64
WORKLOADS = ("wht_small", "touch_small", "wht_large")

condition = sys.argv[1]
workload = sys.argv[2]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"
assert workload in WORKLOADS, f"unknown workload {workload!r}"


def _wht_small(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buf = rng.standard_normal((CHUNK_ROWS, DIM)) + 1j * rng.standard_normal(
        (CHUNK_ROWS, DIM)
    )
    out = _walsh_hadamard_transform_rows(buf, overwrite_input=True)
    return out.real.ravel()[:SMALL_RESULT].astype(np.float64)


def _touch_small(seed: int) -> np.ndarray:
    """14 full-array RMW passes - same touch count as WHT stages,
    no butterfly dependence structure."""
    rng = np.random.default_rng(seed)
    buf = rng.standard_normal((CHUNK_ROWS, DIM)) + 1j * rng.standard_normal(
        (CHUNK_ROWS, DIM)
    )
    for _ in range(14):
        buf += 1.0
        buf -= 0.5
    return buf.real.ravel()[:SMALL_RESULT].astype(np.float64)


def _wht_large(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buf = rng.standard_normal((CHUNK_ROWS, DIM)) + 1j * rng.standard_normal(
        (CHUNK_ROWS, DIM)
    )
    return _walsh_hadamard_transform_rows(buf, overwrite_input=True)


_TASK = {
    "wht_small": _wht_small,
    "touch_small": _touch_small,
    "wht_large": _wht_large,
}[workload]


def _worker_init(cpu_list, next_pin_index):
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


n_workers, cpu_list = _CONDITIONS[condition]
next_pin_index = multiprocessing.Value("i", 0)

t0 = time.perf_counter()
n_done = 0
with ProcessPoolExecutor(
    max_workers=n_workers,
    initializer=_worker_init,
    initargs=(cpu_list, next_pin_index),
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
