"""Paulikit-FREE synthetic control - the third open question from
code_specificity_findings.md: does the 2-physical-core wall-clock
ceiling reproduce on a workload that shares NOTHING with paulikit
except the coarse shape (many small CPU-bound tasks, small numpy
result payloads, ProcessPoolExecutor's default pickle-over-pipe IPC)?
If yes, the ceiling is a property of this machine's Python-
multiprocessing/OS-scheduling layer in general, not anything specific
to paulikit's own code - confirms code_specificity_findings.md's
"points at generic" conclusion rather than leaving it merely
suggested.

Task shape deliberately mimics one real paulikit chunk at N=150,
chunk_size=2 (full_optimum_sweep_results.jsonl: 91,652,096 total terms,
dim=16384/chunk_size=2 = 8192 chunks, ~2.5ms/chunk average): 8192
independent CPU-bound tasks, each doing a fixed amount of pure-Python-
+numpy busywork (no I/O, no shared state, no relation to Hadamard
transforms or Pauli decomposition whatsoever) and returning a small
numpy array - same number of tasks, same rough per-task result size,
same default ProcessPoolExecutor pickling path paulikit's own
parallel_decompose uses, same in_flight submission pattern (bounded
concurrency via wait(FIRST_COMPLETED), not one giant map()).

Usage:
    OPENBLAS_NUM_THREADS=1 python synthetic_ipc_control.py <condition_name> [chunk_work_size]

condition_name: any key of CONDITIONS/SWEEP_CONFIGS (condition_table.py).
chunk_work_size: number of busywork iterations per task (default tuned
to land near paulikit's own ~2.5ms/chunk average - see BUSYWORK_N).
"""
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import CONDITIONS as _CONDITIONS  # noqa: E402

N_CHUNKS = 8192  # matches paulikit's own N=150/chunk_size=2 chunk count
RESULT_SIZE = 64  # small numpy result array per task, same order of
                   # magnitude as one paulikit chunk's surviving-term output

condition = sys.argv[1]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"
BUSYWORK_N = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def _busywork(n: int, seed: int) -> np.ndarray:
    """Pure CPU-bound work, no I/O, no shared memory, no relation to
    paulikit's math - a fixed-size dense matmul repeated `n` times,
    same order of magnitude of raw FLOPs as one real WHT chunk at this
    scale, then reduced down to a small RESULT_SIZE-length array (so
    the IPC payload size mimics a real chunk's surviving-term output,
    not the full working array).
    """
    rng = np.random.default_rng(seed)
    a = rng.random((48, 48))
    b = rng.random((48, 48))
    acc = np.zeros((48, 48))
    for _ in range(n):
        acc += a @ b
    return acc.ravel()[:RESULT_SIZE]


def _worker_init(cpu_list, next_pin_index):
    """Runs once per worker process - each worker claims the next
    unused index into cpu_list via the shared counter (same mechanism
    paulikit's own _parallel_worker_init uses) and pins itself to that
    one physical core's representative CPU."""
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
results = []
with ProcessPoolExecutor(
    max_workers=n_workers, initializer=_worker_init,
    initargs=(cpu_list, next_pin_index),
) as pool:
    pending = iter(range(N_CHUNKS))
    in_flight = set()
    max_in_flight = n_workers * 2

    def _submit_next():
        i = next(pending, None)
        if i is None:
            return False
        in_flight.add(pool.submit(_busywork, BUSYWORK_N, i))
        return True

    for _ in range(max_in_flight):
        if not _submit_next():
            break

    while in_flight:
        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            results.append(future.result())
            _submit_next()

elapsed = time.perf_counter() - t0
print(f"condition={condition} n_chunks={len(results)} elapsed={elapsed:.4f}s "
      f"busywork_n={BUSYWORK_N}")
