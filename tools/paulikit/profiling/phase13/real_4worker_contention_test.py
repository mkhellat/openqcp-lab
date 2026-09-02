"""Tests the L3-capacity/bandwidth contention finding
(l3_capacity_vs_bandwidth_findings.md) at the REAL 4-worker
parallel_decompose scale, not the synthetic 1-worker+noise proxy -
direct follow-up requested after that finding's own "what this still
does NOT show" item.

Uses the EXACT SAME production functions parallel_decompose itself
calls (_parallel_worker_init, _parallel_worker_chunk) rather than
reimplementing the per-chunk logic - guarantees this test measures the
real code path, not an approximation of it. Four real processes, each
pinned to a distinct physical core, each processing its own real,
disjoint slice of chunk indices (matching how parallel_decompose's own
ProcessPoolExecutor divides work), each individually timed.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses python real_4worker_contention_test.py
"""
import multiprocessing
import os
import time

import numpy as np

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import _parallel_worker_chunk, _parallel_worker_init
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
N_WORKERS = 4


def _worker(cpu: int, worker_index: int, chunk_starts, n_active, results, barrier) -> None:
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass

    spring_constants = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = fwht._prepare_operator_for_fwht(
        padded
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_p_nz = p_nz[order]
    sorted_q_nz = q_nz[order]
    z_indices = np.arange(dim)[np.newaxis, :]

    _parallel_worker_init(
        operator, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
        active_x, dim, n_qubits, z_indices, 1e-10, None, None,
    )

    my_chunk_indices = list(range(worker_index, len(chunk_starts), N_WORKERS))

    barrier.wait()  # all 4 processes start their measured work at the same instant
    t0 = time.perf_counter()
    total_terms = 0
    for chunk_index in my_chunk_indices:
        chunk_start = chunk_starts[chunk_index]
        chunk_end = min(chunk_start + CHUNK_SIZE, n_active)
        _, chunk_x_out, _, _ = _parallel_worker_chunk(chunk_index, chunk_start, chunk_end)
        total_terms += len(chunk_x_out)
    elapsed = time.perf_counter() - t0
    results[worker_index] = (elapsed, total_terms)


if __name__ == "__main__":
    spring_constants = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = fwht._prepare_operator_for_fwht(
        padded
    )
    active_x, _ = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    chunk_starts = list(range(0, n_active, CHUNK_SIZE))

    manager = multiprocessing.Manager()
    results = manager.dict()
    barrier = multiprocessing.Barrier(N_WORKERS)

    procs = []
    for i in range(N_WORKERS):
        p = multiprocessing.Process(
            target=_worker, args=(i, i, chunk_starts, n_active, results, barrier)
        )
        p.start()
        procs.append(p)

    t0 = time.perf_counter()
    for p in procs:
        p.join()
    overall_elapsed = time.perf_counter() - t0

    print(f"overall_wall_clock={overall_elapsed:.4f}s n_active={n_active} n_chunks={len(chunk_starts)}")
    for i in range(N_WORKERS):
        elapsed, terms = results[i]
        print(f"  worker {i} (physical core {'ABCD'[i]}): elapsed={elapsed:.4f}s terms={terms} "
              f"chunks_assigned={len(list(range(i, len(chunk_starts), N_WORKERS)))}")
