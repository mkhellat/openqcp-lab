"""Separates L3-CAPACITY contention from memory-BANDWIDTH contention -
directly requested follow-up to l3_contention_isolation_test.py, which
proved cross-core contention is real but could not distinguish these
two mechanisms (both are consistent with a tripled LLC-miss ratio).

This machine's real cache sizes (checked directly via lscpu, not
assumed): per-core L2 = 256 KiB, shared L3 = 8 MiB total (1 instance
across all 4 physical cores).

Three conditions, one script, run fresh in one experiment (per direct
instruction not to rely on old data):

1. alone: no noise processes.
2. noise_l2_bound: 3 noise processes (physical cores B, C, D), each
   repeatedly touching a 64 KiB buffer - well under this core's own
   256 KiB L2, so each noise process's own working set NEVER spills
   into L3 or DRAM. If the measured worker still degrades under this
   condition, that is evidence of BANDWIDTH/interconnect contention
   (something even L2-resident traffic contends for), not L3 capacity.
3. noise_l3_exceeding: 3 noise processes, each touching a 64 MB
   buffer - far exceeding the entire 8 MiB shared L3, guaranteeing
   real eviction pressure on whatever the measured worker has resident
   in L3, plus real DRAM traffic. If degradation here is meaningfully
   WORSE than the l2_bound condition, that is evidence of a genuine
   L3-CAPACITY-specific effect on top of any bandwidth effect.

Usage (foreground only, one job at a time):
    OPENBLAS_NUM_THREADS=1 perf stat --no-inherit -e task-clock,cycles,\
instructions,cache-references,cache-misses,L1-dcache-loads,\
L1-dcache-load-misses,LLC-loads,LLC-load-misses \
python l3_capacity_vs_bandwidth_test.py <alone|noise_l2_bound|noise_l3_exceeding>
"""
import multiprocessing
import os
import sys
import time

import numpy as np

from paulikit.algorithms.fwht import _iter_chunked_coefficients, _prepare_operator_for_fwht
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
N_WORKERS_SIMULATED = 4  # this worker does 1/4 of the real workload

mode = sys.argv[1]
assert mode in ("alone", "noise_l2_bound", "noise_l3_exceeding")

# Buffer sizes in float64 elements (8 bytes each).
L2_BOUND_ELEMENTS = 8_000       # 64 KiB - well under this core's 256 KiB L2
L3_EXCEEDING_ELEMENTS = 8_000_000  # ~64 MB - far exceeds the 8 MiB shared L3


def _noise_worker(cpu: int, n_elements: int, stop_flag) -> None:
    """Pinned to one physical core, continuously touches a buffer of
    the given size - sized to stay within L2 (noise_l2_bound) or to
    far exceed shared L3 (noise_l3_exceeding), per the caller."""
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass
    buf = np.random.rand(n_elements)
    while not stop_flag.value:
        buf *= 1.0000001
        buf += 1e-12


def _measured_worker_share():
    spring_constants = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(padded)
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]

    share_end = n_active // N_WORKERS_SIMULATED

    total_terms = 0
    for chunk_x, chunk_z, chunk_coeff in _iter_chunked_coefficients(
        operator, is_sparse_input, active_x, inverse, p_nz, q_nz, dim, n_qubits,
        n_active, z_indices, CHUNK_SIZE, 1e-10, None,
    ):
        total_terms += len(chunk_x)
        if chunk_x.size and active_x[share_end - 1] < chunk_x[0]:
            break

    return total_terms


try:
    os.sched_setaffinity(0, {0})  # measured worker pinned to physical core A (logical CPU 0)
except (AttributeError, OSError):
    pass

noise_procs = []
stop_flag = None
if mode != "alone":
    n_elements = L2_BOUND_ELEMENTS if mode == "noise_l2_bound" else L3_EXCEEDING_ELEMENTS
    stop_flag = multiprocessing.Value("b", False)
    for cpu in (1, 2, 3):  # physical cores B, C, D
        p = multiprocessing.Process(target=_noise_worker, args=(cpu, n_elements, stop_flag))
        p.start()
        noise_procs.append(p)
    time.sleep(1.0)

t0 = time.perf_counter()
n = _measured_worker_share()
elapsed = time.perf_counter() - t0

if mode != "alone":
    stop_flag.value = True
    for p in noise_procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()

print(f"mode={mode} terms={n} elapsed={elapsed:.4f}s")
