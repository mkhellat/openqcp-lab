"""Controlled L3-contention isolation test - direct evidence, not
inference by elimination (per direct user instruction: "do you have
full evidence for the L3 contention? ... Lets collect evidence
instead of assuming").

Design: run ONE pinned worker processing exactly 1/4 of the real
N=150 workload (matching parallel_decompose's real per-worker share
at n_workers=4), alone vs. with 3 CPU-bound "noise" processes pinned
to the OTHER 3 physical cores. The noise processes touch large memory
buffers to generate real cache/memory traffic but never share a
physical core (hence zero L1/L2 sharing) with the measured worker -
isolating whether cross-core L3/memory-bandwidth pressure ALONE
(no hyperthread-sibling sharing at all) degrades the measured
worker's cache-miss ratio. If it does, that is direct L3 evidence,
not elimination-by-absence.

Usage (foreground only, one job at a time, per established discipline):
    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,LLC-loads,LLC-load-misses \
python l3_contention_isolation_test.py alone
    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,LLC-loads,LLC-load-misses \
python l3_contention_isolation_test.py with_noise
"""
import multiprocessing
import os
import sys
import time

from paulikit.algorithms.fwht import _iter_chunked_coefficients, _prepare_operator_for_fwht
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

import numpy as np

N_OSCILLATORS = 150
CHUNK_SIZE = 2
N_WORKERS_SIMULATED = 4  # this worker does 1/4 of the real workload

mode = sys.argv[1]
assert mode in ("alone", "with_noise")


def _noise_worker(cpu: int, stop_flag) -> None:
    """Pinned to one physical core (distinct from the measured
    worker's core), continuously touches a large buffer to generate
    real cache/memory traffic - big enough to spill past L1/L2/L3, so
    this is genuine memory-subsystem pressure, not a no-op spin loop."""
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass
    buf = np.random.rand(8_000_000)  # ~64 MB, well past any per-core L2/L3 share
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

    # This process's own share: chunks [0, n_active//4) - the same
    # slice worker 0 would get in a real 4-way parallel_decompose run.
    share_end = n_active // N_WORKERS_SIMULATED

    total_terms = 0
    for chunk_x, chunk_z, chunk_coeff in _iter_chunked_coefficients(
        operator, is_sparse_input, active_x, inverse, p_nz, q_nz, dim, n_qubits,
        n_active, z_indices, CHUNK_SIZE, 1e-10, None,
    ):
        total_terms += len(chunk_x)
        if chunk_x.size and active_x[share_end - 1] < chunk_x[0]:
            break  # past our 1/4 share

    return total_terms


try:
    os.sched_setaffinity(0, {0})  # measured worker pinned to physical core A (logical CPU 0)
except (AttributeError, OSError):
    pass

noise_procs = []
stop_flag = None
if mode == "with_noise":
    stop_flag = multiprocessing.Value("b", False)
    for cpu in (1, 2, 3):  # physical cores B, C, D - distinct from A, zero L1/L2 sharing with worker
        p = multiprocessing.Process(target=_noise_worker, args=(cpu, stop_flag))
        p.start()
        noise_procs.append(p)
    time.sleep(1.0)  # let noise processes ramp up before measuring

t0 = time.perf_counter()
n = _measured_worker_share()
elapsed = time.perf_counter() - t0

if mode == "with_noise":
    stop_flag.value = True
    for p in noise_procs:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()

print(f"mode={mode} terms={n} elapsed={elapsed:.4f}s")
