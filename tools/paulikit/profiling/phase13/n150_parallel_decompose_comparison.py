"""Real N=150 wall-clock comparison: sequential fwht_pauli_terms_iter
vs. parallel_decompose - same methodology as
n100_parallel_decompose_comparison.py, at the larger scale where
Phase 12 found the auto-tuned chunk_size climbs to a larger per-chunk
working set (more per-task work, a better test of whether pool
overhead is masking real parallel speedup at N=100's much smaller
chunk_size=3).
"""
import time

from paulikit.algorithms import autotune, fwht
from paulikit.algorithms.fwht import (
    _detect_available_worker_count,
    fwht_pauli_terms_iter,
    parallel_decompose,
)
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
REPS = 2

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
dim = padded.shape[0]

auto_chunk_size = autotune.recommended_chunk_size(dim)
n_workers = _detect_available_worker_count()
per_worker_budget = autotune.per_worker_memory_budget_bytes(n_workers)

print(f"N={N_OSCILLATORS} dim={dim} auto_chunk_size={auto_chunk_size} "
      f"n_workers={n_workers} "
      f"available_memory_bytes={autotune.available_memory_bytes():,} "
      f"per_worker_memory_budget_bytes={per_worker_budget:,}")


def time_sequential(reps):
    def run():
        total = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=auto_chunk_size):
            total += len(chunk)
        return total

    times = []
    n = None
    for _ in range(reps):
        t0 = time.perf_counter()
        n = run()
        times.append(time.perf_counter() - t0)
    return times, n


def time_parallel(reps, workers):
    def run():
        total = 0
        for chunk in parallel_decompose(padded, chunk_size=auto_chunk_size, n_workers=workers):
            total += len(chunk)
        return total

    times = []
    n = None
    for _ in range(reps):
        t0 = time.perf_counter()
        n = run()
        times.append(time.perf_counter() - t0)
    return times, n


seq_times, n_terms_seq = time_sequential(REPS)
par_times, n_terms_par = time_parallel(REPS, n_workers)
assert n_terms_seq == n_terms_par, (n_terms_seq, n_terms_par)

mean_seq = sum(seq_times) / len(seq_times)
mean_par = sum(par_times) / len(par_times)

print(f"terms={n_terms_seq}")
print(f"  sequential (1 process):            mean={mean_seq:.4f}s  "
      f"individual={[f'{t:.4f}' for t in seq_times]}")
print(f"  parallel_decompose ({n_workers} workers):  mean={mean_par:.4f}s  "
      f"individual={[f'{t:.4f}' for t in par_times]}")
print(f"  speedup: {mean_seq/mean_par:.3f}x "
      f"({'parallel FASTER' if mean_par < mean_seq else 'parallel SLOWER'})")
print(f"  (n_workers={n_workers}, ideal linear speedup would be {n_workers:.1f}x)")
