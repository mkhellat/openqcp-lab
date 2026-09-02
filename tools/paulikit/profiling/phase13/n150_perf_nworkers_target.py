"""perf stat target: parallel_decompose at a fixed chunk_size=2,
sweeping n_workers instead of chunk_size (companion to
n150_perf_chunk_size_parallel.py, which fixes n_workers=8 and sweeps
chunk_size) - used to answer whether cache-miss ratio actually scales
with n_workers (see n_workers_placement_and_cache_findings.md). Run
under:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses python n150_perf_nworkers_target.py <n_workers>
"""
import sys

from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
n_workers = int(sys.argv[1])
chunk_size = 2

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)

total_terms = 0
for chunk_terms in parallel_decompose(padded, chunk_size=chunk_size, n_workers=n_workers):
    total_terms += len(chunk_terms)
print(f"n_workers={n_workers} total_terms={total_terms}")
