"""perf stat target: SEQUENTIAL fwht_pauli_terms_iter at a given
chunk_size, N=150 - post-bounded-submission-fix baseline for
comparison against n150_perf_chunk_size_parallel.py's parallel
numbers. Same event set as every other perf stat measurement in this
project (see profiling/phase12/n150_perf_chunk_size.py). Run under:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python n150_perf_chunk_size_sequential.py <chunk_size>
"""
import sys

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
chunk_size = int(sys.argv[1])

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)

total_terms = 0
for chunk_terms in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
    total_terms += len(chunk_terms)
print(f"chunk_size={chunk_size} total_terms={total_terms}")
