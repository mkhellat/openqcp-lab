"""perf stat target for a given chunk_size at N=200. Run under:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python n200_perf_chunk_size.py <chunk_size>
"""
import sys

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 200
chunk_size = int(sys.argv[1])

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)

total_terms = 0
for chunk_terms in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
    total_terms += len(chunk_terms)
print(f"chunk_size={chunk_size} total_terms={total_terms}")
