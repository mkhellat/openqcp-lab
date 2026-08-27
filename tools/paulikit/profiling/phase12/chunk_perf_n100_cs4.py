"""perf stat target, chunk_size=4, N=100 - one of three chunk_size
points (4/32/256) measured for chunk_size_cache_locality_findings.md
in this directory. Run under:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python chunk_perf_n100_cs4.py

(same event set as every other perf stat measurement in this project).
"""

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

spring_constants = _default_spring_constants(100)
masses = _default_masses(100)
unpadded = build_hamiltonian(100, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

total_terms = 0
for chunk_terms in fwht_pauli_terms_iter(padded, chunk_size=4):
    total_terms += len(chunk_terms)
print(f"total_terms={total_terms}")
