"""perf stat target, chunk_size=256, N=100 - see chunk_perf_n100_cs4.py's
docstring for the invocation and chunk_size_cache_locality_findings.md
for the results.
"""

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

spring_constants = _default_spring_constants(100)
masses = _default_masses(100)
unpadded = build_hamiltonian(100, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

total_terms = 0
for chunk_terms in fwht_pauli_terms_iter(padded, chunk_size=256):
    total_terms += len(chunk_terms)
print(f"total_terms={total_terms}")
