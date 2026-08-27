"""Whole-pipeline perf stat target (parallel/TBB labels) - see
n150_pipeline_perf_serial.py's docstring for the perf stat invocation
(same command, this script's filename in place of the serial one) and
full_pipeline_n150_findings.md for the results/analysis.
"""

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

n = 150
spring_constants = {(i, j): 1.0 + 0.1*(i+j) for i in range(n) for j in range(i, n)}
masses = [1.0 + 0.05*i for i in range(n)]

sparse = build_hamiltonian(n, spring_constants, masses, sparse=True)
padded_sparse, n_qubits = pad_to_power_of_two(sparse, sparse=True)

total_terms = 0
for chunk_terms in fwht_pauli_terms_iter(padded_sparse, chunk_size=256, parallel_labels=True):
    total_terms += len(chunk_terms)
print(f"total_terms={total_terms}")
