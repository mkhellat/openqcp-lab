"""N=150 end-to-end attempt using Phase 8's sparse Hamiltonian input,
Phase 9's chunked/checkpointed accumulator, and Phase 10's streaming
output (fwht_pauli_terms_iter) - see phase10_streaming_findings.md in
this directory for what this script was used to establish.

Not a pytest test (deliberately) - meant to be run under a memory
cap, e.g.:

    bash -c "ulimit -v 2000000; python n150_streaming_test.py"

with a `free -m` polling loop guarding against real system memory
exhaustion (same harness used throughout this project).
"""

import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

n = 150
spring_constants = {(i, j): 1.0 + 0.1*(i+j) for i in range(n) for j in range(i, n)}
masses = [1.0 + 0.05*i for i in range(n)]

t0 = time.perf_counter()
sparse = build_hamiltonian(n, spring_constants, masses, sparse=True)
t1 = time.perf_counter()
print(f"build_hamiltonian(sparse=True): {t1-t0:.2f}s")

padded_sparse, n_qubits = pad_to_power_of_two(sparse, sparse=True)
t2 = time.perf_counter()
print(f"pad_to_power_of_two(sparse=True): {t2-t1:.2f}s, n_qubits={n_qubits}")

total_terms = 0
n_chunks = 0
for chunk_terms in fwht_pauli_terms_iter(
    padded_sparse, chunk_size=256, parallel_labels=True
):
    n_chunks += 1
    total_terms += len(chunk_terms)
    if n_chunks % 20 == 0:
        print(f"  ...chunk {n_chunks}, running total {total_terms:,} terms")
t3 = time.perf_counter()
print(f"fwht_pauli_terms_iter streamed: {t3-t2:.2f}s")
print(f"chunks={n_chunks}, total_terms={total_terms:,}")
print("SUCCESS")
