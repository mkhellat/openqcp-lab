"""N=150 end-to-end attempt using Phase 8's sparse Hamiltonian input
plus Phase 9's chunked/checkpointed accumulator - see
``phase9_findings.md`` in this directory for what this script was
used to establish.

Not a pytest test (deliberately) - meant to be run under a memory
cap, e.g.:

    bash -c "ulimit -v 10000000; python n150_chunked_accumulator_test.py"

with a `free -m` polling loop guarding against real system memory
exhaustion (see ``phase9_findings.md`` for the exact harness used).
"""

import time

from paulikit.algorithms.fwht import fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

n = 150
spring_constants = {(i, j): 1.0 + 0.1*(i+j) for i in range(n) for j in range(i, n)}
masses = [1.0 + 0.05*i for i in range(n)]

t0 = time.perf_counter()
sparse = build_hamiltonian(n, spring_constants, masses, sparse=True)
t1 = time.perf_counter()
print(f"build_hamiltonian(sparse=True): {t1-t0:.2f}s, nnz={sparse.nnz}, shape={sparse.shape}")

padded_sparse, n_qubits = pad_to_power_of_two(sparse, sparse=True)
t2 = time.perf_counter()
print(f"pad_to_power_of_two(sparse=True): {t2-t1:.2f}s, n_qubits={n_qubits}, shape={padded_sparse.shape}, nnz={padded_sparse.nnz}")

terms = fwht_pauli_terms(padded_sparse, chunk_size=256)
t3 = time.perf_counter()
print(f"fwht_pauli_terms(chunk_size=256): {t3-t2:.2f}s, n_terms={len(terms)}")
print("SUCCESS")
