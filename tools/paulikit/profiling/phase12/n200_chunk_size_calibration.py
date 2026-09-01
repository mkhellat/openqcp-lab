"""Single-run calibration pass at N=200 - first ever N=200 measurement
in this project. Checks real term count and a single chunk_size=8
timing before committing to a full repeated sweep, since N=150->200
term-count scaling was not yet known (N=100->150 grew ~4.51x).
"""
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 200

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

t_build0 = time.perf_counter()
sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]
t_build = time.perf_counter() - t_build0

print(f"N={N_OSCILLATORS} dim={dim} n_qubits={n_qubits} build_time={t_build:.2f}s", flush=True)

t0 = time.perf_counter()
total = 0
n_chunks = 0
for chunk in fwht_pauli_terms_iter(padded, chunk_size=8):
    total += len(chunk)
    n_chunks += 1
elapsed = time.perf_counter() - t0
print(f"chunk_size=8  elapsed={elapsed:.2f}s  terms={total:,}  chunks={n_chunks:,}")
