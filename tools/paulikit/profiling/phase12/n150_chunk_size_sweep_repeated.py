"""Repeated-run (not single-shot) chunk_size sweep at N=150, for the
narrow range the initial single-run sweep (n150_chunk_size_sweep.py)
found competitive (1, 2, 4, 8) - confirms whether chunk_size=2's
apparent win there is real or single-run noise, per direct instruction
to re-verify broadly before changing the floor constant.
"""
import sys
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
REPS = 3

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]

chunk_sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 4, 8]

print(f"N={N_OSCILLATORS} dim={dim} reps={REPS}")
for chunk_size in chunk_sizes:
    times = []
    n_terms = None
    for r in range(REPS):
        t0 = time.perf_counter()
        total = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
            total += len(chunk)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if n_terms is None:
            n_terms = total
        assert total == n_terms
    mean_t = sum(times) / len(times)
    print(f"chunk_size={chunk_size:>3}  mean={mean_t:7.2f}s  "
          f"individual={[f'{t:.2f}' for t in times]}  terms={n_terms:,}", flush=True)
