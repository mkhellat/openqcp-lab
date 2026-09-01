"""Repeated-run chunk_size sweep at N=200 for chunk_size=2/4/8 - checks
run-to-run stability at this larger scale before finalizing
_min_chunk_size_floor()'s new value, per direct instruction. N=200's
single-run discovery pass (n200_chunk_size_calibration.py +
adhoc sweep) found a clean monotonic increase from chunk_size=1
upward, unlike N=150 (where chunk_size=2 was the clear winner) - this
confirms whether that pattern is stable, not single-run noise.
"""
import sys
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 200
REPS = 3

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]

chunk_sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [2, 4, 8]

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
