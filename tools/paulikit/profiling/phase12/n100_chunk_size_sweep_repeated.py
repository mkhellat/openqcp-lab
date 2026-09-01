"""Repeated-run chunk_size sweep at N=100 for small values (1, 2, 4, 8)
- fills in the gap the original chunk_size_cache_locality_findings.md
sweep left (it tested 4/8/32/128/256 at N=100, single-run only for
most; this adds chunk_size=1/2 and repeats for direct comparison
against the N=150 repeated sweep, before changing the floor constant).
"""
import sys
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 100
REPS = 3

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
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
