"""N=150, n_workers in {2 (half physical), 4 (physical cores), 8
(logical/hyperthreads)}, chunk_size=2 - direct test of the user's
hypothesis that n_workers should match physical core count (4), not
logical CPU count (8), on this hyperthreaded machine (i7-8550U: 4
physical cores, 8 logical CPUs via lscpu). Foreground only, 2 reps
each, absolute times reported.
"""
import statistics
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter, parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 2

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
dim = padded.shape[0]

print(f"N={N_OSCILLATORS} dim={dim} chunk_size={CHUNK_SIZE} reps={REPS}", flush=True)

seq_times = []
for _ in range(REPS):
    t0 = time.perf_counter()
    n = 0
    for chunk in fwht_pauli_terms_iter(padded, chunk_size=CHUNK_SIZE):
        n += len(chunk)
    seq_times.append(time.perf_counter() - t0)
seq_mean = statistics.mean(seq_times)
seq_stdev = statistics.stdev(seq_times)
print(f"sequential: mean={seq_mean:.4f}s stdev={seq_stdev:.4f} runs={[f'{t:.3f}' for t in seq_times]}", flush=True)

for n_workers in [2, 4, 8]:
    times = []
    n_par = None
    for _ in range(REPS):
        t0 = time.perf_counter()
        total = 0
        for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
            total += len(chunk)
        times.append(time.perf_counter() - t0)
        n_par = total
    mean = statistics.mean(times)
    stdev = statistics.stdev(times)
    speedup = seq_mean / mean
    print(f"n_workers={n_workers}: mean={mean:.4f}s stdev={stdev:.4f} "
          f"speedup={speedup:.3f}x runs={[f'{t:.3f}' for t in times]} terms={n_par}", flush=True)
