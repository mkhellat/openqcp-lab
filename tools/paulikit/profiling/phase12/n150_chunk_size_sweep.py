"""Real N=150 chunk_size sweep - the original chunk_size_cache_locality_
findings.md sweep only covered N=25/50/100; N=150 has only ever been
tested at exactly two points (256 and the auto-tuned floor value 8) in
every prior finding. This closes that gap: does chunk_size=8 (the
current auto-tuned value, which is actually the fallback FLOOR at
N=150 - the cache-boundary math itself degenerates to <1 at this dim,
see PLAN.md Phase 12's own note) sit near the real minimum, or is
there a better value nearby that was never tested at this scale?

Single run per chunk_size (not repeated), matching every other N=150
driver's own convention in this project (each pass costs 30-70s).
"""
import sys
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]

chunk_sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1, 2, 4, 8, 16, 32, 64, 128, 256]

print(f"N={N_OSCILLATORS} dim={dim}")
results = []
for chunk_size in chunk_sizes:
    t0 = time.perf_counter()
    total = 0
    n_chunks = 0
    for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
        total += len(chunk)
        n_chunks += 1
    elapsed = time.perf_counter() - t0
    results.append((chunk_size, elapsed, total, n_chunks))
    print(f"chunk_size={chunk_size:>4}  elapsed={elapsed:7.2f}s  "
          f"terms={total:,}  chunks={n_chunks:,}", flush=True)

best = min(results, key=lambda r: r[1])
print(f"\nBEST: chunk_size={best[0]} elapsed={best[1]:.2f}s")
