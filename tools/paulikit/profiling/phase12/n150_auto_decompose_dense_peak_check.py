"""Isolated check: how much real virtual memory does auto_decompose()'s
dense path actually need at N=150? The first combined run
(n150_autotuned_chunk_size_comparison.py) hit an ArrayMemoryError under
`ulimit -v 8000000` (~7.6 GiB) - this isolates just the dense-path call
under a much larger cap to find the real requirement, and to check
whether autotune's own dim**2*16 = 4 GiB estimate (the basis for its
streaming-vs-dense safety threshold) is an underestimate.
"""
import resource
import time

from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]

estimated_dense_bytes = dim * dim * 16
print(f"dim={dim} estimated_dense_bytes (dim**2*16) = {estimated_dense_bytes:,} "
      f"({estimated_dense_bytes/2**30:.2f} GiB)")

t0 = time.perf_counter()
result = fwht_pauli_terms(padded)  # forces the dense path directly, no chunk_size
elapsed = time.perf_counter() - t0

peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(f"dense path completed: terms={len(result):,} elapsed={elapsed:.2f}s")
print(f"peak RSS: {peak_rss_kb:,} KiB ({peak_rss_kb/2**20:.2f} GiB)")
print(f"peak RSS / estimated_dense_bytes ratio: "
      f"{(peak_rss_kb*1024)/estimated_dense_bytes:.2f}x")
print("SUCCESS")
