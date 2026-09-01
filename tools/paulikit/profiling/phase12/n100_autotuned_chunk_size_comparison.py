"""Real N=100 wall-clock comparison: fixed chunk_size=256 (the old
example default used throughout Phase 9/10/11's own docs) vs.
autotune.recommended_chunk_size(dim) (Phase 12's auto-tuned value) -
same methodology as chunk_size_cache_locality_findings.md's own sweep
(same-process, multiple repeats, mean reported). Also runs
auto_decompose() itself and confirms its wall-clock matches calling
fwht_pauli_terms_iter directly with the same auto-tuned chunk_size.
"""
import time

from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import auto_decompose, fwht_pauli_terms_iter
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 100
REPS = 5

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
dim = padded.shape[0]

auto_chunk_size = autotune.recommended_chunk_size(dim)
budget = autotune.available_memory_bytes()
estimated_dense_bytes = dim * dim * 16

print(f"N={N_OSCILLATORS} dim={dim} auto_chunk_size={auto_chunk_size} "
      f"available_memory_bytes={budget:,} estimated_dense_bytes={estimated_dense_bytes:,}")


def time_streaming(chunk_size, reps):
    def run():
        total = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
            total += len(chunk)
        return total

    warm = run()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        n = run()
        times.append(time.perf_counter() - t0)
    assert n == warm
    return times, n


fixed_times, n_terms_fixed = time_streaming(256, REPS)
auto_times, n_terms_auto = time_streaming(auto_chunk_size, REPS)
assert n_terms_fixed == n_terms_auto

mean_fixed = sum(fixed_times) / len(fixed_times)
mean_auto = sum(auto_times) / len(auto_times)

print(f"terms={n_terms_fixed}")
print(f"  chunk_size=256 (old fixed):        mean={mean_fixed:.4f}s  "
      f"individual={[f'{t:.4f}' for t in fixed_times]}")
print(f"  chunk_size={auto_chunk_size} (auto-tuned):   mean={mean_auto:.4f}s  "
      f"individual={[f'{t:.4f}' for t in auto_times]}")
print(f"  auto/fixed ratio: {mean_auto/mean_fixed:.3f}x "
      f"({'auto FASTER' if mean_auto < mean_fixed else 'auto SLOWER'})")

# auto_decompose() itself - confirm which path it picks and that its
# wall-clock matches calling fwht_pauli_terms_iter directly with the
# same auto-tuned chunk_size (i.e. auto_decompose adds no meaningful
# overhead of its own beyond the underlying call).
def run_auto_decompose():
    result = auto_decompose(padded)
    if isinstance(result, dict):
        return "dense", len(result)
    total = 0
    for chunk in result:
        total += len(chunk)
    return "streaming", total

path, warm_n = run_auto_decompose()
ad_times = []
for _ in range(REPS):
    t0 = time.perf_counter()
    p, n = run_auto_decompose()
    ad_times.append(time.perf_counter() - t0)
    assert p == path and n == warm_n

mean_ad = sum(ad_times) / len(ad_times)
print(f"auto_decompose() picked path={path!r} terms={warm_n} mean={mean_ad:.4f}s "
      f"individual={[f'{t:.4f}' for t in ad_times]}")
