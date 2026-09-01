"""Real N=150 wall-clock comparison: fixed chunk_size=256 (the value
used throughout every prior N=150 findings doc, e.g.
profiling/phase11/n150_post_implementation_findings.md's 70.77s
baseline) vs. autotune.recommended_chunk_size(dim) (Phase 12's
auto-tuned value) - both through fwht_pauli_terms_iter for a fair
comparison. Also runs auto_decompose() itself and records which path
it picks at this scale (not assumed - see PLAN.md Phase 12's own note
that this needed real-number verification).

Run under a memory cap, mirroring every other N=150 driver in this
project:

    bash -c "ulimit -v 6000000; python n150_autotuned_chunk_size_comparison.py"

with free -h monitoring before/after.
"""
import time

from paulikit.algorithms import autotune, fwht
from paulikit.algorithms.fwht import auto_decompose, fwht_pauli_terms_iter
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

spring_constants = {(i, j): 1.0 + 0.1 * (i + j)
                     for i in range(N_OSCILLATORS) for j in range(i, N_OSCILLATORS)}
masses = [1.0 + 0.05 * i for i in range(N_OSCILLATORS)]

sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
dim = padded.shape[0]

auto_chunk_size = autotune.recommended_chunk_size(dim)
budget = autotune.available_memory_bytes()
# Post Bug-1-fix accounting (see dense_memory_estimate_fix_findings.md) -
# matches what auto_decompose() itself actually compares against.
estimated_dense_bytes = dim * dim * 16 * fwht._DENSE_MEMORY_MULTIPLIER
dense_threshold = budget * fwht._DENSE_MEMORY_SAFETY_FRACTION

print(f"N={N_OSCILLATORS} dim={dim} auto_chunk_size={auto_chunk_size} "
      f"available_memory_bytes={budget:,} "
      f"estimated_dense_bytes={estimated_dense_bytes:,.0f} "
      f"dense_threshold={dense_threshold:,.0f}")


def time_streaming_once(chunk_size):
    t0 = time.perf_counter()
    total = 0
    n_chunks = 0
    for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
        total += len(chunk)
        n_chunks += 1
    elapsed = time.perf_counter() - t0
    return elapsed, total, n_chunks


print("--- chunk_size=256 (old fixed) ---")
t_fixed, n_fixed, chunks_fixed = time_streaming_once(256)
print(f"  elapsed={t_fixed:.2f}s terms={n_fixed:,} chunks={chunks_fixed:,}")

print(f"--- chunk_size={auto_chunk_size} (auto-tuned) ---")
t_auto, n_auto, chunks_auto = time_streaming_once(auto_chunk_size)
print(f"  elapsed={t_auto:.2f}s terms={n_auto:,} chunks={chunks_auto:,}")

assert n_fixed == n_auto, "term count mismatch between chunk_size choices!"
print(f"auto/fixed ratio: {t_auto/t_fixed:.3f}x "
      f"({'auto FASTER' if t_auto < t_fixed else 'auto SLOWER'})")

print("--- auto_decompose() ---")
t0 = time.perf_counter()
result = auto_decompose(padded)
if isinstance(result, dict):
    path = "dense"
    n_ad = len(result)
else:
    path = "streaming"
    n_ad = 0
    for chunk in result:
        n_ad += len(chunk)
t_ad = time.perf_counter() - t0
print(f"  picked path={path!r} elapsed={t_ad:.2f}s terms={n_ad:,}")
assert n_ad == n_fixed, "auto_decompose term count mismatch!"

print("SUCCESS")
