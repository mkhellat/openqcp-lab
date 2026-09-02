"""Statistical noise check for the pinned_2/unpinned_2 vs.
pinned_4/unpinned_4 sign-flip found in full_matrix_findings.md - the
matrix was single-run per condition, and the user directly and
correctly questioned whether a ~6-10% wall-clock gap is real or just
run-to-run noise before any mechanism is discussed. 3 reps per
condition, wall-clock only (no perf stat - pure timing repeats, to
answer the noise question specifically and quickly).
"""
import statistics
import time

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 3

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

_real_physical_core_representative_cpus = fwht._physical_core_representative_cpus


def run(n_workers: int, pinned: bool) -> float:
    if pinned:
        fwht._physical_core_representative_cpus = _real_physical_core_representative_cpus
    else:
        fwht._physical_core_representative_cpus = lambda: None
    t0 = time.perf_counter()
    total = 0
    for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
        total += len(chunk)
    elapsed = time.perf_counter() - t0
    assert total == 91652096, total
    return elapsed


conditions = [
    ("pinned_2", 2, True),
    ("unpinned_2", 2, False),
    ("pinned_4", 4, True),
    ("unpinned_4", 4, False),
]

results = {}
for name, n_workers, pinned in conditions:
    times = [run(n_workers, pinned) for _ in range(REPS)]
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    results[name] = (mean, stdev, times)
    print(f"{name}: mean={mean:.4f}s stdev={stdev:.4f} "
          f"runs={[f'{t:.4f}' for t in times]}", flush=True)

print("\nSummary:")
for name, (mean, stdev, times) in results.items():
    cv = 100 * stdev / mean
    print(f"  {name:12} mean={mean:7.4f}s  stdev={stdev:.4f}  cv={cv:.2f}%")

p2_mean, p2_stdev, _ = results["pinned_2"]
u2_mean, u2_stdev, _ = results["unpinned_2"]
p4_mean, p4_stdev, _ = results["pinned_4"]
u4_mean, u4_stdev, _ = results["unpinned_4"]

print(f"\npinned_2 vs unpinned_2: diff={u2_mean-p2_mean:+.4f}s "
      f"({'pinned faster' if p2_mean < u2_mean else 'unpinned faster'}), "
      f"combined stdev~{(p2_stdev**2+u2_stdev**2)**0.5:.4f}")
print(f"pinned_4 vs unpinned_4: diff={u4_mean-p4_mean:+.4f}s "
      f"({'pinned faster' if p4_mean < u4_mean else 'unpinned faster'}), "
      f"combined stdev~{(p4_stdev**2+u4_stdev**2)**0.5:.4f}")
