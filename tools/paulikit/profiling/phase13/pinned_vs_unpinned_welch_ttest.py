"""Proper statistical test for the pinned_2/unpinned_2 and
pinned_4/unpinned_4 wall-clock comparisons - direct correction of an
earlier, methodologically weak check (pinned_vs_unpinned_noise_check.py)
that only eyeballed "diff vs. a naive combined stdev" at n=3, which is
NOT a real hypothesis test and cannot support a claim of "no real
difference" (absence of significant evidence is not evidence of
absence, especially at n=3). This uses Welch's t-test (unequal-
variance two-sample t-test, appropriate since nothing guarantees the
two conditions have equal variance) with 5 reps per condition, and
reports the actual p-value and 95% CI on the difference of means, not
an eyeballed comparison.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python pinned_vs_unpinned_welch_ttest.py
"""
import statistics
import time

import scipy.stats

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 5

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
    stdev = statistics.stdev(times)
    sem = stdev / (REPS ** 0.5)
    results[name] = times
    print(f"{name}: mean={mean:.4f}s stdev={stdev:.4f} sem={sem:.4f} "
          f"runs={[f'{t:.4f}' for t in times]}", flush=True)


def welch_report(name_a, name_b):
    a, b = results[name_a], results[name_b]
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    t_stat, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite CI on the difference of means.
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    n_a, n_b = len(a), len(b)
    se_diff = (var_a / n_a + var_b / n_b) ** 0.5
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = scipy.stats.t.ppf(0.975, df)
    diff = mean_b - mean_a
    ci_low, ci_high = diff - t_crit * se_diff, diff + t_crit * se_diff
    print(f"\n{name_a} vs {name_b} (Welch's t-test, n={n_a}/{n_b}):")
    print(f"  mean {name_a}={mean_a:.4f}s  mean {name_b}={mean_b:.4f}s  diff={diff:+.4f}s")
    print(f"  t={t_stat:.3f}  df={df:.2f}  p={p_value:.4f}")
    print(f"  95% CI on difference: [{ci_low:+.4f}, {ci_high:+.4f}]")
    if p_value < 0.05:
        print(f"  -> statistically significant at alpha=0.05")
    else:
        print(f"  -> NOT statistically significant at alpha=0.05 "
              f"(this does NOT prove no difference exists - it means this "
              f"sample size/effect size combination cannot distinguish it "
              f"from zero with 95% confidence; a larger n or a true "
              f"equivalence test would be needed to actually claim 'no "
              f"difference', which was not done here)")


welch_report("pinned_2", "unpinned_2")
welch_report("pinned_4", "unpinned_4")
