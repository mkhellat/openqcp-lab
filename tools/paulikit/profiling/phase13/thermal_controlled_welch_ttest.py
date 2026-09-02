"""Thermal-controlled replication of the pinned/unpinned Welch's
t-test - direct follow-up after discovering the earlier runs were
confounded by real thermal throttling (package temp climbing to 100C
during a single run, tracked via x86_pkg_temp - see
freq_scaling_check.py's own findings). Setting the intel_pstate
governor to 'performance' did NOT fix this (intel_pstate still lets
hardware throttle below the requested P-state under thermal limits) -
this is genuine physics on a 15W-TDP laptop CPU under sustained
multi-core load, not a misconfiguration.

Two controls added, per direct instruction:
1. Cooldown between runs: wait until package temp drops below a
   baseline threshold (55C, chosen from direct empirical observation -
   the chip settles to 52-58C within ~90s idle) before starting the
   next timed run, with a timeout safeguard.
2. Thermal state logged as a covariate for every run: mean/min/max
   temp during the run, plus the STARTING temp (before the run began) -
   so if timing still correlates with starting/mean temp, that is
   now visible and checkable rather than invisible confound.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python thermal_controlled_welch_ttest.py
"""
import statistics
import threading
import time

import scipy.stats

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 5
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

_real_physical_core_representative_cpus = fwht._physical_core_representative_cpus


def _read_pkg_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown():
    start = time.perf_counter()
    while True:
        temp = _read_pkg_temp_c()
        if temp is not None and temp <= COOLDOWN_TARGET_C:
            return temp
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return temp  # give up, report whatever temp it settled at
        time.sleep(2)


def run(n_workers: int, pinned: bool) -> dict:
    if pinned:
        fwht._physical_core_representative_cpus = _real_physical_core_representative_cpus
    else:
        fwht._physical_core_representative_cpus = lambda: None

    temp_before = _read_pkg_temp_c()
    temp_samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            t = _read_pkg_temp_c()
            if t is not None:
                temp_samples.append(t)
            time.sleep(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    total = 0
    for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
        total += len(chunk)
    elapsed = time.perf_counter() - t0

    stop.set()
    thread.join(timeout=2)
    assert total == 91652096, total

    return {
        "elapsed": elapsed,
        "temp_before": temp_before,
        "temp_mean": statistics.mean(temp_samples) if temp_samples else None,
        "temp_max": max(temp_samples) if temp_samples else None,
    }


conditions = [
    ("pinned_2", 2, True),
    ("unpinned_2", 2, False),
    ("pinned_4", 4, True),
    ("unpinned_4", 4, False),
]

results = {name: [] for name, _, _ in conditions}

for rep in range(REPS):
    for name, n_workers, pinned in conditions:
        settled_temp = cooldown()
        r = run(n_workers, pinned)
        results[name].append(r)
        print(f"rep={rep} {name}: elapsed={r['elapsed']:.4f}s "
              f"cooldown_settled={settled_temp:.1f}C "
              f"temp_before={r['temp_before']:.1f}C "
              f"temp_mean={r['temp_mean']:.1f}C temp_max={r['temp_max']:.1f}C", flush=True)

print("\n=== Raw data ===")
for name, runs in results.items():
    elapsed_vals = [r["elapsed"] for r in runs]
    temp_before_vals = [r["temp_before"] for r in runs]
    temp_mean_vals = [r["temp_mean"] for r in runs]
    print(f"\n{name}:")
    print(f"  elapsed: {[f'{v:.4f}' for v in elapsed_vals]}")
    print(f"  temp_before: {[f'{v:.1f}' for v in temp_before_vals]}")
    print(f"  temp_mean: {[f'{v:.1f}' for v in temp_mean_vals]}")


def welch_report(name_a, name_b):
    a = [r["elapsed"] for r in results[name_a]]
    b = [r["elapsed"] for r in results[name_b]]
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    var_a, var_b = statistics.variance(a), statistics.variance(b)
    n_a, n_b = len(a), len(b)
    t_stat, p_value = scipy.stats.ttest_ind(a, b, equal_var=False)
    se_diff = (var_a / n_a + var_b / n_b) ** 0.5
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    t_crit = scipy.stats.t.ppf(0.975, df)
    diff = mean_b - mean_a
    ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)
    pooled_sd = (((n_a - 1) * statistics.stdev(a) ** 2 + (n_b - 1) * statistics.stdev(b) ** 2)
                 / (n_a + n_b - 2)) ** 0.5
    cohens_d = diff / pooled_sd
    print(f"\n=== {name_a} (n={n_a}) vs {name_b} (n={n_b}), THERMAL-CONTROLLED ===")
    print(f"{name_a}: mean={mean_a:.4f}s stdev={statistics.stdev(a):.4f}")
    print(f"{name_b}: mean={mean_b:.4f}s stdev={statistics.stdev(b):.4f}")
    print(f"diff ({name_b}-{name_a}) = {diff:+.4f}s")
    print(f"t={t_stat:.3f}  df={df:.2f}  p={p_value:.6f}")
    print(f"95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"significant at alpha=0.05: {p_value < 0.05}")


welch_report("pinned_2", "unpinned_2")
welch_report("pinned_4", "unpinned_4")
