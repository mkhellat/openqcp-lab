"""Direct comparison requested by the user: pinned_4 locking 4 logical
CPUs from 4 DISTINCT physical cores (pinned_4_4cores, cores A/B/C/D
via cpus 0,1,2,3 - identical to the standing pinned_4 condition) vs.
pinned_4 locking 4 logical CPUs from only 2 physical cores (
pinned_4_2cores, both hyperthread siblings of cores A and B via cpus
0,4,1,5, leaving cores C and D completely unused).

This isolates a real, cleanly interpretable question: does spreading
4 workers across 4 independent physical cores (no hyperthread sharing
at all) actually outperform packing the same 4 workers onto 2
physical cores (2 pairs of hyperthread siblings, each pair sharing
L1/L2)? If hyperthread L1/L2 sharing were the dominant cost, 2cores
should be clearly worse. If it's the L3/memory-bandwidth-across-all-
cores story from earlier findings, the two conditions might be much
closer, since both keep the SAME total core-package power/thermal
footprint... or use FEWER physical cores (2cores) which could reduce
L3/bandwidth pressure precisely because 2 physical cores are left
completely idle.

Collects, 5 reps per condition, real Welch's t-tests on each:
  - wall-clock (elapsed)
  - cycles (raw hardware cycle count - requested explicitly, immune
    to DVFS/frequency-reporting noise unlike wall-clock)
  - cache-miss ratio, LLC-miss ratio (L3 group)
  - peak RSS (memory footprint, via full_matrix_target.py's own
    RssMonitor, printed in its stdout)

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python pinned4_4cores_vs_2cores_welch_ttest.py
"""
import os
import re
import statistics
import subprocess
import sys

import scipy.stats

REPS = 5
EVENTS = "task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"
TARGET_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_matrix_target.py")
PYTHON = sys.executable


def run_once(condition: str) -> dict:
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        ["perf", "stat", "--no-inherit", "-e", EVENTS, PYTHON, TARGET_SCRIPT, condition],
        capture_output=True, text=True, env=env,
    )
    stderr = result.stderr
    stdout = result.stdout

    counters = {}
    for line in stderr.splitlines():
        m = re.match(r"\s*([\d,]+)\s+([\w.\-]+):u", line)
        if m:
            counters[m.group(2)] = int(m.group(1).replace(",", ""))

    m_elapsed = re.search(r"elapsed=([\d.]+)s", stdout)
    m_rss = re.search(r"peak_rss_mib=([\d.]+)", stdout)
    elapsed = float(m_elapsed.group(1)) if m_elapsed else None
    peak_rss_mib = float(m_rss.group(1)) if m_rss else None

    cache_refs = counters.get("cache-references")
    cache_misses = counters.get("cache-misses")
    llc_loads = counters.get("LLC-loads")
    llc_misses = counters.get("LLC-load-misses")

    return {
        "elapsed": elapsed,
        "peak_rss_mib": peak_rss_mib,
        "cycles": counters.get("cycles"),
        "instructions": counters.get("instructions"),
        "cache_miss_ratio": 100 * cache_misses / cache_refs if cache_refs else None,
        "llc_miss_ratio": 100 * llc_misses / llc_loads if llc_loads else None,
        "stdout": stdout,
        "raw_stderr": stderr,
    }


if __name__ == "__main__":
    conditions = ["pinned_4_4cores", "pinned_4_2cores"]
    results = {c: [] for c in conditions}

    for condition in conditions:
        for rep in range(REPS):
            r = run_once(condition)
            results[condition].append(r)
            print(f"{condition} rep={rep}: elapsed={r['elapsed']} cycles={r['cycles']} "
                  f"cache_miss_ratio={r['cache_miss_ratio']:.3f}% "
                  f"llc_miss_ratio={r['llc_miss_ratio']:.3f}% "
                  f"peak_rss_mib={r['peak_rss_mib']}", flush=True)

    print("\n=== Raw data ===")
    for condition, runs in results.items():
        print(f"\n{condition}:")
        print(f"  elapsed: {[r['elapsed'] for r in runs]}")
        print(f"  cycles: {[r['cycles'] for r in runs]}")
        print(f"  cache_miss_ratio: {[round(r['cache_miss_ratio'], 3) for r in runs]}")
        print(f"  llc_miss_ratio: {[round(r['llc_miss_ratio'], 3) for r in runs]}")
        print(f"  peak_rss_mib: {[r['peak_rss_mib'] for r in runs]}")

    def welch_report(metric_name, label=None):
        label = label or metric_name
        a = [r[metric_name] for r in results["pinned_4_4cores"]]
        b = [r[metric_name] for r in results["pinned_4_2cores"]]
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
        cohens_d = diff / pooled_sd if pooled_sd else float("nan")
        print(f"\n=== {label}: pinned_4_4cores (n={n_a}) vs pinned_4_2cores (n={n_b}) ===")
        print(f"pinned_4_4cores: mean={mean_a:.4f} stdev={statistics.stdev(a):.4f}")
        print(f"pinned_4_2cores: mean={mean_b:.4f} stdev={statistics.stdev(b):.4f}")
        print(f"diff (2cores - 4cores) = {diff:+.4f}")
        print(f"t={t_stat:.3f}  df={df:.2f}  p={p_value:.6f}")
        print(f"95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"significant at alpha=0.05: {p_value < 0.05}")

    welch_report("elapsed", "wall-clock (s)")
    welch_report("cycles", "cycles (raw count)")
    welch_report("cache_miss_ratio", "cache-miss ratio (%)")
    welch_report("llc_miss_ratio", "LLC-miss ratio (%)")
    welch_report("peak_rss_mib", "peak RSS (MiB)")
