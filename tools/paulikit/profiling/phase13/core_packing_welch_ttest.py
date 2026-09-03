"""Generic core-packing comparison driver - runs the same Welch's
t-test battery (wall-clock, cycles, cache-miss ratio, LLC-miss ratio,
peak RSS) used for pinned_4_4cores vs. pinned_4_2cores, for any pair
of conditions defined in full_matrix_target.py's _CONDITIONS table.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python core_packing_welch_ttest.py <condition_a> <condition_b> [reps]
"""
import os
import re
import statistics
import subprocess
import sys

import scipy.stats

TARGET_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_matrix_target.py")
PYTHON = sys.executable
EVENTS = "task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"


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
        "cache_miss_ratio": 100 * cache_misses / cache_refs if cache_refs else None,
        "llc_miss_ratio": 100 * llc_misses / llc_loads if llc_loads else None,
        "raw_stderr": stderr,
        "stdout": stdout,
    }


def welch_report(results, cond_a, cond_b, metric_name, label=None):
    label = label or metric_name
    a = [r[metric_name] for r in results[cond_a]]
    b = [r[metric_name] for r in results[cond_b]]
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
    print(f"\n=== {label}: {cond_a} (n={n_a}) vs {cond_b} (n={n_b}) ===")
    print(f"{cond_a}: mean={mean_a:.4f} stdev={statistics.stdev(a):.4f}")
    print(f"{cond_b}: mean={mean_b:.4f} stdev={statistics.stdev(b):.4f}")
    print(f"diff ({cond_b} - {cond_a}) = {diff:+.4f}")
    print(f"t={t_stat:.3f}  df={df:.2f}  p={p_value:.6f}")
    print(f"95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"Cohen's d: {cohens_d:.3f}")
    print(f"significant at alpha=0.05: {p_value < 0.05}")


if __name__ == "__main__":
    cond_a, cond_b = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    results = {cond_a: [], cond_b: []}
    for condition in (cond_a, cond_b):
        for rep in range(reps):
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

    welch_report(results, cond_a, cond_b, "elapsed", "wall-clock (s)")
    welch_report(results, cond_a, cond_b, "cycles", "cycles (raw count)")
    welch_report(results, cond_a, cond_b, "cache_miss_ratio", "cache-miss ratio (%)")
    welch_report(results, cond_a, cond_b, "llc_miss_ratio", "LLC-miss ratio (%)")
    welch_report(results, cond_a, cond_b, "peak_rss_mib", "peak RSS (MiB)")
