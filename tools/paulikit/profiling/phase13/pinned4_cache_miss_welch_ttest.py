"""Proper, statistically repeated cache-miss comparison for pinned_4
vs unpinned_4 - direct follow-up after the earlier full_matrix_findings.md
cache-miss claim (pinned_4 has the worst cache-miss%/LLC-miss% in the
whole matrix) was correctly flagged as single-run, never repeated, and
after the user pointed out flawed reasoning about L3 being a SHARED
constant between the two conditions (so it cannot explain the
DIFFERENCE between them - only something that actually differs
between pinned and unpinned can).

Runs `perf stat` 5 times per condition (L3 event group only, to keep
this run count manageable - task-clock, cycles, instructions,
cache-references, cache-misses, LLC-loads, LLC-load-misses), parses
the real output, computes a proper Welch's t-test on the cache-miss
ratio (cache-misses/cache-references) and LLC-miss ratio
(LLC-load-misses/LLC-loads) - not an eyeballed single-sample
comparison.

This is a DRIVER SCRIPT meant to be invoked with each perf stat run
piped through this parser - see the __main__ block for the actual
orchestration, which shells out to `perf stat` itself via subprocess
(so it can capture and parse perf's stderr output programmatically,
unlike the earlier manual perf stat invocations in this investigation).

Usage (foreground only, needs perf on PATH and the target script
alongside it):
    OPENBLAS_NUM_THREADS=1 python pinned4_cache_miss_welch_ttest.py
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
        [
            "perf", "stat", "--no-inherit", "-e", EVENTS,
            PYTHON, TARGET_SCRIPT, condition,
        ],
        capture_output=True, text=True, env=env,
    )
    stderr = result.stderr
    stdout = result.stdout

    counters = {}
    for line in stderr.splitlines():
        m = re.match(r"\s*([\d,]+)\s+([\w.\-]+):u", line)
        if m:
            value = int(m.group(1).replace(",", ""))
            counters[m.group(2)] = value
        else:
            m2 = re.match(r"\s*([\d.]+)\s+msec task-clock:u", line)
            if m2:
                counters["task-clock-msec"] = float(m2.group(1))

    m_elapsed = re.search(r"elapsed=([\d.]+)s", stdout)
    elapsed = float(m_elapsed.group(1)) if m_elapsed else None

    return {"counters": counters, "elapsed": elapsed, "raw_stderr": stderr}


if __name__ == "__main__":
    results = {"pinned_4": [], "unpinned_4": []}
    for condition in ("pinned_4", "unpinned_4"):
        for rep in range(REPS):
            r = run_once(condition)
            c = r["counters"]
            cache_refs = c.get("cache-references")
            cache_misses = c.get("cache-misses")
            llc_loads = c.get("LLC-loads")
            llc_misses = c.get("LLC-load-misses")
            cache_miss_ratio = 100 * cache_misses / cache_refs if cache_refs else None
            llc_miss_ratio = 100 * llc_misses / llc_loads if llc_loads else None
            results[condition].append({
                "elapsed": r["elapsed"],
                "cache_miss_ratio": cache_miss_ratio,
                "llc_miss_ratio": llc_miss_ratio,
                "counters": c,
            })
            print(f"{condition} rep={rep}: elapsed={r['elapsed']} "
                  f"cache_miss_ratio={cache_miss_ratio:.3f}% "
                  f"llc_miss_ratio={llc_miss_ratio:.3f}%", flush=True)

    print("\n=== Raw data ===")
    for condition, runs in results.items():
        print(f"\n{condition}:")
        print(f"  elapsed: {[r['elapsed'] for r in runs]}")
        print(f"  cache_miss_ratio: {[round(r['cache_miss_ratio'], 3) for r in runs]}")
        print(f"  llc_miss_ratio: {[round(r['llc_miss_ratio'], 3) for r in runs]}")


    def welch_report(metric_name):
        a = [r[metric_name] for r in results["pinned_4"]]
        b = [r[metric_name] for r in results["unpinned_4"]]
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
        print(f"\n=== {metric_name}: pinned_4 (n={n_a}) vs unpinned_4 (n={n_b}) ===")
        print(f"pinned_4: mean={mean_a:.4f} stdev={statistics.stdev(a):.4f}")
        print(f"unpinned_4: mean={mean_b:.4f} stdev={statistics.stdev(b):.4f}")
        print(f"diff (unpinned_4 - pinned_4) = {diff:+.4f}")
        print(f"t={t_stat:.3f}  df={df:.2f}  p={p_value:.6f}")
        print(f"95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"Cohen's d: {cohens_d:.3f}")
        print(f"significant at alpha=0.05: {p_value < 0.05}")


    welch_report("cache_miss_ratio")
    welch_report("llc_miss_ratio")
    welch_report("elapsed")
