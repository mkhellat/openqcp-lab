"""Publication-grade, fully reproducible sweep across every distinct
(n_workers, n_physical_cores_used) configuration on this machine, to
determine the actual wall-clock optimum - not just adjacent pairwise
comparisons (full_optimum_sweep_findings.md's own methodology
section has the full writeup and citation-ready details).

14 configurations (w1_c1 .. w8_c4, see full_matrix_target.py's
_SWEEP_CONFIGS), 10 reps each = 140 runs total. Each run: cooldown to
55C package temperature first (same protocol as every other
thermally-controlled measurement in this investigation), then a real
`parallel_decompose()` call via full_matrix_target.py, timed with
perf stat for cycles/instructions/cache-miss/LLC-miss data too (same
event set as every other perf stat measurement in this project).

Designed to survive being run in chunks across multiple background
invocations spanning hours: results are appended to a JSON Lines file
after EVERY single run (not held in memory until the end), and the
driver skips any (config, rep) pair already present in that file on
startup - safe to interrupt and re-launch at any point.

Usage (foreground - will move to background automatically for long
runs; safe to re-invoke, it resumes):
    OPENBLAS_NUM_THREADS=1 python full_optimum_sweep.py [--reps N] [--configs c1,c2,...]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import SWEEP_CONFIGS as _SWEEP_CONFIGS  # noqa: E402

TARGET_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_matrix_target.py")
PYTHON = sys.executable
EVENTS = "task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "full_optimum_sweep_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

# Deterministic, documented run order: by n_workers ascending, then
# n_cores ascending within each n_workers - makes partial-progress
# logs easy to read and the "what's left" question easy to answer.
CONFIG_ORDER = sorted(
    _SWEEP_CONFIGS.keys(),
    key=lambda k: (int(k.split("_")[0][1:]), int(k.split("_")[1][1:])),
)


def _read_pkg_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown() -> float | None:
    start = time.perf_counter()
    while True:
        temp = _read_pkg_temp_c()
        if temp is not None and temp <= COOLDOWN_TARGET_C:
            return temp
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return temp
        time.sleep(2)


def load_completed() -> set[tuple[str, int]]:
    completed = set()
    if not os.path.exists(RESULTS_PATH):
        return completed
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed.add((rec["condition"], rec["rep"]))
    return completed


def run_once(condition: str) -> dict:
    settled_temp = cooldown()
    temp_before = _read_pkg_temp_c()

    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        ["perf", "stat", "--no-inherit", "-e", EVENTS, PYTHON, TARGET_SCRIPT, condition],
        capture_output=True, text=True, env=env,
    )
    temp_after = _read_pkg_temp_c()
    stderr = result.stderr
    stdout = result.stdout

    counters = {}
    for line in stderr.splitlines():
        m = re.match(r"\s*([\d,]+)\s+([\w.\-]+):u", line)
        if m:
            counters[m.group(2)] = int(m.group(1).replace(",", ""))

    m_elapsed = re.search(r"elapsed=([\d.]+)s", stdout)
    m_rss = re.search(r"peak_rss_mib=([\d.]+)", stdout)
    m_terms = re.search(r"total_terms=(\d+)", stdout)
    elapsed = float(m_elapsed.group(1)) if m_elapsed else None
    peak_rss_mib = float(m_rss.group(1)) if m_rss else None
    total_terms = int(m_terms.group(1)) if m_terms else None

    cache_refs = counters.get("cache-references")
    cache_misses = counters.get("cache-misses")
    llc_loads = counters.get("LLC-loads")
    llc_misses = counters.get("LLC-load-misses")
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")

    return {
        "elapsed": elapsed,
        "peak_rss_mib": peak_rss_mib,
        "total_terms": total_terms,
        "cycles": cycles,
        "instructions": instructions,
        "ipc": instructions / cycles if (instructions and cycles) else None,
        "cache_miss_ratio": 100 * cache_misses / cache_refs if cache_refs else None,
        "llc_miss_ratio": 100 * llc_misses / llc_loads if llc_loads else None,
        "settled_temp": settled_temp,
        "temp_before": temp_before,
        "temp_after": temp_after,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--configs", type=str, default=None,
                         help="comma-separated subset of config names, default: all 14")
    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs else CONFIG_ORDER
    for c in configs:
        assert c in _SWEEP_CONFIGS, f"unknown config {c!r}"

    completed = load_completed()
    total_planned = len(configs) * args.reps
    total_done_already = sum(1 for c in configs for r in range(args.reps) if (c, r) in completed)
    print(f"Resuming: {total_done_already}/{total_planned} runs already recorded in "
          f"{RESULTS_PATH}", flush=True)

    for condition in configs:
        n_workers, cpu_list = _SWEEP_CONFIGS[condition]
        for rep in range(args.reps):
            if (condition, rep) in completed:
                continue
            r = run_once(condition)
            r["condition"] = condition
            r["rep"] = rep
            r["n_workers"] = n_workers
            r["n_cores"] = len(set(c % 4 for c in cpu_list))
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{condition} (n_workers={n_workers}, n_cores={r['n_cores']}) rep={rep}: "
                  f"elapsed={r['elapsed']} ipc={r['ipc']:.4f} "
                  f"cache_miss_ratio={r['cache_miss_ratio']:.3f}% "
                  f"settled_temp={r['settled_temp']} temp_after={r['temp_after']}", flush=True)

    print("\nSweep complete (or this chunk's requested configs are done).")
