"""Screening pass: does ANY multi-physical-core config beat w2_c1's
record (mean 20.537s at chunk_size=2, full_optimum_sweep_results.jsonl)
at some OTHER chunk_size? Answers the tuning/avoidability question
raised after the full optimum sweep - Phase 12's chunk_size floor
(_min_chunk_size_floor in autotune.py) was tuned exclusively on a
single uncontended process; chunk_size_floor_scale_dependence_findings.md's
own "does NOT show" list explicitly flags multi-core contention as
untested. This is that test.

Deliberately a screening pass, not a publication-grade sweep (direct
user decision): 1 rep per (config, chunk_size) with a thermal cooldown
between every run, restricted to the 12 configs that use MORE than one
physical core (w2_c1 itself doesn't need re-testing - it's already
characterized with n=10 in full_optimum_sweep_results.jsonl). If any
cell here beats 20.537s, that candidate gets a proper repeated-run
follow-up before any conclusion is drawn; if none do, that is itself
the answer to "is the 2-core ceiling avoidable via chunk_size alone".

Usage:
    OPENBLAS_NUM_THREADS=1 python contended_chunk_size_screen.py
"""
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
                             "contended_chunk_size_screen_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180
CHUNK_SIZES = [1, 2, 4, 8]

# Every sweep config that uses more than 1 physical core - w2_c1 is
# excluded, it's already the n=10 record to beat, not a candidate.
MULTI_CORE_CONFIGS = sorted(
    (c for c, (_, cpus) in _SWEEP_CONFIGS.items()
     if len(set(cpu % 4 for cpu in cpus)) > 1),
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
            completed.add((rec["condition"], rec["chunk_size"]))
    return completed


def run_once(condition: str, chunk_size: int) -> dict:
    settled_temp = cooldown()
    temp_before = _read_pkg_temp_c()

    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        ["perf", "stat", "--no-inherit", "-e", EVENTS, PYTHON, TARGET_SCRIPT,
         condition, str(chunk_size)],
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
        "returncode": result.returncode,
        "stderr_tail": stderr[-500:] if result.returncode != 0 else None,
    }


if __name__ == "__main__":
    completed = load_completed()
    total_planned = len(MULTI_CORE_CONFIGS) * len(CHUNK_SIZES)
    total_done = len(completed)
    print(f"Resuming: {total_done}/{total_planned} cells already recorded in "
          f"{RESULTS_PATH}", flush=True)
    print(f"Configs ({len(MULTI_CORE_CONFIGS)}): {MULTI_CORE_CONFIGS}", flush=True)
    print("Baseline to beat: w2_c1 mean=20.537s (n=10, chunk_size=2, "
          "full_optimum_sweep_results.jsonl)\n", flush=True)

    for condition in MULTI_CORE_CONFIGS:
        n_workers, cpu_list = _SWEEP_CONFIGS[condition]
        n_cores = len(set(c % 4 for c in cpu_list))
        for chunk_size in CHUNK_SIZES:
            if (condition, chunk_size) in completed:
                continue
            r = run_once(condition, chunk_size)
            r["condition"] = condition
            r["chunk_size"] = chunk_size
            r["n_workers"] = n_workers
            r["n_cores"] = n_cores
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            beats = (r["elapsed"] is not None and r["elapsed"] < 20.537)
            marker = " <<< BEATS w2_c1 RECORD" if beats else ""
            print(f"{condition} (n_workers={n_workers}, n_cores={n_cores}) "
                  f"chunk_size={chunk_size}: elapsed={r['elapsed']}{marker}", flush=True)

    print("\nScreening pass complete (or already done).")
