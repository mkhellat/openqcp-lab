"""Direct test of the memory-bandwidth hypothesis from
code_specificity_findings.md's Correction section: paulikit's 2-
physical-core ceiling reversed on a paulikit-free synthetic workload
(w8_c4 beat w2_c1 by ~3x there), reasoned to be because paulikit's
chunks touch full dim-sized (16384-element) arrays - memory-
bandwidth/cache-heavy - while the synthetic workload's 48x48 matrices
are cache-resident and FLOP-bound. This script measures that
mechanism directly via perf stat's cache/LLC event set (the same set
used throughout this investigation), across a real 2x2:
    {paulikit, synthetic} x {w2_c1 (1 physical core), w8_c4 (4 cores)}

**Methodology fix, found and corrected same day**: every prior perf
stat measurement in this investigation used `--no-inherit`, which
EXCLUDES child-process counters - since both paulikit's
parallel_decompose and this script's own ProcessPoolExecutor do all
their real work in worker SUBPROCESSES, `--no-inherit` was only ever
measuring the tiny, mostly-idle launcher process (confirmed directly:
task-clock=797ms with --no-inherit vs 27332ms without it, on a run
whose wall-clock was 12.5s - the aggregated inherited number correctly
reflects ~2 workers x ~13s each). This script does NOT pass
--no-inherit, so counters correctly aggregate across all worker
subprocesses. Per direct user decision, older findings docs are not
retroactively re-run for this - flagged as a known caveat there
instead; this script and any NEW findings from here on use the
correct (inheriting) invocation.

Usage:
    OPENBLAS_NUM_THREADS=1 python bandwidth_hypothesis_sweep.py [--reps N]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PYTHON = sys.executable
DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS = "task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"
RESULTS_PATH = os.path.join(DIR, "bandwidth_hypothesis_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

# (workload label, script, condition, extra args)
CELLS = [
    ("paulikit", "full_matrix_target.py", "w2_c1", ["2"]),
    ("paulikit", "full_matrix_target.py", "w8_c4", ["2"]),
    ("synthetic", "synthetic_ipc_control.py", "w2_c1", []),
    ("synthetic", "synthetic_ipc_control.py", "w8_c4", []),
]


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


def load_completed() -> set[tuple[str, str, int]]:
    completed = set()
    if not os.path.exists(RESULTS_PATH):
        return completed
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed.add((rec["workload"], rec["condition"], rec["rep"]))
    return completed


def run_once(workload: str, script: str, condition: str, extra_args: list[str]) -> dict:
    settled_temp = cooldown()
    temp_before = _read_pkg_temp_c()

    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    target = os.path.join(DIR, script)
    # NOTE: deliberately NOT --no-inherit - see module docstring.
    cmd = ["perf", "stat", "-e", EVENTS, PYTHON, target, condition, *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    temp_after = _read_pkg_temp_c()
    stderr = result.stderr
    stdout = result.stdout

    counters = {}
    for line in stderr.splitlines():
        # perf stat lines: "<value>[ msec] <event-name>[:u] ..." - the
        # optional "msec" unit token (only task-clock has it) sits
        # between the value and the event name, so it must be skipped
        # explicitly rather than assumed to be the event name itself
        # (a real bug caught here: an earlier version of this regex
        # captured "msec" as the event name for task-clock and left
        # task_clock_ms unparsed).
        m = re.match(r"\s*([\d,]+(?:\.\d+)?)\s+(?:msec\s+)?([\w.\-]+)", line)
        if m:
            key = m.group(2).split(":")[0]
            counters[key] = float(m.group(1).replace(",", ""))

    m_elapsed = re.search(r"elapsed=([\d.]+)s", stdout)
    elapsed = float(m_elapsed.group(1)) if m_elapsed else None

    cache_refs = counters.get("cache-references")
    cache_misses = counters.get("cache-misses")
    llc_loads = counters.get("LLC-loads")
    llc_misses = counters.get("LLC-load-misses")
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")
    task_clock_ms = counters.get("task-clock")

    return {
        "workload": workload,
        "condition": condition,
        "elapsed": elapsed,
        "task_clock_ms": task_clock_ms,
        "cycles": cycles,
        "instructions": instructions,
        "ipc": instructions / cycles if (instructions and cycles) else None,
        "cache_references": cache_refs,
        "cache_misses": cache_misses,
        "cache_miss_ratio": 100 * cache_misses / cache_refs if cache_refs else None,
        "llc_loads": llc_loads,
        "llc_misses": llc_misses,
        "llc_miss_ratio": 100 * llc_misses / llc_loads if llc_loads else None,
        # cache-references per ms of aggregated CPU time - a rough proxy
        # for memory-traffic INTENSITY (not just miss ratio), comparable
        # across workloads/conditions with very different total work.
        "cache_refs_per_ms": cache_refs / task_clock_ms if (cache_refs and task_clock_ms) else None,
        "settled_temp": settled_temp,
        "temp_before": temp_before,
        "temp_after": temp_after,
        "returncode": result.returncode,
        "stderr_tail": stderr[-500:] if result.returncode != 0 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    completed = load_completed()
    total_planned = len(CELLS) * args.reps
    total_done = sum(1 for _ in completed)
    print(f"Resuming: {total_done}/{total_planned} runs already recorded in "
          f"{RESULTS_PATH}", flush=True)

    for workload, script, condition, extra in CELLS:
        for rep in range(args.reps):
            if (workload, condition, rep) in completed:
                continue
            r = run_once(workload, script, condition, extra)
            r["rep"] = rep
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{workload:10s} {condition:8s} rep={rep}: elapsed={r['elapsed']} "
                  f"cache_miss_ratio={r['cache_miss_ratio']:.3f}% "
                  f"llc_miss_ratio={r['llc_miss_ratio']:.3f}% "
                  f"cache_refs_per_ms={r['cache_refs_per_ms']:.1f}", flush=True)

    print("\nBandwidth hypothesis sweep complete (or already done).")
