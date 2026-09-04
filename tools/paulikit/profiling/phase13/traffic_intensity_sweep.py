"""Thermal-controlled sweep for the traffic-intensity suspect.

Cells: {wht_small, touch_small, wht_large} x {w2_c1, w8_c4}, 5 reps,
cooldown to 55C, child-inheriting perf stat (same protocol as
bandwidth_hypothesis_sweep.py).

Prediction if traffic intensity is the roadblock:
  - wht_small and touch_small should FAIL to scale w2->w8 (like
    paulikit), despite tiny IPC payloads.
  - If only wht_large fails, the roadblock is large result IPC, not
    compute-side traffic.
  - If all three scale like the old 48x48 synthetic, traffic alone is
    insufficient and the suspect is elsewhere (gather/operator copies,
    term filtering, etc.).

Usage:
    OPENBLAS_NUM_THREADS=1 python traffic_intensity_sweep.py [--reps N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
EVENTS = (
    "task-clock,cycles,instructions,cache-references,cache-misses,"
    "LLC-loads,LLC-load-misses"
)
RESULTS_PATH = os.path.join(DIR, "traffic_intensity_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

CELLS = [
    ("wht_small", "w2_c1"),
    ("wht_small", "w8_c4"),
    ("touch_small", "w2_c1"),
    ("touch_small", "w8_c4"),
    ("wht_large", "w2_c1"),
    ("wht_large", "w8_c4"),
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
    completed: set[tuple[str, str, int]] = set()
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


def run_once(workload: str, condition: str) -> dict:
    settled_temp = cooldown()
    temp_before = _read_pkg_temp_c()
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = os.path.join(DIR, "..", "..", "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    target = os.path.join(DIR, "traffic_intensity_target.py")
    cmd = ["perf", "stat", "-e", EVENTS, PYTHON, target, condition, workload]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    temp_after = _read_pkg_temp_c()
    stderr = result.stderr
    stdout = result.stdout

    counters: dict[str, float] = {}
    for line in stderr.splitlines():
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
        "ipc": (instructions / cycles) if (instructions and cycles) else None,
        "cache_references": cache_refs,
        "cache_misses": cache_misses,
        "cache_miss_ratio": (
            100 * cache_misses / cache_refs if cache_refs else None
        ),
        "llc_loads": llc_loads,
        "llc_misses": llc_misses,
        "llc_miss_ratio": (
            100 * llc_misses / llc_loads if llc_loads else None
        ),
        "cache_refs_per_ms": (
            cache_refs / task_clock_ms if (cache_refs and task_clock_ms) else None
        ),
        "settled_temp": settled_temp,
        "temp_before": temp_before,
        "temp_after": temp_after,
        "returncode": result.returncode,
        "stderr_tail": stderr[-500:] if result.returncode != 0 else None,
        "stdout_tail": stdout[-300:] if result.returncode != 0 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    completed = load_completed()
    total_planned = len(CELLS) * args.reps
    print(
        f"Resuming: {len(completed)}/{total_planned} runs already in "
        f"{RESULTS_PATH}",
        flush=True,
    )

    for workload, condition in CELLS:
        for rep in range(args.reps):
            if (workload, condition, rep) in completed:
                continue
            r = run_once(workload, condition)
            r["rep"] = rep
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"{workload:12s} {condition:8s} rep={rep}: "
                f"elapsed={r['elapsed']} rc={r['returncode']} "
                f"cache_miss%={r['cache_miss_ratio']} "
                f"eff_proxy_task_clock={r['task_clock_ms']}",
                flush=True,
            )
            if r["returncode"] != 0:
                print(f"  STDERR: {r['stderr_tail']}", flush=True)
                print(f"  STDOUT: {r['stdout_tail']}", flush=True)

    print("\nTraffic-intensity sweep complete (or already done).")
