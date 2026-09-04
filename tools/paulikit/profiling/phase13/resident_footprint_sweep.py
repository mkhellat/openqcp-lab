"""Driver for resident_footprint_target.py - the operator/setup-array
resident-footprint isolation experiment (the last untested item from
traffic_intensity_findings.md's decision tree, after
gather_pattern_findings.md's mixed result). Same protocol as every
prior measurement: w2_c1 vs w8_c4, thermal cooldown before every run,
5 reps/cell, Welch's t-test on the result.

Usage:
    OPENBLAS_NUM_THREADS=1 python resident_footprint_sweep.py [--reps N]
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
TARGET = os.path.join(DIR, "resident_footprint_target.py")
RESULTS_PATH = os.path.join(DIR, "resident_footprint_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

CONDITIONS = ["w2_c1", "w8_c4"]


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
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    result = subprocess.run(
        [PYTHON, TARGET, condition], capture_output=True, text=True, env=env,
    )
    m = re.search(r"elapsed=([\d.]+)s", result.stdout)
    elapsed = float(m.group(1)) if m else None
    return {
        "elapsed": elapsed,
        "settled_temp": settled_temp,
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-500:] if result.returncode != 0 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    completed = load_completed()
    total_planned = len(CONDITIONS) * args.reps
    total_done = sum(1 for c in CONDITIONS for r in range(args.reps) if (c, r) in completed)
    print(
        f"Resuming: {total_done}/{total_planned} runs already recorded in "
        f"{RESULTS_PATH}",
        flush=True,
    )

    for condition in CONDITIONS:
        for rep in range(args.reps):
            if (condition, rep) in completed:
                continue
            r = run_once(condition)
            r["condition"] = condition
            r["rep"] = rep
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"{condition} rep={rep}: elapsed={r['elapsed']} "
                f"settled_temp={r['settled_temp']}",
                flush=True,
            )

    print("\nResident-footprint sweep complete (or already done).")
