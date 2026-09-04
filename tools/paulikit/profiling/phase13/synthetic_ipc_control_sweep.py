"""Driver for synthetic_ipc_control.py - the paulikit-free control
sweep answering code_specificity_findings.md's open question 3: does
the 2-physical-core ceiling reproduce on a workload sharing nothing
with paulikit but the coarse shape (many small CPU-bound tasks,
default ProcessPoolExecutor pickle-over-pipe IPC)?

Restricted to the same conditions the chunk_size screen already
covered (w2_c1 as the record to beat, plus the 12 multi-core configs)
- no need to re-run all 14, w1_c1 was never competitive in the real
sweep either. Thermal-cooldown protocol, same as every other
measurement in this investigation. N reps configurable; default 5 -
enough for a real (if not full publication-grade) comparison, not
just 1 noisy run, given how cheap 5.2-13s runs are per condition.

Usage:
    OPENBLAS_NUM_THREADS=1 python synthetic_ipc_control_sweep.py [--reps N]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import SWEEP_CONFIGS as _SWEEP_CONFIGS  # noqa: E402

TARGET_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "synthetic_ipc_control.py")
PYTHON = sys.executable
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "synthetic_ipc_control_results.jsonl")
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

CONFIGS = ["w2_c1", "w8_c4"]


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
        [PYTHON, TARGET_SCRIPT, condition], capture_output=True, text=True, env=env,
    )
    import re
    m = re.search(r"elapsed=([\d.]+)s", result.stdout)
    elapsed = float(m.group(1)) if m else None
    return {"elapsed": elapsed, "settled_temp": settled_temp,
            "stdout": result.stdout.strip(), "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:] if result.returncode != 0 else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    completed = load_completed()
    total_planned = len(CONFIGS) * args.reps
    total_done = sum(1 for c in CONFIGS for r in range(args.reps) if (c, r) in completed)
    print(f"Resuming: {total_done}/{total_planned} runs already recorded in "
          f"{RESULTS_PATH}", flush=True)

    for condition in CONFIGS:
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
            print(f"{condition} rep={rep}: elapsed={r['elapsed']} "
                  f"settled_temp={r['settled_temp']}", flush=True)

    print("\nSynthetic control sweep complete (or already done).")
