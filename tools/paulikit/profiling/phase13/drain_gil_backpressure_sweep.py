"""Driver for the Step 1 R3 mechanism test (see
drain_gil_backpressure_target.py for the hypothesis and the
falsification criterion).

2x2 design: {bare, drain_work} x {w2_c1, w8_c4}, N reps each,
thermal cooldown to 55C before EVERY run, results appended
incrementally to JSON-Lines so a killed run loses nothing and can be
resumed (same resume-skip discipline as full_optimum_sweep.py).

Statistics: Welch's unequal-variance t-test on the w2-vs-w8 contrast
WITHIN each mode - never an eyeballed comparison of two means, per
this investigation's own standing rule that single-run and
"ranges overlap" comparisons have repeatedly produced false findings
here.

Foreground-only; refuses to start if another measurement job is
already running.
"""

import json
import os
import statistics
import subprocess
import sys
import time

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TARGET = os.path.join(HERE, "drain_gil_backpressure_target.py")
RESULTS = os.path.join(HERE, "drain_gil_backpressure_results.jsonl")

CONDITIONS = ("w2_c1", "w8_c4")
MODES = ("bare", "drain_work", "arrays_only")
REPS = int(os.environ.get("REPS", "5"))
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"


def read_temp():
    try:
        with open(TEMP_PATH) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown():
    start = time.perf_counter()
    while True:
        t = read_temp()
        if t is not None and t <= COOLDOWN_TARGET_C:
            return t
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return t
        time.sleep(2)


def load_completed():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated trailing line
                done.add((r["condition"], r["mode"], r["rep"]))
    return done


def run_one(condition, mode, rep):
    settled = cooldown()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    t_before = read_temp()
    proc = subprocess.run(
        [PYTHON, TARGET, condition, mode],
        capture_output=True, text=True, env=env,
    )
    t_after = read_temp()
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr.strip()[:400]}")
        return None
    elapsed = None
    for tok in proc.stdout.split():
        if tok.startswith("elapsed="):
            elapsed = float(tok.split("=", 1)[1].rstrip("s"))
    if elapsed is None:
        print(f"  no elapsed in output: {proc.stdout.strip()[:200]}")
        return None
    rec = dict(condition=condition, mode=mode, rep=rep, elapsed=elapsed,
               settled_temp=settled, temp_before=t_before, temp_after=t_after)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {condition:>6} {mode:>10} rep{rep}: {elapsed:7.3f}s "
          f"(temp {t_before}->{t_after}C)")
    return rec


def main():
    running = subprocess.run(
        ["pgrep", "-fc", "drain_gil_backpressure_target|full_matrix_target"],
        capture_output=True, text=True,
    ).stdout.strip()
    if running.isdigit() and int(running) > 0:
        raise SystemExit("another measurement job is running - refusing to start")

    done = load_completed()
    for mode in MODES:
        for condition in CONDITIONS:
            for rep in range(REPS):
                if (condition, mode, rep) in done:
                    print(f"  {condition:>6} {mode:>10} rep{rep}: skip (done)")
                    continue
                run_one(condition, mode, rep)

    rows = [json.loads(l) for l in open(RESULTS)]
    print("\n" + "=" * 64)
    print(f"{'mode':>11} {'w2_c1':>9} {'w8_c4':>9} {'speedup':>9} "
          f"{'Welch p':>10}")
    summary = {}
    for mode in MODES:
        a = [r["elapsed"] for r in rows if r["mode"] == mode and r["condition"] == "w2_c1"]
        b = [r["elapsed"] for r in rows if r["mode"] == mode and r["condition"] == "w8_c4"]
        if len(a) < 2 or len(b) < 2:
            continue
        ma, mb = statistics.mean(a), statistics.mean(b)
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        summary[mode] = ma / mb
        print(f"{mode:>11} {ma:>9.3f} {mb:>9.3f} {ma/mb:>9.3f}x {p:>10.2e}")

    if {"bare", "drain_work"} <= summary.keys():
        print()
        print(f"bare       speedup (w2->w8): {summary['bare']:.3f}x")
        print(f"drain_work speedup (w2->w8): {summary['drain_work']:.3f}x")
        print("reference: wht_large 2.23x (scales), paulikit 0.89x (collapses)")
        print()
        if summary["drain_work"] < 1.0 <= summary["bare"]:
            print("--> R3 CONFIRMED: adding GIL-held drain work, and nothing")
            print("    else, destroys the control's multi-core scaling.")
        elif summary["drain_work"] > 1.8:
            print("--> R3 FALSIFIED: the control still scales with drain work.")
            print("    GIL-held drain work is NOT what breaks paulikit.")
        else:
            print("--> PARTIAL: scaling degraded but not collapsed - drain")
            print("    work contributes but is not the whole mechanism.")

    if "arrays_only" in summary:
        ao = summary["arrays_only"]
        print()
        print("=" * 64)
        print("API-CHANGE DECISION TEST (arrays_only)")
        print(f"  arrays_only speedup (w2->w8): {ao:.3f}x")
        print()
        print("  This mode keeps the Hermiticity check but drops label")
        print("  and dict construction - i.e. exactly what an")
        print("  array-yielding API would do on the drain side.")
        print()
        if ao >= 1.8:
            print("  --> RECOVERS SCALING. The label/dict construction is")
            print("      confirmed as the specific cause of the multi-core")
            print("      collapse, by direct measurement rather than by")
            print("      inference from a microbenchmark. An array-yielding")
            print("      API is justified.")
        elif ao < 1.2:
            print("  --> DOES NOT RECOVER SCALING. Removing label/dict work")
            print("      is NOT sufficient: something else in the drain path")
            print("      is responsible. Do NOT proceed with the API change")
            print("      on this evidence - the premise is falsified.")
        else:
            print("  --> PARTIAL RECOVERY. Removing label/dict work helps but")
            print("      does not restore the control's own scaling. The API")
            print("      change would be a real but incomplete fix; decide")
            print("      with that caveat explicit, not hidden.")


if __name__ == "__main__":
    main()
