"""Does a COLD first run inflate w4_c4 the way it inflates w1_c1?

THE PREDICTION UNDER TEST. The ~28% inflation seen on the first run of
a cold session was attributed to cold-SESSION state rather than to
anything about the single-worker configuration. Only one cell out of
eight ever occupied that slot - N=150 w1_c1, at +27.6% - while every
w4_c4 cell measured so far ran on an already-warm machine and showed
nothing (-1.4% to +4.3%).

If the attribution is right, running w4_c4 FIRST on a genuinely cold
machine must inflate it too - roughly +25-30%, so ~10.1-10.5s against
its warm ~8.1s.

If cold w4_c4 comes out at ~8.1s, the attribution is WRONG: something
is specific to the single-worker case, and every efficiency figure
that leans on a w1_c1 baseline needs rethinking rather than a
warm-up rule.

This is the falsifying test, not a confirming one: the outcome that
refutes the theory is the cheap and likely one.

DESIGN. Two orderings, each after a long quiet period:

  round A: w4_c4 cold-first, then w1_c1, then w4_c4 again
  round B: w1_c1 cold-first, then w4_c4, then w1_c1 again

Round A is the actual test. Round B reproduces the known w1_c1 effect
in the same session, which matters: if round B ALSO shows no inflation,
the machine was not truly cold and round A proves nothing. Round B is
the positive control.

The quiet period must exceed 90s - a 90s gap was measured to be
insufficient to restore the cold state (warmup_cause_results.jsonl),
and 120s idle cost only ~2.5s of the ~6s effect. This uses 600s by
default, and reports the temperature each run started at.

Usage:
    OPENBLAS_NUM_THREADS=1 python cold_first_run_prediction.py [quiet_s]
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "cold_first_run_results.jsonl")
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"

# Warm reference values from core_scaling_replicated_results.jsonl.
WARM = {"w1_c1": 23.35, "w4_c4": 8.07}


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def cooldown(target=55.0, cap=240):
    """Every run starts at the same temperature.

    The first version of this script had no cooldown, so temperature
    climbed 48 -> 68 -> 82C across a round and rode along with
    position: its third run looked +33.8% inflated, which was thermal
    throttling, not cold-start. Position and temperature must be
    separated or the test measures the wrong thing.
    """
    start = time.perf_counter()
    while read_temp() > target:
        if time.perf_counter() - start > cap:
            return False
        time.sleep(2)
    return True


def run(condition, tag, rnd):
    reached = cooldown()
    t_start = read_temp()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS="150")
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", condition, "2"],
        capture_output=True, text=True, env=env,
    )
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    elapsed = float(out["elapsed"].rstrip("s"))
    infl = (elapsed - WARM[condition]) / WARM[condition] * 100
    rec = dict(round=rnd, tag=tag, condition=condition, elapsed=elapsed,
               warm_reference=WARM[condition], inflation_pct=round(infl, 1),
               start_temp=t_start, cooldown_reached=reached,
               terms=int(out["total_terms"]))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  [{time.strftime('%H:%M:%S')}] {tag:>16} {condition}: "
          f"{elapsed:7.2f}s  (warm ref {WARM[condition]:.2f}s -> "
          f"{infl:+6.1f}%)  start {t_start:.0f}C", flush=True)
    return rec


def quiet(seconds):
    print(f"  [{time.strftime('%H:%M:%S')}] going quiet for "
          f"{seconds:.0f}s ...", flush=True)
    time.sleep(seconds)


def main():
    quiet_s = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0

    print("Does a cold first run inflate w4_c4 as it does w1_c1?", flush=True)
    print(f"warm references: w1_c1 {WARM['w1_c1']:.2f}s, "
          f"w4_c4 {WARM['w4_c4']:.2f}s", flush=True)
    print(f"quiet period between rounds: {quiet_s:.0f}s\n", flush=True)

    rows = []
    print("ROUND A - w4_c4 FIRST (the test)", flush=True)
    quiet(quiet_s)
    rows.append(run("w4_c4", "cold-first", "A"))
    rows.append(run("w1_c1", "second", "A"))
    rows.append(run("w4_c4", "third(warm)", "A"))

    print("\nROUND B - w1_c1 FIRST (positive control)", flush=True)
    quiet(quiet_s)
    rows.append(run("w1_c1", "cold-first", "B"))
    rows.append(run("w4_c4", "second", "B"))
    rows.append(run("w1_c1", "third(warm)", "B"))

    print("\n" + "=" * 70, flush=True)
    a_cold = rows[0]["inflation_pct"]
    b_cold = rows[3]["inflation_pct"]
    print(f"cold-first w4_c4 inflation: {a_cold:+.1f}%", flush=True)
    print(f"cold-first w1_c1 inflation: {b_cold:+.1f}%   (positive control)",
          flush=True)
    print(flush=True)
    if b_cold < 10:
        print("POSITIVE CONTROL FAILED: w1_c1 was not inflated either, so the",
              flush=True)
        print("machine was not truly cold. This run proves nothing about w4.",
              flush=True)
    elif a_cold >= 10:
        print("Both inflated -> the effect is cold-SESSION state, not "
              "specific to", flush=True)
        print("the single-worker case. The warm-up rule stands.", flush=True)
    else:
        print("w1 inflated but w4 did NOT -> the attribution is WRONG. "
              "Something is", flush=True)
        print("specific to the single-worker path and needs its own "
              "explanation.", flush=True)
    print(f"\nterm counts: {({r['terms'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
