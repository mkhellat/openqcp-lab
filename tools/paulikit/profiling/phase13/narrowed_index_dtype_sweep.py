"""Measure the real end-to-end effect of narrowing the (x, z) index
arrays that cross the ProcessPoolExecutor boundary (PLAN.md Phase 13,
Step 2 of the R3 work).

CONTEXT. drain_gil_backpressure_results.jsonl established that the
multi-core roadblock is GIL-held drain work in the parent starving the
result pipe: adding it to an otherwise-scaling control collapsed 2.22x
-> 0.87x. Narrowing the payload attacks the same bottleneck from the
other side - less to serialize, less to push through the single
lock/pipe/feeder path, less for the parent to unpickle.

WHAT WAS ACTUALLY CHANGED, and what was deliberately NOT:
  - `chunk_x_out` and `z_idx`: intp (8B) -> uint16 at dim=16384, via
    `_index_dtype_for_dim(dim)`. Exact, not lossy: both are provably
    < dim. The dtype is derived from dim rather than hardcoded because
    uint16 wraps for n_qubits > 16, which would corrupt Pauli labels
    rather than error.
  - `chunk_coeff_out`: LEFT complex128. Narrowing it to its real part
    would cut the payload a further ~1.7x, and was rejected on
    correctness grounds: `_build_real_terms` detects a non-Hermitian
    operator by checking the IMAGINARY part, in the parent, after this
    value has crossed IPC. Sending only the real part would silently
    convert a raised ValueError into a wrong answer.

Measured pickled payload at the real per-chunk term count (16,381):
  current (intp, intp, complex128) : 512.1 KiB/chunk -> 2.93 GB total
  narrowed indices only            : 320.2 KiB/chunk -> 1.83 GB total
  = 1.60x less traffic through the serialized result path.

HONEST EXPECTATION, stated before the run. This is NOT expected to
restore full scaling. The single-variable test showed the collapse is
driven by how long the parent holds the GIL, and drain_work collapsed
at a payload IDENTICAL to bare's. Narrowing reduces how much crosses
the pipe, not how long the parent is unavailable to service it. A
real but partial improvement is the honest prediction; the drain-side
GIL cost (d5 dict-build, 60.5% GIL-held) is Step 3's target.

The dominant risk this script exists to rule out: the `.astype()` cast
itself runs in the WORKER, so it costs real per-chunk time. If that
cost exceeds the IPC saving, this is a net regression - exactly the
shape of the reverted labeling-relocation fix (9c5f1c6), where a
theoretically-motivated change lost to an unmodeled per-chunk cost.
w1_c1 is included specifically as the control that would expose it:
with one worker there is minimal pipe contention, so w1_c1 should be
neutral-to-slightly-negative if the cast is expensive.

Protocol matches every other measurement in this investigation:
foreground, thermal cooldown to 55C before every run, N reps,
Welch's t-test, incremental JSON-Lines with resume-skip.

Usage:
    REPS=5 OPENBLAS_NUM_THREADS=1 python narrowed_index_dtype_sweep.py

Compare against the pre-change baseline recorded in
full_optimum_sweep_results.jsonl (w1_c1 26.366s, w2_c1 20.537s,
w8_c4 23.964s, n=10 each, same cooldown protocol).
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
TARGET = os.path.join(HERE, "full_matrix_target.py")
RESULTS = os.path.join(HERE, "narrowed_index_dtype_results.jsonl")

CONDITIONS = ("w1_c1", "w2_c1", "w8_c4")
CHUNK_SIZE = "2"
REPS = int(os.environ.get("REPS", "5"))
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"

# Pre-change means from full_optimum_sweep_results.jsonl (n=10 each,
# identical cooldown protocol and target script).
BASELINE = {"w1_c1": 26.366, "w2_c1": 20.537, "w8_c4": 23.964}


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
                    continue
                done.add((r["condition"], r["rep"]))
    return done


def run_one(condition, rep):
    settled = cooldown()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    t_before = read_temp()
    proc = subprocess.run(
        [PYTHON, TARGET, condition, CHUNK_SIZE],
        capture_output=True, text=True, env=env,
    )
    t_after = read_temp()
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr.strip()[:300]}")
        return
    elapsed = terms = None
    for tok in proc.stdout.split():
        if tok.startswith("elapsed="):
            elapsed = float(tok.split("=", 1)[1].rstrip("s"))
        elif tok.startswith("total_terms="):
            terms = int(tok.split("=", 1)[1])
    if elapsed is None:
        print(f"  no elapsed: {proc.stdout.strip()[:200]}")
        return
    rec = dict(condition=condition, rep=rep, elapsed=elapsed, total_terms=terms,
               settled_temp=settled, temp_before=t_before, temp_after=t_after)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    base = BASELINE[condition]
    print(f"  {condition:>6} rep{rep}: {elapsed:7.3f}s "
          f"(baseline {base:.3f}s, {base/elapsed:5.3f}x) terms={terms}")


def main():
    done = load_completed()
    for condition in CONDITIONS:
        for rep in range(REPS):
            if (condition, rep) in done:
                print(f"  {condition:>6} rep{rep}: skip (done)")
                continue
            run_one(condition, rep)

    rows = [json.loads(l) for l in open(RESULTS)]
    print("\n" + "=" * 66)
    print(f"{'cond':>7} {'baseline':>9} {'narrowed':>9} {'speedup':>9} {'sd':>6} {'n':>3}")
    for c in CONDITIONS:
        v = [r["elapsed"] for r in rows if r["condition"] == c]
        if not v:
            continue
        m = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{c:>7} {BASELINE[c]:>9.3f} {m:>9.3f} "
              f"{BASELINE[c]/m:>8.3f}x {sd:>6.3f} {len(v):>3}")

    # Correctness is not optional: every run must produce the exact
    # same term count, or the narrowing lost information.
    counts = {r["total_terms"] for r in rows if r["total_terms"] is not None}
    print(f"\nterm counts across all runs: {counts}")
    print("(must be a single value - a second value means the dtype "
          "narrowing dropped or corrupted terms)")

    print("\nNOTE: the baseline is n=10 from a prior sweep, not re-run "
          "here, so these are\nnot paired samples - treat the ratio as "
          "indicative and confirm with a\nWelch test against the raw "
          "baseline rows if a precise claim is needed.")


if __name__ == "__main__":
    main()
