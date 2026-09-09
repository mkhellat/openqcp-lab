"""Why does a single-worker run land in two distinct time bands?

THE PUZZLE. `w1_c1` (one worker, pinned to cpu0) at N=150 measures
~22-25s in some contexts and ~30-31s in others. Two explanations were
already falsified: "it is fast only when a w4_c4 ran just before it"
(four consecutive w1_c1 runs with no w4_c4 still gave 24.8-25.3s) and
"a long idle gap makes it slow" (120s idle cost only ~2.5s, not 6s).

Since the baseline is the numerator of every parallel-efficiency
figure, a 25s-vs-31s ambiguity moves efficiency between 72% and 93%.
It has to be resolved before any efficiency number can be quoted.

WHAT THIS VARIES. Every earlier w1_c1 measurement pinned to cpu0, so
"which core" was a constant and could not be seen. cpu0 is not
interchangeable with the others on Linux: it handles the bulk of
interrupts (measured here: 12.2M against 3.8-5.5M for every other
CPU, a 2.7x skew). A run pinned there competes with interrupt
handling in a way a run on cpu3 does not.

This rotates the pinned CPU across repetitions and records, per run:
the CPU used, wall time, CPU time (user+sys), voluntary and
involuntary context switches, and start temperature. If the bands
track the core, the cause is core asymmetry. If they do not, the core
is exonerated and the covariates say where to look next -
involuntary context switches would point at scheduler preemption,
a cpu/wall ratio below 1 at stalling or descheduling.

Cores 0-3 are one hyperthread each of the four physical cores
(siblings are 4-7), so this compares physical cores, never siblings.

Usage:
    OPENBLAS_NUM_THREADS=1 python w1_core_rotation.py [reps_per_cpu] [N]
"""

import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "w1_core_rotation_results.jsonl")

CPUS = (0, 1, 2, 3)  # one hyperthread of each physical core
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 240
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def cooldown():
    start = time.perf_counter()
    while read_temp() > COOLDOWN_TARGET_C:
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return False
        time.sleep(2)
    return True


def run(n_osc, cpu, rep):
    reached = cooldown()
    t_start = read_temp()
    # A condition name is required by the target, but the CPU it pins
    # to is injected here instead, so the same single-worker workload
    # can be placed on any core. w1_c1's own entry is (1, [0]).
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS=str(n_osc),
               PAULIKIT_FORCE_CPU=str(cpu))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", "w1_c1", "2"],
        capture_output=True, text=True, env=env,
    )
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if proc.returncode != 0:
        print(f"  FAILED cpu{cpu}: {proc.stderr.strip()[-200:]}", flush=True)
        return None
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    rec = dict(
        n_osc=n_osc, cpu=cpu, rep=rep,
        elapsed=float(out["elapsed"].rstrip("s")),
        wall=round(wall, 3),
        cpu_time=round((after.ru_utime - before.ru_utime)
                       + (after.ru_stime - before.ru_stime), 3),
        vol_cs=after.ru_nvcsw - before.ru_nvcsw,
        invol_cs=after.ru_nivcsw - before.ru_nivcsw,
        terms=int(out["total_terms"]),
        start_temp=t_start, cooldown_reached=reached,
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  cpu{cpu} rep{rep}: {rec['elapsed']:6.2f}s  "
          f"cpu={rec['cpu_time']:6.2f}s  ratio={rec['cpu_time']/wall:.2f}  "
          f"ctx sw vol={rec['vol_cs']:5d} invol={rec['invol_cs']:5d}  "
          f"{t_start:.0f}C", flush=True)
    return rec


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_osc = int(sys.argv[2]) if len(sys.argv) > 2 else 150

    print(f"N={n_osc}, one worker, pinned CPU ROTATED across "
          f"{CPUS}, {reps} reps each", flush=True)
    print("cpu0 handles ~2.7x the interrupts of any other CPU on this "
          "machine\n", flush=True)

    rows = []
    for rep in range(reps):
        for cpu in CPUS:  # rotate, so drift cannot align with one core
            r = run(n_osc, cpu, rep)
            if r:
                rows.append(r)

    print("\n" + "=" * 70, flush=True)
    import statistics
    print(f"{'cpu':>5} {'mean':>9} {'sd':>7} {'min':>8} {'max':>8} "
          f"{'cpu_time':>9} {'invol_cs':>9}", flush=True)
    for cpu in CPUS:
        vals = [r["elapsed"] for r in rows if r["cpu"] == cpu]
        if not vals:
            continue
        ct = statistics.mean([r["cpu_time"] for r in rows if r["cpu"] == cpu])
        ics = statistics.mean([r["invol_cs"] for r in rows if r["cpu"] == cpu])
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{cpu:>5} {statistics.mean(vals):>8.2f}s {sd:>6.2f} "
              f"{min(vals):>7.2f} {max(vals):>7.2f} {ct:>8.2f}s {ics:>9.0f}",
              flush=True)

    allv = [r["elapsed"] for r in rows]
    print(f"\noverall: {min(allv):.2f}-{max(allv):.2f}s, "
          f"spread {max(allv)/min(allv):.2f}x", flush=True)
    counts = {r["terms"] for r in rows}
    print(f"term counts: {counts}", flush=True)


if __name__ == "__main__":
    main()
