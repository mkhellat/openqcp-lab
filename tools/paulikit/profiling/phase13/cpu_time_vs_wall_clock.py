"""Is CPU time immune to the cold-start penalty that corrupts wall clock?

THE PROPOSAL (user's). Wall-clock is contaminated by a ~28% cold-start
penalty whose cause is not isolated. CPU time - seconds of core
occupancy, from getrusage(RUSAGE_CHILDREN) - might be a better basis
for efficiency, since it counts work done rather than time elapsed.

WHY IT MIGHT WORK. If the cold-start penalty is the cores running at a
lower clock, the same instruction stream takes more WALL time but the
same number of CPU-seconds is not obviously implied - CPU time counts
scheduled time, not cycles, so a slow core still accrues CPU-seconds at
one per second. That makes the prediction non-trivial in both
directions and worth measuring rather than assuming.

WHY IT MIGHT NOT. `ru_utime + ru_stime` measures how long a core was
ASSIGNED to the process, in wall seconds. If the core is simply slower,
the work takes longer and CPU time rises with wall time - inheriting
the exact contamination the proposal hopes to escape.

WHAT IS MEASURED. First run of a genuinely cold session, then repeats,
recording wall time, CPU time, and their ratio. The ratio is mean
cores busy - for one worker it should sit near 1.0 regardless of speed,
so if the RATIO is stable while both components inflate, the ratio is
the robust quantity even though CPU time alone is not.

  cpu time flat, wall inflated -> CPU time is the better metric.
  both inflated, ratio flat    -> the RATIO is the robust quantity.
  all three inflated           -> the proposal fails; nothing here
                                  escapes the cold-start effect.

Run this on a machine that has been quiet: the first row IS the
experiment, and it cannot be repeated without another long quiet
period.

Usage:
    OPENBLAS_NUM_THREADS=1 python cpu_time_vs_wall_clock.py [reps] [condition]
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
RESULTS = os.path.join(HERE, "cpu_time_vs_wall_results.jsonl")
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def core_temps():
    d = "/sys/class/hwmon/hwmon5"
    try:
        out = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith("_label"):
                label = open(f"{d}/{fn}").read().strip()
                if label.startswith("Core "):
                    out[label] = int(
                        open(f"{d}/{fn[:-6]}_input").read().strip()) / 1000.0
        return out or None
    except OSError:
        return None


def run(condition, rep, n_osc=150):
    t_start = read_temp()
    cores = core_temps()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", condition, "2"],
        capture_output=True, text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                 PAULIKIT_N_OSCILLATORS=str(n_osc)),
    )
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ((after.ru_utime - before.ru_utime)
           + (after.ru_stime - before.ru_stime))
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    elapsed = float(out["elapsed"].rstrip("s"))
    rec = dict(
        condition=condition, rep=rep, n_osc=n_osc,
        elapsed=elapsed, driver_wall=round(wall, 3),
        cpu_time=round(cpu, 3), cpu_per_wall=round(cpu / wall, 3),
        start_temp=t_start, core_temps=cores,
        terms=int(out["total_terms"]),
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tag = "COLD (first)" if rep == 0 else f"rep{rep}"
    print(f"  {tag:>12}: wall {elapsed:7.2f}s  cpu {cpu:7.2f}s  "
          f"cpu/wall {cpu / wall:5.2f}  start {t_start:.0f}C", flush=True)
    return rec


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    condition = sys.argv[2] if len(sys.argv) > 2 else "w1_c1"

    print(f"Does CPU time escape the cold-start penalty?  "
          f"condition {condition}", flush=True)
    print("The FIRST row is the experiment - it requires a quiet "
          "machine.\n", flush=True)

    rows = [run(condition, r) for r in range(reps)]

    print("\n" + "=" * 66, flush=True)
    import statistics
    first = rows[0]
    rest = rows[1:]
    if not rest:
        return
    for name, key in (("wall", "elapsed"), ("cpu time", "cpu_time"),
                      ("cpu/wall", "cpu_per_wall")):
        later = [r[key] for r in rest]
        m = statistics.mean(later)
        infl = (first[key] - m) / m * 100
        sd = statistics.stdev(later) if len(later) > 1 else 0.0
        print(f"  {name:>9}: first {first[key]:8.2f}  later mean "
              f"{m:8.2f} (sd {sd:5.2f})  first is {infl:+6.1f}%", flush=True)

    print(flush=True)
    w = (first["elapsed"] - statistics.mean([r["elapsed"] for r in rest])) \
        / statistics.mean([r["elapsed"] for r in rest]) * 100
    c = (first["cpu_time"] - statistics.mean([r["cpu_time"] for r in rest])) \
        / statistics.mean([r["cpu_time"] for r in rest]) * 100
    ratio = (first["cpu_per_wall"]
             - statistics.mean([r["cpu_per_wall"] for r in rest])) \
        / statistics.mean([r["cpu_per_wall"] for r in rest]) * 100

    if abs(c) < 5 and abs(w) >= 10:
        print("CPU TIME IS THE BETTER METRIC: wall inflated, cpu time flat.",
              flush=True)
    elif abs(ratio) < 5 and abs(w) >= 10:
        print("THE RATIO IS THE ROBUST QUANTITY: both components inflate "
              "together,", flush=True)
        print("so cpu/wall stays flat even though neither alone does.",
              flush=True)
    elif abs(w) < 10:
        print("NO COLD-START EFFECT IN THIS RUN - the machine was not "
              "quiet enough.", flush=True)
        print("This decides nothing; repeat after a genuinely idle period.",
              flush=True)
    else:
        print("PROPOSAL FAILS: cpu time inherits the same contamination "
              "as wall clock.", flush=True)
    print(f"\nterm counts: {({r['terms'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
