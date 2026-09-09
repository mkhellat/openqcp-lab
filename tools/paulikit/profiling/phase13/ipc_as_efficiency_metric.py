"""Is IPC (instructions per cycle) a better efficiency quantity than
wall-clock speedup?

THE PROPOSAL (user's). If the instruction count is constant across
runs - the same algorithm on the same data - then IPC measures how
efficiently those instructions execute, independent of how long the
run happened to take. That would sidestep the cold-start penalty and
the thermal variation that contaminate wall-clock.

PRIOR EVIDENCE FOR IT. `instructions_ipc_accounting_findings.md`
(2026-09-03) already established the premise on this machine: across
four worker counts, retired `instructions` differed by under 0.2% and
was never significant, while IPC dropped robustly (p<0.0001 in 3 of 4)
when workers spread across more cores. The work really is constant;
what changes is stalling.

WHAT THIS ADDS. That earlier study compared core PACKINGS at a fixed
worker count. It never asked the question that matters here: is IPC
stable across the COLD-START boundary that corrupts wall-clock? If it
is, it is a genuinely better basis for reported efficiency. If it
inflates like everything else, it inherits the same contamination.

WHAT IS MEASURED, per run: retired instructions, cycles, IPC, wall
time, and task-clock, via `perf stat` (works unprivileged here,
perf_event_paranoid=2, counting user-space :u events). The first run
follows a quiet machine; later runs are warm.

  instructions constant, IPC stable -> IPC is the better metric.
  instructions constant, IPC inflated on the cold run -> IPC inherits
      the contamination; it measures the same slowdown wall-clock does.
  instructions NOT constant -> the premise fails and IPC cannot be
      compared across these runs at all.

A CAVEAT THE NUMBERS CANNOT SETTLE. IPC is a measure of pipeline
efficiency, not of parallel efficiency. Two workers each running at
IPC 2.0 have the same IPC as one worker at IPC 2.0, yet do twice the
work per second. So IPC cannot replace speedup as the *definition* of
parallel efficiency - at best it explains WHY speedup falls short, and
serves as a stability check on individual runs.

Usage:
    OPENBLAS_NUM_THREADS=1 python ipc_as_efficiency_metric.py [reps] [condition]
"""

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "ipc_metric_results.jsonl")
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"

EVENTS = "instructions:u,cycles:u,task-clock"


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def parse_perf(stderr):
    """perf prints counters to stderr, with thousands separators."""
    vals = {}
    for line in stderr.splitlines():
        # perf prints task-clock as "128.79 msec task-clock:u" - the
        # unit sits between the number and the counter name, so a
        # pattern expecting them adjacent silently misses it and the
        # value comes back 0.
        m = re.match(r"\s*([\d,\.]+)\s+(?:msec\s+)?([\w\-:]+)", line)
        if m:
            try:
                vals[m.group(2)] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return vals


def run(condition, rep, n_osc=150):
    t_start = read_temp()
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["perf", "stat", "-e", EVENTS, "--",
         PYTHON, TARGET, "arrays_no_checkpoint", condition, "2"],
        capture_output=True, text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                 PAULIKIT_N_OSCILLATORS=str(n_osc)),
    )
    wall = time.perf_counter() - t0
    counters = parse_perf(proc.stderr)
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    if "elapsed" not in out:
        print(f"  run failed: {proc.stdout[:150]} {proc.stderr[-200:]}",
              flush=True)
        return None
    instr = counters.get("instructions:u", 0.0)
    cycles = counters.get("cycles:u", 0.0)
    ipc = instr / cycles if cycles else 0.0
    rec = dict(
        condition=condition, rep=rep, n_osc=n_osc,
        elapsed=float(out["elapsed"].rstrip("s")),
        driver_wall=round(wall, 3),
        instructions=instr, cycles=cycles, ipc=round(ipc, 4),
        task_clock_ms=counters.get("task-clock:u", counters.get("task-clock", 0.0)),
        start_temp=t_start, terms=int(out["total_terms"]),
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tag = "COLD (first)" if rep == 0 else f"rep{rep}"
    print(f"  {tag:>12}: wall {rec['elapsed']:7.2f}s  "
          f"instr {instr / 1e9:8.2f}G  cycles {cycles / 1e9:8.2f}G  "
          f"IPC {ipc:5.3f}  {t_start:.0f}C", flush=True)
    return rec


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    condition = sys.argv[2] if len(sys.argv) > 2 else "w1_c1"

    print(f"Is IPC stable across the cold-start boundary?  "
          f"condition {condition}", flush=True)
    print("The FIRST row follows a quiet machine.\n", flush=True)

    rows = [r for r in (run(condition, i) for i in range(reps)) if r]
    if len(rows) < 2:
        print("not enough successful runs", flush=True)
        return

    print("\n" + "=" * 72, flush=True)
    import statistics
    first, rest = rows[0], rows[1:]
    for name, key in (("wall", "elapsed"), ("instructions", "instructions"),
                      ("cycles", "cycles"), ("IPC", "ipc")):
        later = [r[key] for r in rest]
        m = statistics.mean(later)
        sd = statistics.stdev(later) if len(later) > 1 else 0.0
        infl = (first[key] - m) / m * 100 if m else 0.0
        allv = [r[key] for r in rows]
        spread = max(allv) / min(allv) if min(allv) else 0.0
        print(f"  {name:>13}: first {first[key]:12.4g}  later mean "
              f"{m:12.4g} (sd {sd:9.4g})  first {infl:+6.1f}%  "
              f"spread {spread:.3f}x", flush=True)

    instr_spread = (max(r["instructions"] for r in rows)
                    / min(r["instructions"] for r in rows))
    ipc_all = [r["ipc"] for r in rows]
    wall_all = [r["elapsed"] for r in rows]
    ipc_spread = max(ipc_all) / min(ipc_all)
    wall_spread = max(wall_all) / min(wall_all)

    print(flush=True)
    if instr_spread > 1.02:
        print(f"PREMISE FAILS: instruction count varies {instr_spread:.3f}x "
              "across runs,", flush=True)
        print("so IPC values are not comparable between them.", flush=True)
    elif ipc_spread < wall_spread:
        print(f"IPC IS MORE STABLE than wall clock "
              f"({ipc_spread:.3f}x vs {wall_spread:.3f}x) and the", flush=True)
        print(f"instruction count is constant to {instr_spread:.3f}x - the "
              "premise holds.", flush=True)
    else:
        print(f"IPC IS NOT MORE STABLE ({ipc_spread:.3f}x vs wall "
              f"{wall_spread:.3f}x):", flush=True)
        print("it inherits the same contamination.", flush=True)
    print(f"\nterm counts: {({r['terms'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
