"""Physical-core scaling with replication and interleaving.

WHY THIS EXISTS. Every earlier core-scaling number in this phase was
computed from n=1 per cell, and each one was wrong. The `w1_c1`
baseline in particular turned out to vary far more than the parallel
runs: 24.31s (n=5, sd 0.99) against single runs of 30.37s and 37.49s
for the identical configuration. Since the baseline is the NUMERATOR
of every efficiency figure, a bad baseline corrupts the whole table -
and three successive "corrections" were each built on another n=1
measurement.

WHAT IS DIFFERENT HERE.

1. REPLICATION. Every cell gets `reps` runs, and the report carries
   mean, sd, min, max and spread. A cell whose spread exceeds ~1.15x
   is flagged: at that point the mean is not a summary of anything
   and the cell should not be quoted.

2. INTERLEAVING. Conditions alternate within each rep
   (w1, w4, w1, w4, ...) rather than running all w1 then all w4. If
   the machine drifts over a session - warming, background load, page
   cache - a blocked design assigns that drift entirely to whichever
   condition ran during it, manufacturing an effect. Interleaving
   spreads drift across both conditions instead.

3. A REAL TEST. Welch's unequal-variance t-test on the two conditions,
   plus the ratio of means with a min/max adversarial bound. The bound
   is the honest number to quote when the distributions are wide.

4. WARM-UP DISCARD. The first rep of each condition is recorded but
   excluded from the statistics by default, since cold page cache and
   first-touch allocation plausibly explain part of the outlier
   pattern. `--keep-first` includes it; the report always shows both
   so the choice is visible rather than hidden.

Thermal protocol is phase 13's own: cooldown to 55C before every timed
run, and the temperature the run ACTUALLY started at is logged. 55C is
reachable on this machine (it idles in the mid-40s); cooldowns only
hit their ceiling inside a dense session where the previous run's heat
has not dissipated. Check the logged start temperatures, not the
setting.

No checkpointing anywhere - this measures the decomposition alone.

Usage:
    OPENBLAS_NUM_THREADS=1 python core_scaling_replicated.py N [reps] [chunk_size]
"""

import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "core_scaling_replicated_results.jsonl")

CONDITIONS = ("w1_c1", "w4_c4")
COOLDOWN_TARGET_C = float(os.environ.get("PAULIKIT_COOLDOWN_C", "55.0"))
COOLDOWN_TIMEOUT_S = 240
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
SPREAD_ALARM = 1.15


def read_temp():
    try:
        with open(TEMP_PATH) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def log(msg):
    print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cooldown():
    start = time.perf_counter()
    while True:
        t = read_temp()
        if t is not None and t <= COOLDOWN_TARGET_C:
            return t, time.perf_counter() - start, True
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return t, time.perf_counter() - start, False
        time.sleep(2)


def run_one(n_osc, condition, chunk_size, rep):
    start_temp, waited, reached = cooldown()
    if not reached:
        log(f"WARNING: cooldown timed out at {start_temp:.0f}C - this run "
            f"is NOT thermally matched")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS=str(n_osc))
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", condition, str(chunk_size)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        log(f"FAILED {condition}: {proc.stderr.strip()[-200:]}")
        return None
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    rec = dict(
        n_osc=n_osc, condition=condition, chunk_size=chunk_size, rep=rep,
        elapsed=float(out["elapsed"].rstrip("s")),
        terms=int(out["total_terms"]),
        peak_rss_mib=float(out["peak_rss_mib"]),
        start_temp=start_temp, cooled_s=round(waited, 1),
        cooldown_reached=reached, end_temp=read_temp(),
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"{condition} rep{rep}: {rec['elapsed']:7.2f}s  "
        f"rss={rec['peak_rss_mib']:.0f}MiB  start {start_temp:.0f}C")
    return rec


def describe(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, sd, min(values), max(values), max(values) / min(values)


def main():
    n_osc = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    keep_first = "--keep-first" in sys.argv

    if chunk_size == 0:
        from paulikit.algorithms import autotune
        from paulikit.cli import _default_masses, _default_spring_constants
        from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
        unpadded = build_hamiltonian(
            n_osc, _default_spring_constants(n_osc),
            _default_masses(n_osc), sparse=True)
        padded, _ = pad_to_power_of_two(unpadded, sparse=True)
        chunk_size = autotune.recommended_chunk_size(padded.shape[0])

    print(f"N={n_osc}, chunk_size={chunk_size}, {reps} reps per condition, "
          f"INTERLEAVED, no checkpointing", flush=True)
    print(f"cooldown to {COOLDOWN_TARGET_C:.0f}C before every run\n",
          flush=True)

    rows = {c: [] for c in CONDITIONS}
    for rep in range(reps):
        for condition in CONDITIONS:  # interleaved, not blocked
            r = run_one(n_osc, condition, chunk_size, rep)
            if r is not None:
                rows[condition].append(r)

    print("\n" + "=" * 78, flush=True)
    for label, drop_first in (("all reps", False), ("first rep dropped", True)):
        if drop_first and reps < 2:
            continue
        print(f"\n--- {label} ---", flush=True)
        stats = {}
        for c in CONDITIONS:
            vals = [r["elapsed"] for r in rows[c]]
            if drop_first:
                vals = vals[1:]
            if not vals:
                continue
            m, sd, lo, hi, spread = describe(vals)
            stats[c] = vals
            flag = "  <-- UNSTABLE, do not quote" if spread > SPREAD_ALARM else ""
            print(f"  {c}: {m:8.2f}s  sd {sd:5.2f}  "
                  f"[{lo:.2f}, {hi:.2f}]  spread {spread:.2f}x  "
                  f"n={len(vals)}{flag}", flush=True)
        if len(stats) == 2:
            a, b = stats["w1_c1"], stats["w4_c4"]
            ma, mb = statistics.mean(a), statistics.mean(b)
            print(f"  speedup {ma / mb:.3f}x  -> efficiency "
                  f"{ma / mb / 4 * 100:.1f}%", flush=True)
            print(f"  adversarial bound (min w1 / max w4): "
                  f"{min(a) / max(b):.3f}x = {min(a) / max(b) / 4 * 100:.1f}%",
                  flush=True)
            if len(a) > 1 and len(b) > 1:
                try:
                    from scipy import stats as st
                    p = st.ttest_ind(a, b, equal_var=False).pvalue
                    print(f"  Welch p={p:.2e}", flush=True)
                except ImportError:
                    pass

    temps = [r["start_temp"] for c in CONDITIONS for r in rows[c]]
    missed = [r for c in CONDITIONS for r in rows[c]
              if not r["cooldown_reached"]]
    print(f"\nstart temps: {min(temps):.0f}-{max(temps):.0f}C; "
          f"{len(missed)} run(s) started above target", flush=True)
    counts = {r["terms"] for c in CONDITIONS for r in rows[c]}
    print(f"term counts: {counts}", flush=True)
    if len(counts) > 1:
        print("*** CORRECTNESS FAILURE ***", flush=True)


if __name__ == "__main__":
    main()
