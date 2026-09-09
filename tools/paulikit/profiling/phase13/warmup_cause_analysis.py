"""What warms up between the first run of a session and the next?

THE EFFECT. `w1_c1` at N=150 takes ~30s on the first run after the
machine has been quiet and ~24s on every run after
(`w1_baseline_warmup_findings.md`). Each measurement is a separate
subprocess, so nothing carries over inside the process. Already
falsified: the preceding condition (four consecutive w1_c1 runs stay
fast) and idle gaps (120s idle costs ~2.5s, not 6s). The core is also
exonerated - the effect follows the first run, not the CPU.

CANDIDATE CAUSES, each removed independently.

  C1  CPU FREQUENCY RAMP. The governor is `powersave` on
      `intel_pstate`, and the CPUs idle at 800 MHz against a 4000 MHz
      maximum - a 5x ramp. A cold run starts slow and climbs; a run
      following recent work starts already boosted.
      REMOVED BY: a busy-loop on the target CPU immediately before the
      timed run, which raises the frequency without touching page
      cache or any file the workload reads.

  C2  PAGE CACHE for the venv's compiled extension modules and the
      Python/NumPy/SciPy libraries the process imports.
      REMOVED BY: reading every .so the process maps into page cache
      before the timed run, without doing any CPU work.

  C3  IMPORT/BUILD WORK inside the target - meson-python's editable
      install re-runs ninja on import, and the Hamiltonian is built
      before the timer starts. If this were the cause the reported
      `elapsed` would be unaffected, since it excludes setup.
      TESTED BY: comparing reported `elapsed` against the driver's own
      wall measurement of the whole subprocess.

DESIGN. Each arm gets a genuinely cold start - the machine is left
quiet, then exactly ONE timed run happens under that arm's treatment.
Repeating within an arm would warm the machine and destroy the very
condition under test, so arms are compared across separate cold
starts, several rounds each, and the round order is rotated so a
monotone drift cannot align with one arm.

  cold        - nothing before the run (the control; should be slow)
  freq        - busy-loop first, removing C1 only
  cache       - read the .so files first, removing C2 only
  both        - both treatments (should look like a warm run)

If `freq` alone recovers the fast time, C1 is the cause. If `cache`
alone does, C2 is. If neither does but `both` does, they interact. If
none recover it, all three candidates are wrong.

Usage:
    OPENBLAS_NUM_THREADS=1 python warmup_cause_analysis.py [rounds]
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "warmup_cause_results.jsonl")

# Long enough that the governor has certainly ramped, short enough not
# to heat the machine into throttling.
FREQ_WARM_SECONDS = 6.0
# Quiet period before each arm, so every arm starts genuinely cold.
QUIET_SECONDS = float(os.environ.get("PAULIKIT_QUIET_S", "90"))
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
CPU = 2  # a core with low interrupt load; the effect is core-independent


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def clock_proxy_ms():
    """A fixed CPU workload, timed. `scaling_cur_freq` is unreliable on
    intel_pstate - read back to back it reported 3000 MHz at idle and
    400 MHz right after a busy loop, i.e. backwards. This measures the
    clock by its EFFECT instead: the same arithmetic loop takes ~124ms
    cold and ~110ms warm on this machine, a 13% ramp.
    """
    t = time.perf_counter()
    x = 0.0
    for _ in range(3_000_000):
        x += 1.000001
    return round((time.perf_counter() - t) * 1000, 1)


def warm_frequency():
    """Removes C1: burn CPU on the target core to raise the clock.

    Touches no file the workload reads, so page cache is unaffected.
    """
    os.sched_setaffinity(0, {CPU})
    end = time.perf_counter() + FREQ_WARM_SECONDS
    x = 0.0
    while time.perf_counter() < end:
        for _ in range(100000):
            x += 1.000001
    os.sched_setaffinity(0, set(range(os.cpu_count())))
    return x


def warm_page_cache():
    """Removes C2: pull every shared object the venv would map into
    page cache, using almost no CPU."""
    n = 0
    for root, _dirs, files in os.walk(
            os.path.expanduser("~/.venvs/paulikit/lib")):
        for fn in files:
            if fn.endswith(".so") or ".so." in fn:
                try:
                    with open(os.path.join(root, fn), "rb") as f:
                        while f.read(1 << 20):
                            pass
                    n += 1
                except OSError:
                    pass
    return n


def timed_run(arm):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS="150", PAULIKIT_FORCE_CPU=str(CPU))
    # The clock proxy is measured AFTER the timed run, never before:
    # it burns CPU, so running it first would partially warm the
    # frequency in every arm and contaminate the `cold` control - the
    # exact variable under test. Measured after, it reports the state
    # the machine ENDED in, which still distinguishes the arms.
    t_start = read_temp()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", "w1_c1", "2"],
        capture_output=True, text=True, env=env,
    )
    wall = time.perf_counter() - t0
    f_start = clock_proxy_ms()
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    return dict(
        arm=arm, elapsed=float(out["elapsed"].rstrip("s")),
        subprocess_wall=round(wall, 3),
        setup_overhead=round(wall - float(out["elapsed"].rstrip("s")), 3),
        terms=int(out["total_terms"]),
        start_temp=t_start, clock_proxy_ms=f_start,
    )


ARMS = ("cold", "freq", "cache", "both")


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"N=150, one worker on cpu{CPU}, {rounds} round(s) per arm",
          flush=True)
    print(f"each arm preceded by {QUIET_SECONDS:.0f}s quiet so it starts "
          f"genuinely cold\n", flush=True)

    rows = []
    for rnd in range(rounds):
        # Rotate arm order per round so drift cannot align with an arm.
        order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
        for arm in order:
            print(f"  [{time.strftime('%H:%M:%S')}] quiet "
                  f"{QUIET_SECONDS:.0f}s before '{arm}' ...", flush=True)
            time.sleep(QUIET_SECONDS)
            pre = time.perf_counter()
            if arm in ("freq", "both"):
                warm_frequency()
            if arm in ("cache", "both"):
                warm_page_cache()
            prep = time.perf_counter() - pre
            r = timed_run(arm)
            r["round"] = rnd
            r["prep_s"] = round(prep, 2)
            rows.append(r)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  [{time.strftime('%H:%M:%S')}] {arm:>5}: "
                  f"{r['elapsed']:6.2f}s  setup={r['setup_overhead']:5.2f}s  "
                  f"clock={r['clock_proxy_ms']:.0f}ms  "
                  f"{r['start_temp']:.0f}C", flush=True)

    print("\n" + "=" * 68, flush=True)
    import statistics
    print(f"{'arm':>7} {'mean elapsed':>14} {'runs':>5} "
          f"{'mean setup':>12} {'clock after':>12}", flush=True)
    for arm in ARMS:
        v = [r["elapsed"] for r in rows if r["arm"] == arm]
        if not v:
            continue
        s = statistics.mean([r["setup_overhead"] for r in rows
                             if r["arm"] == arm])
        fq = statistics.mean([r["clock_proxy_ms"] for r in rows
                              if r["arm"] == arm])
        print(f"{arm:>7} {statistics.mean(v):>13.2f}s {len(v):>5} "
              f"{s:>11.2f}s {fq:>10.0f}ms", flush=True)
    print(f"\nterm counts: {({r['terms'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
