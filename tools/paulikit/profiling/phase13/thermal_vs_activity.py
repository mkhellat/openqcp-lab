"""Is the cold-start penalty about TEMPERATURE or about prior ACTIVITY?

THE CONFOUND. Every inflated run in `cold_first_run_results.jsonl`
started at 45-48C and every clean run started at exactly 55C. That
looks like a temperature effect but is useless as evidence: the
cooldown waited for <=55C, so an idle machine was already BELOW it and
ran immediately, while a machine that had just run cooled DOWN to 55C
and stopped there. "45-48C" means "idle"; "55C" means "just executed
something". Temperature and prior activity move together.

THE SEPARATION. Reach the SAME temperature by the SAME means in both
arms, and vary only whether paulikit ran beforehand:

  cold    - baseline temperature, no burn (low-temperature reference).
  heated  - external burn to the target, no extra paulikit run first.
  warm    - a real paulikit run, cooled back to baseline, then the
            SAME external burn to the same target.

`heated` and `warm` therefore start at an identical temperature
reached identically. The only difference is the extra paulikit
execution in `warm`.

  heated == warm -> prior activity is irrelevant; temperature explains
      it, and pre-heating IS a sufficient protocol.
  heated worse than warm -> temperature is not the whole story and
      prior execution matters (kernel state, page cache, memory
      fragmentation back on the table).

WHAT WAS WRONG BEFORE. An earlier version idled 600s before each arm
and cooled DOWN to the target between runs. Both were mistakes. The
long idles were never needed - what makes a machine "cold" here is
that paulikit has not run, not that ten minutes elapsed. And cooling
down to the target in one arm while burning up to it in another meant
the two reached the same number by opposite routes, with the cooling
arm taking its heat from the very run under test: the exact confound
this script exists to break. Cooling now goes to BASELINE only; the
target is always reached by the burn.

Per-core, not package: a package reading can sit at 60C while
individual cores differ by 14C (observed), leaving the cores a run
actually uses unmatched.

Usage:
    OPENBLAS_NUM_THREADS=1 python thermal_vs_activity.py [rounds]
"""

import json
import multiprocessing
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "thermal_vs_activity_results.jsonl")
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"

CONDITION = "w1_c1"      # the cell with the largest, clearest effect
WARM_REFERENCE = 23.35   # steady-state w1_c1 at N=150, n=5
TARGET_C = float(os.environ.get("PAULIKIT_TARGET_C", "60.0"))
# 55C, not 50C. After a burn the hottest core sat at 74C and 50C was
# simply not reachable while the machine was in use, so every cooldown
# ran its full 300s cap and gave up - 15 of round 0's ~20 minutes were
# spent waiting for a temperature that never arrived. 55C is reachable
# in ~30s and still sits clearly below the 60C target, which is all the
# design requires: the target must be reached by the BURN, not by
# cooling down to it.
BASELINE_C = float(os.environ.get("PAULIKIT_BASELINE_C", "55.0"))

CORETEMP_DIR = None
for _d in sorted(os.listdir("/sys/class/hwmon")):
    _p = f"/sys/class/hwmon/{_d}"
    try:
        if open(f"{_p}/name").read().strip() == "coretemp":
            CORETEMP_DIR = _p
            break
    except OSError:
        pass


def read_temp():
    with open(TEMP_PATH) as f:
        return int(f.read().strip()) / 1000.0


def read_core_temps():
    if CORETEMP_DIR is None:
        return None
    temps = {}
    for fn in sorted(os.listdir(CORETEMP_DIR)):
        if not fn.endswith("_label"):
            continue
        label = open(f"{CORETEMP_DIR}/{fn}").read().strip()
        if not label.startswith("Core "):
            continue
        base = fn[: -len("_label")]
        temps[label] = int(
            open(f"{CORETEMP_DIR}/{base}_input").read().strip()) / 1000.0
    return temps or None


def hottest_core():
    t = read_core_temps()
    return max(t.values()) if t else read_temp()


def coolest_core():
    t = read_core_temps()
    return min(t.values()) if t else read_temp()


def _burn(stop_at, duty_value):
    """Arithmetic at a duty cycle the parent adjusts live via shared
    memory. A FIXED duty cannot work here: 100% took this laptop
    50C -> 94C in one second, 35% never reached 60C in 120s, and 85%
    overshot to 74C inside a single poll interval. The target is only
    holdable with feedback."""
    x = 0.0
    while time.time() < stop_at:
        duty = max(0.02, min(1.0, duty_value.value))
        t0 = time.perf_counter()
        for _ in range(5000):
            x += 1.000001
        busy = time.perf_counter() - t0
        if duty < 1.0:
            time.sleep(busy * (1.0 - duty) / duty)
    return x


def heat_to(target_c, cap_s=180, hold_s=2.0, tol_c=6.0):
    """Drive every core to the target with a feedback loop, then HOLD.

    External heat only: touches no file the workload reads and imports
    nothing from paulikit, so it warms silicon and nothing else.

    Gates on the COOLEST core, so every core reaches the target, and
    backs the duty cycle off as the target is approached - an
    open-loop burn either undershoots or overshoots by 14C, and an
    overshoot makes the `heated` arm hotter than the `warm` arm it is
    supposed to match. Holds briefly at temperature so all four cores
    settle, then returns immediately (this CPU sheds heat within
    seconds, so any longer wait drops the run back below target).
    """
    start = time.perf_counter()
    duty_value = multiprocessing.Value("d", 0.9)
    procs = []
    for _ in range(max(1, multiprocessing.cpu_count() // 2)):
        p = multiprocessing.Process(
            target=_burn, args=(time.time() + cap_s, duty_value))
        p.start()
        procs.append(p)

    reached_at = None
    last_report = 0.0
    while time.perf_counter() - start < cap_s:
        coolest, hottest = coolest_core(), hottest_core()
        # Proportional back-off: full power while far away, gentle once
        # within 5C, and idle if any core runs past the target.
        if hottest > target_c + 1:
            duty_value.value = 0.02
        elif coolest >= target_c:
            duty_value.value = 0.1
        elif coolest > target_c - 5:
            duty_value.value = 0.35
        else:
            duty_value.value = 0.9

        # tol_c=6, not 3: the natural inter-core spread on this CPU is
        # 3-4C typically and spikes to 12C while idle (measured over 10
        # samples). A 3C window is NARROWER THAN THE HARDWARE'S OWN
        # VARIATION, so it was frequently unsatisfiable and the loop ran
        # its full cap without ever declaring success. The requirement
        # that matters is that every core is AT LEAST at the target;
        # the upper bound only guards against a gross overshoot.
        if coolest >= target_c and hottest <= target_c + tol_c:
            if reached_at is None:
                reached_at = time.perf_counter()
            elif time.perf_counter() - reached_at >= hold_s:
                break
        else:
            reached_at = None

        waited = time.perf_counter() - start
        if waited - last_report >= 15:
            last_report = waited
            print(f"  [{time.strftime('%H:%M:%S')}] heating: "
                  f"{coolest:.0f}-{hottest:.0f}C -> {target_c:.0f}C "
                  f"(duty {duty_value.value:.2f}, {waited:.0f}s)", flush=True)
        time.sleep(0.02)

    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=5)
    return reached_at is not None, time.perf_counter() - start


def cool_to_baseline(cap_s=180):
    """Down to BASELINE, never to the target - the target is always
    reached by the burn, so both arms get there the same way.

    Progress is printed every 15s. A cooldown that is not converging is
    the failure mode that silently ate round 0, so it must be VISIBLE
    while running, not diagnosed afterwards from a stalled log.
    """
    start = time.perf_counter()
    last_report = 0.0
    while hottest_core() > BASELINE_C:
        waited = time.perf_counter() - start
        if waited > cap_s:
            print(f"  [{time.strftime('%H:%M:%S')}] COOLDOWN GAVE UP after "
                  f"{waited:.0f}s at {hottest_core():.0f}C "
                  f"(target {BASELINE_C:.0f}C) - runs are NOT matched",
                  flush=True)
            return False
        if waited - last_report >= 15:
            last_report = waited
            print(f"  [{time.strftime('%H:%M:%S')}] cooling: hottest "
                  f"{hottest_core():.0f}C -> {BASELINE_C:.0f}C "
                  f"({waited:.0f}s)", flush=True)
        time.sleep(2)
    return True


def timed_run(arm, rnd, note):
    cores = read_core_temps()
    pkg = read_temp()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS="150")
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", CONDITION, "2"],
        capture_output=True, text=True, env=env,
    )
    out = dict(t.split("=", 1) for t in proc.stdout.split() if "=" in t)
    elapsed = float(out["elapsed"].rstrip("s"))
    infl = (elapsed - WARM_REFERENCE) / WARM_REFERENCE * 100
    rec = dict(round=rnd, arm=arm, elapsed=elapsed,
               inflation_pct=round(infl, 1), start_temp=pkg,
               core_temps=cores, note=note, terms=int(out["total_terms"]))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ct = "/".join(f"{v:.0f}" for v in cores.values()) if cores else "n/a"
    print(f"  [{time.strftime('%H:%M:%S')}] {arm:>6}: {elapsed:7.2f}s "
          f"({infl:+6.1f}%)  cores {ct}C   {note}", flush=True)
    return rec


def run_paulikit_untimed():
    subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", CONDITION, "2"],
        capture_output=True, text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1",
                 PAULIKIT_N_OSCILLATORS="150"),
    )


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Temperature or prior activity?  target {TARGET_C:.0f}C, "
          f"baseline {BASELINE_C:.0f}C, condition {CONDITION}", flush=True)
    print(f"warm reference {WARM_REFERENCE:.2f}s\n", flush=True)

    rows = []
    for rnd in range(rounds):
        print(f"--- round {rnd} ---", flush=True)

        # COLD: baseline temperature, no burn. In round 0 this is also
        # the session's first paulikit run; in later rounds it is not,
        # which is itself informative.
        cool_to_baseline()
        rows.append(timed_run("cold", rnd, "baseline temp, no burn"))

        # HEATED: erase that run's heat, then reach the target by burn.
        cool_to_baseline()
        reached, took = heat_to(TARGET_C)
        print(f"  [{time.strftime('%H:%M:%S')}] burned to "
              f"{coolest_core():.0f}C in {took:.0f}s "
              f"({'ok' if reached else 'TIMED OUT'})", flush=True)
        rows.append(timed_run("heated", rnd, "external burn to target"))

        # WARM: an extra real paulikit run, then baseline, then the SAME
        # burn. Identical temperature by an identical route; only the
        # extra prior execution differs from `heated`. No cooldown
        # before the untimed run - it is not measured, so its starting
        # temperature is irrelevant; only the cooldown AFTER it matters,
        # since that is what puts the timed run on the same footing as
        # `heated`.
        run_paulikit_untimed()
        cool_to_baseline()
        heat_to(TARGET_C)
        rows.append(timed_run("warm", rnd, "paulikit ran, then same burn"))

    print("\n" + "=" * 72, flush=True)
    import statistics
    means = {}
    for arm in ("cold", "heated", "warm"):
        v = [r["inflation_pct"] for r in rows if r["arm"] == arm]
        if not v:
            continue
        means[arm] = statistics.mean(v)
        print(f"  {arm:>6}: {means[arm]:+6.1f}% mean inflation  n={len(v)}",
              flush=True)

    print(flush=True)
    if "heated" in means and "warm" in means:
        gap = means["heated"] - means["warm"]
        if abs(gap) < 5:
            print("heated == warm at the SAME temperature reached the SAME "
                  "way", flush=True)
            print("-> prior activity is irrelevant; temperature explains it, "
                  "and", flush=True)
            print("   pre-heating is a sufficient protocol.", flush=True)
        else:
            print(f"heated is {gap:+.1f} points worse than warm at the SAME "
                  "temperature", flush=True)
            print("-> temperature is NOT the whole story; prior execution "
                  "matters.", flush=True)
    print(f"\nterm counts: {({r['terms'] for r in rows})}", flush=True)


if __name__ == "__main__":
    main()
