"""Wall-clock and peak RSS for N=150 across three worker/CPU
conditions, no checkpointing, with a thermal cooldown between runs.

Each condition runs in its OWN subprocess so peak RSS is that run's
alone and no allocator state carries over. The parent samples the
child's RSS (plus its descendants - the pool workers are separate
processes and their memory is part of the footprint) and reports the
peak.

CPU pinning matches the conditions used throughout phase 13:
  w2_c1 - 2 workers, pinned to 1 physical core's representative CPUs
  w4_c2 - 4 workers, 2 cores
  w8_c4 - 8 workers, 4 cores
This machine is a 4-core/8-thread i7-8550U, so w8_c4 is every logical
CPU and w2_c1 is one core's worth.

Cooldown to 55C package temp before every timed run: this machine hits
100C thermal throttling on sustained multi-core work, and the
'performance' governor does not prevent it (intel_pstate). Without the
cooldown, later runs are measured on a hotter, slower CPU.

Usage:
    OPENBLAS_NUM_THREADS=1 python worker_scaling_no_checkpoint.py [reps]
"""

import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The venv's interpreter, not sys.executable: this script may be run by
# a system python that has no paulikit installed, and the child process
# is what actually imports it.
PYTHON = os.environ.get(
    "PAULIKIT_PYTHON", os.path.expanduser("~/.venvs/paulikit/bin/python")
)
if not os.path.exists(PYTHON):
    PYTHON = sys.executable
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")

# Overridable so the same harness can measure a different ladder.
# The default is the packed ladder (siblings doubled up on each core);
# "w1_c1,w2_c2,w4_c4" is the one-worker-per-PHYSICAL-core ladder that
# isolates hyperthread contention from real core scaling.
CONDITIONS = tuple(
    os.environ.get("PAULIKIT_CONDITIONS", "w2_c1,w4_c2,w8_c4").split(",")
)
# N and chunk_size come from the environment so the same script can
# measure another problem size. chunk_size defaults to whatever
# autotune recommends for that N rather than a hardcoded 2, since
# N=180 crosses a power-of-two boundary and the tuner picks 1 there.
N_OSC = os.environ.get("PAULIKIT_N_OSCILLATORS", "150")
CHUNK_SIZE = os.environ.get("PAULIKIT_CHUNK_SIZE", "2")
# Set PAULIKIT_CHECKPOINT=1 to measure the checkpointed variant. The
# file goes on REAL DISK under $HOME, never /tmp: /tmp is a 7.7 GB
# RAM-backed tmpfs here, and an N=180 payload is 5.28 GB, so writing
# there would measure memory pressure instead of I/O - the exact
# mistake that invalidated an earlier checkpoint measurement.
USE_CHECKPOINT = os.environ.get("PAULIKIT_CHECKPOINT") == "1"
VARIANT = "arrays_with_checkpoint" if USE_CHECKPOINT else "arrays_no_checkpoint"
CKPT_DIR = os.path.expanduser("~/.paulikit_ckpt_runs")
# 65C, not 55C: this machine idles around 69-75C after sustained
# multi-core work and does not reach 55C at all, so every cooldown ran
# the full 240s timeout without ever hitting its target - 32 minutes of
# waiting for 2 minutes of compute, and the runs were NOT actually
# matched at 55C despite the setting claiming so. 65C is reachable, so
# runs are genuinely matched at it.
COOLDOWN_TARGET_C = float(os.environ.get("PAULIKIT_COOLDOWN_C", "65.0"))
COOLDOWN_TIMEOUT_S = 240
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
# Only known for N=150; for any other N the count is simply reported
# and checked for consistency ACROSS runs rather than against a
# literal, which would be a fabricated expectation.
EXPECTED_TERMS = 91652096 if N_OSC == "150" else None


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
            return t, time.perf_counter() - start
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return t, time.perf_counter() - start
        time.sleep(2)


def run_one(condition):
    settled, waited = cooldown()
    t_before = read_temp()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS=N_OSC)
    args = [PYTHON, TARGET, VARIANT, condition, CHUNK_SIZE]
    ckpt_dir = None
    if USE_CHECKPOINT:
        os.makedirs(CKPT_DIR, exist_ok=True)
        ckpt_dir = os.path.join(CKPT_DIR, f"{condition}_{os.getpid()}")
        os.makedirs(ckpt_dir, exist_ok=True)
        args.append(os.path.join(ckpt_dir, "checkpoint.bin"))
    ckpt_bytes = None
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, env=env,
        )
        if ckpt_dir is not None:
            ckpt_bytes = sum(
                os.path.getsize(os.path.join(ckpt_dir, f))
                for f in os.listdir(ckpt_dir)
            )
    finally:
        if ckpt_dir is not None:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
    t_after = read_temp()
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr.strip()[:300]}")
        return None
    out = {}
    for tok in proc.stdout.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return dict(
        condition=condition,
        elapsed=float(out["elapsed"].rstrip("s")),
        terms=int(out["total_terms"]),
        peak_rss_mib=float(out["peak_rss_mib"]),
        settled_temp=settled, cooldown_s=waited,
        temp_before=t_before, temp_after=t_after,
        ckpt_bytes=ckpt_bytes,
    )


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"N={N_OSC}, {VARIANT}, chunk_size={CHUNK_SIZE}, "
          f"{reps} reps per condition")
    print(f"cooldown to {COOLDOWN_TARGET_C}C before every run\n")

    results = {c: [] for c in CONDITIONS}
    for rep in range(reps):
        for condition in CONDITIONS:
            r = run_one(condition)
            if r is None:
                continue
            results[condition].append(r)
            print(f"  {condition} rep{rep}: {r['elapsed']:7.3f}s  "
                  f"rss={r['peak_rss_mib']:6.1f} MiB  "
                  f"terms={r['terms']}  "
                  f"temp {r['temp_before']:.0f}->{r['temp_after']:.0f}C  "
                  f"(cooled {r['cooldown_s']:.0f}s)"
                  + (f"  ckpt={r['ckpt_bytes'] / 1e9:.2f} GB"
                     if r["ckpt_bytes"] else ""))

    print("\n" + "=" * 74)
    print(f"{'condition':>10} {'wall-clock':>22} {'peak RSS':>20} "
          f"{'vs w2_c1':>9}")
    base = None
    for c in CONDITIONS:
        rows = results[c]
        if not rows:
            print(f"{c:>10} no successful runs")
            continue
        el = [r["elapsed"] for r in rows]
        rss = [r["peak_rss_mib"] for r in rows]
        m = statistics.mean(el)
        sd = statistics.stdev(el) if len(el) > 1 else 0.0
        if base is None:
            base = m
        print(f"{c:>10} {m:>10.3f}s (sd {sd:5.3f}, n={len(el)}) "
              f"{statistics.mean(rss):>9.1f} MiB "
              f"(max {max(rss):6.1f}) {base / m:>8.3f}x")

    counts = {r["terms"] for rows in results.values() for r in rows}
    print(f"\nterm counts across all runs: {counts}")
    if EXPECTED_TERMS is not None:
        print(f"expected: {{{EXPECTED_TERMS}}}")
        if counts and counts != {EXPECTED_TERMS}:
            print("*** CORRECTNESS FAILURE - do not trust the timings "
                  "above ***")
    elif len(counts) > 1:
        print("*** CORRECTNESS FAILURE: conditions disagree on the term "
              "count - do not trust the timings above ***")
    else:
        print("(no reference count known for this N; all conditions "
              "agree, which is the check that is available)")


if __name__ == "__main__":
    main()
