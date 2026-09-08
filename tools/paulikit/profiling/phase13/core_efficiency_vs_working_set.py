"""Does 4-physical-core efficiency track the per-core working set?

THE HYPOTHESIS. Measured on one worker per PHYSICAL core (w1_c1,
w2_c2, w4_c4, so no hyperthread siblings anywhere), parallel
efficiency at 4 cores fell from 88% at N=150 to 73% at N=180. The
proposed explanation was cache: N=180 doubles dim to 32768, so the
working set per worker grows and cores contend for the shared 8 MiB
L3 instead of being served by their private 1 MiB L2s.

WHY THAT EXPLANATION IS ALREADY IN DOUBT. The auto-tuner halves
chunk_size to 1 at N=180 precisely to keep the per-chunk buffer at
chunk_size*dim*16 = 512 KiB - the SAME per-worker buffer as N=150 at
chunk_size=2. If efficiency were a function of that buffer alone, the
two would match. They do not (88% vs 73%). So either the relevant
working set is something other than the chunk buffer, or the cause is
not cache at all.

THE TEST. Hold N fixed and sweep chunk_size, which is the only knob
that changes the per-worker buffer without changing the problem. For
each chunk_size, measure w1_c1 and w4_c4 and compute efficiency
(speedup / 4).

  If the hypothesis holds, efficiency falls monotonically as the
  4-worker aggregate buffer crosses L3 (8 MiB): at N=180 that is
  chunk_size=4, so chunk_size 1 and 2 should look healthy and 4 and 8
  should degrade.

  If efficiency is flat across chunk_size, the working set is NOT what
  governs it, and the N=150 -> N=180 drop must be explained by
  something else (per-term work, IPC volume, chunk count, thermals).

Falsification, not confirmation: a flat row refutes the hypothesis
outright, which is the outcome this script is designed to be able to
produce.

Cooldown to 55C before every timed run, real disk, foreground only.

Usage:
    OPENBLAS_NUM_THREADS=1 python core_efficiency_vs_working_set.py [N]
"""

import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")

COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 240
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
L3_BYTES = 8 * 1024 * 1024


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


def run(n_osc, condition, chunk_size):
    cooldown()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               PAULIKIT_N_OSCILLATORS=str(n_osc))
    proc = subprocess.run(
        [PYTHON, TARGET, "arrays_no_checkpoint", condition, str(chunk_size)],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        print(f"    FAILED: {proc.stderr.strip()[-200:]}")
        return None
    out = dict(
        tok.split("=", 1) for tok in proc.stdout.split() if "=" in tok
    )
    return float(out["elapsed"].rstrip("s")), int(out["total_terms"])


def main():
    n_osc = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    dim = 32768 if n_osc == 180 else 16384
    # Which chunk sizes to run this invocation. The full sweep exceeds
    # a single foreground window once cooldowns are counted, so it is
    # run in pieces rather than backgrounded - background timing jobs
    # are exactly how duplicate runs and mis-attributed results happen.
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    sizes = [int(c) for c in only] if only else [1, 2, 4, 8]

    print(f"N={n_osc} (dim={dim}), one worker per PHYSICAL core")
    print(f"L2 1 MiB/core, L3 {L3_BYTES // 1024 // 1024} MiB shared\n")
    print(f"{'chunk':>6} {'buf/worker':>11} {'4x buf':>9} {'w1_c1':>9} "
          f"{'w4_c4':>9} {'speedup':>8} {'efficiency':>11}")

    terms_seen = set()
    for cs in sizes:
        buf = cs * dim * 16
        r1 = run(n_osc, "w1_c1", cs)
        r4 = run(n_osc, "w4_c4", cs)
        if r1 is None or r4 is None:
            continue
        t1, n1 = r1
        t4, n4 = r4
        terms_seen.update({n1, n4})
        sp = t1 / t4
        print(f"{cs:>6} {buf / 1024:>8.0f} KiB {4 * buf / 1024:>6.0f} KiB "
              f"{t1:>8.2f}s {t4:>8.2f}s {sp:>7.3f}x {sp / 4 * 100:>10.1f}%"
              + ("  <- 4x buf exceeds L3" if 4 * buf > L3_BYTES else ""))

    print(f"\nterm counts across all runs: {terms_seen}")
    if len(terms_seen) > 1:
        print("*** CORRECTNESS FAILURE: runs disagree on the term count ***")


if __name__ == "__main__":
    main()
