"""Task 7 (final task, array-yielding-parallel-decompose plan):
measures whether the SHIPPED parallel_decompose_arrays actually
delivers the 2.191x multi-core scaling that a SYNTHETIC arrays-only
drain loop (no checkpointing, no real chunking/auto-tuning/pinning
machinery - drain_gil_backpressure_results.jsonl) showed in a
controlled single-variable experiment.

THE QUESTION. That earlier number was a proof that removing
dict-building from the drain loop *can* restore scaling. It said
nothing about whether the actual shipped API, wired into the real
pipeline - real chunk auto-tuning, real CPU pinning, and (for callers
who want resumability) real per-chunk checkpoint I/O written by the
PARENT process - delivers the same number. Per-chunk checkpoint
writing is parent-side I/O on the same single-threaded path that the
2.191x result depended on being empty; it is the most plausible thing
that would erode the number.

DESIGN. 3x2 cells, 5 reps each, N=150, chunk_size=2:
  variant  in {dict, arrays_no_checkpoint, arrays_with_checkpoint}
  condition in {w2_c1, w8_c4}
  dict                    -> parallel_decompose (existing API)
  arrays_no_checkpoint    -> parallel_decompose_arrays, checkpoint_path=None
  arrays_with_checkpoint  -> parallel_decompose_arrays, checkpoint_path=<fresh file per rep>

Within each variant, the w2_c1 -> w8_c4 speedup is the scaling number
to compare against 2.191x. The dict cell reproduces (approximately)
the well-established ~1.21x cap as a sanity check that this harness
measures what the rest of the investigation has always measured.
Comparing arrays_no_checkpoint against arrays_with_checkpoint at the
SAME condition isolates the checkpoint I/O cost directly.

CORRECTNESS GATE. Every run must report total_terms=91652096 (N=150's
known exact term count). A different count on ANY row means the
arrays path lost or duplicated terms - a correctness bug, and this
script deliberately keeps every row (dict and arrays alike) so that
check runs over the whole file, not just the new API's rows.

Protocol matches every other measurement in this investigation:
foreground only, thermal cooldown to 55C before every run, N reps,
Welch's t-test, incremental JSON-Lines with resume-skip - modeled
directly on narrowed_index_dtype_sweep.py.

Usage:
    OPENBLAS_NUM_THREADS=1 python arrays_vs_dict_sweep.py
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TARGET = os.path.join(HERE, "arrays_vs_dict_target.py")
RESULTS = os.path.join(HERE, "arrays_vs_dict_results.jsonl")

VARIANTS = (
    "dict",
    "arrays_no_checkpoint",
    "arrays_with_checkpoint",
    # Added after the first sweep measured arrays_with_checkpoint at
    # ~298s versus ~18s without, to ask whether that cost is INHERITED
    # from the shared, unmodified _append_parallel_checkpoint_chunk or
    # INTRODUCED by the array path.
    #
    # SUPERSEDED 2026-09-08 - this cell is no longer needed and is kept
    # only so the sweep can still reproduce it on request.
    # checkpoint_cost_attribution.py answered the question far more
    # cheaply by falsifying each candidate cause directly at small N:
    # 94.8% of the writer's cost is GIL-held per-term serialization
    # (.tolist() + dict literal + json.dumps), only 1.1% is disk I/O.
    # That cost lives entirely inside the shared writer both paths
    # call, so it is PRE-EXISTING. Running this 10-rep, ~50-minute cell
    # at N=150 would only re-confirm that at much greater expense.
    "dict_with_checkpoint",
)
CONDITIONS = ("w2_c1", "w8_c4")
CHUNK_SIZE = "2"
REPS = int(os.environ.get("REPS", "5"))
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
EXPECTED_TERMS = 91652096


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
                done.add((r["variant"], r["condition"], r["rep"]))
    return done


def run_one(variant, condition, rep):
    settled = cooldown()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    args = [PYTHON, TARGET, variant, condition, CHUNK_SIZE]
    ckpt_dir = None
    if variant in ("arrays_with_checkpoint", "dict_with_checkpoint"):
        ckpt_dir = tempfile.mkdtemp(prefix=f"{variant}_")
        args.append(os.path.join(ckpt_dir, "checkpoint.jsonl"))
    t_before = read_temp()
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, env=env,
        )
    finally:
        if ckpt_dir is not None:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
    t_after = read_temp()
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr.strip()[:300]}")
        return
    elapsed = terms = peak_rss = None
    for tok in proc.stdout.split():
        if tok.startswith("elapsed="):
            elapsed = float(tok.split("=", 1)[1].rstrip("s"))
        elif tok.startswith("total_terms="):
            terms = int(tok.split("=", 1)[1])
        elif tok.startswith("peak_rss_mib="):
            peak_rss = float(tok.split("=", 1)[1])
    if elapsed is None:
        print(f"  no elapsed: {proc.stdout.strip()[:200]}")
        return
    rec = dict(
        variant=variant, condition=condition, rep=rep, elapsed=elapsed,
        total_terms=terms, peak_rss_mib=peak_rss, settled_temp=settled,
        temp_before=t_before, temp_after=t_after,
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {variant:>22}/{condition:>6} rep{rep}: {elapsed:7.3f}s "
          f"rss={peak_rss:.1f}MiB terms={terms}")


def main():
    done = load_completed()
    for variant in VARIANTS:
        for condition in CONDITIONS:
            for rep in range(REPS):
                if (variant, condition, rep) in done:
                    print(f"  {variant:>22}/{condition:>6} rep{rep}: skip (done)")
                    continue
                run_one(variant, condition, rep)

    if not os.path.exists(RESULTS):
        print("no results written")
        return
    rows = [json.loads(l) for l in open(RESULTS)]

    print("\n" + "=" * 78)
    print(f"{'variant':>22} {'w2_c1':>9} {'w8_c4':>9} {'speedup':>9} "
          f"{'p-value':>10} {'n2':>3} {'n8':>3}")
    for v in VARIANTS:
        v2 = [r["elapsed"] for r in rows if r["variant"] == v and r["condition"] == "w2_c1"]
        v8 = [r["elapsed"] for r in rows if r["variant"] == v and r["condition"] == "w8_c4"]
        if not v2 or not v8:
            print(f"{v:>22} incomplete (n2={len(v2)} n8={len(v8)})")
            continue
        m2 = statistics.mean(v2)
        m8 = statistics.mean(v8)
        speedup = m2 / m8
        if len(v2) > 1 and len(v8) > 1:
            _, pval = stats.ttest_ind(v2, v8, equal_var=False)
        else:
            pval = float("nan")
        print(f"{v:>22} {m2:>8.3f}s {m8:>8.3f}s {speedup:>8.3f}x "
              f"{pval:>10.2e} {len(v2):>3} {len(v8):>3}")

    print("\nPeak RSS (MiB), mean by variant/condition:")
    for v in VARIANTS:
        for c in CONDITIONS:
            vals = [r["peak_rss_mib"] for r in rows
                    if r["variant"] == v and r["condition"] == c
                    and r.get("peak_rss_mib") is not None]
            if vals:
                print(f"  {v:>22}/{c:>6}: {statistics.mean(vals):>8.1f} MiB (n={len(vals)})")

    print("\nCheckpoint cost isolation (arrays_no_checkpoint vs "
          "arrays_with_checkpoint, same condition):")
    for c in CONDITIONS:
        nck = [r["elapsed"] for r in rows
               if r["variant"] == "arrays_no_checkpoint" and r["condition"] == c]
        wck = [r["elapsed"] for r in rows
               if r["variant"] == "arrays_with_checkpoint" and r["condition"] == c]
        if nck and wck:
            m_nck, m_wck = statistics.mean(nck), statistics.mean(wck)
            if len(nck) > 1 and len(wck) > 1:
                _, pval = stats.ttest_ind(nck, wck, equal_var=False)
            else:
                pval = float("nan")
            print(f"  {c:>6}: no_ckpt={m_nck:.3f}s  with_ckpt={m_wck:.3f}s  "
                  f"delta={m_wck - m_nck:+.3f}s ({(m_wck/m_nck - 1)*100:+.1f}%)  p={pval:.2e}")

    counts = {r["total_terms"] for r in rows if r["total_terms"] is not None}
    print(f"\nterm counts across ALL runs (dict + arrays, all conditions): {counts}")
    print(f"expected: {{{EXPECTED_TERMS}}}")
    if counts != {EXPECTED_TERMS}:
        print("*** CORRECTNESS FAILURE: term count mismatch. Do not trust "
              "any timing above until this is root-caused. ***")


if __name__ == "__main__":
    main()
