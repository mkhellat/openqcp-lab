"""Sizes the progress-marker write against the frame write, now that
the binary chunk-framed format made the frame ~500x cheaper.

THE QUESTION. The Phase 13 redesign replaced the JSONL payload writer
with binary frames (511x cheaper at 500K terms), but left the PROGRESS
MARKER untouched: `_append_parallel_checkpoint_chunk` still does

    json.dump({"completed_chunk_indices": sorted(completed)}, f)

on EVERY completed chunk, rewriting the whole set each time. That is
O(n) work per chunk and so O(n^2) total bytes written across a run.
When the frame write cost ~4.2 us/term this was noise; at ~0.009
us/term it may now dominate. Before changing checkpoint granularity
(the open question left by the redesign), size which half actually
costs anything - changing granularity is the wrong lever if the marker
is the cost.

REAL SCALE. N=150 is n_qubits=14, dim=16384, n_active=11,189 distinct
x values -> 5,595 chunks at the sweep's chunk_size=2. That count, not
a round number, is what the marker is rewritten this many times for.

WHAT IS MEASURED. Both halves of one chunk's checkpoint write, at a
realistic per-chunk term count, swept over how many chunks have
already completed (the marker's cost depends on the SET SIZE, the
frame's does not):

  frame_only   - _append_checkpoint_frame alone.
  marker_only  - the sorted() + json.dump of a completed-set of size k.
  append       - the append-only 8-byte record that replaced the
                 rewrite (the fourth arm, added after the change).

Sweeping k reveals the growth: a flat marker cost would mean the O(n^2)
concern is theoretical, a rising one confirms it.

MEASUREMENT INTEGRITY. The append arm's record file is built ONCE per
k, outside the timed region, and each rep times a single 8-byte append
onto it. Rebuilding the k-record file per rep dirties page cache in
proportion to k immediately before the timed write, and that harness
artifact alone produced a false "rising, O(1) falsified" verdict during
design. Keep every per-rep setup out of the timer.

Writes to real disk under $HOME, never /tmp (a RAM-backed tmpfs here;
writing into it measures memory exhaustion, not I/O).

Usage:
    python progress_marker_cost.py [reps]
"""

import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from paulikit.algorithms.fwht import (  # noqa: E402
    _PROGRESS_RECORD,
    _append_checkpoint_frame,
    _append_progress_record,
    _parallel_checkpoint_progress_path,
)

DISK_DIR = os.path.expanduser("~/.paulikit_marker_bench")

# N=150 at chunk_size=2, measured: n_active=11,189 -> 5,595 chunks.
N150_CHUNKS = 5595
# 91,652,096 surviving terms spread over those chunks.
N150_TERMS_PER_CHUNK = 91_652_096 // N150_CHUNKS


def time_frame(path, x, z, coeff, idx_dtype, _completed):
    t0 = time.perf_counter()
    _append_checkpoint_frame(path, 0, x, z, coeff, idx_dtype)
    return time.perf_counter() - t0


def time_marker(path, _x, _z, _coeff, _idx_dtype, completed):
    progress_path = _parallel_checkpoint_progress_path(path)
    t0 = time.perf_counter()
    with open(progress_path, "w") as f:
        json.dump({"completed_chunk_indices": sorted(completed)}, f)
    return time.perf_counter() - t0


def time_marker_append(path, _x, _z, _coeff, _idx_dtype, completed):
    """The append-only marker that replaced the rewrite.

    The record file it appends onto is built by the caller ONCE per k,
    outside this function and outside the timed region.
    """
    progress_path = _parallel_checkpoint_progress_path(path)
    t0 = time.perf_counter()
    _append_progress_record(progress_path, len(completed))
    return time.perf_counter() - t0


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    os.makedirs(DISK_DIR, exist_ok=True)

    n_terms = N150_TERMS_PER_CHUNK
    rng = np.random.default_rng(0)
    idx_dtype = np.dtype(np.uint16)  # dim=16384 -> uint16
    x = rng.integers(0, 16384, n_terms).astype(idx_dtype)
    z = rng.integers(0, 16384, n_terms).astype(idx_dtype)
    coeff = rng.standard_normal(n_terms).astype(complex)

    print(f"N=150 shape: {N150_CHUNKS:,} chunks, "
          f"~{n_terms:,} terms/chunk, idx_dtype={idx_dtype}")
    print(f"reps={reps}  disk={DISK_DIR}\n")

    print(f"{'completed set size':>19} {'frame':>10} {'marker':>10} "
          f"{'append':>10} {'marker/frame':>13} {'marker/append':>14} "
          f"{'marker bytes':>13}")

    frame_times = []
    append_times = []
    rows = []
    for k in (1, 100, 1000, N150_CHUNKS // 2, N150_CHUNKS):
        completed = set(range(k))
        path = os.path.join(DISK_DIR, f"c{k}.bin")
        # The append arm gets its own checkpoint path so the rewrite
        # arm's JSON never lands on the record file being appended to.
        apath = os.path.join(DISK_DIR, f"a{k}.bin")
        paths = (path, str(_parallel_checkpoint_progress_path(path)),
                 apath, str(_parallel_checkpoint_progress_path(apath)))
        for p in paths:
            if os.path.exists(p):
                os.unlink(p)

        # SETUP, OUTSIDE THE TIMED REGION: build the k-record file once.
        # Rebuilding it per rep would dirty page cache in proportion to
        # k right before the timed write and fake an O(k) append.
        with open(_parallel_checkpoint_progress_path(apath), "wb") as f:
            f.write(b"".join(_PROGRESS_RECORD.pack(i) for i in range(k)))

        ft, mt, at = [], [], []
        for _ in range(reps):
            if os.path.exists(path):
                os.unlink(path)
            ft.append(time_frame(path, x, z, coeff, idx_dtype, completed))
            mt.append(time_marker(path, x, z, coeff, idx_dtype, completed))
            at.append(time_marker_append(
                apath, x, z, coeff, idx_dtype, completed))
        mf, mm = statistics.mean(ft), statistics.mean(mt)
        ma = statistics.mean(at)
        marker_bytes = os.path.getsize(_parallel_checkpoint_progress_path(path))
        frame_times.append(mf)
        append_times.append(ma)
        rows.append((k, mf, mm, marker_bytes, ma))
        print(f"{k:>19,} {mf * 1e3:>9.3f}ms {mm * 1e3:>9.3f}ms "
              f"{ma * 1e3:>9.3f}ms {mm / mf:>12.2f}x {mm / ma:>13.1f}x "
              f"{marker_bytes:>12,}")

        for p in paths:
            if os.path.exists(p):
                os.unlink(p)

    print("\n" + "=" * 72)
    print("Extrapolated cost of a FULL N=150 run (5,595 chunks):")
    # Frame cost is flat in k, so one mean covers every chunk.
    frame_total = statistics.mean(frame_times) * N150_CHUNKS
    # The marker is rewritten once per chunk with a set that grows
    # 1..N, so the run's total is the integral, not N * the final cost.
    # Trapezoid over the measured points, in chunk-index space.
    pts = sorted((k, m) for k, _f, m, _b, _a in rows)
    marker_total = 0.0
    for (k0, m0), (k1, m1) in zip(pts, pts[1:]):
        marker_total += (m0 + m1) / 2 * (k1 - k0)
    # The append is flat in k, so one mean covers every chunk.
    append_total = statistics.mean(append_times) * N150_CHUNKS
    print(f"  frames : {frame_total:8.2f}s  "
          f"({statistics.mean(frame_times) * 1e3:.3f}ms x {N150_CHUNKS:,})")
    print(f"  markers: {marker_total:8.2f}s  (trapezoid over growing set)")
    print(f"  appends: {append_total:8.2f}s  "
          f"({statistics.mean(append_times) * 1e6:.1f}us x {N150_CHUNKS:,})")
    total = frame_total + marker_total
    if total > 0:
        print(f"  marker share of OLD checkpoint cost: "
              f"{marker_total / total * 100:.1f}%")
    new_total = frame_total + append_total
    print(f"  checkpoint total: {total:.2f}s -> {new_total:.2f}s "
          f"({(1 - new_total / total) * 100:.1f}% cheaper)")

    lo, hi = min(append_times), max(append_times)
    print(f"\n  append flatness over k=1..{N150_CHUNKS:,}: "
          f"{lo * 1e6:.1f}us..{hi * 1e6:.1f}us, spread {hi / lo:.2f}x")
    if hi / lo > 3.0:
        print("  WARNING: append is NOT flat - suspect the harness "
              "before the design.")

    final_bytes = rows[-1][3]
    print(f"\n  marker bytes rewritten across the run "
          f"(~n^2/2): {final_bytes * N150_CHUNKS / 2 / 1e9:.2f} GB")
    print(f"  final marker file size: {final_bytes / 1e6:.2f} MB")
    print(f"  append-only file size after {N150_CHUNKS:,} chunks: "
          f"{_PROGRESS_RECORD.size * N150_CHUNKS / 1e3:.1f} kB")

    try:
        os.rmdir(DISK_DIR)
    except OSError:
        pass


if __name__ == "__main__":
    main()
