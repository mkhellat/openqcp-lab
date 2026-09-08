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
  both         - what _append_parallel_checkpoint_chunk actually does.

Sweeping k reveals the growth: a flat marker cost would mean the O(n^2)
concern is theoretical, a rising one confirms it.

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
    _append_checkpoint_frame,
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
          f"{'marker/frame':>13} {'marker bytes':>13}")

    frame_times = []
    rows = []
    for k in (1, 100, 1000, N150_CHUNKS // 2, N150_CHUNKS):
        completed = set(range(k))
        path = os.path.join(DISK_DIR, f"c{k}.bin")
        for p in (path, str(_parallel_checkpoint_progress_path(path))):
            if os.path.exists(p):
                os.unlink(p)

        ft, mt = [], []
        for _ in range(reps):
            if os.path.exists(path):
                os.unlink(path)
            ft.append(time_frame(path, x, z, coeff, idx_dtype, completed))
            mt.append(time_marker(path, x, z, coeff, idx_dtype, completed))
        mf, mm = statistics.mean(ft), statistics.mean(mt)
        marker_bytes = os.path.getsize(_parallel_checkpoint_progress_path(path))
        frame_times.append(mf)
        rows.append((k, mf, mm, marker_bytes))
        print(f"{k:>19,} {mf * 1e3:>9.3f}ms {mm * 1e3:>9.3f}ms "
              f"{mm / mf:>12.2f}x {marker_bytes:>12,}")

        for p in (path, str(_parallel_checkpoint_progress_path(path))):
            if os.path.exists(p):
                os.unlink(p)

    print("\n" + "=" * 72)
    print("Extrapolated cost of a FULL N=150 run (5,595 chunks):")
    # Frame cost is flat in k, so one mean covers every chunk.
    frame_total = statistics.mean(frame_times) * N150_CHUNKS
    # The marker is rewritten once per chunk with a set that grows
    # 1..N, so the run's total is the integral, not N * the final cost.
    # Trapezoid over the measured points, in chunk-index space.
    pts = sorted((k, m) for k, _f, m, _b in rows)
    marker_total = 0.0
    for (k0, m0), (k1, m1) in zip(pts, pts[1:]):
        marker_total += (m0 + m1) / 2 * (k1 - k0)
    print(f"  frames : {frame_total:8.2f}s  "
          f"({statistics.mean(frame_times) * 1e3:.3f}ms x {N150_CHUNKS:,})")
    print(f"  markers: {marker_total:8.2f}s  (trapezoid over growing set)")
    total = frame_total + marker_total
    if total > 0:
        print(f"  marker share of checkpoint cost: "
              f"{marker_total / total * 100:.1f}%")
    final_bytes = rows[-1][3]
    print(f"\n  marker bytes rewritten across the run "
          f"(~n^2/2): {final_bytes * N150_CHUNKS / 2 / 1e9:.2f} GB")
    print(f"  final marker file size: {final_bytes / 1e6:.2f} MB")

    try:
        os.rmdir(DISK_DIR)
    except OSError:
        pass


if __name__ == "__main__":
    main()
