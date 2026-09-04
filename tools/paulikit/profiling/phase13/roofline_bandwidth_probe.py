"""Measures this machine's real, achievable single-core DRAM read+write
bandwidth - the ceiling the Roofline calculation needs to bound how
much a cache-blocked WHT could possibly help (see roofline_analysis.md).

Single-threaded (not `-a`/system-wide) deliberately: paulikit's own
per-chunk WHT runs single-threaded within one worker process (NumPy
vectorizes the whole (chunk_size, dim) array on one core - see
dag_gst_master_analysis.md section 1, DAG B), so the relevant ceiling
for THIS kernel is one core's achievable bandwidth, not the aggregate
across all cores contending for the shared memory controller (that
aggregate ceiling is a different, already partially-explored question
- see the uncore_imc attempt in bandwidth_hypothesis_sweep.py, blocked
without deeper root access).

Method: STREAM-like triad-style read+write over an array far larger
than the 8 MiB shared L3 (256 MiB per array here, ~32x L3) so results
are dominated by real DRAM traffic, not cache hits. Uses the same
thermal-cooldown protocol as every other measurement in this
investigation.

Usage:
    OPENBLAS_NUM_THREADS=1 python roofline_bandwidth_probe.py [--reps N]
"""
import argparse
import os
import time

import numpy as np

COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180
ARRAY_BYTES = 256 * 1024 * 1024  # 256 MiB per array, ~32x the 8 MiB L3


def _read_pkg_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown() -> float | None:
    start = time.perf_counter()
    while True:
        temp = _read_pkg_temp_c()
        if temp is not None and temp <= COOLDOWN_TARGET_C:
            return temp
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return temp
        time.sleep(2)


def run_once() -> dict:
    settled_temp = cooldown()
    n = ARRAY_BYTES // 8  # float64
    a = np.random.default_rng(0).random(n)
    b = np.random.default_rng(1).random(n)

    t0 = time.perf_counter()
    # Triad-like: c = a + scalar*b - one read of a, one read of b, one
    # write of c per element; 2 FLOPs (multiply, add) per element.
    c = a + 2.0 * b
    elapsed = time.perf_counter() - t0
    assert c.sum() != 0  # prevent the optimizer from eliding the work

    bytes_moved = 3 * n * 8  # 2 reads (a, b) + 1 write (c), float64
    flops = 2 * n  # multiply + add per element
    bandwidth_gbps = bytes_moved / elapsed / 1e9
    gflops = flops / elapsed / 1e9

    return {
        "elapsed": elapsed,
        "bytes_moved": bytes_moved,
        "bandwidth_gbps": bandwidth_gbps,
        "gflops": gflops,
        "settled_temp": settled_temp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    results = []
    for rep in range(args.reps):
        r = run_once()
        results.append(r)
        print(
            f"rep={rep}: elapsed={r['elapsed']:.4f}s "
            f"bandwidth={r['bandwidth_gbps']:.2f} GB/s "
            f"gflops={r['gflops']:.3f} settled_temp={r['settled_temp']}",
            flush=True,
        )

    bws = [r["bandwidth_gbps"] for r in results]
    print(f"\nmean bandwidth: {np.mean(bws):.2f} GB/s (stdev {np.std(bws):.2f})")
