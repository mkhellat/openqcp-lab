"""Attributes the cost of _append_parallel_checkpoint_chunk to its two
candidate causes, by falsifying each independently.

THE QUESTION. Yesterday's arrays_with_checkpoint sweep cell measured
~298s versus ~18s without checkpointing, but it wrote into /tmp - a
7.7 GB RAM-backed tmpfs on this machine - and hit ENOSPC ("Disk quota
exceeded") after 4.55 GB. That measurement is retracted
(arrays_vs_dict_findings.md). This script does NOT re-run the 30-run
thermal sweep to recover the number; it isolates the MECHANISM at
small N, where the writer is the same shared, unmodified code.

TWO CANDIDATE CAUSES for the checkpoint cost:
  C1 (I/O)  - bytes reaching the filesystem: ~91.6M JSON lines at N=150.
  C2 (CPU)  - the per-term CPython loop in the writer: .tolist()
              materializing Python ints/complexes, a dict literal, and a
              json.dumps call per surviving term, all GIL-held in the
              PARENT's drain loop.

FALSIFICATION DESIGN (per feedback_falsify_every_candidate_cause):
each variant REMOVES exactly one candidate and keeps the other, so a
cost that survives removal did not come from the removed cause.

  full          - the real writer, to a real file on real disk. Both
                  causes present. Control.
  no_io         - identical serialization, written to /dev/null.
                  REMOVES C1, keeps C2. If the cost is I/O, this
                  collapses; if it is CPU, it barely moves.
  no_format     - the same bytes, pre-serialized ONCE outside the timed
                  region, then written to a real file. REMOVES C2,
                  keeps C1 (identical byte volume). The mirror test.
  raw_bytes     - as no_format but to /dev/null: both removed. Floor.

Timing the writer directly (not through a full decomposition) keeps the
measurement about the writer and avoids a multi-minute run per rep.
Byte volume is held identical across variants by construction and
asserted, so no variant can win by writing less.

Usage:
    python checkpoint_cost_attribution.py [n_terms] [reps]
"""

import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from paulikit.algorithms.fwht import (  # noqa: E402
    _append_parallel_checkpoint_chunk,
)

# Real disk, never /tmp: /tmp is a RAM-backed tmpfs here and writing
# gigabytes into it measures memory exhaustion, not I/O.
DISK_DIR = os.path.expanduser("~/.paulikit_ckpt_bench")


def make_chunk(n_terms, seed=0):
    """A chunk shaped like a real one: intp index arrays and a complex
    coefficient array, with values in a realistic range."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2**20, size=n_terms).astype(np.intp)
    z = rng.integers(0, 2**20, size=n_terms).astype(np.intp)
    coeff = (rng.standard_normal(n_terms)
             + 1j * rng.standard_normal(n_terms) * 1e-18)
    return x, z, coeff


def serialize(x, z, coeff):
    """The exact per-term formatting the real writer performs."""
    out = []
    for xv, zv, cv in zip(x.tolist(), z.tolist(), coeff.tolist()):
        out.append(json.dumps(
            {"x": xv, "z": zv, "re": cv.real, "im": cv.imag}) + "\n")
    return "".join(out)


def time_full(x, z, coeff, path):
    completed = set()
    t0 = time.perf_counter()
    _append_parallel_checkpoint_chunk(path, completed, 0, x, z, coeff)
    return time.perf_counter() - t0


def time_no_io(x, z, coeff, _path):
    """Removes C1: identical serialization, discarded at /dev/null."""
    t0 = time.perf_counter()
    with open(os.devnull, "a") as f:
        for xv, zv, cv in zip(x.tolist(), z.tolist(), coeff.tolist()):
            f.write(json.dumps(
                {"x": xv, "z": zv, "re": cv.real, "im": cv.imag}) + "\n")
    return time.perf_counter() - t0


def time_no_format(_x, _z, _coeff, path, payload=None):
    """Removes C2: identical bytes, formatted once OUTSIDE the timer."""
    t0 = time.perf_counter()
    with open(path, "a") as f:
        f.write(payload)
    return time.perf_counter() - t0


def time_raw_bytes(_x, _z, _coeff, _path, payload=None):
    t0 = time.perf_counter()
    with open(os.devnull, "a") as f:
        f.write(payload)
    return time.perf_counter() - t0


def main():
    n_terms = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    os.makedirs(DISK_DIR, exist_ok=True)
    x, z, coeff = make_chunk(n_terms)
    payload = serialize(x, z, coeff)
    n_bytes = len(payload.encode())
    print(f"n_terms={n_terms:,}  reps={reps}  "
          f"payload={n_bytes / 1e6:.1f} MB  disk={DISK_DIR}")
    print(f"bytes/term={n_bytes / n_terms:.1f}\n")

    variants = [
        ("full", time_full, True),
        ("no_io", time_no_io, False),
        ("no_format", lambda *a: time_no_format(*a, payload=payload), True),
        ("raw_bytes", lambda *a: time_raw_bytes(*a, payload=payload), False),
    ]

    results = {}
    for name, fn, writes_disk in variants:
        times = []
        for rep in range(reps):
            fd, path = tempfile.mkstemp(dir=DISK_DIR, suffix=".jsonl")
            os.close(fd)
            try:
                dt = fn(x, z, coeff, path)
                if writes_disk:
                    got = os.path.getsize(path)
                    # The real writer also emits a progress file; only
                    # the triples file is compared here.
                    assert abs(got - n_bytes) < 0.02 * n_bytes, (
                        f"{name}: wrote {got} bytes, expected ~{n_bytes} "
                        "- variants must write identical volume")
                times.append(dt)
            finally:
                os.unlink(path)
                prog = path + ".parallel_progress.json"
                if os.path.exists(prog):
                    os.unlink(prog)
        results[name] = times
        print(f"  {name:>10}: {statistics.mean(times):7.3f}s  "
              f"(sd {statistics.stdev(times) if len(times) > 1 else 0:.3f})")

    full = statistics.mean(results["full"])
    no_io = statistics.mean(results["no_io"])
    no_fmt = statistics.mean(results["no_format"])

    print(f"\n{'=' * 62}\nAttribution at n_terms={n_terms:,}:")
    print(f"  removing I/O  (full -> no_io)     leaves "
          f"{no_io / full * 100:5.1f}% of the cost")
    print(f"  removing CPU  (full -> no_format) leaves "
          f"{no_fmt / full * 100:5.1f}% of the cost")
    print(f"\n  per-term serialization cost: "
          f"{no_io / n_terms * 1e6:.3f} us/term")
    print(f"  extrapolated to N=150 (91,652,096 terms): "
          f"{no_io / n_terms * 91_652_096:.1f}s of GIL-held CPU")
    print(f"  extrapolated payload: "
          f"{n_bytes / n_terms * 91_652_096 / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
