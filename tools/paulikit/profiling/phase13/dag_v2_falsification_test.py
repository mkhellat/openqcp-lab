"""Phase 2: FALSIFY (or confirm) dag_extraction_v2_execution_tier.md.

This script exists to give v2 a chance to be WRONG. v1
(`dag_extraction_and_parallelism.md`, parallelism ~=3.0) and v2
(`dag_extraction_v2_execution_tier.md`) make DIRECTLY OPPOSED
predictions about one measurable quantity, so a single measurement
decides between them:

    P1  v2: dict-build (d5) costs MORE than labeling (d4+d4').
        v1: labeling costs 14x dict-build (it weights d4 at
            Theta(t*n_qubits) and d5 at Theta(t)).

    P2  v2: d1 (unpickle) + d4' (str materialization) + d5 (dict
        inserts) - the tier-P nodes - are the large majority of
        drain-loop time.

    P3  v2: total drain time is comparable to or larger than total
        worker compute at w8_c4 (the serial region really is the
        ceiling).

    P4  v2: d2 (_submit_next bookkeeping) is negligible (<1%), which
        would directly explain why the batched-submit fix moved
        almost nothing.

If P1 comes out the other way, v2 is wrong and gets retracted exactly
the way v1 is being retracted - no salvaging.

METHOD NOTES (why the instrumentation is shaped like this):

- `_pauli_label_batch` FUSES tier-V and tier-P in one call: the C
  kernel (`pauli_label.c`) then the Cython wrapper's own Python loop
  building t_i str objects (`pauli_label_native.pyx:94-98`). The
  shipped extension exposes NO C-kernel-only entry point, so d4 and
  d4' CANNOT be separated from inside this loop - the bucket
  `d4_plus_d4prime_label` is honestly named as their SUM. This does
  not affect P1, which compares total labeling against total
  dict-build; it only means the labeling number cannot be attributed
  between its C and Python halves here. A separate standalone
  microbenchmark (`--split-label`) does that attribution outside the
  drain loop, where adding an isolated C-only timing is safe.

- d1 (unpickle) is measured as `future.result()` wall-time. At the
  moment we call it the future is ALREADY complete (wait() returned
  it in `done`), so this is dominated by deserialization, not by
  waiting on the worker.

- Worker compute (for P3) is summed from per-task timings returned by
  the worker itself, so it excludes IPC and is directly comparable
  against the drain-side buckets.

- Thermal cooldown before every timed run, same protocol as every
  other measurement in this investigation
  (thermal_controlled_welch_ttest.py).

Foreground-only, per this project's standing execution rule.
"""

import multiprocessing
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from unittest import mock

import numpy as np

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import (
    _build_real_terms,
    _parallel_worker_init,
    _pauli_label_batch,
    _prepare_operator_for_fwht,
)
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

try:
    from paulikit._native import pauli_label_native as _native
except ImportError:
    _native = None

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 3
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180


def _read_pkg_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown():
    start = time.perf_counter()
    while True:
        temp = _read_pkg_temp_c()
        if temp is not None and temp <= COOLDOWN_TARGET_C:
            return temp
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return temp
        time.sleep(2)


def _timed_worker_chunk(chunk_index, chunk_start, chunk_end):
    """`_parallel_worker_chunk` plus its own compute time, for P3.

    Deliberately NOT importing the production function's body - it
    calls the real one, so the measured compute is the real thing.
    """
    t0 = time.perf_counter()
    result = fwht._parallel_worker_chunk(chunk_index, chunk_start, chunk_end)
    return result + (time.perf_counter() - t0,)


def _setup_shared_state(operator, chunk_size):
    op, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(
        operator
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]
    order = np.argsort(inverse, kind="stable")
    chunk_starts = list(range(0, n_active, chunk_size))
    pending = [
        (i, start, min(start + chunk_size, n_active))
        for i, start in enumerate(chunk_starts)
    ]
    return (
        op, is_sparse_input, inverse[order], p_nz[order], q_nz[order],
        active_x, dim, n_qubits, z_indices, pending,
    )


def run_instrumented(operator, chunk_size, n_workers, pin_cpus_override):
    """The REAL production drain loop, per-node instrumented."""
    (op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
     active_x, dim, n_qubits, z_indices, pending) = _setup_shared_state(
        operator, chunk_size
    )
    max_in_flight = max(1, 2 * n_workers)
    next_pin_index = multiprocessing.Value("i", 0)

    acc = {
        "d1_unpickle": 0.0,
        "d2_submit": 0.0,
        "d4_plus_d4prime_label": 0.0,
        "d5_dict_build": 0.0,
        "worker_compute_sum": 0.0,
    }
    total_terms = 0
    n_chunks_seen = 0

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus_override
    ), ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
                  active_x, dim, n_qubits, z_indices, 1e-10,
                  pin_cpus_override, next_pin_index),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_next():
            item = next(pending_iter, None)
            if item is None:
                return False
            ci, cstart, cend = item
            in_flight.add(pool.submit(_timed_worker_chunk, ci, cstart, cend))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        t_start = time.perf_counter()
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                t = time.perf_counter()
                (_ci, chunk_x_out, z_idx, chunk_coeff_out,
                 worker_time) = future.result()
                acc["d1_unpickle"] += time.perf_counter() - t
                acc["worker_compute_sum"] += worker_time

                t = time.perf_counter()
                _submit_next()
                acc["d2_submit"] += time.perf_counter() - t

                t = time.perf_counter()
                labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)
                acc["d4_plus_d4prime_label"] += time.perf_counter() - t

                t = time.perf_counter()
                term_dict = _build_real_terms(labels, chunk_coeff_out, 1e-10)
                acc["d5_dict_build"] += time.perf_counter() - t

                total_terms += len(term_dict)
                n_chunks_seen += 1
        elapsed = time.perf_counter() - t_start

    return elapsed, total_terms, n_chunks_seen, acc


def split_label_tiers(n_terms=16384, n_qubits=14, reps=200):
    """Attribute labeling between tier-V (d4, C kernel) and tier-P
    (d4', Python str materialization), OUTSIDE the drain loop.

    The extension has no C-only entry point, so the split is done by
    reimplementing the wrapper's own two halves here against the SAME
    kernel: `pauli_label_batch` gives (C kernel + str loop); a bare
    `bytes` slice+decode loop over a preallocated buffer gives the
    str-loop half alone at the same t_i and n_qubits. This is a
    reconstruction of `pauli_label_native.pyx:90-98`, not a guess -
    it mirrors that code line for line.
    """
    if _native is None:
        print("native extension unavailable - cannot split label tiers")
        return
    rng = np.random.default_rng(0)
    xm = rng.integers(0, 2**n_qubits, n_terms, dtype=np.uint32)
    zm = rng.integers(0, 2**n_qubits, n_terms, dtype=np.uint32)

    t = time.perf_counter()
    for _ in range(reps):
        _native.pauli_label_batch(xm, zm, n_qubits)
    full = (time.perf_counter() - t) / reps

    # The tier-P half alone: same slice/decode/list-store the wrapper
    # does, over an already-filled buffer (no C kernel involved).
    buf = bytes(n_terms * n_qubits)
    t = time.perf_counter()
    for _ in range(reps):
        out = [None] * n_terms
        for i in range(n_terms):
            out[i] = buf[i * n_qubits:(i + 1) * n_qubits].decode("ascii")
    strloop = (time.perf_counter() - t) / reps

    print(f"\n{'='*66}\nLabeling tier split ({n_terms} terms, n_qubits={n_qubits})"
          f"\n{'='*66}")
    print(f"  full pauli_label_batch (d4 + d4')  : {full*1e3:8.3f} ms")
    print(f"  str-materialization loop alone (d4'): {strloop*1e3:8.3f} ms")
    print(f"  implied C kernel (d4)               : {(full-strloop)*1e3:8.3f} ms")
    if full > 0:
        print(f"  --> d4' is {100*strloop/full:.1f}% of labeling "
              f"(v2 predicts the Python half dominates)")


def main():
    sc = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, sc, masses, sparse=True)
    padded, _ = pad_to_power_of_two(unpadded, sparse=True)

    real_pin_cpus = fwht._physical_core_representative_cpus()
    if real_pin_cpus is None or len(real_pin_cpus) < 4:
        raise SystemExit(f"expected >=4 physical cores, got {real_pin_cpus}")

    conditions = [
        ("w1_c1", 1, real_pin_cpus[:1]),
        ("w8_c4", 8, real_pin_cpus[:4]),
    ]

    for name, n_workers, pins in conditions:
        print(f"\n{'='*66}\n{name}: n_workers={n_workers} pins={pins}\n{'='*66}")
        runs = []
        for rep in range(REPS):
            temp = cooldown()
            elapsed, terms, nchunks, acc = run_instrumented(
                padded, CHUNK_SIZE, n_workers, pins
            )
            runs.append((elapsed, acc))
            print(f"  rep {rep}: elapsed={elapsed:7.2f}s terms={terms} "
                  f"chunks={nchunks} start_temp={temp}")

        elapsed_m = statistics.mean(r[0] for r in runs)
        print(f"\n  mean drain-loop elapsed: {elapsed_m:.2f}s")
        print(f"  {'node':<28} {'mean s':>9} {'% drain':>9}")
        for key in ("d1_unpickle", "d2_submit", "d4_plus_d4prime_label",
                    "d5_dict_build", "worker_compute_sum"):
            m = statistics.mean(r[1][key] for r in runs)
            print(f"  {key:<28} {m:>9.2f} {100*m/elapsed_m:>8.1f}%")

        lab = statistics.mean(r[1]["d4_plus_d4prime_label"] for r in runs)
        dct = statistics.mean(r[1]["d5_dict_build"] for r in runs)
        print(f"\n  P1  dict/label ratio = {dct/lab:.2f}x")
        print(f"      v2 predicts >1 (dict dominates); "
              f"v1 predicts ~0.07 (labeling 14x dict)")
        print(f"      --> {'v2 SUPPORTED' if dct > lab else 'v2 FALSIFIED'}")


if __name__ == "__main__":
    import sys
    if "--split-label" in sys.argv:
        split_label_tiers()
    else:
        split_label_tiers()
        main()
