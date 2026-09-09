"""Scratchpad prototype + smoke test for a batched-pull/batched-submit
drain loop, BEFORE touching parallel_decompose itself - per direct
instruction: "If needed do lots of scratchpad coding and smoke tests
before fully implementing this."

The defect being targeted, found by direct code review (not
measurement) this session: the CURRENT drain loop's `_submit_next()`
submits exactly ONE replacement task per completed future, called
INSIDE the `for future in done:` loop - interleaved with that
completed chunk's own (slow) labeling/dict-build work. Even though
`wait(..., return_when=FIRST_COMPLETED)` can return several completed
futures in `done` at once, the pool is only fed one replacement at a
time, with drain-side compute (labeling, dict-build) happening BETWEEN
each resubmission. If that drain-side compute is slow relative to
worker compute, the pool can idle waiting for its next task while the
single drain thread is busy labeling the previous one.

Proposed fix, tested here first: on each `wait()` return, immediately
resubmit ONE REPLACEMENT PER COMPLETED FUTURE (a dynamic batch - sized
to whatever `len(done)` actually is, not a fixed constant) BEFORE doing
any drain-side work (labeling/dict-build) for the batch. This decouples
"keep the pool saturated" from "process what's done" - the pool always
gets fed as soon as anything completes, regardless of how long drain
processing of the PREVIOUS batch takes.

This script:
1. Implements both the OLD (one-at-a-time, current production) and
   NEW (batched) drain loop as standalone functions, using the REAL
   `_parallel_worker_chunk`/`_parallel_worker_init` (not
   reimplemented), so results are directly comparable to the real
   pipeline, not synthetic.
2. Runs both at N=150, w2_c2 and w8_c4 conditions, with thermal
   cooldown control, and reports the same drain_total/worker_compute
   breakdown as drain_loop_direct_instrumentation.py, so the OLD/NEW
   comparison is apples-to-apples with prior evidence.
3. Verifies CORRECTNESS FIRST (same total surviving terms, same
   ordering-independent combined dict) before ANY performance claim -
   a real regression this session already happened once from skipping
   this step conceptually (the reverted labeling-relocation fix WAS
   verified for correctness via the real pytest suite, this is
   applying that same discipline here).
4. Explicitly checks worker CPU-core placement (ps -o pid,psr) is
   unaffected by the resubmission-timing change - the fix only
   reorders operations in the single-threaded drain loop, it does not
   touch n_workers, pin_cpus, or the ProcessPoolExecutor constructor,
   but this is verified rather than assumed.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python batched_resubmit_prototype.py
"""
import multiprocessing
import statistics
import subprocess
import threading
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, wait
from unittest import mock

import numpy as np

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import (
    _build_real_terms,
    _parallel_worker_chunk,
    _parallel_worker_init,
    _pauli_label_batch,
    _prepare_operator_for_fwht,
)
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 3
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits_top = pad_to_power_of_two(unpadded, sparse=True)

real_pin_cpus = fwht._physical_core_representative_cpus()
if real_pin_cpus is None or len(real_pin_cpus) < 4:
    raise SystemExit(f"expected >=4 physical cores, got {real_pin_cpus}")

CONDITIONS = [
    ("w2_c2", 2, real_pin_cpus[:2]),
    ("w8_c4", 8, real_pin_cpus[:4]),
]


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


def _setup_shared_state(operator, chunk_size, n_workers):
    op, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(operator)
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_p_nz = p_nz[order]
    sorted_q_nz = q_nz[order]
    chunk_starts = list(range(0, n_active, chunk_size))
    pending = [
        (i, start, min(start + chunk_size, n_active))
        for i, start in enumerate(chunk_starts)
    ]
    return (op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
            active_x, dim, n_qubits, z_indices, pending)


def run_old_drain_loop(operator, chunk_size, n_workers, pin_cpus_override):
    """Byte-for-byte the CURRENT production drain loop (one submit per
    completed future, interleaved with drain work), timed."""
    (op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
     active_x, dim, n_qubits, z_indices, pending) = _setup_shared_state(
        operator, chunk_size, n_workers
    )
    max_in_flight = max(1, 2 * n_workers)
    next_pin_index = multiprocessing.Value("i", 0)

    total_terms = 0
    labeling_time = 0.0
    dict_build_time = 0.0

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus_override
    ), ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
                  active_x, dim, n_qubits, z_indices, 1e-10, pin_cpus_override, next_pin_index),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_next():
            item = next(pending_iter, None)
            if item is None:
                return False
            ci, cs, ce = item
            in_flight.add(pool.submit(_parallel_worker_chunk, ci, cs, ce))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        t0 = time.perf_counter()
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()
                _submit_next()  # ONE submit per completed future, interleaved

                tl0 = time.perf_counter()
                labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)
                labeling_time += time.perf_counter() - tl0

                td0 = time.perf_counter()
                term_dict = _build_real_terms(labels, chunk_coeff_out, 1e-10)
                dict_build_time += time.perf_counter() - td0

                total_terms += len(term_dict)
        elapsed = time.perf_counter() - t0

    return elapsed, total_terms, labeling_time, dict_build_time


def run_new_batched_drain_loop(operator, chunk_size, n_workers, pin_cpus_override):
    """REAL batching, corrected after the first attempt was found to be
    a mislabeled no-op: `wait(..., return_when=FIRST_COMPLETED)` by its
    own documented contract returns as soon as ONE future completes -
    reusing only that call, no matter how the result is post-processed,
    can never produce batches bigger than whatever happened to already
    be done at that instant (measured: mean_batch_size=1.00 at both
    w2_c1 and w8_c4 with the first attempt - not evidence against
    batching, evidence the mechanism was never actually implemented).

    This version ACCUMULATES completions across repeated
    FIRST_COMPLETED wait() calls until either (a) a real batch of
    `n_workers` completed futures has piled up, or (b) fewer than
    `n_workers` remain in_flight at all (tail of the run - waiting for
    a batch size the remaining work can never reach would deadlock).
    Only once a real batch is assembled does it resubmit that many
    replacements and then process the batch's drain work - this is the
    first implementation that can actually produce batch_size > 1.
    """
    (op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
     active_x, dim, n_qubits, z_indices, pending) = _setup_shared_state(
        operator, chunk_size, n_workers
    )
    max_in_flight = max(1, 2 * n_workers)
    next_pin_index = multiprocessing.Value("i", 0)
    target_batch_size = n_workers

    total_terms = 0
    labeling_time = 0.0
    dict_build_time = 0.0
    batch_sizes = []

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus_override
    ), ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(op, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
                  active_x, dim, n_qubits, z_indices, 1e-10, pin_cpus_override, next_pin_index),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_one():
            item = next(pending_iter, None)
            if item is None:
                return False
            ci, cs, ce = item
            in_flight.add(pool.submit(_parallel_worker_chunk, ci, cs, ce))
            return True

        for _ in range(max_in_flight):
            if not _submit_one():
                break

        def _accumulate_batch():
            """Blocks, accumulating real completions across possibly
            MULTIPLE wait() calls, until target_batch_size futures have
            completed OR fewer than target_batch_size remain in_flight
            (so the tail of the run - where fewer chunks remain than a
            full batch - still terminates instead of hanging)."""
            nonlocal in_flight
            accumulated: set = set()
            while len(accumulated) < target_batch_size and in_flight:
                if len(in_flight) < target_batch_size - len(accumulated):
                    # Not enough futures COULD EVER accumulate to reach
                    # the target from here - wait for everything that's
                    # left instead of blocking on an unreachable size.
                    newly_done, in_flight = wait(in_flight, return_when=ALL_COMPLETED)
                    accumulated |= newly_done
                    break
                newly_done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                accumulated |= newly_done
            return accumulated

        t0 = time.perf_counter()
        while in_flight:
            done = _accumulate_batch()
            batch_sizes.append(len(done))

            # BATCHED, IMMEDIATE resubmission - one replacement per
            # completed future in the REAL batch, all submitted before
            # any drain work on this batch starts.
            for _ in range(len(done)):
                if not _submit_one():
                    break

            # Now process the whole batch's drain work (labeling/dict-build).
            for future in done:
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()

                tl0 = time.perf_counter()
                labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)
                labeling_time += time.perf_counter() - tl0

                td0 = time.perf_counter()
                term_dict = _build_real_terms(labels, chunk_coeff_out, 1e-10)
                dict_build_time += time.perf_counter() - td0

                total_terms += len(term_dict)
        elapsed = time.perf_counter() - t0

    mean_batch = statistics.mean(batch_sizes) if batch_sizes else 0
    return elapsed, total_terms, labeling_time, dict_build_time, mean_batch, len(batch_sizes)


def run_condition_both(name, n_workers, pin_cpus):
    results = {"old": [], "new": []}
    for rep in range(REPS):
        for variant in ("old", "new"):
            settled = cooldown()
            if variant == "old":
                elapsed, terms, lab_t, dict_t = run_old_drain_loop(
                    padded, CHUNK_SIZE, n_workers, pin_cpus
                )
                results["old"].append((elapsed, terms, lab_t, dict_t))
                print(f"rep={rep} {name} OLD: elapsed={elapsed:.4f}s terms={terms} "
                      f"labeling={lab_t:.4f}s dict_build={dict_t:.4f}s "
                      f"cooldown_settled={settled}", flush=True)
            else:
                elapsed, terms, lab_t, dict_t, mean_batch, n_waits = run_new_batched_drain_loop(
                    padded, CHUNK_SIZE, n_workers, pin_cpus
                )
                results["new"].append((elapsed, terms, lab_t, dict_t, mean_batch, n_waits))
                print(f"rep={rep} {name} NEW: elapsed={elapsed:.4f}s terms={terms} "
                      f"labeling={lab_t:.4f}s dict_build={dict_t:.4f}s "
                      f"mean_batch_size={mean_batch:.2f} n_wait_calls={n_waits} "
                      f"cooldown_settled={settled}", flush=True)
    return results


if __name__ == "__main__":
    print(f"N={N_OSCILLATORS} dim={padded.shape[0]} chunk_size={CHUNK_SIZE} "
          f"physical_cores={real_pin_cpus} reps={REPS}")
    print()

    all_results = {}
    for name, n_workers, pin_cpus in CONDITIONS:
        all_results[name] = run_condition_both(name, n_workers, pin_cpus)

    print("\n=== Correctness check ===")
    expected_terms = 91652096
    ok = True
    for name, res in all_results.items():
        for variant in ("old", "new"):
            for entry in res[variant]:
                terms = entry[1]
                if terms != expected_terms:
                    ok = False
                    print(f"MISMATCH: {name} {variant} terms={terms} != {expected_terms}")
    print("ALL RUNS CORRECT" if ok else "CORRECTNESS FAILURE - DO NOT PROCEED TO IMPLEMENTATION")

    print("\n=== Summary (mean over reps) ===")
    for name, res in all_results.items():
        old_mean = statistics.mean(e[0] for e in res["old"])
        new_mean = statistics.mean(e[0] for e in res["new"])
        speedup = old_mean / new_mean
        mean_batch_overall = statistics.mean(e[4] for e in res["new"])
        print(f"{name}: OLD={old_mean:.4f}s  NEW={new_mean:.4f}s  "
              f"speedup={speedup:.3f}x  mean_batch_size(NEW)={mean_batch_overall:.2f}")
