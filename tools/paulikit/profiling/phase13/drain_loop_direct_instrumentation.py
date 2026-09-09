"""Direct, non-theoretical measurement of where wall-clock time goes
inside the REAL parallel_decompose execution - no DAG estimate, no
isolated microbenchmark on synthetic data. Requested directly after
two DAG-based claims failed real scrutiny this session: (1) the
labeling-relocation fix's own arithmetic didn't match measured
speedup/slowdown at any condition, and (2) the follow-up IPC-cost
explanation for that regression was shown, by the user's own
back-of-envelope check, to predict WORSE wall-clock than what was
actually measured at w2_c1/w8_c4 - meaning it cannot be the (whole)
explanation either. Both were built from indirect evidence (DAG
node-cost estimates, an isolated pickle.dumps() microbenchmark on
random data outside the real pipeline). This script instead
instruments the REAL, unmodified `_parallel_worker_chunk`/
`_parallel_worker_init` (same production functions, not a
reimplementation - see real_4worker_contention_test.py's own
precedent for this pattern) and re-implements `parallel_decompose`'s
drain loop with timers around every sub-step, to measure directly:

    (a) total time each worker spends actually computing (returned
        back to the main process as a float - cheap to pickle, unlike
        the label-relocation fix's mistake of returning a list[str])
    (b) total time the drain loop spends in `future.result()`
        (receiving a completed future - blocks if nothing is ready)
    (c) total time the drain loop spends in `_pauli_label_batch`
        (labeling, done here in the drain loop, matching the CURRENT
        - reverted - production code)
    (d) total time the drain loop spends building the output dict
        (`_build_real_terms`)
    (e) overall wall-clock

If the drain loop (b+c+d summed) is large and does NOT shrink as
n_workers increases, that is direct, first-party evidence of a serial
bottleneck - independent of any DAG estimate or synthetic
microbenchmark. If it is small, or shrinks, the drain-loop theory
itself needs to be abandoned in favor of a different explanation
(algorithm-level: chunk_size too small; codebase-level: something else
entirely serial; implementation-level: a specific inefficiency not yet
identified).

Thermal control reused directly from thermal_controlled_welch_ttest.py
(same machine, same real finding: this 15W-TDP CPU throttles under
sustained multi-core load even with the 'performance' governor).

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python drain_loop_direct_instrumentation.py
"""
import multiprocessing
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from unittest import mock

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import (
    _build_real_terms,
    _load_parallel_checkpoint,
    _parallel_worker_chunk,
    _parallel_worker_init,
    _pauli_label_batch,
    _per_worker_resident_bytes,
    _prepare_operator_for_fwht,
    _recommended_parallel_chunk_size,
)
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

import numpy as np

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
    ("w1", 1, real_pin_cpus[:1]),
    ("w2_c1", 2, real_pin_cpus[:1]),
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


def _timed_worker_chunk(chunk_index, chunk_start, chunk_end):
    """Wraps the REAL _parallel_worker_chunk unmodified, only adds a
    wall-clock timer around the call and returns it as a 5th, cheap
    (float) element - not a list[str] like the reverted fix."""
    t0 = time.perf_counter()
    chunk_index, chunk_x_out, z_idx, chunk_coeff_out = _parallel_worker_chunk(
        chunk_index, chunk_start, chunk_end
    )
    worker_elapsed = time.perf_counter() - t0
    return chunk_index, chunk_x_out, z_idx, chunk_coeff_out, worker_elapsed


def instrumented_parallel_decompose(operator, chunk_size, n_workers, pin_cpus_override):
    """Re-implementation of parallel_decompose's own drain loop
    (fwht.py:1450-1683 as of this session), byte-for-byte the same
    control flow, with timers added around each sub-step. Checkpointing
    is not exercised here (checkpoint_path=None always) - out of scope
    for this measurement."""
    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(
        operator
    )
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

    max_in_flight = max(1, 2 * n_workers)
    next_pin_index = multiprocessing.Value("i", 0)

    timings = {
        "future_result": 0.0,
        "labeling": 0.0,
        "dict_build": 0.0,
        "worker_compute_sum": 0.0,
    }
    total_terms = 0

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus_override
    ), ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(
            operator, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
            active_x, dim, n_qubits, z_indices, 1e-10, pin_cpus_override, next_pin_index,
        ),
    ) as pool:
        pending_iter = iter(pending)
        in_flight: set = set()

        def _submit_next():
            item = next(pending_iter, None)
            if item is None:
                return False
            ci, cs, ce = item
            in_flight.add(pool.submit(_timed_worker_chunk, ci, cs, ce))
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        t_start = time.perf_counter()
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                t0 = time.perf_counter()
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out, worker_elapsed = future.result()
                timings["future_result"] += time.perf_counter() - t0
                timings["worker_compute_sum"] += worker_elapsed

                _submit_next()

                t0 = time.perf_counter()
                labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)
                timings["labeling"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                term_dict = _build_real_terms(labels, chunk_coeff_out, 1e-10)
                timings["dict_build"] += time.perf_counter() - t0

                total_terms += len(term_dict)
        overall_elapsed = time.perf_counter() - t_start

    return overall_elapsed, total_terms, timings


def run_once(n_workers, pin_cpus):
    temp_before = _read_pkg_temp_c()
    temp_samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            t = _read_pkg_temp_c()
            if t is not None:
                temp_samples.append(t)
            time.sleep(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    overall, n_terms, timings = instrumented_parallel_decompose(
        padded, CHUNK_SIZE, n_workers, pin_cpus
    )

    stop.set()
    thread.join(timeout=2)

    drain_total = timings["future_result"] + timings["labeling"] + timings["dict_build"]
    return {
        "overall": overall,
        "n_terms": n_terms,
        "future_result": timings["future_result"],
        "labeling": timings["labeling"],
        "dict_build": timings["dict_build"],
        "drain_total": drain_total,
        "worker_compute_sum": timings["worker_compute_sum"],
        "temp_before": temp_before,
        "temp_mean": statistics.mean(temp_samples) if temp_samples else None,
        "temp_max": max(temp_samples) if temp_samples else None,
    }


if __name__ == "__main__":
    print(f"N={N_OSCILLATORS} dim={padded.shape[0]} chunk_size={CHUNK_SIZE} "
          f"physical_cores_detected={real_pin_cpus} reps={REPS}")
    print("Direct instrumentation of the CURRENT (reverted) production drain loop.")
    print()

    results = {name: [] for name, _, _ in CONDITIONS}

    for rep in range(REPS):
        for name, n_workers, pin_cpus in CONDITIONS:
            settled = cooldown()
            r = run_once(n_workers, pin_cpus)
            results[name].append(r)
            print(
                f"rep={rep} {name:8s} overall={r['overall']:.4f}s "
                f"drain_total={r['drain_total']:.4f}s "
                f"(future_result={r['future_result']:.4f}s "
                f"labeling={r['labeling']:.4f}s "
                f"dict_build={r['dict_build']:.4f}s) "
                f"worker_compute_sum={r['worker_compute_sum']:.4f}s "
                f"terms={r['n_terms']} cooldown_settled={settled} "
                f"temp_mean={r['temp_mean']}",
                flush=True,
            )

    print("\n=== Summary (mean over reps) ===")
    for name, _, _ in CONDITIONS:
        runs = results[name]
        mean_overall = statistics.mean(r["overall"] for r in runs)
        mean_drain = statistics.mean(r["drain_total"] for r in runs)
        mean_future_result = statistics.mean(r["future_result"] for r in runs)
        mean_labeling = statistics.mean(r["labeling"] for r in runs)
        mean_dict_build = statistics.mean(r["dict_build"] for r in runs)
        mean_worker_sum = statistics.mean(r["worker_compute_sum"] for r in runs)
        print(
            f"{name:8s}: overall={mean_overall:.4f}s  "
            f"drain_total={mean_drain:.4f}s ({100*mean_drain/mean_overall:.1f}% of overall)  "
            f"[future_result={mean_future_result:.4f}s labeling={mean_labeling:.4f}s "
            f"dict_build={mean_dict_build:.4f}s]  "
            f"worker_compute_sum={mean_worker_sum:.4f}s "
            f"(would-be single-thread total if serialized)"
        )
