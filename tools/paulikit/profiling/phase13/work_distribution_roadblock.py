"""What is rationing work across cores, and why doesn't it distribute
more aggressively? Algorithm roadblock, code roadblock, or scheduling?

This does NOT measure wall-clock or temperature. It measures the
CONTROL STRUCTURE that decides how much work is available to workers
at each instant - which is a property of the code, observable directly.

THREE CANDIDATE ROADBLOCKS, each separately testable:

R1 ALGORITHM. Are the chunks genuinely independent, and are there
   enough of them? If the math forced ordering, no scheduler could
   help. (Prior work says chunks ARE independent and C=5595 >> 8, so
   this is expected to be RULED OUT - included so the elimination is
   on the record rather than assumed.)

R2 CODE / QUEUE STARVATION. `parallel_decompose` caps in-flight work
   at `max_in_flight = 2 * n_workers` (fwht.py:1624) and refills it
   ONE task per completed future, from the single drain thread
   (`_submit_next()`, fwht.py:1654-1660). If the drain thread is busy
   labeling/dict-building (measured: ~21.5s of the ~24s run), it is
   NOT refilling the queue. Workers can then run dry even though
   thousands of chunks remain pending. THIS IS THE PRIME SUSPECT: it
   is a code-level rationing decision, not a hardware limit.

R3 SCHEDULING. Even with work queued, the OS may not place it well.
   (Already partly measured: avg scheduler delay 0.020 -> 0.054 ms.)

WHAT THIS SCRIPT MEASURES (the R2 test, directly):
  - queue depth (len(in_flight)) sampled continuously, from a
    background thread, so the drain thread's own blocking does not
    hide the sampling;
  - how often depth falls BELOW n_workers, i.e. how often there is
    literally not enough queued work to keep every worker busy;
  - the fraction of run time spent starved.

FALSIFIABLE PREDICTION:
  If R2 is the roadblock, w8_c4 must spend a LARGE fraction of the run
  with depth < n_workers (workers idle for want of queued work), and
  w2_c1 must spend much less - because 2*2=4 in-flight slots are far
  easier for the drain thread to keep full than 2*8=16.

  If BOTH conditions keep their queues full, R2 is FALSE - the drain
  thread keeps up, and the problem is elsewhere (R3).

Deliberately reuses the REAL production `_parallel_worker_chunk` /
`_parallel_worker_init`, not a reimplementation.
"""

import multiprocessing
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
SAMPLE_HZ = 2000  # 0.5 ms - fine enough to catch brief starvation


def _setup(operator, chunk_size):
    op, is_sparse, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(operator)
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]
    order = np.argsort(inverse, kind="stable")
    starts = list(range(0, n_active, chunk_size))
    pending = [(i, s, min(s + chunk_size, n_active)) for i, s in enumerate(starts)]
    return (op, is_sparse, inverse[order], p_nz[order], q_nz[order],
            active_x, dim, n_qubits, z_indices, pending)


def run(operator, chunk_size, n_workers, pin_cpus):
    (op, is_sparse, sorted_inverse, sorted_p_nz, sorted_q_nz,
     active_x, dim, n_qubits, z_indices, pending) = _setup(operator, chunk_size)

    max_in_flight = max(1, 2 * n_workers)
    next_pin_index = multiprocessing.Value("i", 0)

    depth_samples: list[int] = []
    stop = threading.Event()
    in_flight: set = set()
    lock = threading.Lock()

    def sampler():
        """Sample queue depth from a SEPARATE thread - the drain thread
        spends most of its time in GIL-holding dict work, so sampling
        from inside it would systematically miss starvation."""
        period = 1.0 / SAMPLE_HZ
        while not stop.is_set():
            with lock:
                depth_samples.append(len(in_flight))
            time.sleep(period)

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus
    ), ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_parallel_worker_init,
        initargs=(op, is_sparse, sorted_inverse, sorted_p_nz, sorted_q_nz,
                  active_x, dim, n_qubits, z_indices, 1e-10,
                  pin_cpus, next_pin_index),
    ) as pool:
        pending_iter = iter(pending)

        def _submit_next():
            item = next(pending_iter, None)
            if item is None:
                return False
            ci, cs, ce = item
            fut = pool.submit(_parallel_worker_chunk, ci, cs, ce)
            with lock:
                in_flight.add(fut)
            return True

        for _ in range(max_in_flight):
            if not _submit_next():
                break

        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()
        t0 = time.perf_counter()
        total_terms = 0

        while True:
            with lock:
                current = set(in_flight)
            if not current:
                break
            done, _ = wait(current, return_when=FIRST_COMPLETED)
            with lock:
                in_flight.difference_update(done)
            for future in done:
                _ci, x_out, z_idx, coeff = future.result()
                _submit_next()
                labels = _pauli_label_batch(x_out, z_idx, n_qubits)
                total_terms += len(_build_real_terms(labels, coeff, 1e-10))

        elapsed = time.perf_counter() - t0
        stop.set()
        sampler_thread.join(timeout=2)

    return elapsed, total_terms, depth_samples


def report(name, n_workers, elapsed, terms, samples):
    n = len(samples)
    starved = sum(1 for d in samples if d < n_workers)
    empty = sum(1 for d in samples if d == 0)
    mean_depth = sum(samples) / n if n else 0
    print(f"\n--- {name} (n_workers={n_workers}, max_in_flight={2*n_workers}) ---")
    print(f"  elapsed          : {elapsed:.2f}s   terms={terms}")
    print(f"  queue samples    : {n}")
    print(f"  mean depth       : {mean_depth:.2f}  (capacity {2*n_workers})")
    print(f"  depth < n_workers: {100*starved/n:5.1f}%  <-- STARVED "
          f"(not enough queued work to fill every worker)")
    print(f"  depth == 0       : {100*empty/n:5.1f}%  <-- fully empty")
    return dict(name=name, n_workers=n_workers, elapsed=elapsed,
                mean_depth=mean_depth, starved_pct=100*starved/n)


def main():
    sc = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, sc, masses, sparse=True)
    padded, _ = pad_to_power_of_two(unpadded, sparse=True)

    pins = fwht._physical_core_representative_cpus()
    if pins is None or len(pins) < 4:
        raise SystemExit(f"expected >=4 physical cores, got {pins}")

    print("R2 test: is the drain thread starving the worker pool?")
    print(f"chunks pending = 5595 (C), so the ALGORITHM (R1) always has")
    print(f"work available - any starvation is a CODE-level rationing.")

    results = []
    for name, nw, p in (("w2_c1", 2, pins[:1] * 2), ("w8_c4", 8, pins[:4])):
        elapsed, terms, samples = run(padded, CHUNK_SIZE, nw, p)
        results.append(report(name, nw, elapsed, terms, samples))

    print("\n" + "=" * 62)
    a, b = results
    print(f"starvation: {a['name']}={a['starved_pct']:.1f}%  "
          f"{b['name']}={b['starved_pct']:.1f}%")
    if b["starved_pct"] > 50 and b["starved_pct"] > 2 * a["starved_pct"]:
        print("--> R2 SUPPORTED: the wider config starves far more. The")
        print("    drain thread cannot refill 16 slots while doing ~21.5s")
        print("    of labeling/dict work. This is a CODE roadblock.")
    elif b["starved_pct"] < 20:
        print("--> R2 FALSIFIED: queues stay full; work IS available to")
        print("    workers. The roadblock is not queue starvation.")
    else:
        print("--> PARTIAL: starvation present but not dominant.")


if __name__ == "__main__":
    main()
