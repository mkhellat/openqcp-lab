"""Measures Work(D) for dag_gst_master_analysis.md Section 3f: the
main-process drain loop's per-chunk `_pauli_label_batch` cost, at the
real average surviving-term count per chunk for the N=150 workload
(total_terms / n_chunks = 91,652,096 / 5595 ~ 16,381 terms/chunk,
from full_matrix_target.py's own total_terms counter).

This is a LOWER BOUND on the drain loop's true main-process cost: it
does not include _append_parallel_checkpoint_chunk's file I/O (not
exercised here - full_matrix_target.py never passes checkpoint_path)
or the dict-construction step in parallel_decompose's non-Hermitian
yield path - both additional main-process-serial cost not measured
by this script.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/desadm/Projects/__0__science-tools__/openqcp-lab/tools/paulikit/src")
from paulikit.algorithms.fwht import _native, _pauli_label_batch

N_QUBITS = 14
REPS = 200
TERM_COUNTS = [1000, 5000, 16381, 32000]


def main() -> None:
    print(f"native extension loaded: {_native is not None}")
    rng = np.random.default_rng(0)
    for n_terms in TERM_COUNTS:
        x = rng.integers(0, 2**N_QUBITS, size=n_terms).astype(np.intp)
        z = rng.integers(0, 2**N_QUBITS, size=n_terms).astype(np.intp)
        t0 = time.perf_counter()
        for _ in range(REPS):
            _pauli_label_batch(x, z, N_QUBITS)
        t1 = time.perf_counter()
        per_call_ms = (t1 - t0) / REPS * 1e3
        print(f"n_terms={n_terms:6d}: {per_call_ms:8.4f} ms/call")

    n_chunks = 5595
    total_terms = 91_652_096
    avg_terms = total_terms / n_chunks
    print(f"\navg surviving terms/chunk (real N=150 workload): {avg_terms:.1f}")


if __name__ == "__main__":
    main()
