"""Precomputes the real N=150 gather-index pattern once, so the gather-
pattern isolation experiment (gather_pattern_target.py) can replay the
ACTUAL irregular scatter positions paulikit's own worker uses per
chunk, without needing to rebuild the real Hamiltonian (or its complex
values) inside every worker process.

Why this matters (traffic_intensity_findings.md's own "Actual next
isolation step", and dag_gst_master_analysis.md section 3e): the
dense-traffic controls (wht_small/touch_small/wht_large) matched
paulikit's per-chunk BUFFER SIZE and stage-touch COUNT but used a
freshly-random dense buffer, not the real IRREGULAR gather (sparse
operator[p_nz, q_nz] values scattered into a zeroed dense buffer at
arbitrary (row, q) positions - see _parallel_worker_chunk,
fwht.py:1246-1255) - and all three scaled fine, refuting dense-traffic-
volume as sufficient. This isolates the access PATTERN itself: same
chunking, same real column positions (sorted_q_nz) and row offsets
(sorted_inverse - chunk_start) every real chunk actually writes to,
but synthetic (fast-to-generate) values instead of real Hamiltonian
entries - the access pattern is what's under test, not the physics.

Output: gather_pattern_chunks.npz - one entry per real N=150/
chunk_size=2 chunk, each a (row_offsets, columns) pair giving exactly
where that chunk's real _parallel_worker_chunk gather/scatter writes
land in its (chunk_size, dim) buffer. Run once; the isolation target
loads this file rather than rebuilding the real Hamiltonian per
worker.

Usage:
    OPENBLAS_NUM_THREADS=1 python gather_pattern_precompute.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paulikit.algorithms.fwht import _prepare_operator_for_fwht
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gather_pattern_chunks.npz")

if __name__ == "__main__":
    spring_constants = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(
        padded
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_q_nz = q_nz[order]

    chunk_starts = list(range(0, n_active, CHUNK_SIZE))
    row_offsets_per_chunk = []
    columns_per_chunk = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + CHUNK_SIZE, n_active)
        lo = int(np.searchsorted(sorted_inverse, chunk_start))
        hi = int(np.searchsorted(sorted_inverse, chunk_end))
        row_offsets_per_chunk.append(
            (sorted_inverse[lo:hi] - chunk_start).astype(np.int32)
        )
        columns_per_chunk.append(sorted_q_nz[lo:hi].astype(np.int32))

    n_chunks = len(chunk_starts)
    nnz_per_chunk = [len(c) for c in columns_per_chunk]
    print(
        f"dim={dim} n_active={n_active} n_chunks={n_chunks} "
        f"nnz_per_chunk: min={min(nnz_per_chunk)} max={max(nnz_per_chunk)} "
        f"mean={np.mean(nnz_per_chunk):.2f}"
    )

    # np.savez with object arrays (ragged per-chunk lengths) - one key
    # per chunk is simplest and fastest to load per-task in the target.
    save_kwargs = {"dim": np.array([dim]), "n_chunks": np.array([n_chunks])}
    for i, (rows, cols) in enumerate(zip(row_offsets_per_chunk, columns_per_chunk)):
        save_kwargs[f"rows_{i}"] = rows
        save_kwargs[f"cols_{i}"] = cols
    np.savez(OUT_PATH, **save_kwargs)
    print(f"wrote {OUT_PATH}")
