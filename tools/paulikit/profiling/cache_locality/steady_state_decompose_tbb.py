#!/usr/bin/env python3
"""Same driver as steady_state_decompose.py, but forces the TBB-
parallelized label-generation kernel (`pauli_label_batch_parallel`)
instead of the serial one (`pauli_label_batch`) that
`fwht_pauli_terms` actually calls today.

Why this exists: `tbb_not_actually_used_finding.md` found the TBB
entry point isn't invoked in the production path at all -
`_pauli_label_batch` calls the serial kernel, per a Phase 3a finding
that parallelizing this specific loop "barely helps end to end"
(wall-clock only, not measured against cache-miss ratio or stall
cycles - the metrics this investigation actually cares about). Before
proceeding to Phase 6's sparse-representation redesign, the user
asked to test TBB execution properly first with the same detailed
cache-locality methodology used throughout this directory, rather
than take Phase 3a's wall-clock-only "barely helps" finding at face
value for a question (cache locality) it was never measuring.

This does NOT modify `paulikit.algorithms.fwht` or any shipped code -
it monkeypatches `paulikit.algorithms.fwht._pauli_label_batch` at
runtime, in this standalone script's own process, to call
`pauli_label_batch_parallel` instead of `pauli_label_batch`. Compare
its perf output directly against `steady_state_decompose.py`'s (same
event set, same N values, same warm-up/reps protocol) to isolate the
effect of TBB parallelization specifically.

Usage:
    perf stat -e ... -- python steady_state_decompose_tbb.py --n-oscillators 50 --reps 5
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import paulikit.algorithms.fwht as fwht_module
from paulikit._native import pauli_label_native as _native
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _pauli_label_batch_tbb(x_indices, z_indices, n_qubits):
    """Drop-in replacement for fwht_module._pauli_label_batch that
    always uses the TBB-parallel kernel, bypassing the serial-vs-native
    fallback logic entirely (this script requires the native extension
    to be built - see the precondition check in main())."""
    x_masks = np.asarray(x_indices, dtype=np.uint32)
    z_masks = np.asarray(z_indices, dtype=np.uint32)
    return _native.pauli_label_batch_parallel(x_masks, z_masks, n_qubits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-oscillators", type=int, required=True)
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Timed repetitions after the untimed warm-up call (default: 5).",
    )
    args = parser.parse_args()

    if args.n_oscillators < 1:
        print("error: --n-oscillators must be >= 1", file=sys.stderr)
        return 1
    if args.reps < 1:
        print("error: --reps must be >= 1", file=sys.stderr)
        return 1

    if _native is None:
        print(
            "error: paulikit's native extension is not available - this "
            "script specifically requires it (to test the TBB path), "
            "unlike the pure-Python-fallback-tolerant steady_state_decompose.py",
            file=sys.stderr,
        )
        return 1

    # Monkeypatch: only affects this process, not the installed package.
    fwht_module._pauli_label_batch = _pauli_label_batch_tbb

    spring_constants = _default_spring_constants(args.n_oscillators)
    masses = _default_masses(args.n_oscillators)
    unpadded = build_hamiltonian(args.n_oscillators, spring_constants, masses)
    operator, n_qubits = pad_to_power_of_two(unpadded)

    warm_terms = fwht_module.fwht_pauli_terms(operator)

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        terms = fwht_module.fwht_pauli_terms(operator)
        times.append(time.perf_counter() - t0)

    if len(terms) != len(warm_terms):
        print(
            f"error: term count changed between warm-up ({len(warm_terms)}) "
            f"and timed runs ({len(terms)}) - non-deterministic result, "
            "investigate before trusting these timings",
            file=sys.stderr,
        )
        return 1

    mean_time = sum(times) / len(times)
    print(
        f"N={args.n_oscillators} oscillators, {n_qubits} qubits, "
        f"{operator.shape[0]}x{operator.shape[1]} padded Hamiltonian "
        f"[TBB-parallel label kernel]"
    )
    print(f"Nonzero Pauli terms: {len(terms)}")
    print(
        f"Steady-state mean time over {args.reps} reps: {mean_time:.4f}s "
        f"(individual: {[f'{t:.4f}' for t in times]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
