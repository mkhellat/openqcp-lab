#!/usr/bin/env python3
"""In-process, warmed-up decompose driver using the OLD dense-array
code path, for a before/after comparison against Phase 6's fix.

`fwht_pauli_terms` itself now always uses `fwht_pauli_coefficients(...,
sparse=True)` internally (see PLAN.md Phase 6) - there is no longer a
way to exercise the old dense-array-plus-re-scan behavior through the
public API. This script reconstructs fwht_pauli_terms's *entire*
pre-Phase-6 body explicitly (dense `fwht_pauli_coefficients(operator)`
call, full-array `np.nonzero` re-scan, label generation, dict
construction - everything the real function did, not just the
coefficients step) so it's a fair, complete comparison against
`steady_state_decompose.py` (which now measures the new sparse path
via the real `fwht_pauli_terms`) under the same perf event set - the
actual A/B comparison Phase 6's plan requires before claiming an
improvement.

Usage:
    perf stat -e ... -- python steady_state_decompose_dense.py --n-oscillators 50 --reps 5
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_coefficients
from paulikit.algorithms.fwht import _pauli_label_batch


def _decompose_dense(operator, atol, n_qubits):
    """Replicates fwht_pauli_terms's pre-Phase-6 body in full: dense
    fwht_pauli_coefficients call, full-array re-scan, label generation,
    dict construction - matching what the real function did before
    Phase 6 switched it to the sparse path."""
    coefficients = fwht_pauli_coefficients(operator, sparse=False)
    x_nonzero, z_nonzero = np.nonzero(np.abs(coefficients) > atol)
    labels = _pauli_label_batch(x_nonzero, z_nonzero, n_qubits)
    terms = {
        label: complex(coefficients[x, z])
        for label, x, z in zip(labels, x_nonzero.tolist(), z_nonzero.tolist())
    }
    return terms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-oscillators", type=int, required=True)
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Timed repetitions after the untimed warm-up call (default: 5).",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    if args.n_oscillators < 1:
        print("error: --n-oscillators must be >= 1", file=sys.stderr)
        return 1
    if args.reps < 1:
        print("error: --reps must be >= 1", file=sys.stderr)
        return 1

    spring_constants = _default_spring_constants(args.n_oscillators)
    masses = _default_masses(args.n_oscillators)
    unpadded = build_hamiltonian(args.n_oscillators, spring_constants, masses)
    operator, n_qubits = pad_to_power_of_two(unpadded)

    warm_terms = _decompose_dense(operator, args.atol, n_qubits)

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        terms = _decompose_dense(operator, args.atol, n_qubits)
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
        f"{operator.shape[0]}x{operator.shape[1]} padded Hamiltonian [DENSE path]"
    )
    print(f"Nonzero Pauli terms: {len(terms)}")
    print(
        f"Steady-state mean time over {args.reps} reps: {mean_time:.4f}s "
        f"(individual: {[f'{t:.4f}' for t in times]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
