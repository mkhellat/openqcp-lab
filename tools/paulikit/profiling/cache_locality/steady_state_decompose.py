#!/usr/bin/env python3
"""In-process, warmed-up decompose driver for perf measurements.

Why this exists: invoking the `paulikit` CLI once per measurement (as
the earlier baseline_perf_stat.md / n_scaling_findings.md did)
conflates process startup (Python interpreter init, module imports,
first-touch page faults - see compiler_flags_findings.md and the
N=25 perf-record localization that found ~12% of sampled cache
misses in kernel_init_pages / memcg accounting / gc_collect_main, not
the algorithm at all) with the actual decomposition work being
measured. That conflation is proportionally worse at small N (where
the real work is fast) than at large N - which is itself a
confound when comparing across N, since it makes small-N look more
"startup-dominated" for reasons unrelated to the algorithm.

This script builds the Hamiltonian once, calls
`fwht_pauli_terms` once un-timed as a warm-up (pays for first-call
costs: any lazy imports, allocator warm-up, etc.), then times
`--reps` further calls in the same process and reports the mean.
Run the whole script under `perf stat`/`perf record` to measure
steady-state algorithm behavior specifically.

Usage:
    perf stat -e ... -- python steady_state_decompose.py --n-oscillators 50 --reps 5
"""

from __future__ import annotations

import argparse
import sys
import time

from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms


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

    spring_constants = _default_spring_constants(args.n_oscillators)
    masses = _default_masses(args.n_oscillators)
    unpadded = build_hamiltonian(args.n_oscillators, spring_constants, masses)
    operator, n_qubits = pad_to_power_of_two(unpadded)

    # Untimed warm-up: pays for first-call costs (lazy imports inside
    # fwht.py if any, allocator warm-up, page faults for freshly
    # allocated arrays) so the timed loop below measures steady-state
    # behavior only.
    warm_terms = fwht_pauli_terms(operator)

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        terms = fwht_pauli_terms(operator)
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
        f"{operator.shape[0]}x{operator.shape[1]} padded Hamiltonian"
    )
    print(f"Nonzero Pauli terms: {len(terms)}")
    print(
        f"Steady-state mean time over {args.reps} reps: {mean_time:.4f}s "
        f"(individual: {[f'{t:.4f}' for t in times]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
