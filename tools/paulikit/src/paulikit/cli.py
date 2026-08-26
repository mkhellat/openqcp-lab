#!/usr/bin/env python3
"""Command-line interface for the paulikit package.

The other modules (``paulikit.hamiltonian``, ``paulikit.pauli_utils``,
``paulikit.testing.fixtures``, ``paulikit.algorithms.fwht``) are plain
importable library code with no CLI of their own by design - keeping
them free of argument-parsing concerns makes them easier to unit test
and reuse as a library. This module is the one place that wires them
together into runnable subcommands.

Usage
-----
Once installed (``pip install -e . --no-build-isolation`` from the
package root, or ``pip install paulikit`` once published), the
``paulikit`` console script is available directly::

    paulikit --help
    paulikit decompose --help
    paulikit decompose --n-oscillators 4
    paulikit benchmark --n-oscillators 16 30 50
    paulikit regenerate-fixtures

Without installing, it can also be run as a module from the package's
``src/`` directory::

    python -m paulikit.cli --help

See README.md for what each subcommand is for and when you'd use it.
"""

import argparse
import sys
import time

from paulikit import __version__
from paulikit.algorithms.fwht import fwht_pauli_coefficients, fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def _default_spring_constants(n_oscillators):
    """A simple, deterministic (not physically meaningful) parameter
    set used by the decompose/benchmark subcommands when the caller
    doesn't need a specific physical system - just something concrete
    and reproducible to decompose. Scales with N so larger N doesn't
    degenerate into all-zero or all-equal matrices."""
    constants = {}
    for i in range(n_oscillators):
        for j in range(i, n_oscillators):
            constants[(i, j)] = 1.0 + 0.1 * (i + j)
    return constants


def _default_masses(n_oscillators):
    return [1.0 + 0.05 * i for i in range(n_oscillators)]


def cmd_decompose(args):
    """Build a synthetic N-oscillator Hamiltonian and Pauli-decompose it."""
    n = args.n_oscillators
    spring_constants = _default_spring_constants(n)
    masses = _default_masses(n)

    unpadded = build_hamiltonian(n, spring_constants, masses)
    padded, n_qubits = pad_to_power_of_two(unpadded)

    print(f"N={n} oscillators, {n_qubits} qubits, {padded.shape[0]}x{padded.shape[0]} "
          f"padded Hamiltonian")

    if args.sparse_output:
        # Exercises fwht_pauli_coefficients(sparse=True) directly -
        # the raw (active_x, active_coefficients) form fwht_pauli_terms
        # itself now always uses internally (see PLAN.md Phase 6). This
        # flag is for inspecting/timing that raw output shape, not a
        # switch on fwht_pauli_terms's own algorithm.
        start = time.perf_counter()
        active_x, active_coefficients = fwht_pauli_coefficients(
            padded, sparse=True, chunk_size=args.chunk_size
        )
        elapsed = time.perf_counter() - start

        print(f"Decomposition time (sparse output): {elapsed:.4f}s")
        print(f"Active rows: {len(active_x)} of {padded.shape[0]} possible")

        if args.show_terms:
            terms = fwht_pauli_terms(
                padded, atol=args.atol, chunk_size=args.chunk_size
            )
            for label in sorted(terms):
                print(f"  {label}: {terms[label]!r}")
        return 0

    start = time.perf_counter()
    terms = fwht_pauli_terms(padded, atol=args.atol, chunk_size=args.chunk_size)
    elapsed = time.perf_counter() - start

    print(f"Decomposition time: {elapsed:.4f}s")
    print(f"Nonzero Pauli terms: {len(terms)}")

    if args.show_terms:
        for label in sorted(terms):
            print(f"  {label}: {terms[label]!r}")

    return 0


def cmd_benchmark(args):
    """Time the FWHT decomposition across a sweep of N (oscillator count) values."""
    if args.compare_dense_sparse:
        print(f"{'N':>5} {'qubits':>7} {'dim':>6} {'active':>8} "
              f"{'dense (s)':>10} {'sparse (s)':>11}")
        for n in args.n_oscillators:
            spring_constants = _default_spring_constants(n)
            masses = _default_masses(n)
            unpadded = build_hamiltonian(n, spring_constants, masses)
            padded, n_qubits = pad_to_power_of_two(unpadded)

            start = time.perf_counter()
            dense_coefficients = fwht_pauli_coefficients(padded, sparse=False)
            dense_elapsed = time.perf_counter() - start

            start = time.perf_counter()
            active_x, _ = fwht_pauli_coefficients(padded, sparse=True)
            sparse_elapsed = time.perf_counter() - start

            print(f"{n:>5} {n_qubits:>7} {padded.shape[0]:>6} {len(active_x):>8} "
                  f"{dense_elapsed:>10.4f} {sparse_elapsed:>11.4f}")
            del dense_coefficients

        return 0

    print(f"{'N':>5} {'qubits':>7} {'dim':>6} {'terms':>8} {'time (s)':>10}")
    for n in args.n_oscillators:
        spring_constants = _default_spring_constants(n)
        masses = _default_masses(n)
        unpadded = build_hamiltonian(n, spring_constants, masses)
        padded, n_qubits = pad_to_power_of_two(unpadded)

        start = time.perf_counter()
        terms = fwht_pauli_terms(padded, atol=args.atol, chunk_size=args.chunk_size)
        elapsed = time.perf_counter() - start

        print(f"{n:>5} {n_qubits:>7} {padded.shape[0]:>6} {len(terms):>8} {elapsed:>10.4f}")

    return 0


def cmd_regenerate_fixtures(args):
    """Regenerate paulikit.testing.fixtures's expected_terms constants.

    Thin wrapper around ``paulikit.testing.fixtures.generate_fixture_data``;
    kept here so it's discoverable alongside the other subcommands and
    documented in one place (--help).
    """
    from paulikit.testing.fixtures import generate_fixture_data

    generate_fixture_data()
    print(
        "\nReminder: this prints regenerated constants but does not "
        "edit fixtures.py automatically. Review the output and paste "
        "into paulikit/testing/fixtures.py's FIXTURE_N2/FIXTURE_N4 "
        "definitions by hand if they should change - see that module's "
        "docstring.",
        file=sys.stderr,
    )
    return 0


def build_parser():
    """Construct the top-level argparse.ArgumentParser for the paulikit CLI."""
    parser = argparse.ArgumentParser(
        prog="paulikit",
        description=(
            "Performance-engineering tools for Pauli decomposition, "
            "built for the coupled_harmonic_oscillators Hamiltonian-"
            "simulation tutorial in the openqcp-lab repository. See "
            "PLAN.md for the full research background and phased plan."
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decompose_parser = subparsers.add_parser(
        "decompose",
        help="Build a synthetic N-oscillator Hamiltonian and decompose it",
        description=(
            "Builds a coupled-oscillator Hamiltonian for a given N using "
            "a fixed, deterministic (not physically calibrated) set of "
            "spring constants and masses, pads it to a power-of-two "
            "dimension, and runs the FWHT-based Pauli decomposition on "
            "it, reporting timing and term count."
        ),
    )
    decompose_parser.add_argument(
        "--n-oscillators", "-n", type=int, default=2,
        help="Number of coupled oscillators, N (default: 2)",
    )
    decompose_parser.add_argument(
        "--atol", type=float, default=1e-10,
        help="Absolute coefficient threshold below which a term is "
             "dropped as zero (default: 1e-10)",
    )
    decompose_parser.add_argument(
        "--show-terms", action="store_true",
        help="Print every nonzero Pauli term and its coefficient "
             "(omitted by default since term counts grow quickly with N)",
    )
    decompose_parser.add_argument(
        "--sparse-output", action="store_true",
        help="Call fwht_pauli_coefficients(sparse=True) directly and "
             "report its timing/active-row count, instead of the usual "
             "label->coefficient dict via fwht_pauli_terms (which "
             "already uses the sparse path internally either way - "
             "this flag is for inspecting the raw sparse output shape "
             "itself, see PLAN.md Phase 6)",
    )
    decompose_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Process active rows in blocks of at most this size instead "
             "of one (n_active, dim) array all at once, bounding peak "
             "memory to roughly chunk_size * dim complex entries - needed "
             "at large N where the whole-array approach exhausts memory "
             "(default: None, i.e. no chunking; see PLAN.md Phase 6 "
             "follow-up on N=150). Not yet auto-tuned - pick a value, or "
             "omit this flag if N is small enough not to need it.",
    )
    decompose_parser.set_defaults(func=cmd_decompose)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Time the decomposition across a sweep of N values",
        description=(
            "Runs the decompose workflow across multiple N values and "
            "prints a timing table, for comparison against the "
            "PennyLane reference numbers recorded in PLAN.md/README.md."
        ),
    )
    benchmark_parser.add_argument(
        "--n-oscillators", "-n", type=int, nargs="+", default=[2, 4, 8, 16, 30],
        help="Space-separated list of N values to benchmark "
             "(default: 2 4 8 16 30)",
    )
    benchmark_parser.add_argument(
        "--atol", type=float, default=1e-10,
        help="Absolute coefficient threshold below which a term is "
             "dropped as zero (default: 1e-10)",
    )
    benchmark_parser.add_argument(
        "--compare-dense-sparse", action="store_true",
        help="Instead of the usual fwht_pauli_terms timing table, time "
             "fwht_pauli_coefficients's dense (sparse=False) and sparse "
             "(sparse=True) output modes directly, side by side, at "
             "each N - a quick command-line way to reproduce the Phase "
             "6 dense-vs-sparse comparison "
             "(profiling/cache_locality/ has the full perf-counter-based "
             "methodology; this is wall-clock timing only)",
    )
    benchmark_parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Passed through to fwht_pauli_terms - see decompose "
             "--chunk-size's help. Ignored when --compare-dense-sparse is "
             "set (that path calls fwht_pauli_coefficients directly "
             "without chunking).",
    )
    benchmark_parser.set_defaults(func=cmd_benchmark)

    regenerate_parser = subparsers.add_parser(
        "regenerate-fixtures",
        help="Regenerate testing/fixtures.py's expected_terms constants (prints only)",
        description=(
            "Recomputes the N=2/N=4 expected Pauli decompositions using "
            "PennyLane as an independent oracle and prints them in a "
            "form ready to paste into paulikit/testing/fixtures.py. Does "
            "not modify that file automatically - see its module "
            "docstring for when you'd need to do this (only if "
            "paulikit.hamiltonian's Hamiltonian construction changes)."
        ),
    )
    regenerate_parser.set_defaults(func=cmd_regenerate_fixtures)

    return parser


def main(argv=None):
    """Entry point registered as the ``paulikit`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
