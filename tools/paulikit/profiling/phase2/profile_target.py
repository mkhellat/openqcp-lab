"""Shared setup for Phase 2 profiling (PLAN.md section 5).

Builds the same matched-N real coupled-oscillator Hamiltonian used by
``tests/test_benchmark_reference.py``, so profiling results are
directly comparable to the recorded benchmark numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_benchmark_reference import _synthetic_masses, _synthetic_spring_constants

from paulikit.algorithms.fwht import fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 50
"""N=50 (2048x2048, 11 qubits): the largest matched-benchmark size
that still runs in single-digit seconds (~6.2s), fast enough to
profile repeatedly while still slow enough for cProfile/py-spy
sampling to have plenty to look at. N=100 (~8192x8192) was too slow
to iterate on directly during profiling; see the Phase 2 write-up for
timing notes at multiple N."""


def build_target():
    spring_constants = _synthetic_spring_constants(N_OSCILLATORS)
    masses = _synthetic_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses)
    padded, n_qubits = pad_to_power_of_two(unpadded)
    return padded, n_qubits


def run_once():
    padded, _ = build_target()
    return fwht_pauli_terms(padded)


if __name__ == "__main__":
    terms = run_once()
    print(f"N={N_OSCILLATORS}: {len(terms)} nonzero Pauli terms")
