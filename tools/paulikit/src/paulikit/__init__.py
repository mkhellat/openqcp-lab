"""paulikit: performance-engineering tools for Pauli decomposition.

An original, from-scratch fast Pauli decomposition implementation
built to scale the coupled_harmonic_oscillators Hamiltonian-simulation
tutorial (in the parent openqcp-lab repository) to larger N than its
original O(4^n) symbolic brute-force approach allows.

Currently implements the Fast Walsh-Hadamard Transform (FWHT) based
algorithm, O(N^2 log N) for an N x N matrix. See
``paulikit.algorithms`` for the algorithm implementations and
``PLAN.md`` (package root) for the full research background, phased
plan, and planned additional algorithms (Tensorized Pauli
Decomposition, PHASE, C-ported variants).

Public API
----------
    paulikit.hamiltonian.build_hamiltonian
    paulikit.hamiltonian.pad_to_power_of_two
    paulikit.pauli_utils.pauli_string_to_matrix
    paulikit.pauli_utils.reconstruct_from_terms
    paulikit.algorithms.fwht.fwht_pauli_coefficients
    paulikit.algorithms.fwht.fwht_pauli_terms
    paulikit.testing.fixtures.ALL_FIXTURES
"""

__version__ = "0.1.0"
