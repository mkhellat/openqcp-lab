"""paulikit: performance-engineering tools for Pauli decomposition.

Exact Pauli decomposition of arbitrary complex matrices, built for
the regime where materialising the full 4^n coefficient set is the
binding constraint rather than the transform itself. Output is
streamed, so peak resident memory is bounded by the chunk size rather
than by the term count.

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
    paulikit.algorithms.fwht.fwht_pauli_terms_iter
    paulikit.algorithms.fwht.auto_decompose
    paulikit.algorithms.fwht.parallel_decompose
    paulikit.algorithms.fwht.parallel_decompose_arrays
    paulikit.algorithms.fwht.terms_from_arrays
    paulikit.testing.fixtures.ALL_FIXTURES
"""

__version__ = "0.1.0"
