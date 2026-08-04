"""Numeric construction of the coupled-oscillator Hamiltonian.

This is an independent NumPy reimplementation of the Hamiltonian
construction in ``N_coupled_harmonic_oscillators_1_D.ipynb``'s
``prepare_hmatrix(N)`` (which builds the matrix symbolically with
SymPy). It exists so that pauli_perf's correctness fixtures and
benchmarks do not depend on importing a notebook, and so the matrix
can be built directly in floating point for large N without symbolic
overhead.

Cross-checked against the notebook's own ``prepare_hmatrix`` at N=2
and N=4: matches to machine epsilon (~1e-16) for both.
"""

import numpy as np


def build_hamiltonian(n_oscillators, spring_constants, masses):
    """Build the coupled-oscillator Hamiltonian matrix.

    Reproduces the structure described in
    ``coupled_harmonic_oscillators/README.md`` and implemented
    symbolically in ``N_coupled_harmonic_oscillators_1_D.ipynb``: an
    ``(N + N*(N+1)/2)``-dimensional matrix with a block-antidiagonal
    "B" coupling block, built from a "sqrt(k/m)" encoding of the
    physical spring constants and masses.

    Args:
        n_oscillators: Number of coupled oscillators, N.
        spring_constants: Mapping from ``(i, j)`` with ``i <= j`` to
            the spring constant k_ij. Diagonal entries ``(i, i)`` are
            the on-site spring constants; off-diagonal entries are
            the coupling terms between oscillator i and j.
        masses: Sequence of N masses, ``masses[i]`` for oscillator i.

    Returns:
        A real-valued, symmetric ``numpy.ndarray`` of shape
        ``(N + N*(N+1)//2, N + N*(N+1)//2)``.

    Note:
        Only oscillators 0 and 1 contribute coupling-block entries,
        matching the notebook's own ``if i < 2`` branch. This is a
        property of the original algorithm's encoding, not a bug in
        this reimplementation - see
        ``N_coupled_harmonic_oscillators_1_D.ipynb``, cell defining
        ``prepare_hmatrix``, for the symbolic version this mirrors.
    """
    n = n_oscillators
    coupling_count = n * (n + 1) // 2
    size = n + coupling_count
    hamiltonian = np.zeros((size, size))

    # Column index -> (i, j) pair mapping (i < j), matching the
    # notebook's own iteration order: for l in range(N): for m in
    # range(l+1, N).
    pairs = [(l, m) for l in range(n) for m in range(l + 1, n)]

    for i in range(n):
        on_site = -np.sqrt(spring_constants[(i, i)] / masses[i])
        hamiltonian[i, n + i] = on_site
        hamiltonian[n + i, i] = on_site

        if i < 2:
            sign = (-1) ** i
            for col_offset, (l, m) in enumerate(pairs):
                key = (l, m) if l <= m else (m, l)
                coupling = -sign * np.sqrt(spring_constants[key] / masses[i])
                hamiltonian[i, n + n + col_offset] = coupling
                hamiltonian[n + n + col_offset, i] = coupling

    return hamiltonian


def pad_to_power_of_two(matrix):
    """Zero-pad a square matrix so its dimension is a power of two.

    Required before Pauli decomposition, since an n-qubit operator
    must have dimension exactly 2**n.

    Args:
        matrix: A square ``numpy.ndarray``.

    Returns:
        A tuple ``(padded_matrix, n_qubits)`` where ``padded_matrix``
        has shape ``(2**n_qubits, 2**n_qubits)`` and contains
        ``matrix`` in its top-left block, zero elsewhere.
    """
    size = matrix.shape[0]
    n_qubits = int(np.ceil(np.log2(size)))
    dim = 2 ** n_qubits
    padded = np.zeros((dim, dim))
    padded[:size, :size] = matrix
    return padded, n_qubits
