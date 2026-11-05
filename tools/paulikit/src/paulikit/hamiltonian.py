"""Numeric construction of the coupled-oscillator Hamiltonian.

This is a NumPy implementation of the coupled-oscillator Hamiltonian,
validated against an independent symbolic (SymPy) construction of the
same matrix. It exists so that
paulikit's correctness fixtures and benchmarks do not depend on
importing a notebook, and so the matrix can be built directly in
floating point for large N without symbolic overhead.

Checked against Newton's equations by ``tests/test_hamiltonian_physics.py``.
Before 0.1.0+hamiltonianfix this module, and the notebook's
``prepare_hmatrix`` it was cross-checked against, both carried the same
wrong layout for N >= 3 (see ``build_hamiltonian``); agreeing with each
other did not make either right.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.sparse as _sp
except ImportError:
    _sp = None


def build_hamiltonian(
    n_oscillators: int,
    spring_constants: dict[tuple[int, int], float],
    masses: list[float],
    sparse: bool = False,
) -> NDArray[np.floating]:
    """Build the coupled-oscillator Hamiltonian matrix.

    An ``(N + N*(N+1)/2)``-dimensional real symmetric matrix
    ``H = -[[0, B], [B^T, 0]]``, following Babbush, Berry, Kothari,
    Somma, Wiebe, Phys. Rev. X 13, 041041 (2023), arXiv:2303.13012,
    Sec. III. ``B`` has one column per spring: the N wall springs
    ``(i, i)`` first, then every coupling ``(j, k)`` with ``j < k`` in
    the order ``for j in range(N), for k in range(j+1, N)``. Column
    ``(i, i)`` holds ``sqrt(k_ii/m_i)`` in row i; column ``(j, k)``
    holds ``sqrt(k_jk/m_j)`` in row j and ``-sqrt(k_jk/m_k)`` in row k.
    Then ``B B^T = M^{-1/2} F M^{-1/2}``, with F the stiffness matrix of
    Newton's equations.

    Args:
        n_oscillators: Number of coupled oscillators, N.
        spring_constants: Mapping from ``(i, j)`` with ``i <= j`` to
            the spring constant k_ij. Diagonal entries ``(i, i)`` are
            the on-site spring constants; off-diagonal entries are
            the coupling terms between oscillator i and j.
        masses: Sequence of N masses, ``masses[i]`` for oscillator i.
        sparse: If ``False`` (default), returns a dense
            ``numpy.ndarray`` - unchanged behavior. If ``True``,
            returns a ``scipy.sparse.coo_matrix`` instead, built by
            assembling the same nonzero entries directly rather than
            scattering into a dense array first - avoids ever
            allocating the ``O(N**2)`` dense matrix (see PLAN.md Phase
            8: at N=150 the dense Hamiltonian alone costs ~4GiB once
            padded and upcast to complex, despite being only 0.034%
            dense). Requires the ``scipy`` package (install via the
            ``paulikit[sparse]`` extra); raises ``ImportError`` if
            ``scipy`` is not installed - deliberately not a silent
            fallback to dense, since that would defeat the purpose of
            an explicit sparse request.

    Returns:
        A real-valued, symmetric matrix of shape
        ``(N + N*(N+1)//2, N + N*(N+1)//2)`` - a ``numpy.ndarray`` if
        ``sparse=False``, a ``scipy.sparse.coo_matrix`` if
        ``sparse=True``.

    Note:
        Before 0.1.0+hamiltonianfix every coupling column was placed in
        rows 0 and 1 regardless of which oscillators the spring joins.
        That agrees with the construction above only for N=2; for
        N >= 3 it produced the wrong dynamics. The same fix shipped in
        the standalone paulikit 0.1.4.
    """
    if sparse and _sp is None:
        raise ImportError(
            "sparse=True requires scipy, which is not installed. "
            "Install it via `pip install paulikit[sparse]`."
        )

    n = n_oscillators
    size = n + n * (n + 1) // 2

    # (row, col, value) of the upper-right block of H = -[[0,B],[B^T,0]];
    # the lower-left block is its transpose.
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i in range(n):
        rows.append(i)
        cols.append(n + i)
        values.append(-np.sqrt(spring_constants[(i, i)] / masses[i]))
    col = 2 * n
    for j in range(n):
        for k in range(j + 1, n):
            kappa = spring_constants[(j, k)]
            rows += [j, k]
            cols += [col, col]
            values += [-np.sqrt(kappa / masses[j]), np.sqrt(kappa / masses[k])]
            col += 1

    if not sparse:
        hamiltonian = np.zeros((size, size))
        hamiltonian[rows, cols] = values
        hamiltonian[cols, rows] = values
        return hamiltonian

    return _sp.coo_matrix(
        (values + values, (rows + cols, cols + rows)), shape=(size, size)
    )


def pad_to_power_of_two(
    matrix: NDArray[np.floating],
    sparse: bool = False,
) -> tuple[NDArray[np.floating], int]:
    """Zero-pad a square matrix so its dimension is a power of two.

    Required before Pauli decomposition, since an n-qubit operator
    must have dimension exactly 2**n.

    Args:
        matrix: A square matrix - a ``numpy.ndarray`` if
            ``sparse=False``, or any ``scipy.sparse`` matrix (e.g. the
            ``coo_matrix`` returned by ``build_hamiltonian(...,
            sparse=True)``) if ``sparse=True``.
        sparse: If ``False`` (default), ``matrix`` and the return
            value are dense ``numpy.ndarray``s - unchanged behavior.
            If ``True``, ``matrix`` is treated as a ``scipy.sparse``
            matrix and the padded result is built via COO
            reconstruction at the new shape rather than
            ``csr_matrix.resize()`` - the latter mutates its input in
            place (verified directly), which would break this
            function's existing non-mutating contract. Requires the
            ``scipy`` package (see ``build_hamiltonian``'s
            ``sparse`` parameter for the same requirement/rationale).

    Returns:
        A tuple ``(padded_matrix, n_qubits)`` where ``padded_matrix``
        has shape ``(2**n_qubits, 2**n_qubits)`` and contains
        ``matrix`` in its top-left block, zero elsewhere -
        a ``numpy.ndarray`` if ``sparse=False``, a
        ``scipy.sparse.coo_matrix`` if ``sparse=True``.
    """
    if sparse and _sp is None:
        raise ImportError(
            "sparse=True requires scipy, which is not installed. "
            "Install it via `pip install paulikit[sparse]`."
        )

    size = matrix.shape[0]
    n_qubits = int(np.ceil(np.log2(size)))
    dim = 2**n_qubits

    if not sparse:
        padded = np.zeros((dim, dim))
        padded[:size, :size] = matrix
        return padded, n_qubits

    coo = matrix.tocoo()
    padded = _sp.coo_matrix((coo.data, (coo.row, coo.col)), shape=(dim, dim))
    return padded, n_qubits
