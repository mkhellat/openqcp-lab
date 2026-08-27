"""Numeric construction of the coupled-oscillator Hamiltonian.

This is an independent NumPy reimplementation of the Hamiltonian
construction in the parent ``openqcp-lab`` repository's
``coupled_harmonic_oscillators/N_coupled_harmonic_oscillators_1_D_N_2.ipynb``
notebook (which builds the matrix symbolically with SymPy, in a
function there named ``prepare_hmatrix(N)``). It exists so that
paulikit's correctness fixtures and benchmarks do not depend on
importing a notebook, and so the matrix can be built directly in
floating point for large N without symbolic overhead.

Cross-checked against the notebook's own ``prepare_hmatrix`` at N=2
and N=4: matches to machine epsilon (~1e-16) for both.
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

    Reproduces the structure described in the parent repository's
    ``coupled_harmonic_oscillators/README.md`` and implemented
    symbolically in its tutorial notebooks: an
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
        Only oscillators 0 and 1 contribute coupling-block entries,
        matching the source notebook's own ``if i < 2`` branch. This
        is a property of the original algorithm's encoding, not a bug
        in this reimplementation.
    """
    if sparse and _sp is None:
        raise ImportError(
            "sparse=True requires scipy, which is not installed. "
            "Install it via `pip install paulikit[sparse]`."
        )

    n = n_oscillators
    coupling_count = n * (n + 1) // 2
    size = n + coupling_count

    # Column index -> (i, j) pair mapping (i < j), matching the
    # notebook's own iteration order: for l in range(N): for m in
    # range(l+1, N).
    pairs = [(l, m) for l in range(n) for m in range(l + 1, n)]

    if not sparse:
        hamiltonian = np.zeros((size, size))
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

    # Sparse path: accumulate (row, col, value) triples directly,
    # never touching an O(size**2) dense array.
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for i in range(n):
        on_site = -np.sqrt(spring_constants[(i, i)] / masses[i])
        rows += [i, n + i]
        cols += [n + i, i]
        values += [on_site, on_site]

        if i < 2:
            sign = (-1) ** i
            for col_offset, (l, m) in enumerate(pairs):
                key = (l, m) if l <= m else (m, l)
                coupling = -sign * np.sqrt(spring_constants[key] / masses[i])
                rows += [i, n + n + col_offset]
                cols += [n + n + col_offset, i]
                values += [coupling, coupling]

    return _sp.coo_matrix((values, (rows, cols)), shape=(size, size))


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
