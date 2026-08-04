"""Original Pauli decomposition via the Fast Walsh-Hadamard Transform (FWHT).

Implements the O(N^2 log N) algorithm (N = 2**n) described in
"Pauli decomposition via the fast Walsh-Hadamard transform"
(https://iopscience.iop.org/article/10.1088/1367-2630/adb44d), as an
alternative to an O(4^n)-with-symbolic-overhead brute-force approach.

This is an original implementation: the algorithm's three steps were
independently re-derived and verified against a from-scratch,
definition-level brute-force decomposition (Frobenius inner product
against every tensor-product Pauli string) before writing the fast
version, rather than transcribed from the paper. See
``tests/test_fwht.py`` (package root) for that verification.

Mathematical basis
-------------------
Using the symplectic (X/Z) representation of an n-qubit Pauli string,
indexed by bitmasks x, z in [0, 2**n):

    P(x, z) = bigotimes_{j=0}^{n-1} i**(x_j & z_j) * X**x_j * Z**z_j

(qubit j corresponds to bit (n-1-j) of x and z, matching the row/column
order produced by a left-to-right ``numpy.kron`` chain).

The matrix element ``<p| X**x Z**z |q>`` equals
``(-1)**popcount(q & z)`` when ``p == q ^ x``, and 0 otherwise. So the
Frobenius inner product coefficient is:

    c(x, z) = (1 / dim) * conj(i**popcount(x & z))
              * sum_q H[q ^ x, q] * (-1)**popcount(q & z)

For fixed x, the inner sum over q, as a function of z, is exactly the
(unnormalized, +-1 butterfly) Walsh-Hadamard Transform of the
"anti-diagonal gather" g_x(q) = H[q ^ x, q]. This is the paper's three
steps: (1) the XOR-index gather/permutation building g_x for every x,
(2) the Walsh-Hadamard Transform applied to each g_x, (3) the
phase-factor multiplication by conj(i**popcount(x & z)) / dim.

Complexity: building all 2**n gathers and running a length-2**n WHT on
each costs O(2**n * n * 2**n) = O(n * 4**n) -- i.e. O(N^2 log N) for
N = 2**n -- versus a naive approach's O(4**n) *symbolic* trace
evaluations, each of which itself costs O(2**n) or worse with symbolic
(e.g. SymPy) overhead. The fast approach also parallelizes trivially
across x (each row's gather + WHT is independent), though this
module's initial version is single-threaded NumPy; see PLAN.md
(package root) for the planned parallelization step.

Hermitian and non-Hermitian operators
---------------------------------------
Nothing in the derivation above requires H to be Hermitian: the Pauli
strings span the full space of 2**n x 2**n complex matrices, so
``fwht_pauli_coefficients`` decomposes *any* complex matrix exactly,
producing complex coefficients in general (real coefficients are the
special case that results from Hermitian input). ``fwht_pauli_terms``
defaults to ``assume_hermitian=True`` for convenience with this
package's primary use case (real-symmetric coupled-oscillator
Hamiltonians) but supports ``assume_hermitian=False`` for arbitrary
operators - relevant for, e.g., non-Hermitian effective Hamiltonians
of open/dissipative systems, PT-symmetric Hamiltonians, Liouvillian
superoperators, or individual non-Hermitian summands of a Hermitian
total.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _popcount_array(values: NDArray[np.integer], n_bits: int) -> NDArray[np.integer]:
    """Vectorized population count for integers in [0, 2**n_bits).

    Args:
        values: Integer numpy array.
        n_bits: Number of bits to examine (values are assumed to fit).

    Returns:
        An integer array of the same shape as ``values``, containing
        the number of set bits in each element.
    """
    values = values.copy()
    count = np.zeros_like(values)
    for _ in range(n_bits):
        count += values & 1
        values = values >> 1
    return count


def _walsh_hadamard_transform_rows(
    array: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Apply the unnormalized Walsh-Hadamard Transform along axis 1.

    Uses the standard in-place butterfly algorithm: at each of
    log2(dim) stages, pairs of elements a distance ``h`` apart are
    replaced by their sum and difference. No normalization is applied
    here (that is folded into the phase-factor step); this matches
    the convention (H^{\\otimes n})^2 = 2^n * I^{\\otimes n}.

    Args:
        array: A 2D complex array of shape ``(rows, dim)`` where
            ``dim`` is a power of two. Each row is transformed
            independently.

    Returns:
        A new array of the same shape with the transform applied to
        each row.
    """
    transformed = array.copy()
    dim = array.shape[1]
    span = 1
    while span < dim:
        transformed = transformed.reshape(
            transformed.shape[0], dim // (2 * span), 2, span
        )
        left = transformed[:, :, 0, :].copy()
        right = transformed[:, :, 1, :].copy()
        transformed[:, :, 0, :] = left + right
        transformed[:, :, 1, :] = left - right
        transformed = transformed.reshape(transformed.shape[0], dim)
        span *= 2
    return transformed


def fwht_pauli_coefficients(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Decompose an operator into Pauli-string coefficients via FWHT.

    Works for **any** complex ``(2**n, 2**n)`` matrix, Hermitian or
    not - see the module docstring's "Hermitian and non-Hermitian
    operators" section. Hermitian input (e.g. a real-symmetric
    coupled-oscillator Hamiltonian) yields real coefficients; general
    input yields complex coefficients.

    Args:
        operator: A ``(2**n, 2**n)`` matrix (real or complex).
            Dimension must be an exact power of two - pad with
            ``paulikit.hamiltonian.pad_to_power_of_two`` first if not.

    Returns:
        A complex ``numpy.ndarray`` of shape ``(2**n, 2**n)`` where
        entry ``[x, z]`` is the coefficient of the Pauli string
        ``P(x, z)`` (see module docstring for the x/z encoding). Most
        entries will be exactly or near zero for structured/sparse
        input; callers that want only the nonzero terms should filter
        by magnitude (see ``fwht_pauli_terms``).

    Raises:
        ValueError: If ``operator`` is not square or its dimension is
            not a power of two.
    """
    dim = operator.shape[0]
    if operator.shape != (dim, dim):
        raise ValueError(f"operator must be square, got shape {operator.shape}")
    n_qubits = int(round(np.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError(
            f"operator dimension {dim} is not a power of two; "
            "pad it first with paulikit.hamiltonian.pad_to_power_of_two"
        )

    operator = np.asarray(operator, dtype=complex)
    q_indices = np.arange(dim)
    x_indices = np.arange(dim)[:, np.newaxis]
    p_indices = q_indices[np.newaxis, :] ^ x_indices  # p_indices[x, q] = q ^ x

    # Step 1: XOR-index gather. gathered[x, q] = operator[q ^ x, q].
    gathered = operator[p_indices, q_indices[np.newaxis, :]]

    # Step 2: Walsh-Hadamard Transform of each row (each fixed x).
    transformed = _walsh_hadamard_transform_rows(gathered)

    # Step 3: phase-factor multiplication.
    z_indices = np.arange(dim)[np.newaxis, :]
    xz_and = x_indices & z_indices
    phase = 1j ** _popcount_array(xz_and, n_qubits)
    coefficients = transformed * np.conj(phase) / dim

    return coefficients


def pauli_label(x_mask: int, z_mask: int, n_qubits: int) -> str:
    """Convert an (x, z) symplectic bitmask pair to an IXYZ label string.

    Per qubit j (bit position ``n_qubits - 1 - j`` of the masks, matching
    this module's row/column convention - see module docstring):
    (x_j, z_j) = (0,0) -> 'I', (1,0) -> 'X', (0,1) -> 'Z', (1,1) -> 'Y'.

    Args:
        x_mask: Integer in [0, 2**n_qubits).
        z_mask: Integer in [0, 2**n_qubits).
        n_qubits: Number of qubits.

    Returns:
        A string of length ``n_qubits``, leftmost character = qubit 0,
        matching the convention used by
        ``paulikit.testing.fixtures.pauli_word_to_label``.
    """
    letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    chars = []
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit
        xj = (x_mask >> bit) & 1
        zj = (z_mask >> bit) & 1
        chars.append(letters[(xj, zj)])
    return "".join(chars)


def fwht_pauli_terms(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    atol: float = 1e-10,
    assume_hermitian: bool = True,
) -> dict[str, complex] | dict[str, float]:
    """Decompose an operator into a label -> coefficient dict.

    Convenience wrapper around ``fwht_pauli_coefficients`` that filters
    to nonzero terms and converts (x, z) bitmask pairs to IXYZ label
    strings, matching the format used by
    ``paulikit.testing.fixtures``.

    Works for **any** complex ``(2**n, 2**n)`` matrix, not just
    Hermitian operators: the Pauli strings (including Y) span the
    full space of complex matrices, so an arbitrary (non-Hermitian,
    non-normal, non-diagonalizable - anything) linear operator has a
    well-defined Pauli decomposition with, in general, complex
    coefficients. This matters beyond mathematical completeness:
    physically, not every operator of interest is Hermitian - e.g.
    non-Hermitian effective Hamiltonians for open/dissipative quantum
    systems (complex energies encode finite lifetimes), PT-symmetric
    Hamiltonians, Liouvillian superoperators, or terms in a sum where
    only the total is Hermitian even though individual summands are
    not.

    Args:
        operator: A ``(2**n, 2**n)`` complex or real matrix. No
            Hermiticity is assumed unless ``assume_hermitian=True``.
        atol: Terms with ``abs(coefficient) <= atol`` are dropped.
        assume_hermitian: If True, asserts every returned
            coefficient's imaginary part is within ``atol`` of zero
            (raising ``ValueError`` otherwise) and returns real
            (``float``) coefficients. If False (the default), returns
            ``complex`` coefficients as-is - correct for both
            Hermitian and non-Hermitian input, since a Hermitian
            operator's Pauli coefficients are real to begin with (the
            imaginary parts are then just zero, not dropped).

    Returns:
        A dict mapping Pauli-string label (e.g. ``"IXZ"``) to its
        coefficient: ``float`` if ``assume_hermitian=True``,
        ``complex`` otherwise.

    Raises:
        ValueError: If ``assume_hermitian=True`` and a term's
            coefficient has a non-negligible imaginary part.
    """
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    coefficients = fwht_pauli_coefficients(operator)

    x_nonzero, z_nonzero = np.nonzero(np.abs(coefficients) > atol)
    if assume_hermitian:
        real_terms: dict[str, float] = {}
        for x, z in zip(x_nonzero.tolist(), z_nonzero.tolist()):
            c = coefficients[x, z]
            if abs(c.imag) > max(atol, 1e-6 * abs(c)):
                raise ValueError(
                    f"term {pauli_label(x, z, n_qubits)!r} has non-negligible "
                    f"imaginary part {c.imag!r} - operator may not be Hermitian; "
                    "pass assume_hermitian=False to decompose it anyway"
                )
            real_terms[pauli_label(x, z, n_qubits)] = float(c.real)
        return real_terms

    complex_terms: dict[str, complex] = {
        pauli_label(x, z, n_qubits): complex(coefficients[x, z])
        for x, z in zip(x_nonzero.tolist(), z_nonzero.tolist())
    }
    return complex_terms
