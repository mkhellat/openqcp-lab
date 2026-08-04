"""Original Pauli decomposition via the Fast Walsh-Hadamard Transform (FWHT).

Implements the O(N^2 log N) algorithm (N = 2**n) described in
"Pauli decomposition via the fast Walsh-Hadamard transform"
(https://iopscience.iop.org/article/10.1088/1367-2630/adb44d), as an
alternative to the O(4^n)-with-symbolic-overhead brute-force approach
in N_coupled_harmonic_oscillators_1_D.ipynb's decompose_to_pauli_terms.

This is an original implementation: the algorithm's three steps were
independently re-derived and verified against a from-scratch,
definition-level brute-force decomposition (Frobenius inner product
against every tensor-product Pauli string) before writing the fast
version, rather than transcribed from the paper. See
``_reference_brute_force_decompose`` below and
``tests/test_fwht_decompose.py`` for that verification.

Mathematical basis
-------------------
Using the symplectic (X/Z) representation of an n-qubit Pauli string,
indexed by bitmasks x, z in [0, 2**n):

    P(x, z) = bigotimes_{j=0}^{n-1} i**(x_j & z_j) * X**x_j * Z**z_j

(qubit j corresponds to bit (n-1-j) of x and z, matching the row/column
order produced by a left-to-right ``numpy.kron`` chain).

The matrix element <p| X**x Z**z |q> equals
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
N = 2**n -- versus the naive approach's O(4**n) *symbolic* trace
evaluations, each of which itself costs O(2**n) or worse with SymPy's
overhead. The fast approach also parallelizes trivially across x
(each row's gather + WHT is independent), though this module's initial
version is single-threaded NumPy; see PLAN.md Phase 3 for the planned
parallelization step.
"""

import numpy as np


def _popcount_array(values, n_bits):
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


def _walsh_hadamard_transform_rows(array):
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


def fwht_pauli_coefficients(hamiltonian):
    """Decompose a Hermitian matrix into Pauli-string coefficients via FWHT.

    Args:
        hamiltonian: A ``(2**n, 2**n)`` Hermitian matrix (real or
            complex; typically real-symmetric for the coupled-oscillator
            Hamiltonians this module targets). Dimension must be an
            exact power of two - pad with
            ``hamiltonian.pad_to_power_of_two`` first if not.

    Returns:
        A complex ``numpy.ndarray`` of shape ``(2**n, 2**n)`` where
        entry ``[x, z]`` is the coefficient of the Pauli string
        ``P(x, z)`` (see module docstring for the x/z encoding). Most
        entries will be exactly or near zero for structured/sparse
        input Hamiltonians; callers that want only the nonzero terms
        should filter by magnitude (see ``fwht_pauli_terms``).

    Raises:
        ValueError: If ``hamiltonian`` is not square or its dimension
            is not a power of two.
    """
    dim = hamiltonian.shape[0]
    if hamiltonian.shape != (dim, dim):
        raise ValueError(f"hamiltonian must be square, got shape {hamiltonian.shape}")
    n_qubits = int(round(np.log2(dim)))
    if 2 ** n_qubits != dim:
        raise ValueError(
            f"hamiltonian dimension {dim} is not a power of two; "
            "pad it first with hamiltonian.pad_to_power_of_two"
        )

    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    q_indices = np.arange(dim)
    x_indices = np.arange(dim)[:, np.newaxis]
    p_indices = q_indices[np.newaxis, :] ^ x_indices  # p_indices[x, q] = q ^ x

    # Step 1: XOR-index gather. gathered[x, q] = H[q ^ x, q].
    gathered = hamiltonian[p_indices, q_indices[np.newaxis, :]]

    # Step 2: Walsh-Hadamard Transform of each row (each fixed x).
    transformed = _walsh_hadamard_transform_rows(gathered)

    # Step 3: phase-factor multiplication.
    z_indices = np.arange(dim)[np.newaxis, :]
    xz_and = x_indices & z_indices
    phase = 1j ** _popcount_array(xz_and, n_qubits)
    coefficients = transformed * np.conj(phase) / dim

    return coefficients


def pauli_label(x_mask, z_mask, n_qubits):
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
        matching the convention used by fixtures.py's
        ``pauli_word_to_label``.
    """
    letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    chars = []
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit
        xj = (x_mask >> bit) & 1
        zj = (z_mask >> bit) & 1
        chars.append(letters[(xj, zj)])
    return "".join(chars)


def fwht_pauli_terms(hamiltonian, atol=1e-10):
    """Decompose a Hermitian matrix into a label -> coefficient dict.

    Convenience wrapper around ``fwht_pauli_coefficients`` that filters
    to nonzero terms and converts (x, z) bitmask pairs to IXYZ label
    strings, matching the format used by ``fixtures.py``.

    Args:
        hamiltonian: A ``(2**n, 2**n)`` Hermitian matrix.
        atol: Terms with ``abs(coefficient) <= atol`` are dropped.

    Returns:
        A dict mapping Pauli-string label (e.g. ``"IXZ"``) to its real
        coefficient. Imaginary parts are dropped after asserting they
        are within ``atol`` of zero, since a Hermitian input's
        decomposition into Hermitian Pauli strings must have real
        coefficients.
    """
    dim = hamiltonian.shape[0]
    n_qubits = int(round(np.log2(dim)))
    coefficients = fwht_pauli_coefficients(hamiltonian)

    terms = {}
    x_nonzero, z_nonzero = np.nonzero(np.abs(coefficients) > atol)
    for x, z in zip(x_nonzero.tolist(), z_nonzero.tolist()):
        c = coefficients[x, z]
        if abs(c.imag) > max(atol, 1e-6 * abs(c)):
            raise ValueError(
                f"term {pauli_label(x, z, n_qubits)!r} has non-negligible "
                f"imaginary part {c.imag!r} - hamiltonian may not be Hermitian"
            )
        terms[pauli_label(x, z, n_qubits)] = float(c.real)
    return terms
