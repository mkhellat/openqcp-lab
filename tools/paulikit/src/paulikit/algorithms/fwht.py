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

See the full documentation site (``docs/``) for a much more detailed
treatment: :doc:`/background` (the physical problem this solves),
:doc:`/theory` (this module's derivation, worked step by step with a
hand-verified example), and :doc:`/non_hermitian` (physical examples
of non-Hermitian operators and a worked complex-coefficient example).

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

import warnings

import numpy as np
from numpy.typing import NDArray

try:
    from paulikit._native import pauli_label_native as _native
except ImportError:
    _native = None

try:
    import scipy.sparse as _sp
except ImportError:
    _sp = None


_POPCOUNT_BYTE_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_array(values: NDArray[np.integer], n_bits: int) -> NDArray[np.integer]:
    """Vectorized population count for integers in [0, 2**n_bits).

    Uses an 8-bit lookup table over successive byte slices rather than
    a bit-serial Python-level loop over ``n_bits`` (see
    ``profiling/phase3b/README.md`` Section 3 - this constant-factor win is
    unconditional, independent of any sparsity assumption, and was
    confirmed via profiling to be one of the dense implementation's
    non-negligible costs at N=50/100).

    Args:
        values: Integer numpy array.
        n_bits: Number of bits to examine (values are assumed to fit).

    Returns:
        An integer array of the same shape as ``values``, containing
        the number of set bits in each element.
    """
    values = values.astype(np.uint32)
    count = np.zeros(values.shape, dtype=np.int64)
    for shift in range(0, n_bits, 8):
        count += _POPCOUNT_BYTE_LUT[(values >> shift) & 0xFF]
    return count


def _walsh_hadamard_transform_rows(
    array: NDArray[np.complexfloating],
    overwrite_input: bool = False,
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
        overwrite_input: If ``False`` (default), ``array`` is copied
            first so the caller's array is left untouched - safe for
            any caller. If ``True``, the transform is applied directly
            to ``array`` (still returned, for API symmetry with the
            ``False`` case) without an extra full-size copy; only pass
            ``True`` when the caller no longer needs ``array`` in its
            original form after this call (e.g. it was gathered solely
            to be transformed) - at N=150 this copy alone is ~2.73GiB,
            found to be the dominant memory bottleneck for large N,
            upstream of and separate from the dense-output cost
            ``fwht_pauli_coefficients(..., sparse=True)`` avoids.

    Returns:
        An array of the same shape with the transform applied to each
        row - a new array if ``overwrite_input=False``, otherwise
        ``array`` itself (mutated in place).
    """
    transformed = array if overwrite_input else array.copy()
    dim = array.shape[1]
    span = 1
    while span < dim:
        transformed = transformed.reshape(
            transformed.shape[0], dim // (2 * span), 2, span
        )
        left = transformed[:, :, 0, :]
        right = transformed[:, :, 1, :]
        left, right = left + right, left - right
        transformed[:, :, 0, :] = left
        transformed[:, :, 1, :] = right
        transformed = transformed.reshape(transformed.shape[0], dim)
        span *= 2
    return transformed


def fwht_pauli_coefficients(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    sparse: bool = False,
    chunk_size: int | None = None,
) -> NDArray[np.complexfloating] | tuple[NDArray[np.intp], NDArray[np.complexfloating]]:
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
        sparse: If ``False`` (default), returns the full dense
            ``(dim, dim)`` array described below - unchanged behavior,
            kept as the default so existing callers are unaffected. If
            ``True``, returns only the active (nonzero-row) data
            without ever materializing the dense array - see Returns.
            The dense array's O(dim**2) memory/re-scan cost dwarfs L3
            cache well before it dwarfs the operator's own sparsity
            (confirmed via hardware performance counters, see
            ``profiling/cache_locality/``); ``sparse=True`` is the fix
            for callers that only need the nonzero terms, such as
            ``fwht_pauli_terms``. Both modes compute identically up to
            the final step - this only changes what's returned, not
            the sparsity-aware computation itself (preserved from
            Phase 3b either way).
        chunk_size: Only used when ``sparse=True``. If ``None``
            (default), all active rows are gathered and transformed at
            once (one ``(n_active, dim)`` array live in memory - fine
            for moderate N). If set to a positive integer, active rows
            are processed in blocks of at most ``chunk_size`` rows, so
            peak memory scales with ``chunk_size * dim`` rather than
            ``n_active * dim`` - the tiling technique from MIT 6.172
            lecture 1 (block the computation to bound working-set
            size), applied here for memory-footprint reduction rather
            than cache reuse, since each row transforms independently
            (see ``_walsh_hadamard_transform_rows``) with no cross-row
            reduction to preserve. Trades some Python-loop overhead
            for the ability to complete at N where the whole-array
            approach runs out of memory (see
            ``profiling/cache_locality/`` Phase 6 N=150 findings).

    Returns:
        If ``sparse=False``: a complex ``numpy.ndarray`` of shape
        ``(2**n, 2**n)`` where entry ``[x, z]`` is the coefficient of
        the Pauli string ``P(x, z)`` (see module docstring for the
        x/z encoding). Most entries will be exactly or near zero for
        structured/sparse input; callers that want only the nonzero
        terms should filter by magnitude (see ``fwht_pauli_terms``).

        If ``sparse=True``: a tuple ``(active_x, active_coefficients)``
        where ``active_x`` is a 1-D integer array of the ``x`` values
        with at least one nonzero coefficient, and
        ``active_coefficients`` is a complex array of shape
        ``(len(active_x), dim)`` such that
        ``active_coefficients[i, z]`` is the coefficient of
        ``P(active_x[i], z)``. Rows for ``x`` not in ``active_x`` are
        implicitly all-zero and are not represented at all.

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

    # operator may itself be a scipy.sparse matrix (e.g. from
    # paulikit.hamiltonian.build_hamiltonian(..., sparse=True)) - kept
    # sparse rather than densified here, since densifying+upcasting to
    # complex is exactly the ~4GiB N=150 cost this input path exists
    # to avoid (see PLAN.md Phase 8). np.nonzero() and fancy indexing
    # both dispatch correctly to scipy.sparse's own implementations
    # without densifying (verified directly - see PLAN.md Phase 8
    # question 4), except that sparse fancy indexing returns a
    # numpy.matrix of shape (1, nnz) rather than a flat (nnz,) array,
    # which is why the gather below goes through np.asarray(...).ravel().
    is_sparse_input = _sp is not None and _sp.issparse(operator)
    if is_sparse_input:
        # COO (e.g. from pad_to_power_of_two(..., sparse=True)) does
        # not support fancy indexing (operator[p_nz, q_nz] below) -
        # CSR does, and np.nonzero() dispatches efficiently on CSR too.
        operator = operator.tocsr().astype(complex)
    else:
        operator = np.asarray(operator, dtype=complex)

    # Step 1: XOR-index gather, restricted to the operator's nonzero
    # entries. gathered[x, q] = operator[q ^ x, q] is nonzero only when
    # p = q ^ x is a nonzero entry of operator, i.e. x = p ^ q. Only
    # scattering those (x, q) cells - rather than gathering the full
    # dense (dim, dim) array via fancy indexing - avoids O(dim**2) work
    # for the O(N)-nonzero Hamiltonians this package targets. See
    # ``profiling/phase3b/README.md`` for the profiling/design work behind this
    # (an operator-sparsity-independent all-dense fallback would still
    # be correct here, but measurably slower on sparse input and no
    # faster on dense input).
    p_nz, q_nz = np.nonzero(operator)
    x_nz = p_nz ^ q_nz
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)

    z_indices = np.arange(dim)[np.newaxis, :]

    if sparse and chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        # Tiled variant: process active_x in row-blocks of at most
        # chunk_size, so at most one (chunk_size, dim) array is live
        # at a time instead of one (n_active, dim) array - see the
        # chunk_size docstring entry above. inverse is sorted first so
        # each chunk's nonzero entries (p_nz[lo:hi], q_nz[lo:hi]) are
        # a contiguous slice, found via searchsorted on chunk
        # boundaries - avoiding an O(nnz) boolean mask per chunk.
        order = np.argsort(inverse, kind="stable")
        sorted_inverse = inverse[order]
        sorted_p_nz = p_nz[order]
        sorted_q_nz = q_nz[order]

        active_coefficients = np.empty((n_active, dim), dtype=complex)
        for chunk_start in range(0, n_active, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_active)
            lo = int(np.searchsorted(sorted_inverse, chunk_start))
            hi = int(np.searchsorted(sorted_inverse, chunk_end))

            gathered_chunk = np.zeros((chunk_end - chunk_start, dim), dtype=complex)
            gathered_values = operator[sorted_p_nz[lo:hi], sorted_q_nz[lo:hi]]
            if is_sparse_input:
                # scipy.sparse fancy indexing returns a numpy.matrix of
                # shape (1, nnz), not a flat (nnz,) ndarray - verified
                # directly (PLAN.md Phase 8 question 4).
                gathered_values = np.asarray(gathered_values).ravel()
            gathered_chunk[
                sorted_inverse[lo:hi] - chunk_start, sorted_q_nz[lo:hi]
            ] = gathered_values

            transformed_chunk = _walsh_hadamard_transform_rows(
                gathered_chunk, overwrite_input=True
            )

            chunk_x = active_x[chunk_start:chunk_end, np.newaxis]
            phase = 1j ** _popcount_array(chunk_x & z_indices, n_qubits)
            active_coefficients[chunk_start:chunk_end] = (
                transformed_chunk * np.conj(phase) / dim
            )

        return active_x, active_coefficients

    gathered_active = np.zeros((n_active, dim), dtype=complex)
    gathered_values = operator[p_nz, q_nz]
    if is_sparse_input:
        # scipy.sparse fancy indexing returns a numpy.matrix of shape
        # (1, nnz), not a flat (nnz,) ndarray - verified directly
        # (PLAN.md Phase 8 question 4).
        gathered_values = np.asarray(gathered_values).ravel()
    gathered_active[inverse, q_nz] = gathered_values

    # Step 2: Walsh-Hadamard Transform of each active row (each fixed
    # x with at least one nonzero gathered entry). Rows with no
    # nonzero entries transform to all-zero and are skipped entirely.
    # overwrite_input=True: gathered_active is never read again after
    # this call, so transforming it in place avoids a second full-size
    # copy (~2.73GiB at N=150) that was otherwise the dominant memory
    # cost in this function - see _walsh_hadamard_transform_rows.
    transformed_active = _walsh_hadamard_transform_rows(
        gathered_active, overwrite_input=True
    )

    # Step 3: phase-factor multiplication, computed only for active x.
    xz_and = active_x[:, np.newaxis] & z_indices
    phase = 1j ** _popcount_array(xz_and, n_qubits)
    active_coefficients = transformed_active * np.conj(phase) / dim

    if sparse:
        return active_x, active_coefficients

    coefficients = np.zeros((dim, dim), dtype=complex)
    coefficients[active_x] = active_coefficients
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
    chunk_size: int | None = None,
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
        chunk_size: Passed through to ``fwht_pauli_coefficients`` - see
            its docstring. ``None`` (default) processes all active
            rows at once; a positive integer bounds peak memory to
            roughly ``chunk_size * dim`` complex entries, needed for
            large N where the whole-array approach exhausts memory.

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
    active_x, active_coefficients = fwht_pauli_coefficients(
        operator, sparse=True, chunk_size=chunk_size
    )

    # Re-scan only the active rows fwht_pauli_coefficients already
    # identified, never the full (dim, dim) array - see
    # profiling/cache_locality/README.md for why the dense re-scan was
    # a measured cache-locality/robustness problem (OOMs at N=150).
    row_idx, z_nonzero = np.nonzero(np.abs(active_coefficients) > atol)
    x_nonzero = active_x[row_idx]
    coefficient_values = active_coefficients[row_idx, z_nonzero]
    labels = _pauli_label_batch(x_nonzero, z_nonzero, n_qubits)

    if assume_hermitian:
        real_terms: dict[str, float] = {}
        for label, c in zip(labels, coefficient_values.tolist()):
            if abs(c.imag) > max(atol, 1e-6 * abs(c)):
                raise ValueError(
                    f"term {label!r} has non-negligible "
                    f"imaginary part {c.imag!r} - operator may not be Hermitian; "
                    "pass assume_hermitian=False to decompose it anyway"
                )
            real_terms[label] = float(c.real)
        return real_terms

    complex_terms: dict[str, complex] = {
        label: complex(c) for label, c in zip(labels, coefficient_values.tolist())
    }
    return complex_terms


_WARNED_NO_NATIVE = False


def _pauli_label_batch(
    x_indices: NDArray[np.integer], z_indices: NDArray[np.integer], n_qubits: int
) -> list[str]:
    """Batch IXYZ labels for parallel arrays of (x, z) indices.

    Uses the compiled ``pauli_label_native`` extension when available
    (built from the same C kernel benchmarked in Phase 3a - see
    ``bindings/README.md``); falls back to the pure-Python
    ``pauli_label`` loop otherwise, since paulikit must stay
    pip-installable without a C++ toolchain (see PLAN.md's packaging
    note). The fallback is NOT silent: paulikit's whole purpose is
    fast Pauli decomposition, so running the slow path unknowingly
    would defeat the point of the package - a warning fires once per
    process the first time the fallback is actually used.
    """
    if _native is not None:
        x_masks = np.asarray(x_indices, dtype=np.uint32)
        z_masks = np.asarray(z_indices, dtype=np.uint32)
        return _native.pauli_label_batch(x_masks, z_masks, n_qubits)

    global _WARNED_NO_NATIVE
    if not _WARNED_NO_NATIVE:
        warnings.warn(
            "paulikit's compiled pauli_label fast path is not available "
            "(built with -Dnative=disabled, or a C++ compiler/oneTBB were "
            "missing at build time) - using the pure-Python pauli_label "
            "loop, which is substantially slower for large term counts. "
            "Rebuild paulikit with a C++ compiler and oneTBB available "
            "(see PLAN.md's packaging note) to get the compiled fast path.",
            stacklevel=3,
        )
        _WARNED_NO_NATIVE = True

    return [
        pauli_label(int(x), int(z), n_qubits)
        for x, z in zip(x_indices.tolist(), z_indices.tolist())
    ]
