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

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

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


class _GrowableArray:
    """Amortized-doubling growable 1-D array.

    Appending ``n`` elements at a time and doubling capacity on
    overflow costs O(total elements) amortized, rather than O(chunks)
    reallocations of ever-larger arrays (the naive ``np.concatenate``
    per chunk) or O(total elements) Python-object overhead (a plain
    list of scalars) - keeps the chunked path's per-chunk append cheap
    relative to that chunk's O(chunk_size * dim * log dim) transform
    cost (see Phase 9, PLAN.md), rather than trading space complexity
    for a new time-complexity regression.
    """

    def __init__(self, dtype, initial_capacity: int = 1024):
        self._data = np.empty(initial_capacity, dtype=dtype)
        self._size = 0

    def extend(self, values: NDArray) -> None:
        n = len(values)
        if n == 0:
            return
        needed = self._size + n
        if needed > len(self._data):
            new_capacity = max(needed, len(self._data) * 2)
            grown = np.empty(new_capacity, dtype=self._data.dtype)
            grown[: self._size] = self._data[: self._size]
            self._data = grown
        self._data[self._size : self._size + n] = values
        self._size += n

    def finalize(self) -> NDArray:
        return self._data[: self._size]


def _checkpoint_progress_path(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + ".progress.json")


def _load_checkpoint(
    checkpoint_path: str | Path | None,
) -> tuple[int, tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.complexfloating]] | None]:
    """Read an existing checkpoint, if any.

    Returns ``(resume_from_chunk_index, (x, z, coeff) | None)``: the
    chunk index to resume from (0 if no checkpoint, or the checkpoint
    is absent/incomplete) and the previously-recorded triples to
    replay into the fresh accumulator, or ``None`` if there is nothing
    to replay.
    """
    if checkpoint_path is None:
        return 0, None
    checkpoint_path = Path(checkpoint_path)
    progress_path = _checkpoint_progress_path(checkpoint_path)
    if not checkpoint_path.exists() or not progress_path.exists():
        return 0, None

    with open(progress_path) as f:
        progress = json.load(f)
    next_chunk = progress["next_chunk"]

    x_vals: list[int] = []
    z_vals: list[int] = []
    coeff_vals: list[complex] = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            x_vals.append(record["x"])
            z_vals.append(record["z"])
            coeff_vals.append(complex(record["re"], record["im"]))

    return next_chunk, (
        np.array(x_vals, dtype=np.intp),
        np.array(z_vals, dtype=np.intp),
        np.array(coeff_vals, dtype=complex),
    )


def _append_checkpoint_chunk(
    checkpoint_path: str | Path,
    next_chunk: int,
    x_out: NDArray[np.intp],
    z_out: NDArray[np.intp],
    coeff_out: NDArray[np.complexfloating],
) -> None:
    """Append one completed chunk's surviving triples to the
    checkpoint file, then update the progress marker.

    The triples are appended before the progress marker is updated,
    so a crash mid-write leaves the progress marker pointing at a
    chunk whose triples may be incompletely written - ``_load_checkpoint``
    is only ever consulted for chunks strictly before ``next_chunk``
    once this function has returned for the resumability guarantee to
    hold; a crash during this function itself simply loses that one
    in-flight chunk's checkpoint, which is then recomputed on resume
    (not silently corrupted), since ``next_chunk`` is only advanced
    (in the progress file) after this file's write succeeds.
    """
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path, "a") as f:
        for x, z, coeff in zip(x_out.tolist(), z_out.tolist(), coeff_out.tolist()):
            f.write(json.dumps({"x": x, "z": z, "re": coeff.real, "im": coeff.imag}) + "\n")

    progress_path = _checkpoint_progress_path(checkpoint_path)
    with open(progress_path, "w") as f:
        json.dump({"next_chunk": next_chunk}, f)


def _iter_chunked_coefficients(
    operator,
    is_sparse_input: bool,
    active_x: NDArray[np.intp],
    inverse: NDArray[np.intp],
    p_nz: NDArray[np.intp],
    q_nz: NDArray[np.intp],
    dim: int,
    n_qubits: int,
    n_active: int,
    z_indices: NDArray[np.intp],
    chunk_size: int,
    atol: float,
    checkpoint_path: str | Path | None,
):
    """Generator over chunks of already-thresholded ``(x, z,
    coefficient)`` triples - the shared tile-producing core of the
    chunked path (PLAN.md Phase 9), used by both
    ``fwht_pauli_coefficients`` (which accumulates every tile into one
    COO triple) and ``fwht_pauli_terms_iter`` (which converts each
    tile to labels and yields it immediately - PLAN.md Phase 10). Each
    chunk is a fully independent sub-problem (divide-and-conquer: no
    cross-chunk combination step exists in the underlying math, unlike
    e.g. tiled matrix multiply's block-sum reduction - see PLAN.md
    Phase 10's design notes) - this generator is what keeps that
    independence visible to callers instead of re-fusing every tile
    before returning, which is the actual fix streaming needed.

    Yields ``(chunk_x, chunk_z, chunk_coeff)`` - three 1-D arrays of
    equal length, one triple per chunk, in chunk order.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    # inverse is sorted first so each chunk's nonzero entries
    # (p_nz[lo:hi], q_nz[lo:hi]) are a contiguous slice, found via
    # searchsorted on chunk boundaries - avoiding an O(nnz) boolean
    # mask per chunk.
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_p_nz = p_nz[order]
    sorted_q_nz = q_nz[order]

    chunk_starts = list(range(0, n_active, chunk_size))
    resume_from, checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is not None and resume_from > 0:
        # Replay already-completed chunks' triples from the checkpoint
        # file rather than recomputing them - the actual resume
        # behavior (see fwht_pauli_coefficients's checkpoint_path
        # docstring). Replayed as one combined "chunk" up front; a
        # caller streaming this (fwht_pauli_terms_iter) sees it as a
        # single larger tile rather than per-original-chunk history,
        # which is fine since the checkpoint file itself does not
        # preserve original chunk boundaries.
        yield checkpoint

    for chunk_index in range(resume_from, len(chunk_starts)):
        chunk_start = chunk_starts[chunk_index]
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
        chunk_coefficients = transformed_chunk * np.conj(phase) / dim

        # Threshold now, before accumulation - the space-complexity
        # fix (PLAN.md Phase 9): only surviving triples are ever held
        # for more than one chunk's lifetime.
        row_idx, z_idx = np.nonzero(np.abs(chunk_coefficients) > atol)
        chunk_x_out = active_x[chunk_start:chunk_end][row_idx]
        chunk_coeff_out = chunk_coefficients[row_idx, z_idx]

        if checkpoint_path is not None:
            _append_checkpoint_chunk(
                checkpoint_path, chunk_index + 1, chunk_x_out, z_idx, chunk_coeff_out
            )

        yield chunk_x_out, z_idx, chunk_coeff_out


def _prepare_operator_for_fwht(operator):
    """Shared validation/setup for ``fwht_pauli_coefficients`` and
    ``fwht_pauli_terms_iter``: shape/power-of-two checks, sparse-input
    detection and CSR conversion, and the XOR-index gather's raw
    nonzero-entry arrays (before deduplicating into ``active_x``,
    which each caller does slightly differently downstream).

    Returns ``(operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz)``.
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
    # which is why callers' gathers go through np.asarray(...).ravel().
    is_sparse_input = _sp is not None and _sp.issparse(operator)
    if is_sparse_input:
        # COO (e.g. from pad_to_power_of_two(..., sparse=True)) does
        # not support fancy indexing (operator[p_nz, q_nz]) - CSR
        # does, and np.nonzero() dispatches efficiently on CSR too.
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
    return operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz


def fwht_pauli_coefficients(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    sparse: bool = False,
    chunk_size: int | None = None,
    atol: float = 1e-10,
    checkpoint_path: str | Path | None = None,
) -> (
    NDArray[np.complexfloating]
    | tuple[NDArray[np.intp], NDArray[np.complexfloating]]
    | tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.complexfloating]]
):
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
            for moderate N; returns the dense-block form described
            below). If set to a positive integer, active rows are
            processed in blocks of at most ``chunk_size`` rows - the
            tiling technique from MIT 6.172 lecture 1 (block the
            computation to bound working-set size), applied here for
            memory-footprint reduction rather than cache reuse, since
            each row transforms independently (see
            ``_walsh_hadamard_transform_rows``) with no cross-row
            reduction to preserve. Unlike ``chunk_size=None``, this
            mode also thresholds each chunk's output against ``atol``
            immediately and accumulates only the surviving ``(x, z,
            coefficient)`` triples (see Returns) - bounding peak memory
            to ``O(chunk_size * dim + n_final_terms)`` rather than
            ``O(n_active * dim)``, since the earlier ``chunk_size``
            design (Phase 6) still allocated one full
            ``(n_active, dim)`` accumulator regardless of
            ``chunk_size`` (see PLAN.md Phase 9 - that accumulator was
            the actual N=150 ceiling, not the per-chunk transient
            ``chunk_size`` was designed to bound).
        atol: Only used when ``chunk_size`` is set. Coefficients with
            ``abs(coefficient) <= atol`` are dropped per-chunk, before
            accumulation - moved here (rather than staying purely in
            ``fwht_pauli_terms``) because thresholding must happen
            before accumulation for the space-complexity fix above to
            work; the dense-block modes (``chunk_size=None``) are
            unaffected and keep returning unthresholded output.
        checkpoint_path: Only used when ``chunk_size`` is set. If
            given, each completed chunk's surviving triples are
            appended to ``checkpoint_path`` (newline-delimited JSON)
            and a sibling ``<checkpoint_path>.progress.json`` file
            records the index of the next chunk to process. If a
            checkpoint already exists at this path when called, chunks
            already recorded there are skipped and their triples are
            read back rather than recomputed - resuming a crashed or
            interrupted run rather than restarting from chunk 0. This
            costs one small file append per chunk (negligible next to
            each chunk's O(chunk_size * dim * log dim) transform cost -
            see PLAN.md Phase 9), so it is opt-in but effectively free
            when enabled; ``None`` (default) does no I/O at all.

    Returns:
        If ``sparse=False``: a complex ``numpy.ndarray`` of shape
        ``(2**n, 2**n)`` where entry ``[x, z]`` is the coefficient of
        the Pauli string ``P(x, z)`` (see module docstring for the
        x/z encoding). Most entries will be exactly or near zero for
        structured/sparse input; callers that want only the nonzero
        terms should filter by magnitude (see ``fwht_pauli_terms``).

        If ``sparse=True`` and ``chunk_size=None``: a tuple
        ``(active_x, active_coefficients)`` where ``active_x`` is a
        1-D integer array of the ``x`` values with at least one
        nonzero coefficient, and ``active_coefficients`` is a complex
        array of shape ``(len(active_x), dim)`` such that
        ``active_coefficients[i, z]`` is the coefficient of
        ``P(active_x[i], z)``. Rows for ``x`` not in ``active_x`` are
        implicitly all-zero and are not represented at all.

        If ``sparse=True`` and ``chunk_size`` is a positive integer: a
        tuple ``(x_out, z_out, coefficient_out)`` of three 1-D arrays
        of equal length - a COO-style triple where entry ``i`` is the
        coefficient ``coefficient_out[i]`` of Pauli string
        ``P(x_out[i], z_out[i])``, already thresholded against
        ``atol`` (unlike the other two return forms).

    Raises:
        ValueError: If ``operator`` is not square or its dimension is
            not a power of two.
    """
    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(
        operator
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)

    z_indices = np.arange(dim)[np.newaxis, :]

    if sparse and chunk_size is not None:
        x_out = _GrowableArray(np.intp)
        z_out = _GrowableArray(np.intp)
        coeff_out = _GrowableArray(complex)
        for chunk_x_out, chunk_z_out, chunk_coeff_out in _iter_chunked_coefficients(
            operator, is_sparse_input, active_x, inverse, p_nz, q_nz, dim, n_qubits,
            n_active, z_indices, chunk_size, atol, checkpoint_path,
        ):
            x_out.extend(chunk_x_out)
            z_out.extend(chunk_z_out)
            coeff_out.extend(chunk_coeff_out)

        return x_out.finalize(), z_out.finalize(), coeff_out.finalize()

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
    checkpoint_path: str | Path | None = None,
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
            rows at once, then thresholds by ``atol`` here. A positive
            integer switches to the chunked, already-thresholded COO
            path (``atol`` is applied per-chunk inside
            ``fwht_pauli_coefficients`` in that case, not here - same
            numerical result, but memory stays bounded at large N; see
            PLAN.md Phase 9).
        checkpoint_path: Only meaningful when ``chunk_size`` is set -
            passed through to ``fwht_pauli_coefficients``, see its
            docstring for the resume behavior.

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

    if chunk_size is not None:
        # Already-thresholded COO triples - see fwht_pauli_coefficients's
        # chunk_size/atol docstring (PLAN.md Phase 9). No further
        # thresholding/gather needed here; that is the whole point of
        # this path.
        x_nonzero, z_nonzero, coefficient_values = fwht_pauli_coefficients(
            operator,
            sparse=True,
            chunk_size=chunk_size,
            atol=atol,
            checkpoint_path=checkpoint_path,
        )
    else:
        active_x, active_coefficients = fwht_pauli_coefficients(operator, sparse=True)
        # Re-scan only the active rows fwht_pauli_coefficients already
        # identified, never the full (dim, dim) array - see
        # profiling/cache_locality/README.md for why the dense re-scan
        # was a measured cache-locality/robustness problem (OOMs at
        # N=150).
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


def fwht_pauli_terms_iter(
    operator: NDArray[np.complexfloating] | NDArray[np.floating],
    chunk_size: int,
    atol: float = 1e-10,
    assume_hermitian: bool = True,
    checkpoint_path: str | Path | None = None,
    parallel_labels: bool = False,
) -> Iterator[dict[str, complex] | dict[str, float]]:
    """Streaming counterpart to ``fwht_pauli_terms`` - PLAN.md Phase
    10. Yields one ``dict`` of terms per chunk instead of building one
    combined ``dict`` for the whole operator.

    Why this exists (the actual problem, not just a memory
    workaround): each chunk of active rows is a fully independent
    sub-problem - there is no cross-chunk combination step anywhere in
    the underlying math (unlike, say, tiled matrix multiply's
    block-sum reduction). ``fwht_pauli_terms`` re-fuses every chunk's
    result into one dict before the caller ever sees it, which is an
    artificial recombination the math does not require, not a
    necessary step - the actual divide-and-conquer strategy for this
    problem is to keep each tile a tile all the way to the caller (see
    PLAN.md Phase 10's design notes). This matters beyond just fitting
    in memory: at N=150, the full combined result is ~134M terms
    (~4.3 GiB for the raw data alone, more once label strings and a
    single dict's hash-table overhead are added - see
    ``profiling/phase9/phase9_findings.md``), too large for a
    dict-returning API to handle at all on modest hardware, regardless
    of any per-chunk memory bound. A caller that only needs to, e.g.,
    write terms to disk, filter them, or fold them into a running sum
    never needs the full combined dict to exist at once.

    Args:
        operator: Same contract as ``fwht_pauli_terms``.
        chunk_size: Required (no default) - unlike
            ``fwht_pauli_terms``, there is no dense/whole-array mode
            here; streaming without chunking is not a meaningful
            combination (see PLAN.md Phase 10 design question 2).
        atol: Same as ``fwht_pauli_terms`` - applied per-chunk inside
            ``fwht_pauli_coefficients``.
        assume_hermitian: Same as ``fwht_pauli_terms``, but checked
            per-chunk rather than for the whole operator at once: a
            ``ValueError`` raised on chunk *k* means chunks
            ``0..k-1`` were already yielded to the caller before the
            error - see PLAN.md Phase 10 design question 4. This is a
            real behavior difference from ``fwht_pauli_terms``, which
            either returns a fully valid dict or raises before
            returning anything; a streaming caller that needs
            all-or-nothing Hermiticity validation should call
            ``fwht_pauli_terms`` (non-streaming) instead.
        checkpoint_path: Same as ``fwht_pauli_terms`` - passed through
            to ``fwht_pauli_coefficients``'s chunked accumulation
            internals for crash/resume (PLAN.md Phase 9). Note this
            checkpoints the underlying coefficient computation, not
            this generator's own iteration state - resuming a
            streaming consumer that was itself interrupted partway
            through consuming chunks means simply calling this
            function again with the same ``checkpoint_path``; already
            checkpointed chunks are replayed as one combined tile (see
            ``_iter_chunked_coefficients``), not re-yielded
            chunk-by-chunk in their original grouping.
        parallel_labels: If ``True``, uses the oneTBB-parallel label
            kernel (``pauli_label_batch_parallel``) per chunk instead
            of the serial kernel. Measured (PLAN.md Phase 10,
            ``profiling/phase10/tbb_labeling_n150_findings.md``) as a
            real ~1.1-1.4x wall-clock win at N=150-representative
            scale, at the cost of a modest cache-locality regression
            (cache-miss/stall percentages each rise a few points) -
            not a universal win, so left opt-in (default ``False``)
            rather than replacing the serial kernel outright.

    Yields:
        One ``dict`` per chunk, same value-type contract as
        ``fwht_pauli_terms`` (``float`` values if
        ``assume_hermitian=True``, ``complex`` otherwise). A chunk
        with no surviving terms above ``atol`` yields an empty dict,
        not a skipped chunk - callers that want to skip empty chunks
        should filter for truthiness themselves.

    Raises:
        ValueError: If ``operator`` is not square or its dimension is
            not a power of two (raised immediately, before the first
            chunk is yielded - this check does not depend on any
            chunk's data). Also raised mid-stream, after already
            yielding zero or more prior chunks, if
            ``assume_hermitian=True`` and a term in the current chunk
            has a non-negligible imaginary part - see the
            ``assume_hermitian`` parameter above.
    """
    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(
        operator
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    z_indices = np.arange(dim)[np.newaxis, :]

    for chunk_x, chunk_z, chunk_coeff in _iter_chunked_coefficients(
        operator, is_sparse_input, active_x, inverse, p_nz, q_nz, dim, n_qubits,
        n_active, z_indices, chunk_size, atol, checkpoint_path,
    ):
        labels = _pauli_label_batch(chunk_x, chunk_z, n_qubits, parallel=parallel_labels)

        if assume_hermitian:
            real_terms: dict[str, float] = {}
            for label, c in zip(labels, chunk_coeff.tolist()):
                if abs(c.imag) > max(atol, 1e-6 * abs(c)):
                    raise ValueError(
                        f"term {label!r} has non-negligible "
                        f"imaginary part {c.imag!r} - operator may not be Hermitian; "
                        "pass assume_hermitian=False to decompose it anyway"
                    )
                real_terms[label] = float(c.real)
            yield real_terms
        else:
            yield {
                label: complex(c) for label, c in zip(labels, chunk_coeff.tolist())
            }


_WARNED_NO_NATIVE = False


def _pauli_label_batch(
    x_indices: NDArray[np.integer],
    z_indices: NDArray[np.integer],
    n_qubits: int,
    parallel: bool = False,
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

    Args:
        parallel: If ``True`` and the native extension is available,
            uses ``pauli_label_batch_parallel`` (oneTBB-parallel)
            instead of the serial ``pauli_label_batch`` kernel. Real
            wall-clock win at large batch sizes (~1.1-1.4x measured at
            40M terms) at the cost of a modest cache-locality
            regression (cache-miss/stall percentages each rise a few
            points - see PLAN.md Phase 10 and
            ``profiling/phase10/tbb_labeling_n150_findings.md`` for the
            measured tradeoff) - not a universal win, so left opt-in
            rather than the default. Ignored (falls back to serial, or
            the pure-Python loop) if the native extension is
            unavailable - the ``parallel`` and native-availability
            questions are independent.
    """
    if _native is not None:
        x_masks = np.asarray(x_indices, dtype=np.uint32)
        z_masks = np.asarray(z_indices, dtype=np.uint32)
        if parallel:
            return _native.pauli_label_batch_parallel(x_masks, z_masks, n_qubits)
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
