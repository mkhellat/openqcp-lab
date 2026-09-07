# Array-Yielding Parallel Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `parallel_decompose_arrays` (yields `(x, z, coeff)` arrays per chunk) and `terms_from_arrays` (renders arrays to a label→coefficient dict), lifting the multi-core speedup ceiling from a measured 1.21× to a measured 2.19×.

**Architecture:** Two new public functions in `paulikit.algorithms.fwht`. `parallel_decompose_arrays` is a near-copy of `parallel_decompose`'s pool/drain machinery whose drain loop does a vectorized Hermiticity check instead of building ~91.6M Python `str` objects and dict entries. `terms_from_arrays` is the opt-in rendering step so callers who *do* want labels have a supported path. `parallel_decompose` is untouched.

**Tech Stack:** Python 3.12, NumPy, `concurrent.futures.ProcessPoolExecutor`, pytest, meson-python build backend.

**Spec:** `docs/superpowers/specs/2026-09-07-array-yielding-parallel-decompose-design.md`

## Global Constraints

- **Build/test only via the project's own system**: `make build` / `make test` from `tools/paulikit/`. A bare `pip install -e .` bakes in an ephemeral build-isolation NumPy include path and breaks subsequent rebuilds. `make test` takes no arguments, so targeted runs are written as `make build && <venv>/bin/pytest <args>` — the `make build` prefix is what keeps the compiled extension current; never invoke pytest without it after touching source.
- **Do not modify `parallel_decompose`.** Existing callers must be unaffected. Its dict contract stays capped at ~1.21× and that is correct.
- **Coefficients stay `complex128` across IPC.** Narrowing to the real part deletes the evidence the Hermiticity check needs.
- **Preserve both invariants**: cache-bound working set (`chunk_size` auto-tuning untouched) and bounded streaming memory (one chunk live at a time).
- **Atomic commits with full multi-paragraph bodies**, matching this repo's `git log` style. One logical change per commit. No `Co-Authored-By` trailer.
- **Push every commit to both remotes via proxychains**: `proxychains4 -q git push github main` and `proxychains4 -q git push origin main`. Never push unproxied.
- **Measurement protocol** (Task 7 only): foreground, thermal cooldown to 55 °C before every timed run, ≥5 reps, Welch's t-test. Never a single-run comparison.
- **`n_qubits` is `int(log2(dim))`** where `dim = operator.shape[0]`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/paulikit/algorithms/fwht.py` (modify) | Both new public functions plus a shared `_check_hermitian_violation` helper |
| `tests/test_array_yielding.py` (create) | All correctness/hermiticity/checkpoint/dtype tests for the new API |
| `profiling/phase13/arrays_vs_dict_sweep.py` (create) | The N=150 performance sweep |
| `src/paulikit/__init__.py` (modify) | Public API docstring list |
| `docs/api/*`, `docs/tutorial.md`, `PLAN.md` (modify) | Doc sweep |

New tests go in their own file rather than appending to `tests/test_parallel_decompose.py`, which is already 300+ lines covering a different function.

---

### Task 1: Shared Hermiticity-check helper

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py` (add helper next to `_build_real_terms`, ~line 746)
- Test: `tests/test_array_yielding.py` (create)

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `_check_hermitian_violation(coefficient_values: NDArray[np.complexfloating], atol: float, x: NDArray[np.integer], z: NDArray[np.integer], n_qubits: int) -> None` — raises `ValueError` naming the first offending term; returns `None` otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_array_yielding.py`:

```python
"""Tests for the array-yielding parallel decomposition API
(PLAN.md Phase 13): parallel_decompose_arrays and terms_from_arrays.

Correctness is anchored on two independent references: round-trip
equivalence with parallel_decompose (same machinery, different output
format) and agreement with fwht_pauli_terms (an independently derived
sequential path), so a bug shared by both parallel paths cannot pass.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _build_real_terms,
    _check_hermitian_violation,
    _pauli_label_batch,
    fwht_pauli_terms,
    parallel_decompose,
)
from paulikit.testing.fixtures import ALL_FIXTURES


def test_check_hermitian_violation_raises_naming_offending_term():
    # Two terms; only the second has a non-negligible imaginary part.
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.5j])

    with pytest.raises(ValueError, match="imaginary part"):
        _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2)


def test_check_hermitian_violation_message_matches_build_real_terms():
    # The array path has no labels at check time; it must still produce
    # a byte-identical message to the dict path's, by labeling ONLY the
    # single offending term.
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.5j])
    labels = _pauli_label_batch(x, z, 2)

    with pytest.raises(ValueError) as dict_err:
        _build_real_terms(labels, coeff, 1e-10)
    with pytest.raises(ValueError) as array_err:
        _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2)

    assert str(array_err.value) == str(dict_err.value)


def test_check_hermitian_violation_passes_for_hermitian_input():
    x = np.array([0, 1], dtype=np.uint16)
    z = np.array([0, 2], dtype=np.uint16)
    coeff = np.array([1.0 + 0.0j, 2.0 + 0.0j])

    assert _check_hermitian_violation(coeff, 1e-10, x, z, n_qubits=2) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: FAIL with `ImportError: cannot import name '_check_hermitian_violation'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/paulikit/algorithms/fwht.py`, immediately after `_build_real_terms`:

```python
def _check_hermitian_violation(
    coefficient_values: NDArray[np.complexfloating],
    atol: float,
    x: NDArray[np.integer],
    z: NDArray[np.integer],
    n_qubits: int,
) -> None:
    """Raise if any coefficient has a non-negligible imaginary part.

    The array-yielding path's counterpart to the check inside
    ``_build_real_terms`` - same tolerance rule, same error message,
    but without building a label for every term first. On violation it
    labels ONLY the single offending term, so the diagnostic is
    byte-identical at O(1) cost rather than O(t_i).

    The tolerance floor must match ``_build_real_terms`` exactly:
    ``abs(c)`` is the *full complex magnitude*, not ``abs(c.real)``
    (they only agree when the imaginary part is already negligible,
    which is exactly the case this check exists to catch).
    """
    c_abs = np.abs(coefficient_values)
    imag_abs = np.abs(coefficient_values.imag)
    violation = imag_abs > np.maximum(atol, 1e-6 * c_abs)
    if not violation.any():
        return
    first = int(np.nonzero(violation)[0][0])
    label = _pauli_label_batch(x[first:first + 1], z[first:first + 1], n_qubits)[0]
    c = coefficient_values[first]
    raise ValueError(
        f"term {label!r} has non-negligible "
        f"imaginary part {c.imag!r} - operator may not be Hermitian; "
        "pass assume_hermitian=False to decompose it anyway"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Mutation-check the guard**

Temporarily change `if not violation.any():` to `if True:`. Re-run.
Expected: the two raising tests FAIL. Revert the change and confirm they pass again. This proves the tests guard the behavior rather than passing vacuously.

- [ ] **Step 6: Commit**

```bash
git add src/paulikit/algorithms/fwht.py tests/test_array_yielding.py
git commit   # full body: why the array path needs its own check, and
             # why it labels only the offending term
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 2: `terms_from_arrays`

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py`
- Test: `tests/test_array_yielding.py`

**Interfaces:**
- Consumes: `_check_hermitian_violation` (Task 1)
- Produces: `terms_from_arrays(x, z, coeff, n_qubits, assume_hermitian=True, atol=1e-10) -> dict[str, complex] | dict[str, float]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_array_yielding.py`:

```python
def test_terms_from_arrays_matches_build_real_terms():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0, 1, 2], dtype=np.uint16)
    z = np.array([0, 2, 1], dtype=np.uint16)
    coeff = np.array([1.5 + 0.0j, -2.0 + 0.0j, 0.25 + 0.0j])

    expected = _build_real_terms(_pauli_label_batch(x, z, 2), coeff, 1e-10)
    assert terms_from_arrays(x, z, coeff, n_qubits=2) == expected


def test_terms_from_arrays_non_hermitian_raises():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0], dtype=np.uint16)
    z = np.array([0], dtype=np.uint16)
    coeff = np.array([1.0 + 0.5j])

    with pytest.raises(ValueError, match="imaginary part"):
        terms_from_arrays(x, z, coeff, n_qubits=2)


def test_terms_from_arrays_assume_hermitian_false_keeps_complex():
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([0], dtype=np.uint16)
    z = np.array([0], dtype=np.uint16)
    coeff = np.array([1.0 + 0.5j])

    result = terms_from_arrays(x, z, coeff, n_qubits=2, assume_hermitian=False)
    assert result == {"II": 1.0 + 0.5j}


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.intp])
def test_terms_from_arrays_accepts_any_integer_dtype(dtype):
    # Legacy checkpoints hold intp; new results hold uint16. Both must work.
    from paulikit.algorithms.fwht import terms_from_arrays

    x = np.array([1], dtype=dtype)
    z = np.array([2], dtype=dtype)
    coeff = np.array([1.0 + 0.0j])

    assert terms_from_arrays(x, z, coeff, n_qubits=2) == {"XZ": 1.0}


def test_terms_from_arrays_empty_input_returns_empty_dict():
    from paulikit.algorithms.fwht import terms_from_arrays

    empty_i = np.array([], dtype=np.uint16)
    empty_c = np.array([], dtype=complex)

    assert terms_from_arrays(empty_i, empty_i, empty_c, n_qubits=2) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: FAIL with `ImportError: cannot import name 'terms_from_arrays'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/paulikit/algorithms/fwht.py`, after `_check_hermitian_violation`:

```python
def terms_from_arrays(
    x: NDArray[np.integer],
    z: NDArray[np.integer],
    coeff: NDArray[np.complexfloating],
    n_qubits: int,
    assume_hermitian: bool = True,
    atol: float = 1e-10,
) -> dict[str, complex] | dict[str, float]:
    """Render one chunk's ``(x, z, coeff)`` arrays to a label -> coefficient dict.

    The opt-in counterpart to ``parallel_decompose_arrays`` (PLAN.md
    Phase 13): that function yields raw arrays so the ~91.6M Python
    ``str`` objects and dict insertions a full decomposition would
    otherwise need are never built in the parent process - which is
    what lifts the multi-core speedup ceiling from ~1.21x to a measured
    ~2.19x. This function is where a caller opts back IN to labels,
    for as many terms as they actually want.

    Building labels for every term of a large decomposition costs the
    same here as it does inside ``parallel_decompose``; the saving
    comes from calling this on a *subset* (filter by coefficient
    magnitude, take the largest terms, render one chunk) rather than
    on all of them.

    Args:
        x: Per-term x bitmasks, any integer dtype (``uint16`` from a
            fresh run, ``intp`` from a legacy checkpoint - both work).
        z: Per-term z bitmasks, same length and dtype rules as ``x``.
        coeff: Per-term complex coefficients.
        n_qubits: Number of qubits, i.e. ``int(log2(dim))``.
        assume_hermitian: If ``True`` (default), raises ``ValueError``
            when any coefficient has a non-negligible imaginary part
            and returns real coefficients - identical contract and
            identical error message to ``fwht_pauli_terms``. If
            ``False``, returns complex coefficients unchecked.
        atol: Tolerance floor for the Hermiticity check.

    Returns:
        ``dict[str, float]`` when ``assume_hermitian=True``, else
        ``dict[str, complex]``.
    """
    labels = _pauli_label_batch(x, z, n_qubits)
    if assume_hermitian:
        return _build_real_terms(labels, coeff, atol)
    return {label: complex(c) for label, c in zip(labels, coeff.tolist())}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/paulikit/algorithms/fwht.py tests/test_array_yielding.py
git commit   # full body: the opt-in rendering step; why the saving
             # comes from calling it on a subset, not from being faster
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 3: `parallel_decompose_arrays` — core generator

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py` (add after `parallel_decompose`, ~line 1710)
- Test: `tests/test_array_yielding.py`

**Interfaces:**
- Consumes: `_check_hermitian_violation` (Task 1), `terms_from_arrays` (Task 2)
- Produces: `parallel_decompose_arrays(operator, chunk_size=None, n_workers=None, atol=1e-10, assume_hermitian=True, checkpoint_path=None) -> Iterator[tuple[NDArray, NDArray, NDArray]]`

**Implementation note for the engineer:** copy `parallel_decompose`'s body verbatim (lines ~1530-1684) and change only the drain-loop body and the resume replay. Everything else — operator prep, `chunk_size`/`n_workers` auto-tuning, `max_in_flight` bounded submission, CPU pinning, checkpoint append — is identical and must not be re-derived.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_array_yielding.py`:

```python
def _combine_arrays(chunks, n_qubits):
    """Render every yielded chunk and merge, mirroring _combine in
    tests/test_parallel_decompose.py."""
    from paulikit.algorithms.fwht import terms_from_arrays

    combined = {}
    for x, z, coeff in chunks:
        combined.update(terms_from_arrays(x, z, coeff, n_qubits))
    return combined


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_parallel_decompose_arrays_matches_fwht_pauli_terms(fixture, chunk_size):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))
    reference = fwht_pauli_terms(padded)

    combined = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=chunk_size, n_workers=2),
        n_qubits,
    )

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_round_trip_equals_parallel_decompose(fixture):
    # The correctness anchor: rendering the array path must reproduce
    # the dict path exactly, term for term.
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))

    dict_terms = {}
    for chunk in parallel_decompose(padded, chunk_size=2, n_workers=2):
        dict_terms.update(chunk)
    array_terms = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=2, n_workers=2), n_qubits
    )

    assert array_terms == dict_terms


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_parallel_decompose_arrays_no_duplicate_terms(fixture):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = fixture.padded_hamiltonian()
    seen = set()
    total = 0
    for x, z, _coeff in parallel_decompose_arrays(padded, chunk_size=2, n_workers=2):
        for xi, zi in zip(x.tolist(), z.tolist()):
            seen.add((xi, zi))
            total += 1

    assert total == len(seen), "a term was yielded more than once"


def test_parallel_decompose_arrays_yields_three_arrays_of_equal_length():
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    for chunk in parallel_decompose_arrays(padded, chunk_size=2, n_workers=2):
        assert len(chunk) == 3
        x, z, coeff = chunk
        assert len(x) == len(z) == len(coeff)
        assert np.iscomplexobj(coeff), "coefficients must stay complex"


def test_parallel_decompose_arrays_non_hermitian_raises():
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    operator = np.zeros((4, 4), dtype=complex)
    operator[0, 0] = 1.0 + 0.5j

    with pytest.raises(ValueError, match="imaginary part"):
        list(parallel_decompose_arrays(operator, chunk_size=2, n_workers=2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: FAIL with `ImportError: cannot import name 'parallel_decompose_arrays'`

- [ ] **Step 3: Write minimal implementation**

Add `parallel_decompose_arrays` to `src/paulikit/algorithms/fwht.py` after `parallel_decompose`. Copy that function's body, then make exactly these two changes:

Resume replay becomes:

```python
    completed_indices, checkpoint = _load_parallel_checkpoint(checkpoint_path)
    if checkpoint is not None:
        if assume_hermitian:
            _check_hermitian_violation(
                checkpoint[2], atol, checkpoint[0], checkpoint[1], n_qubits
            )
        yield checkpoint
```

Drain-loop body becomes:

```python
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()
                _submit_next()

                if checkpoint_path is not None:
                    _append_parallel_checkpoint_chunk(
                        checkpoint_path, completed_indices, chunk_index,
                        chunk_x_out, z_idx, chunk_coeff_out,
                    )

                if assume_hermitian:
                    _check_hermitian_violation(
                        chunk_coeff_out, atol, chunk_x_out, z_idx, n_qubits
                    )
                yield chunk_x_out, z_idx, chunk_coeff_out
```

Give it this docstring:

```python
    """Multi-core decomposition yielding raw ``(x, z, coeff)`` arrays -
    PLAN.md Phase 13.

    Identical machinery to ``parallel_decompose`` (same chunking, same
    auto-tuning, same bounded submission, same CPU pinning, same
    checkpoint format) with one difference: it yields each chunk's raw
    arrays instead of building a ``dict[str, complex]`` from them.

    That difference is the whole point. Building ~91.6M Python ``str``
    objects and dict entries at N=150 is ~82% of ``parallel_decompose``'s
    total runtime, all of it in the single parent process, which caps
    its speedup at ~1.21x no matter how many cores are available
    (measured: best-ever 1.284x, and 8 workers gives 1.100x). Removing
    that work from the drain loop was measured to restore real
    multi-core scaling - 2.191x, statistically indistinguishable from a
    control doing no drain-side work at all. See
    ``profiling/phase13/drain_gil_backpressure_results.jsonl``.

    Use ``terms_from_arrays`` to render any chunk (or a filtered subset
    of one) to the usual label -> coefficient dict.

    Yields:
        ``(x, z, coeff)`` per completed chunk: two integer arrays of
        symplectic bitmasks and one ``complex128`` coefficient array,
        all the same length. **Order is not guaranteed to match chunk
        order** - same contract as ``parallel_decompose``. A chunk with
        no surviving terms yields three empty arrays rather than being
        skipped, so chunk count is stable.

    Raises:
        ValueError: If ``assume_hermitian=True`` and any coefficient
            has a non-negligible imaginary part. Checked per chunk, so
            this can raise *after* earlier chunks have been yielded -
            the same partial-yield-then-error contract
            ``fwht_pauli_terms_iter`` documents. Coefficients are kept
            ``complex128`` across the process boundary precisely so
            this check remains possible.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: PASS (all new tests)

- [ ] **Step 5: Run the full suite for regressions**

Run: `make test`
Expected: 146 pre-existing tests still pass, plus the new ones.

- [ ] **Step 6: Commit**

```bash
git add src/paulikit/algorithms/fwht.py tests/test_array_yielding.py
git commit   # full body: the 82% serial fraction, the 1.21x Amdahl
             # ceiling, and the measured 2.191x recovery
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 4: Checkpoint interop and resume

**Files:**
- Test: `tests/test_array_yielding.py`

**Interfaces:**
- Consumes: `parallel_decompose_arrays` (Task 3), `terms_from_arrays` (Task 2)
- Produces: no new API — verifies the shared on-disk format the spec claims

- [ ] **Step 1: Write the failing test**

Append to `tests/test_array_yielding.py`:

```python
def test_checkpoint_written_by_dict_path_resumes_under_array_path(tmp_path):
    # Both functions must share one on-disk format. Asserted, not assumed.
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    n_qubits = int(np.log2(padded.shape[0]))
    reference = fwht_pauli_terms(padded)
    checkpoint = tmp_path / "ckpt.jsonl"

    # Consume only the first chunk from the dict path, leaving a
    # partial checkpoint on disk.
    gen = parallel_decompose(padded, chunk_size=2, n_workers=2,
                             checkpoint_path=checkpoint)
    next(gen)
    gen.close()

    combined = _combine_arrays(
        parallel_decompose_arrays(padded, chunk_size=2, n_workers=2,
                                  checkpoint_path=checkpoint),
        n_qubits,
    )

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_checkpoint_written_by_array_path_resumes_under_dict_path(tmp_path):
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    padded = ALL_FIXTURES[1].padded_hamiltonian()
    reference = fwht_pauli_terms(padded)
    checkpoint = tmp_path / "ckpt.jsonl"

    gen = parallel_decompose_arrays(padded, chunk_size=2, n_workers=2,
                                    checkpoint_path=checkpoint)
    next(gen)
    gen.close()

    combined = {}
    for chunk in parallel_decompose(padded, chunk_size=2, n_workers=2,
                                    checkpoint_path=checkpoint):
        combined.update(chunk)

    assert set(combined) == set(reference)


def test_resume_replay_still_checks_hermiticity(tmp_path):
    # The replay path is separate code from the drain loop; it is easy
    # to add the check to one and forget the other.
    from paulikit.algorithms.fwht import parallel_decompose_arrays

    checkpoint = tmp_path / "ckpt.jsonl"
    progress = tmp_path / "ckpt.jsonl.parallel_progress.json"
    # A hand-written checkpoint holding one non-Hermitian term.
    checkpoint.write_text('{"x": 0, "z": 0, "re": 1.0, "im": 0.5}\n')
    progress.write_text('{"completed_chunk_indices": [0]}')

    operator = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="imaginary part"):
        list(parallel_decompose_arrays(operator, chunk_size=2, n_workers=2,
                                       checkpoint_path=checkpoint))
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v -k "checkpoint or resume"`

If the checkpoint tests pass immediately, that confirms the spec's claim that the format is already shared — record that in the commit body. If `test_resume_replay_still_checks_hermiticity` fails, the check is missing from the replay path in Task 3; fix it there.

**Verify the progress-file suffix before running:** confirm `_parallel_checkpoint_progress_path` produces `ckpt.jsonl.parallel_progress.json` and correct the literal filename above if it differs.

- [ ] **Step 3: Fix any failure, then re-run**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_array_yielding.py src/paulikit/algorithms/fwht.py
git commit   # full body: state whether interop worked out of the box
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 5: Export the new API

**Files:**
- Modify: `src/paulikit/__init__.py` (Public API docstring list, ~line 16-24)
- Test: `tests/test_array_yielding.py`

**Interfaces:**
- Consumes: both public functions (Tasks 2, 3)
- Produces: no new code — documentation surface only

- [ ] **Step 1: Write the failing test**

```python
def test_new_api_listed_in_package_docstring():
    import paulikit

    assert "parallel_decompose_arrays" in paulikit.__doc__
    assert "terms_from_arrays" in paulikit.__doc__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py::test_new_api_listed_in_package_docstring -v`
Expected: FAIL on the first assert.

- [ ] **Step 3: Add both names to the Public API list**

In `src/paulikit/__init__.py`, add after the `auto_decompose` line:

```
    paulikit.algorithms.fwht.parallel_decompose_arrays
    paulikit.algorithms.fwht.terms_from_arrays
```

Note `parallel_decompose` is itself absent from that list today; add it too, since omitting it while listing its sibling would be incoherent.

- [ ] **Step 4: Run test to verify it passes**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_array_yielding.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paulikit/__init__.py tests/test_array_yielding.py
git commit
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 6: Documentation sweep

**Files:**
- Modify: `PLAN.md` (Phase 13 status)
- Modify: `docs/tutorial.md`
- Modify: `docs/api/` (whichever file documents `fwht`)
- Modify: `profiling/phase13/README.md`

**Interfaces:**
- Consumes: everything above
- Produces: no code

- [ ] **Step 1: Inspect what exists**

Run: `ls docs/api/` and `grep -rn "parallel_decompose" docs/ PLAN.md profiling/phase13/README.md`
Identify every place that describes the parallel API and would now be incomplete.

- [ ] **Step 2: Update `PLAN.md` Phase 13**

Record: the 0.824 serial fraction and 1.21× Amdahl ceiling as the measured reason `parallel_decompose` does not scale; the new functions as the fix; the measured 2.191× from the `arrays_only` control; and that the dict path remains supported and capped.

- [ ] **Step 3: Add a tutorial section**

Show the array API and when to prefer it:

```python
from paulikit.algorithms.fwht import parallel_decompose_arrays, terms_from_arrays

n_qubits = int(np.log2(padded.shape[0]))
for x, z, coeff in parallel_decompose_arrays(padded):
    big = np.abs(coeff) > 1e-3          # keep only what you need
    terms = terms_from_arrays(x[big], z[big], coeff[big], n_qubits)
```

- [ ] **Step 4: Build the docs**

Run: `cd docs && make html` (or the documented command)
Expected: no new warnings about undocumented/unreferenced names.

- [ ] **Step 5: Commit**

```bash
git add PLAN.md docs/ profiling/phase13/README.md
git commit
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

---

### Task 7: N=150 performance verification

**Files:**
- Create: `profiling/phase13/arrays_vs_dict_sweep.py`
- Create: `profiling/phase13/arrays_vs_dict_results.jsonl` (force-add; `*_results.jsonl` is gitignored)

**Interfaces:**
- Consumes: `parallel_decompose_arrays` (Task 3)
- Produces: the measured answer to whether 2.191× survives in the real pipeline

- [ ] **Step 1: Write the sweep script**

Model it on `profiling/phase13/narrowed_index_dtype_sweep.py`: cooldown to 55 °C before every run, incremental JSON-Lines with resume-skip, Welch's t-test. Cells: `{dict, arrays} × {w2_c1, w8_c4}`, 5 reps, N=150, `chunk_size=2`. Record `elapsed`, `total_terms`, `peak_rss_mib`, and temperatures.

Run **two variants of the arrays cell**: with and without `checkpoint_path`, since checkpoint writing is per-chunk parent-side I/O and is the most likely reason the control's 2.191× would not survive.

- [ ] **Step 2: Confirm the machine is idle**

Run: `pgrep -af "python.*sweep|full_matrix_target"`
Expected: no measurement jobs. Never run two concurrently.

- [ ] **Step 3: Run the sweep in the foreground**

Run: `OPENBLAS_NUM_THREADS=1 python profiling/phase13/arrays_vs_dict_sweep.py`

- [ ] **Step 4: Verify correctness before reading any timing**

Every run must report `total_terms=91652096`. A second value means the array path lost or duplicated terms — stop and fix rather than reporting timings.

- [ ] **Step 5: Write findings**

Create `profiling/phase13/arrays_vs_dict_findings.md` reporting the w2_c1→w8_c4 speedup for each cell, Welch p-values, and peak RSS. **Report the result honestly whichever way it lands.** If the speedup falls well short of 2.191×, say so plainly and state the ship/no-ship question for the user rather than reframing a shortfall as a win. If checkpointing costs a measurable amount, quantify it.

- [ ] **Step 6: Commit**

```bash
git add profiling/phase13/arrays_vs_dict_sweep.py \
        profiling/phase13/arrays_vs_dict_findings.md
git add -f profiling/phase13/arrays_vs_dict_results.jsonl
git commit
proxychains4 -q git push github main
proxychains4 -q git push origin main
```

- [ ] **Step 7: Update the memory checkpoint**

Append a Phase 13 checkpoint to the project memory recording the outcome, **including any failed predictions**.

---

## Self-Review

**Spec coverage:** Both functions (Tasks 2, 3); Hermiticity in both paths incl. resume replay (Tasks 1, 3, 4); error-message parity (Task 1); checkpoint interop (Task 4); all 12 planned tests distributed across Tasks 1-4; performance sweep with the two flagged questions (Task 7); doc sweep (Tasks 5, 6); atomic commits and dual push (every task).

**Placeholders:** none — every code step contains real code.

**Type consistency:** `_check_hermitian_violation(coeff, atol, x, z, n_qubits)` is called with that argument order in Tasks 1, 3, 4. `terms_from_arrays(x, z, coeff, n_qubits, ...)` is consistent in Tasks 2, 3, 4, 6. `parallel_decompose_arrays` yields a 3-tuple throughout.

**Known gap, deliberate:** the spec's "empty chunk yields three empty arrays" is asserted structurally (Task 3, Step 1) but not with a fixture guaranteed to produce an empty chunk, since no such fixture is known to exist. If one is found during implementation, add the direct test.
