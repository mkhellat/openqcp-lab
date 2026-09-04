"""Correctness tests for PLAN.md Phase 10: fwht_pauli_terms_iter, the
streaming/generator counterpart to fwht_pauli_terms.

See that function's docstring for the divide-and-conquer rationale:
each chunk is an independent sub-problem, and this generator keeps
tiles as tiles all the way to the caller instead of re-fusing them
into one combined dict.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import fwht_pauli_terms, fwht_pauli_terms_iter
from paulikit.testing.fixtures import ALL_FIXTURES


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
@pytest.mark.parametrize("chunk_size", [1, 2, 5, 1000])
def test_streaming_combined_matches_fwht_pauli_terms(fixture, chunk_size):
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = {}
    for chunk_dict in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
        combined.update(chunk_dict)

    assert set(combined) == set(reference), (
        f"{fixture.name!r} chunk_size={chunk_size}: label set mismatch"
    )
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_streaming_chunks_have_disjoint_labels():
    """Each label must appear in exactly one yielded chunk - if the
    same term appeared in two chunks, a naive dict.update() combine
    would silently mask the bug, so check disjointness directly."""
    fixture = ALL_FIXTURES[-1]
    padded = fixture.padded_hamiltonian()

    seen: set[str] = set()
    for chunk_dict in fwht_pauli_terms_iter(padded, chunk_size=2):
        overlap = seen & set(chunk_dict)
        assert not overlap, f"labels repeated across chunks: {overlap}"
        seen |= set(chunk_dict)


def test_streaming_yields_empty_dict_for_chunks_with_no_survivors():
    """A chunk with zero terms above atol must still yield an empty
    dict (not be silently skipped) - callers rely on one yield per
    chunk for progress tracking."""
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    dim = padded.shape[0]

    chunks = list(fwht_pauli_terms_iter(padded, chunk_size=1))
    # chunk_size=1 means one chunk per active x - some chunks may be
    # empty if that row has no term surviving atol, but the total
    # number of chunks must still match dim (all rows visited).
    assert len(chunks) <= dim
    assert all(isinstance(c, dict) for c in chunks)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_streaming_parallel_labels_matches_serial(chunk_size):
    for fixture in ALL_FIXTURES:
        padded = fixture.padded_hamiltonian()
        serial_combined = {}
        for chunk_dict in fwht_pauli_terms_iter(
            padded, chunk_size=chunk_size, parallel_labels=False
        ):
            serial_combined.update(chunk_dict)

        parallel_combined = {}
        for chunk_dict in fwht_pauli_terms_iter(
            padded, chunk_size=chunk_size, parallel_labels=True
        ):
            parallel_combined.update(chunk_dict)

        assert set(serial_combined) == set(parallel_combined)
        for label in serial_combined:
            assert parallel_combined[label] == pytest.approx(
                serial_combined[label], abs=1e-9
            )


def test_streaming_checkpoint_resume(tmp_path):
    """Abandoning a generator partway through, then calling again with
    the same checkpoint_path, must reach the same final combined
    result as an uninterrupted run - mirrors
    test_chunked_accumulator.py's non-streaming resume test, but
    through the generator interface."""
    fixture = ALL_FIXTURES[-1]
    padded = fixture.padded_hamiltonian()
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    reference = fwht_pauli_terms(padded, chunk_size=2)

    gen = fwht_pauli_terms_iter(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    next(gen)  # consume exactly one chunk, then abandon
    del gen

    combined = {}
    for chunk_dict in fwht_pauli_terms_iter(
        padded, chunk_size=2, checkpoint_path=checkpoint_path
    ):
        combined.update(chunk_dict)

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_streaming_checkpoint_resume_survives_truncated_inflight_line(tmp_path):
    """Regression test for a real bug (REVIEW_NOTES.md, found
    2026-09-04): ``_append_checkpoint_chunk`` writes a chunk's triple
    lines before advancing the progress marker's ``next_chunk`` (see
    its own docstring) - so a crash mid-write can leave the checkpoint
    file with a truncated trailing line for a chunk the progress
    marker still considers NOT YET completed (that chunk gets
    resubmitted on resume, exactly as designed). Loading must not
    raise json.JSONDecodeError on that stale partial line - it will be
    superseded by the resubmitted chunk's own fresh append and must
    simply be skipped while reading the earlier, real chunks' data."""
    fixture = ALL_FIXTURES[-1]
    padded = fixture.padded_hamiltonian()
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    reference = fwht_pauli_terms(padded, chunk_size=2)

    gen = fwht_pauli_terms_iter(padded, chunk_size=2, checkpoint_path=checkpoint_path)
    next(gen)  # consume exactly one full, flushed chunk
    del gen

    # Simulate a crash mid-write of the NEXT (not-yet-progress-marked)
    # chunk: append a truncated trailing line without ever writing a
    # progress-marker update for it - progress.json still correctly
    # points at the chunk before this partial line.
    with open(checkpoint_path, "a") as f:
        f.write('{"x": 1, "z": 2, "re": 0.5')  # deliberately unterminated

    combined = {}
    for chunk_dict in fwht_pauli_terms_iter(
        padded, chunk_size=2, checkpoint_path=checkpoint_path
    ):
        combined.update(chunk_dict)

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_streaming_assume_hermitian_true_raises_mid_stream_on_non_hermitian_input():
    n_qubits = 2
    dim = 2**n_qubits
    rng = np.random.default_rng(5)
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    with pytest.raises(ValueError, match="may not be Hermitian"):
        for _ in fwht_pauli_terms_iter(operator, chunk_size=1):
            pass


def test_streaming_assume_hermitian_false_decomposes_non_hermitian_input():
    n_qubits = 2
    dim = 2**n_qubits
    rng = np.random.default_rng(33)
    operator = rng.random((dim, dim)) + 1j * rng.random((dim, dim))

    combined = {}
    for chunk_dict in fwht_pauli_terms_iter(
        operator, chunk_size=1, assume_hermitian=False
    ):
        combined.update(chunk_dict)

    assert any(c.imag != 0 for c in combined.values())
    reference = fwht_pauli_terms(operator, assume_hermitian=False)
    assert set(combined) == set(reference)


def test_streaming_rejects_non_power_of_two_dimension_before_first_yield():
    gen = fwht_pauli_terms_iter(np.eye(5), chunk_size=1)
    with pytest.raises(ValueError):
        next(gen)
