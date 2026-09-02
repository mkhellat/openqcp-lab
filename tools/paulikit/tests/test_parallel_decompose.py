"""Tests for PLAN.md Phase 13a's multi-core chunk parallelism
(paulikit.algorithms.fwht.parallel_decompose).

Correctness is checked against fwht_pauli_terms (the known-correct
sequential dense path) on every fixture, same discipline as Phase
12's auto_decompose tests - a parallel implementation that merely
"runs" without a real-output correctness check would not actually
verify the divide-and-conquer decomposition is right.
"""

import json
import os

import pytest

from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import (
    _detect_available_worker_count,
    fwht_pauli_terms,
    parallel_decompose,
)
from paulikit.testing.fixtures import ALL_FIXTURES


def _combine(chunks):
    combined = {}
    for chunk_dict in chunks:
        combined.update(chunk_dict)
    return combined


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_parallel_decompose_matches_fwht_pauli_terms(fixture, chunk_size):
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded, chunk_size=chunk_size, n_workers=2))

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_parallel_decompose_single_worker_matches_sequential():
    # n_workers=1 should give identical results to the multi-worker
    # case (and to fwht_pauli_terms) - sanity check that the pool
    # infrastructure itself introduces no numerical difference.
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded, chunk_size=2, n_workers=1))
    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_parallel_decompose_default_chunk_size_and_workers_run(monkeypatch):
    # chunk_size=None/n_workers=None should fall through to the
    # autotune formula / _detect_available_worker_count without
    # erroring - forced to small deterministic values here so the
    # test doesn't depend on this machine's real hardware.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 2)
    monkeypatch.setattr(
        "paulikit.algorithms.fwht._detect_available_worker_count", lambda: 2
    )
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded))
    assert set(combined) == set(reference)


def test_parallel_decompose_clamps_worker_count_to_chunk_count():
    # More workers requested than chunks exist should not error - the
    # pool is just over-provisioned, clamped internally.
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded, chunk_size=100, n_workers=64))
    assert set(combined) == set(reference)


def test_parallel_decompose_checkpoint_resume(tmp_path):
    fixture = ALL_FIXTURES[1]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    ckpt = tmp_path / "ckpt.jsonl"
    progress = tmp_path / "ckpt.jsonl.parallel_progress.json"

    gen = parallel_decompose(padded, chunk_size=2, n_workers=2, checkpoint_path=str(ckpt))
    next(gen)  # consume exactly one chunk, then abandon the generator
    gen.close()

    assert progress.exists()
    with open(progress) as f:
        first_progress = json.load(f)
    assert len(first_progress["completed_chunk_indices"]) == 1

    combined = _combine(
        parallel_decompose(padded, chunk_size=2, n_workers=2, checkpoint_path=str(ckpt))
    )
    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_parallel_decompose_checkpoint_uses_distinct_file_suffix(tmp_path):
    # The parallel checkpoint format must never collide with the
    # sequential one (fwht_pauli_terms_iter's _checkpoint_progress_path)
    # if a caller reuses the same checkpoint_path between the two.
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    ckpt = tmp_path / "shared.jsonl"

    list(parallel_decompose(padded, chunk_size=2, n_workers=2, checkpoint_path=str(ckpt)))

    assert (tmp_path / "shared.jsonl.parallel_progress.json").exists()
    assert not (tmp_path / "shared.jsonl.progress.json").exists()


def test_parallel_decompose_already_complete_checkpoint_yields_nothing_new(tmp_path):
    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    ckpt = tmp_path / "ckpt.jsonl"

    list(parallel_decompose(padded, chunk_size=2, n_workers=2, checkpoint_path=str(ckpt)))
    # Second call with the same (now-complete) checkpoint should yield
    # exactly one replay chunk (the checkpoint) and submit no new work.
    results = list(
        parallel_decompose(padded, chunk_size=2, n_workers=2, checkpoint_path=str(ckpt))
    )
    assert len(results) == 1


def test_per_worker_memory_budget_bytes_divides_evenly(monkeypatch):
    monkeypatch.setattr(autotune, "_cached_memory_budget_bytes", 8 * 1024**3)
    assert autotune.per_worker_memory_budget_bytes(4) == 2 * 1024**3


def test_per_worker_memory_budget_bytes_rejects_invalid_worker_count():
    with pytest.raises(ValueError):
        autotune.per_worker_memory_budget_bytes(0)


def test_detect_available_worker_count_uses_sched_getaffinity_not_cpu_count(monkeypatch):
    # Regression test for the cpuset-correctness fix (PLAN.md Phase
    # 13, same bug class Phase 12 fixed for memory): must prefer
    # sched_getaffinity over cpu_count when both are available.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert _detect_available_worker_count() == 3


def test_detect_available_worker_count_falls_back_to_cpu_count(monkeypatch):
    def raise_attr_error(pid):
        raise AttributeError

    monkeypatch.setattr(os, "sched_getaffinity", raise_attr_error, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert _detect_available_worker_count() == 4
