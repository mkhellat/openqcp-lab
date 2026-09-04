"""Tests for PLAN.md Phase 13a's multi-core chunk parallelism
(paulikit.algorithms.fwht.parallel_decompose).

Correctness is checked against fwht_pauli_terms (the known-correct
sequential dense path) on every fixture, same discipline as Phase
12's auto_decompose tests - a parallel implementation that merely
"runs" without a real-output correctness check would not actually
verify the divide-and-conquer decomposition is right.
"""

import builtins
import io
import json
import os

import pytest

from paulikit.algorithms import autotune
import numpy as np

from paulikit.algorithms.fwht import (
    _detect_available_worker_count,
    _per_worker_resident_bytes,
    _physical_core_representative_cpus,
    _recommended_parallel_chunk_size,
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


def test_recommended_parallel_chunk_size_respects_per_worker_memory_budget(monkeypatch):
    # Regression test for a real bug (found by the user noticing
    # memory spikes vs. the non-parallel chunked path, same day this
    # was first implemented): auto chunk_size must not just inherit
    # Phase 12's single-process cache-locality formula unchanged - up
    # to n_workers chunks are live simultaneously here, so it must
    # also respect each worker's *share* of the memory budget, not
    # the whole budget.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 999999)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 512 * 1024 * 1024)

    dim = 16384  # N=150's real dim
    n_workers = 4
    chunk_size = _recommended_parallel_chunk_size(dim, n_workers)

    expected_worker_budget = (512 * 1024 * 1024) // n_workers
    expected_max_chunk_size = expected_worker_budget // (dim * 16)
    assert chunk_size == expected_max_chunk_size
    assert chunk_size < 999999, (
        "the memory bound must actually clamp the cache-driven value down, "
        "not just be computed and ignored"
    )


def test_recommended_parallel_chunk_size_more_workers_means_smaller_chunks(monkeypatch):
    # More concurrent workers -> smaller per-worker memory share ->
    # smaller chunk_size, when the memory bound (not the cache bound)
    # is the binding constraint.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 999999)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 512 * 1024 * 1024)

    dim = 16384
    cs_4_workers = _recommended_parallel_chunk_size(dim, n_workers=4)
    cs_8_workers = _recommended_parallel_chunk_size(dim, n_workers=8)
    assert cs_8_workers < cs_4_workers


def test_recommended_parallel_chunk_size_uses_cache_bound_when_smaller(monkeypatch):
    # When the cache-driven value is already smaller than the memory
    # bound, it should win unchanged - the memory bound is a ceiling,
    # not a floor or a replacement.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 2)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 2**40)  # huge

    assert _recommended_parallel_chunk_size(dim=16384, n_workers=4) == 2


def test_recommended_parallel_chunk_size_subtracts_fixed_resident_bytes(monkeypatch):
    # Regression test for a real bug (REVIEW_NOTES.md 2026-09-04):
    # _recommended_parallel_chunk_size only bounded the per-chunk
    # transient buffer against a worker's memory share, never
    # subtracting the fixed operator-copy + setup-array footprint
    # every worker also holds resident - silently overestimating how
    # much budget is actually left for chunk_size.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 999999)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 512 * 1024 * 1024)

    dim = 16384
    n_workers = 4
    worker_budget = (512 * 1024 * 1024) // n_workers
    fixed_bytes = worker_budget // 2  # a real, non-negligible fixed footprint

    without_fixed = _recommended_parallel_chunk_size(dim, n_workers)
    with_fixed = _recommended_parallel_chunk_size(dim, n_workers, fixed_bytes)
    assert with_fixed < without_fixed, (
        "a nonzero fixed_resident_bytes must shrink the recommended "
        "chunk_size relative to ignoring it entirely"
    )

    expected = max(1, (worker_budget - fixed_bytes) // (dim * 16))
    assert with_fixed == expected


def test_recommended_parallel_chunk_size_never_goes_below_one_when_fixed_bytes_exceed_budget(
    monkeypatch,
):
    # If the fixed resident footprint alone already exceeds a
    # worker's share of the budget, the memory-bound chunk_size must
    # clamp to 1 (still make progress, correctness over throughput),
    # not go negative or raise.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 999999)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 1024)

    dim = 16384
    n_workers = 1
    huge_fixed_bytes = 10 * 1024 * 1024  # far exceeds the whole budget
    assert _recommended_parallel_chunk_size(dim, n_workers, huge_fixed_bytes) == 1


def test_per_worker_resident_bytes_dense_matches_operator_nbytes_plus_setup_arrays():
    dim = 8
    operator = np.zeros((dim, dim), dtype=complex)
    operator[0, 1] = 1.0
    operator[2, 3] = 1.0
    nnz = 2

    result = _per_worker_resident_bytes(operator, is_sparse_input=False, nnz=nnz)
    expected = operator.nbytes + 3 * nnz * np.dtype(np.intp).itemsize
    assert result == expected


def test_per_worker_resident_bytes_sparse_uses_csr_buffers_not_dense_equivalent():
    pytest.importorskip("scipy")
    import scipy.sparse as sp

    dim = 8
    dense = np.zeros((dim, dim), dtype=complex)
    dense[0, 1] = 1.0
    dense[2, 3] = 1.0
    operator = sp.csr_matrix(dense)
    nnz = 2

    result = _per_worker_resident_bytes(operator, is_sparse_input=True, nnz=nnz)
    dense_equivalent = dim * dim * 16
    assert result < dense_equivalent, (
        "sparse footprint must be O(nnz), not O(dim**2) - the whole "
        "point of the sparse input path"
    )
    expected = (
        operator.data.nbytes
        + operator.indices.nbytes
        + operator.indptr.nbytes
        + 3 * nnz * np.dtype(np.intp).itemsize
    )
    assert result == expected


def test_parallel_decompose_auto_chunk_size_stays_correct_under_a_tight_memory_budget(
    monkeypatch,
):
    # End-to-end: force a tiny per-worker memory budget (so chunk_size
    # is clamped far below the cache-driven value) and confirm the
    # result is still exactly correct, not just non-crashing.
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 999999)
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 512 * 1024 * 1024)
    monkeypatch.setattr(
        "paulikit.algorithms.fwht._detect_available_worker_count", lambda: 4
    )

    fixture = ALL_FIXTURES[0]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded))
    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


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


def test_physical_core_representative_cpus_groups_hyperthread_siblings(monkeypatch, tmp_path):
    # Simulate a 2-physical-core, 4-logical-CPU hyperthreaded machine:
    # (0,2) share one physical core, (1,3) share another.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)

    siblings = {0: "0,2", 1: "1,3", 2: "0,2", 3: "1,3"}
    real_open = open

    def fake_open(path, *args, **kwargs):
        for cpu, sib in siblings.items():
            marker = f"/cpu{cpu}/topology/thread_siblings_list"
            if marker in str(path):
                return io.StringIO(sib)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    result = _physical_core_representative_cpus()
    assert result == [0, 1], "expected one representative CPU per physical-core sibling group"


def test_physical_core_representative_cpus_returns_none_when_sysfs_unavailable(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)

    def raise_oserror(*args, **kwargs):
        raise OSError

    monkeypatch.setattr(builtins, "open", raise_oserror)
    assert _physical_core_representative_cpus() is None


def test_physical_core_representative_cpus_returns_none_without_sched_getaffinity(monkeypatch):
    def raise_attr_error(pid):
        raise AttributeError

    monkeypatch.setattr(os, "sched_getaffinity", raise_attr_error, raising=False)
    assert _physical_core_representative_cpus() is None


def test_parallel_decompose_with_pinning_still_correct(monkeypatch):
    # End-to-end: pinning must never break correctness, even when the
    # physical-core probe succeeds on this real machine.
    fixture = ALL_FIXTURES[1]
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    combined = _combine(parallel_decompose(padded, chunk_size=2, n_workers=2))
    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)
