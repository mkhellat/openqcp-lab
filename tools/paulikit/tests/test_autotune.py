"""Tests for PLAN.md Phase 12's runtime auto-tuning heuristics
(paulikit.algorithms.autotune) and fwht.auto_decompose's streaming-vs-
dense decision.

Memory/cgroup/cache-detection tests monkeypatch the module's own
internal helpers rather than depending on this machine's actual
hardware/cgroup state - the formulas are what's under test, not any
particular machine's numbers.
"""

import numpy as np
import pytest

from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import auto_decompose, fwht_pauli_terms
from paulikit.testing.fixtures import ALL_FIXTURES


@pytest.fixture(autouse=True)
def _reset_autotune_caches():
    """autotune's memory/chunk_size results are cached per-process
    (module-level) - reset before and after each test so tests don't
    leak state into each other via the cache."""
    autotune._cached_chunk_size = None
    autotune._cached_memory_budget_bytes = None
    yield
    autotune._cached_chunk_size = None
    autotune._cached_memory_budget_bytes = None


def test_available_memory_bytes_uses_meminfo_when_present(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)
    assert autotune.available_memory_bytes() == 8 * 1024**3


def test_available_memory_bytes_falls_back_to_posix_when_no_meminfo(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: None)
    monkeypatch.setattr(autotune, "_posix_available_physical_bytes", lambda: 4 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)
    assert autotune.available_memory_bytes() == 4 * 1024**3


def test_available_memory_bytes_uses_smaller_of_physical_and_cgroup(monkeypatch):
    # The correctness-critical case for shared HPC nodes: a cgroup cap
    # smaller than physically-available memory must win.
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: 2 * 1024**3)
    assert autotune.available_memory_bytes() == 2 * 1024**3


def test_available_memory_bytes_ignores_larger_cgroup_limit(monkeypatch):
    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: 16 * 1024**3)
    assert autotune.available_memory_bytes() == 2 * 1024**3


def test_available_memory_bytes_is_cached_per_process(monkeypatch):
    calls = []

    def fake_meminfo():
        calls.append(1)
        return 8 * 1024**3

    monkeypatch.setattr(autotune, "_read_meminfo_available_bytes", fake_meminfo)
    monkeypatch.setattr(autotune, "_cgroup_memory_limit_bytes", lambda: None)

    first = autotune.available_memory_bytes()
    second = autotune.available_memory_bytes()
    assert first == second
    assert len(calls) == 1, "expected the underlying probe to run exactly once (cached)"


def test_recommended_chunk_size_uses_declared_size_when_no_probe(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: 256 * 1024)
    # dim=1024 -> 1024*16 = 16384 bytes/row; 262144 // 16384 = 16
    assert autotune.recommended_chunk_size(dim=1024) == 16


def test_recommended_chunk_size_respects_floor(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: 1024)  # tiny
    # dim large enough that the cache-driven size would be < floor
    assert autotune.recommended_chunk_size(dim=4096) == autotune._min_chunk_size_floor()


def test_recommended_chunk_size_falls_back_to_32_when_nothing_works(monkeypatch):
    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", lambda: None)
    assert autotune.recommended_chunk_size(dim=64) == 32


def test_recommended_chunk_size_is_cached_per_process(monkeypatch):
    calls = []

    def fake_declared():
        calls.append(1)
        return 256 * 1024

    monkeypatch.setattr(autotune, "_cache_probe", None)
    monkeypatch.setattr(autotune, "_declared_l2_size_bytes", fake_declared)

    first = autotune.recommended_chunk_size(dim=64)
    second = autotune.recommended_chunk_size(dim=64)
    assert first == second
    assert len(calls) == 1


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_auto_decompose_dense_path_matches_fwht_pauli_terms(monkeypatch, fixture):
    # Force the dense path by giving a huge memory budget.
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 2**40)
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    result = auto_decompose(padded)
    assert isinstance(result, dict)
    assert set(result) == set(reference)
    for label in reference:
        assert result[label] == pytest.approx(reference[label], abs=1e-9)


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_auto_decompose_streaming_path_matches_fwht_pauli_terms(monkeypatch, fixture):
    # Force the streaming path by giving a near-zero memory budget.
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 4)
    padded = fixture.padded_hamiltonian()
    reference = fwht_pauli_terms(padded)

    result = auto_decompose(padded)
    assert not isinstance(result, dict)
    combined = {}
    for chunk_dict in result:
        combined.update(chunk_dict)

    assert set(combined) == set(reference)
    for label in reference:
        assert combined[label] == pytest.approx(reference[label], abs=1e-9)


def test_auto_decompose_return_type_is_distinguishable_via_isinstance(monkeypatch):
    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 2**40)
    padded = ALL_FIXTURES[0].padded_hamiltonian()
    dense_result = auto_decompose(padded)
    assert isinstance(dense_result, dict)

    monkeypatch.setattr(autotune, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(autotune, "recommended_chunk_size", lambda dim: 4)
    streaming_result = auto_decompose(padded)
    assert not isinstance(streaming_result, dict)
