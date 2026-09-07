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
