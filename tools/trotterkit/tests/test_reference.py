# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""The reference must converge to exact evolution at its formal order."""

import numpy as np
import pytest
from scipy.linalg import expm

from _systems import random_pauli_sum, random_state
from trotterkit import exact, reference
from trotterkit.schedule import stages

T = 0.5


def _terms_error(x, z, c, H, psi0, steps, order):
    psi = reference.evolve_terms(psi0, x, z, c, stages(T, steps, order),
                                 steps)
    return np.linalg.norm(psi - exact.evolve(H, psi0, T))


@pytest.mark.parametrize("order, steps, lo, hi",
                         [(2, 32, 3.6, 4.4), (4, 8, 13.0, 19.0)])
@pytest.mark.parametrize("complex_coeffs", [False, True])
def test_terms_converges_at_formal_order(order, steps, lo, hi,
                                         complex_coeffs):
    x, z, c, H = random_pauli_sum(4, 12, seed=1,
                                  complex_coeffs=complex_coeffs)
    psi0 = random_state(4, seed=2)
    ratio = (_terms_error(x, z, c, H, psi0, steps, order)
             / _terms_error(x, z, c, H, psi0, 2 * steps, order))
    assert lo < ratio < hi


def test_terms_empty_sum_is_identity():
    psi0 = random_state(3, seed=0)
    empty = np.array([], dtype=np.uint64)
    out = reference.evolve_terms(psi0, empty, empty,
                                 np.array([], dtype=complex),
                                 stages(1.0, 3, 2), 3)
    np.testing.assert_array_equal(out, psi0)


def test_non_hermitian_is_not_renormalised():
    x, z, c, H = random_pauli_sum(4, 12, seed=3, complex_coeffs=True)
    psi0 = random_state(4, seed=4)
    psi = reference.evolve_terms(psi0, x, z, c, stages(T, 256, 2), 256)
    expected = exact.evolve(H, psi0, T)
    assert abs(np.linalg.norm(expected) - 1.0) > 1e-3
    assert np.linalg.norm(psi) == pytest.approx(
        np.linalg.norm(expected), rel=1e-4)


def _grouped_error(H, psi0, steps, order):
    psi = reference.evolve_grouped(psi0, H, stages(T, steps, order),
                                   steps)
    return np.linalg.norm(psi - exact.evolve(H, psi0, T))


@pytest.mark.parametrize("order, steps, lo, hi",
                         [(2, 32, 3.6, 4.4), (4, 8, 13.0, 19.0)])
@pytest.mark.parametrize("complex_coeffs", [False, True])
def test_grouped_converges_at_formal_order(order, steps, lo, hi,
                                           complex_coeffs):
    _, _, _, H = random_pauli_sum(4, 12, seed=5,
                                  complex_coeffs=complex_coeffs)
    psi0 = random_state(4, seed=6)
    ratio = (_grouped_error(H, psi0, steps, order)
             / _grouped_error(H, psi0, 2 * steps, order))
    assert lo < ratio < hi


def test_fragments_partition_h():
    _, _, _, H = random_pauli_sum(3, 10, seed=7, complex_coeffs=True)
    diag, frags = reference.fragments(H)
    rebuilt = np.zeros_like(H)
    for i, d in diag.items():
        rebuilt[i, i] += d
    for x, blocks in frags.items():
        for i, (a, b) in blocks.items():
            rebuilt[i, i ^ x] += a
            rebuilt[i ^ x, i] += b
    np.testing.assert_allclose(rebuilt, H, atol=1e-15)
    assert list(frags) == sorted(frags)


@pytest.mark.parametrize("a, b", [(0.7, 0.7), (0.3 + 0.2j, -0.1j),
                                  (0.5, 0.0), (1e-9, 2e-9)])
def test_block_exp_matches_matrix_exponential(a, b):
    theta = 0.37
    m = np.array([[0, a], [b, 0]], dtype=complex)
    c, s = reference.block_exp(a, b, theta)
    np.testing.assert_allclose(c * np.eye(2) - 1j * s * m,
                               expm(-1j * theta * m), atol=1e-14)
