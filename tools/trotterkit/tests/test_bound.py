# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""The closed-form bound must never undercut the exact commutator
expression it loosens."""

import math

import numpy as np
import pytest

from _systems import random_pauli_sum
from trotterkit import bound
from trotterkit.paulis import pauli_matrix
from trotterkit.reference import fragments


def _exact_alpha(parts):
    def norm(m):
        return np.linalg.norm(m, ord=2)

    alpha = 0.0
    for g, hg in enumerate(parts):
        tail = sum(parts[g + 1:], np.zeros_like(hg))
        inner = tail @ hg - hg @ tail
        alpha += norm(tail @ inner - inner @ tail) / 12
        inner2 = hg @ tail - tail @ hg
        alpha += norm(hg @ inner2 - inner2 @ hg) / 24
    return alpha


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_terms_bound_dominates_exact(seed):
    x, z, c, _ = random_pauli_sum(3, 8, seed=seed)
    parts = [c[g].real * pauli_matrix(int(x[g]), int(z[g]), 3)
             for g in range(len(c))]
    closed = bound.alpha_from_norms(np.abs(c.real))
    assert closed >= _exact_alpha(parts) - 1e-12


def test_steps_formula():
    assert bound.steps_for(2.0, 3.0, 0.5) == math.ceil(math.sqrt(108))
    assert bound.steps_for(2.0, 0.0, 0.5) == 1


def test_refuses_non_hermitian():
    x, z, c, H = random_pauli_sum(3, 8, seed=5, complex_coeffs=True)
    with pytest.raises(ValueError):
        bound.bound_terms(c, 1.0, 1e-3)
    with pytest.raises(ValueError):
        bound.bound_grouped(H, 1.0, 1e-3)


def test_only_order_two():
    _, _, c, _ = random_pauli_sum(3, 4, seed=6)
    with pytest.raises(NotImplementedError):
        bound.bound_terms(c, 1.0, 1e-3, order=4)


def test_alpha_from_norms_indexing():
    # Suffix sums are [5, 3, 0]: a suffix taken from the prefix, or a
    # tail that includes g itself, changes this value.
    assert bound.alpha_from_norms([1.0, 2.0, 3.0]) == pytest.approx(
        103 / 6, rel=1e-15)
    assert bound.alpha_from_norms([2.5]) == 0.0


def _dense_fragments(H, n_qubits):
    dim = 1 << n_qubits
    diag, frags = fragments(H)
    parts = []
    if diag:
        d = np.zeros((dim, dim), dtype=complex)
        for i, v in diag.items():
            d[i, i] = v
        parts.append(d)
    for x, blocks in frags.items():
        m = np.zeros((dim, dim), dtype=complex)
        for i, (a, b) in blocks.items():
            m[i, i ^ x] = a
            m[i ^ x, i] = b
        parts.append(m)
    return parts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_norms_match_exact_and_dominate(seed):
    _, _, _, H = random_pauli_sum(3, 8, seed=seed)
    parts = _dense_fragments(H, 3)
    norms = bound.grouped_norms(H)
    assert len(norms) == len(parts)
    for k, part in enumerate(parts):
        assert norms[k] == pytest.approx(
            np.linalg.norm(part, 2), rel=1e-12)
    assert bound.alpha_from_norms(norms) >= _exact_alpha(parts) - 1e-12
