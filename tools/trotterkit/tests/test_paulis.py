# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Pin trotterkit's Pauli convention to paulikit's."""

import itertools

import numpy as np
import pytest

from trotterkit.paulis import apply, pauli_matrix, phase

fwht = pytest.importorskip("paulikit.algorithms.fwht")
pauli_utils = pytest.importorskip("paulikit.pauli_utils")


def test_phase_values():
    assert phase(0, 0) == 1
    assert phase(1, 1) == 1j
    assert phase(3, 3) == -1
    assert phase(7, 7) == -1j


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_pauli_matrix_matches_paulikit(n_qubits):
    dim = 1 << n_qubits
    for x, z in itertools.product(range(dim), repeat=2):
        label = fwht.pauli_label(x, z, n_qubits)
        np.testing.assert_array_equal(
            pauli_matrix(x, z, n_qubits),
            pauli_utils.pauli_string_to_matrix(label),
        )


def test_apply_matches_matrix():
    rng = np.random.default_rng(0)
    n_qubits, dim = 4, 16
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    for x, z in itertools.product(range(dim), repeat=2):
        np.testing.assert_allclose(
            apply(x, z, psi), pauli_matrix(x, z, n_qubits) @ psi,
            atol=1e-14,
        )


def test_paulikit_coefficients_rebuild_hamiltonian():
    from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

    k = {(i, j): 1.0 + 0.1 * (i + j) for i in range(3) for j in range(i, 3)}
    H = build_hamiltonian(3, k, [1.0, 1.5, 2.0], sparse=True)
    Hp, n_qubits = pad_to_power_of_two(H, sparse=True)
    x, z, c = fwht.fwht_pauli_coefficients(
        Hp.tocsr(), sparse=True, chunk_size=256, atol=1e-12)
    rebuilt = sum(
        complex(cg) * pauli_matrix(int(xg), int(zg), n_qubits)
        for xg, zg, cg in zip(x, z, c)
    )
    np.testing.assert_allclose(rebuilt, Hp.toarray(), atol=1e-12)
