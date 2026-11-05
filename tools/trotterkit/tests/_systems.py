# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Small random systems shared by the tests."""

import numpy as np

from trotterkit.paulis import pauli_matrix


def random_pauli_sum(n_qubits, n_terms, seed, complex_coeffs=False):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    x = rng.integers(0, dim, size=n_terms, dtype=np.uint64)
    z = rng.integers(0, dim, size=n_terms, dtype=np.uint64)
    c = 0.3 * rng.normal(size=n_terms).astype(complex)
    if complex_coeffs:
        c = c + 0.1j * rng.normal(size=n_terms)
    H = sum(cg * pauli_matrix(int(xg), int(zg), n_qubits)
            for xg, zg, cg in zip(x, z, c))
    return x, z, c, H


def random_state(n_qubits, seed):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)
