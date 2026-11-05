# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Pauli-string conventions shared by every trotterkit component.

Masks follow paulikit: for n qubits, label character j (qubit j) is
bit n-1-j of the x and z masks. The operator is
P = i**popcount(x & z) * X**x Z**z, so
P|r> = i**popcount(x & z) * (-1)**popcount(r & z) |r ^ x>.
"""

import numpy as np


def phase(x, z):
    """i**popcount(x & z)."""
    return 1j ** ((int(x) & int(z)).bit_count() % 4)


def _signs(indices, z):
    parity = np.bitwise_count(indices & np.uint64(z)) & 1
    return 1 - 2 * parity.astype(np.int64)


def pauli_matrix(x, z, n_qubits):
    """Dense matrix of the Pauli string with masks (x, z)."""
    dim = 1 << n_qubits
    r = np.arange(dim, dtype=np.uint64)
    m = np.zeros((dim, dim), dtype=complex)
    rows = (r ^ np.uint64(x)).astype(np.intp)
    m[rows, r.astype(np.intp)] = phase(x, z) * _signs(r, z)
    return m


def apply(x, z, psi):
    """P @ psi without forming P.

    (P psi)[j] = phase * (-1)**popcount((j ^ x) & z) * psi[j ^ x].
    """
    dim = psi.shape[0]
    j = np.arange(dim, dtype=np.uint64)
    src = j ^ np.uint64(x)
    return phase(x, z) * _signs(src, z) * psi[src.astype(np.intp)]
