# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Workload files round-trip and reject damaged input."""

import numpy as np
import pytest
import scipy.sparse as sp

from trotterkit import workload
from _systems import random_pauli_sum, random_state


def test_roundtrip(tmp_path):
    x, z, c, H = random_pauli_sum(3, 7, seed=1, complex_coeffs=True)
    psi0 = random_state(3, 2)
    path = tmp_path / "w.tkwl"
    workload.write(path, 3, x, z, c, psi0, H=sp.coo_array(H))
    w = workload.read(path)
    assert w.n_qubits == 3
    np.testing.assert_array_equal(w.x, x)
    np.testing.assert_array_equal(w.z, z)
    np.testing.assert_array_equal(w.coeff, c)
    np.testing.assert_array_equal(w.psi0, psi0)
    dense = np.zeros((8, 8), dtype=complex)
    dense[w.row, w.col] = w.val
    np.testing.assert_array_equal(dense, H)


def test_without_matrix(tmp_path):
    x, z, c, _ = random_pauli_sum(2, 3, seed=3)
    path = tmp_path / "w.tkwl"
    workload.write(path, 2, x, z, c, random_state(2, 4))
    assert len(workload.read(path).row) == 0


def test_corrupt_payload_is_rejected(tmp_path):
    x, z, c, _ = random_pauli_sum(2, 3, seed=5)
    path = tmp_path / "w.tkwl"
    workload.write(path, 2, x, z, c, random_state(2, 6))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        workload.read(path)


def test_bad_magic_is_rejected(tmp_path):
    path = tmp_path / "w.tkwl"
    path.write_bytes(b"XXXX" + bytes(60))
    with pytest.raises(ValueError, match="not a trotterkit workload"):
        workload.read(path)
