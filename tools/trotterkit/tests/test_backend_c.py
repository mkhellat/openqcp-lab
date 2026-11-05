# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""The C backend must reproduce the Python reference exactly."""

import ctypes

import numpy as np
import pytest
import scipy.sparse as sp

import trotterkit
from _systems import random_pauli_sum, random_state
from trotterkit import _backend, reference
from trotterkit.schedule import stages


def test_c_backend_is_built():
    assert "c" in trotterkit.backends()
    assert _backend.load("c").tk_backend_name() == b"c"


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("complex_coeffs", [False, True])
def test_terms_match_reference(order, complex_coeffs):
    x, z, c, _ = random_pauli_sum(5, 20, seed=11,
                                  complex_coeffs=complex_coeffs)
    psi0 = random_state(5, seed=12)
    want = reference.evolve_terms(psi0, x, z, c,
                                  stages(0.7, 6, order), 6)
    got = trotterkit.evolve(psi0, t=0.7, steps=6, terms=(x, z, c),
                            order=order, backend="c")
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_terms_empty_sum_is_identity():
    psi0 = random_state(3, seed=13)
    empty = np.array([], dtype=np.uint64)
    got = trotterkit.evolve(psi0, t=1.0, steps=2,
                            terms=(empty, empty,
                                   np.array([], dtype=complex)),
                            backend="c")
    np.testing.assert_array_equal(got, psi0)


def test_input_state_is_not_modified():
    x, z, c, _ = random_pauli_sum(3, 5, seed=14)
    psi0 = random_state(3, seed=15)
    keep = psi0.copy()
    trotterkit.evolve(psi0, t=1.0, steps=2, terms=(x, z, c),
                      backend="c")
    np.testing.assert_array_equal(psi0, keep)


@pytest.mark.parametrize("kwargs", [
    {"strategy": "tables"}, {"n_threads": -1}, {"tile_bits": 2},
    {"tile_bits": 3}, {"tile_bits": 4}])
def test_unsupported_options_raise_einval(kwargs):
    # 3 qubits: tiles must have >= 8 amplitudes and split the state,
    # so no tile_bits > 0 is valid; tables are not implemented.
    x, z, c, _ = random_pauli_sum(3, 5, seed=16)
    with pytest.raises(trotterkit.TrotterkitError) as err:
        trotterkit.evolve(random_state(3, 0), t=1.0, steps=1,
                          terms=(x, z, c), backend="c", **kwargs)
    assert err.value.code == 1


def test_mask_out_of_range_raises_einval():
    x = np.array([8], dtype=np.uint64)
    z = np.array([0], dtype=np.uint64)
    with pytest.raises(trotterkit.TrotterkitError) as err:
        trotterkit.evolve(random_state(3, 0), t=1.0, steps=1,
                          terms=(x, z, np.array([0.5 + 0j])),
                          backend="c")
    assert err.value.code == 1


def test_too_many_qubits_raises_equbits():
    lib = _backend.load("c")
    psi = np.zeros(2, dtype=complex)
    terms = _backend.Terms(None, None, None, None, 0)
    dt = np.array([0.1])
    sched = _backend.Schedule(dt.ctypes.data_as(ctypes.c_void_p), 1, 1)
    rc = lib.tk_evolve_terms(psi.ctypes.data_as(ctypes.c_void_p), 33,
                             ctypes.byref(terms), ctypes.byref(sched),
                             0, 0, 1)
    assert rc == 3


@pytest.mark.parametrize("bad", ["short_x", "short_z", "2d_x"])
def test_mismatched_term_arrays_raise_valueerror(bad):
    x = np.array([1, 2, 3], dtype=np.uint64)
    z = np.array([0, 1, 2], dtype=np.uint64)
    c = np.array([0.5, 0.25, 0.125], dtype=complex)
    if bad == "short_x":
        x = x[:2]
    elif bad == "short_z":
        z = z[:2]
    else:
        x = x.reshape(3, 1)
    with pytest.raises(ValueError):
        trotterkit.evolve(random_state(3, 0), t=1.0, steps=1,
                          terms=(x, z, c), backend="c")


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("complex_coeffs", [False, True])
def test_grouped_match_reference(order, complex_coeffs):
    _, _, _, H = random_pauli_sum(5, 20, seed=21,
                                  complex_coeffs=complex_coeffs)
    psi0 = random_state(5, seed=22)
    want = reference.evolve_grouped(psi0, H, stages(0.7, 6, order), 6)
    got = trotterkit.evolve(psi0, t=0.7, steps=6, H=sp.csr_array(H),
                            mode="grouped", order=order, backend="c")
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_grouped_one_sided_blocks():
    H = np.zeros((8, 8), dtype=complex)
    H[1, 6] = 0.4
    H[2, 2] = -0.3
    psi0 = random_state(3, seed=23)
    want = reference.evolve_grouped(psi0, H, stages(1.0, 3, 2), 3)
    got = trotterkit.evolve(psi0, t=1.0, steps=3, H=H, mode="grouped",
                            backend="c")
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_grouped_wrong_qubit_count_raises_einval():
    lib = _backend.load("c")
    row = np.array([0], dtype=np.uint64)
    col = np.array([1], dtype=np.uint64)
    v = np.array([1.0])
    handle = ctypes.c_void_p()
    assert lib.tk_groups_build(
        2, row.ctypes.data_as(ctypes.c_void_p),
        col.ctypes.data_as(ctypes.c_void_p),
        v.ctypes.data_as(ctypes.c_void_p),
        np.zeros(1).ctypes.data_as(ctypes.c_void_p), 1,
        ctypes.byref(handle)) == 0
    try:
        psi = np.zeros(8, dtype=complex)
        dt = np.array([0.1])
        sched = _backend.Schedule(dt.ctypes.data_as(ctypes.c_void_p),
                                  1, 1)
        assert lib.tk_evolve_grouped(
            psi.ctypes.data_as(ctypes.c_void_p), 3, handle,
            ctypes.byref(sched), 1) == 1
    finally:
        lib.tk_groups_free(handle)
