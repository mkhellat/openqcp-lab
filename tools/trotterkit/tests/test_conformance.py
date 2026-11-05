# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Conformance suite: every native backend must meet the contract in
include/trotterkit.h. The Python reference and products of dense
matrix exponentials are the authorities."""

import ctypes

import numpy as np
import pytest
import scipy.linalg as la

import trotterkit
from _systems import random_pauli_sum, random_state
from trotterkit import _backend, reference
from trotterkit.paulis import pauli_matrix
from trotterkit.schedule import stages

NATIVE = _backend.available()
WITH_PYTHON = NATIVE + ["python"]

TK_OK, TK_EINVAL, TK_EQUBITS = 0, 1, 3


def _vp(a):
    return a.ctypes.data_as(ctypes.c_void_p)


# 1. Deterministic per-term cases -------------------------------------

def _term_cases(n_qubits):
    """(x, z) pairs covering pure Z, x = 1, 2, 3, the top bit and
    Y-type terms (x & z != 0) that fit in n_qubits."""
    dim = 1 << n_qubits
    top = 1 << (n_qubits - 1)
    cases = {(0, 1), (0, dim - 1), (0, top)}
    for x in (1, 2, 3, top):
        if x >= dim:
            continue
        cases.add((x, 0))            # X-type
        cases.add((x, x))            # Y-type
        cases.add((x, dim - 1))      # mixed, x & z = x
        if (dim - 1) & ~x:
            cases.add((x, (dim - 1) & ~x))  # x & z = 0, z != 0
    return sorted(cases)


_PER_TERM = [(n, x, z, c)
             for n in (1, 2, 5)
             for x, z in _term_cases(n)
             for c in (0.37, 0.37 - 0.21j)]


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("n_qubits,x,z,c", _PER_TERM)
def test_single_term(backend, n_qubits, x, z, c):
    """One term: the product formula is exact, so the backend must
    match both the reference and exp(-i t c P)."""
    psi0 = random_state(n_qubits, seed=100 + n_qubits)
    xs = np.array([x], dtype=np.uint64)
    zs = np.array([z], dtype=np.uint64)
    cs = np.array([c], dtype=complex)
    t, steps = 0.9, 3
    got = trotterkit.evolve(psi0, t=t, steps=steps, terms=(xs, zs, cs),
                            backend=backend)
    want = reference.evolve_terms(psi0, xs, zs, cs,
                                  stages(t, steps, 2), steps)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    exact = la.expm(-1j * t * c * pauli_matrix(x, z, n_qubits)) @ psi0
    np.testing.assert_allclose(got, exact, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("n_qubits", [1, 2, 5])
@pytest.mark.parametrize("complex_coeffs", [False, True])
def test_deterministic_term_sum(backend, n_qubits, complex_coeffs):
    """All the deterministic cases of one size as one ordered sum."""
    pairs = _term_cases(n_qubits)
    xs = np.array([p[0] for p in pairs], dtype=np.uint64)
    zs = np.array([p[1] for p in pairs], dtype=np.uint64)
    cs = 0.11 * (1 + np.arange(len(pairs))).astype(complex)
    if complex_coeffs:
        cs = cs - 0.03j * np.arange(len(pairs))
    psi0 = random_state(n_qubits, seed=200 + n_qubits)
    for order in (2, 4):
        got = trotterkit.evolve(psi0, t=0.8, steps=4,
                                terms=(xs, zs, cs), order=order,
                                backend=backend)
        want = reference.evolve_terms(psi0, xs, zs, cs,
                                      stages(0.8, 4, order), 4)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


# 2. Grouped: complex diagonal, diagonal only, no diagonal ------------

def _grouped_h(kind):
    rng = np.random.default_rng({"complex_diag": 31, "diag_only": 32,
                                 "no_diag": 33}[kind])
    dim = 8
    H = np.zeros((dim, dim), dtype=complex)
    if kind in ("complex_diag", "diag_only"):
        H[np.arange(dim), np.arange(dim)] = (
            rng.normal(size=dim) + 1j * 0.4 * rng.normal(size=dim))
    if kind in ("complex_diag", "no_diag"):
        for r, col in [(0, 1), (1, 0), (2, 7), (5, 6), (3, 4), (6, 2)]:
            H[r, col] = rng.normal() + 1j * rng.normal()
    return H


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("kind", ["complex_diag", "diag_only",
                                  "no_diag"])
@pytest.mark.parametrize("order", [2, 4])
def test_grouped_non_hermitian(backend, kind, order):
    H = _grouped_h(kind)
    psi0 = random_state(3, seed=34)
    got = trotterkit.evolve(psi0, t=0.6, steps=5, H=H, mode="grouped",
                            order=order, backend=backend)
    want = reference.evolve_grouped(psi0, H, stages(0.6, 5, order), 5)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)
    if kind == "diag_only":
        exact = la.expm(-1j * 0.6 * H) @ psi0
        np.testing.assert_allclose(got, exact, rtol=1e-12, atol=1e-13)


# 3. Series switchover of the 2x2 block exponential -------------------

@pytest.mark.parametrize("backend", WITH_PYTHON)
@pytest.mark.parametrize("ratio", [0.999, 0.9, 1.001, 1.1])
@pytest.mark.parametrize("a,b", [(0.5 + 0.2j, 0.3 - 0.1j),
                                 (0.7, 0.7), (0.4j, -1.3 + 0.0j)])
def test_block_series_switchover(backend, ratio, a, b):
    """|theta^2 ab| = ratio * 1e-6 straddles the series threshold. One
    fragment applied twice at theta = dt/2 is exactly exp(-i dt M)."""
    theta = np.sqrt(ratio * 1e-6 / abs(a * b))
    t = 2 * theta
    H = np.array([[0, a], [b, 0]], dtype=complex)
    psi0 = np.array([0.6 + 0.1j, -0.3 + 0.7j])
    got = trotterkit.evolve(psi0, t=t, steps=1, H=H, mode="grouped",
                            backend=backend)
    want = la.expm(-1j * t * H) @ psi0
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)


# 4. Product order pinned against explicit dense factors --------------

def _s2(factors, psi):
    """Apply factors in order, then in reverse order."""
    for f in factors + factors[::-1]:
        psi = f @ psi
    return psi


@pytest.mark.parametrize("backend", WITH_PYTHON)
def test_terms_product_order(backend):
    n = 2
    x = np.array([1, 0, 3], dtype=np.uint64)
    z = np.array([0, 3, 1], dtype=np.uint64)
    c = np.array([0.7, -0.4 + 0.1j, 0.55], dtype=complex)
    dt = 0.9
    psi0 = random_state(n, seed=41)
    mats = [pauli_matrix(int(x[g]), int(z[g]), n) for g in range(3)]
    fac = [la.expm(-1j * (dt / 2) * c[g] * mats[g]) for g in range(3)]
    want = _s2(fac, psi0)
    wrong = _s2(fac[::-1], psi0)
    assert np.linalg.norm(want - wrong) > 1e-3   # order is visible
    got = trotterkit.evolve(psi0, t=dt, steps=1, terms=(x, z, c),
                            backend=backend)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("backend", WITH_PYTHON)
def test_grouped_product_order(backend):
    """Fragments: diagonal, x = 2, x = 5 (ascending), then reversed."""
    dim = 8
    rng = np.random.default_rng(42)
    parts = []
    D = np.diag(rng.normal(size=dim) + 0.3j * rng.normal(size=dim))
    parts.append(D)
    for x in (2, 5):
        F = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            if i < i ^ x:
                F[i, i ^ x] = rng.normal() + 1j * rng.normal()
                F[i ^ x, i] = rng.normal() + 1j * rng.normal()
        parts.append(F)
    H = sum(parts)
    dt = 0.7
    psi0 = random_state(3, seed=43)
    fac = [la.expm(-1j * (dt / 2) * p) for p in parts]
    want = _s2(fac, psi0)
    wrong = _s2([fac[0], fac[2], fac[1]], psi0)
    assert np.linalg.norm(want - wrong) > 1e-3   # order is visible
    got = trotterkit.evolve(psi0, t=dt, steps=1, H=H, mode="grouped",
                            backend=backend)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


# 5. Error contract (raw ABI calls) -----------------------------------

def _raw_terms(lib, psi, n_qubits, x, z, c, strategy=0, tile_bits=0,
               n_threads=1, dt=0.3, steps=2):
    c = np.asarray(c, dtype=complex)
    c_re = np.ascontiguousarray(c.real)
    c_im = np.ascontiguousarray(c.imag)
    dts = np.array([dt])
    terms = _backend.Terms(_vp(x), _vp(z), _vp(c_re), _vp(c_im), len(c))
    sched = _backend.Schedule(_vp(dts), 1, steps)
    return lib.tk_evolve_terms(_vp(psi), n_qubits, ctypes.byref(terms),
                               ctypes.byref(sched), strategy,
                               tile_bits, n_threads)


def _build(lib, n_qubits, row, col, v):
    row = np.asarray(row, dtype=np.uint64)
    col = np.asarray(col, dtype=np.uint64)
    v = np.asarray(v, dtype=complex)
    v_re = np.ascontiguousarray(v.real)
    v_im = np.ascontiguousarray(v.imag)
    handle = ctypes.c_void_p(0xDEAD)    # must be overwritten
    rc = lib.tk_groups_build(n_qubits, _vp(row), _vp(col), _vp(v_re),
                             _vp(v_im), len(row), ctypes.byref(handle))
    return rc, handle


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("bad", [
    {"strategy": 7}, {"strategy": -1}, {"n_threads": -1},
    {"x": 8}, {"z": 8}])
def test_terms_error_leaves_psi_unchanged(backend, bad):
    lib = _backend.load(backend)
    x = np.array([bad.get("x", 1), 2], dtype=np.uint64)
    z = np.array([bad.get("z", 3), 0], dtype=np.uint64)
    opts = {k: v for k, v in bad.items() if k not in ("x", "z")}
    psi0 = random_state(3, seed=51)
    psi = psi0.copy()
    rc = _raw_terms(lib, psi, 3, x, z, [0.4, 0.2 + 0.1j], **opts)
    assert rc == TK_EINVAL
    np.testing.assert_array_equal(psi, psi0)


@pytest.mark.parametrize("backend", NATIVE)
def test_terms_success_changes_psi(backend):
    """Guard for the test above: the same call with valid options
    does write psi."""
    lib = _backend.load(backend)
    x = np.array([1, 2], dtype=np.uint64)
    z = np.array([3, 0], dtype=np.uint64)
    psi0 = random_state(3, seed=51)
    psi = psi0.copy()
    assert _raw_terms(lib, psi, 3, x, z, [0.4, 0.2 + 0.1j]) == TK_OK
    assert not np.array_equal(psi, psi0)


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("n_qubits", [0, 33])
def test_terms_bad_qubit_count_raises_equbits(backend, n_qubits):
    lib = _backend.load(backend)
    psi0 = np.array([0.6 + 0.8j, 0.0])
    psi = psi0.copy()
    terms = _backend.Terms(None, None, None, None, 0)
    dts = np.array([0.1])
    sched = _backend.Schedule(_vp(dts), 1, 1)
    rc = lib.tk_evolve_terms(_vp(psi), n_qubits, ctypes.byref(terms),
                             ctypes.byref(sched), 0, 0, 1)
    assert rc == TK_EQUBITS
    np.testing.assert_array_equal(psi, psi0)


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("row,col", [([0, 8], [1, 0]), ([0, 1], [1, 8]),
                                     ([2**40], [0])])
def test_build_index_out_of_range(backend, row, col):
    rc, handle = _build(_backend.load(backend), 3, row, col,
                        [0.5] * len(row))
    assert rc == TK_EINVAL
    assert handle.value is None


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("n_qubits", [0, 33])
def test_build_bad_qubit_count(backend, n_qubits):
    rc, handle = _build(_backend.load(backend), n_qubits, [0], [0],
                        [1.0])
    assert rc == TK_EQUBITS
    assert handle.value is None


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("bad", ["n_qubits", "n_threads"])
def test_grouped_error_leaves_psi_unchanged(backend, bad):
    lib = _backend.load(backend)
    rc, handle = _build(lib, 3, [0, 1, 4], [1, 0, 4],
                        [0.5, 0.25 + 0.1j, -0.3j])
    assert rc == TK_OK and handle.value is not None
    try:
        psi0 = random_state(3, seed=52)
        dts = np.array([0.4])
        sched = _backend.Schedule(_vp(dts), 1, 2)
        for n_q, n_thr, expect in [
                (2 if bad == "n_qubits" else 3,
                 -1 if bad == "n_threads" else 1, TK_EINVAL),
                (3, 1, TK_OK)]:
            psi = psi0.copy()
            rc = lib.tk_evolve_grouped(_vp(psi), n_q, handle,
                                       ctypes.byref(sched), n_thr)
            assert rc == expect
            if expect == TK_OK:
                assert not np.array_equal(psi, psi0)
            else:
                np.testing.assert_array_equal(psi, psi0)
    finally:
        lib.tk_groups_free(handle)


@pytest.mark.parametrize("backend", NATIVE)
def test_free_null_and_strings(backend):
    lib = _backend.load(backend)
    lib.tk_groups_free(None)
    assert lib.tk_backend_name() == backend.encode()
    for code in (0, 1, 2, 3, 99, -5):
        msg = lib.tk_strerror(code)
        assert msg is not None and len(msg) > 0


@pytest.mark.parametrize("backend", NATIVE)
def test_zero_steps_is_identity(backend):
    lib = _backend.load(backend)
    x = np.array([1, 2], dtype=np.uint64)
    z = np.array([3, 0], dtype=np.uint64)
    psi0 = random_state(3, seed=53)
    psi = psi0.copy()
    assert _raw_terms(lib, psi, 3, x, z, [0.4, 0.2], steps=0) == TK_OK
    np.testing.assert_array_equal(psi, psi0)


# 6. Many steps --------------------------------------------------------

@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("order", [2, 4])
def test_many_steps_terms(backend, order):
    x, z, c, _ = random_pauli_sum(4, 10, seed=61, complex_coeffs=True)
    psi0 = random_state(4, seed=62)
    want = reference.evolve_terms(psi0, x, z, c,
                                  stages(2.0, 200, order), 200)
    got = trotterkit.evolve(psi0, t=2.0, steps=200, terms=(x, z, c),
                            order=order, backend=backend)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("order", [2, 4])
def test_many_steps_grouped(backend, order):
    H = _grouped_h("complex_diag")
    psi0 = random_state(3, seed=63)
    want = reference.evolve_grouped(psi0, H, stages(2.0, 200, order),
                                    200)
    got = trotterkit.evolve(psi0, t=2.0, steps=200, H=H,
                            mode="grouped", order=order,
                            backend=backend)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-12)
