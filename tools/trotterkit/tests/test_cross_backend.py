# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Every native backend reproduces the C backend to the bit.

All backends run the same operations in the same order without FMA
contraction (C and C++ build with -ffp-contract=off; Rust never
contracts) and compute every constant from the same real libm
functions, so their final states are identical, not merely close.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import trotterkit
from _systems import random_state

OTHERS = [b for b in trotterkit.backends() if b not in ("c", "python")]
pytestmark = pytest.mark.skipif(not OTHERS, reason="only C is built")


def _terms(n, count, seed, sort_x):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << n, size=count, dtype=np.uint64)
    x[::5] >>= np.uint64(min(4, n - 1))
    x[::9] = 0
    if sort_x:
        x.sort()
    z = rng.integers(0, 1 << n, size=count, dtype=np.uint64)
    c = rng.normal(size=count) + 0.2j * rng.normal(size=count)
    return x, z, c


def _evolve(backend, x, z, c, n, **kw):
    try:
        return trotterkit.evolve(random_state(n, 7), t=0.9, steps=2,
                                 terms=(x, z, c), backend=backend, **kw)
    except trotterkit.TrotterkitError as err:
        if kw.get("n_threads", 0) > 1 and err.code == 1:
            pytest.skip(f"{backend} built without threads")
        raise


@pytest.mark.parametrize("backend", OTHERS)
@pytest.mark.parametrize("n", [1, 3, 5, 8, 10])
@pytest.mark.parametrize("order", [2, 4])
def test_terms_serial(backend, n, order):
    x, z, c = _terms(n, 80, seed=n, sort_x=False)
    np.testing.assert_array_equal(
        _evolve(backend, x, z, c, n, order=order, n_threads=1),
        _evolve("c", x, z, c, n, order=order, n_threads=1))


@pytest.mark.parametrize("backend", OTHERS)
@pytest.mark.parametrize("n_threads", [2, 4])
@pytest.mark.parametrize("tile_bits", [-1, 0, 6])
@pytest.mark.parametrize("sort_x", [False, True])
def test_terms_threads_tiles(backend, n_threads, tile_bits, sort_x):
    n = 10
    x, z, c = _terms(n, 200, seed=11, sort_x=sort_x)
    np.testing.assert_array_equal(
        _evolve(backend, x, z, c, n, n_threads=n_threads,
                tile_bits=tile_bits),
        _evolve("c", x, z, c, n, n_threads=1))


def _sparse(n, density, seed):
    rng = np.random.default_rng(seed)
    dim = 1 << n
    m = sp.random_array((dim, dim), density=density, rng=rng,
                        dtype=complex, format="csr")
    m.data = rng.normal(size=m.nnz) + 1j * rng.normal(size=m.nnz)
    return m


@pytest.mark.parametrize("backend", OTHERS)
@pytest.mark.parametrize("n", [1, 3, 6, 9])
@pytest.mark.parametrize("order", [2, 4])
def test_grouped(backend, n, order):
    # Non-Hermitian on purpose: every quadrant of ab, one-sided blocks.
    H = _sparse(n, 0.3 if n < 6 else 0.02, seed=n)
    psi = random_state(n, 3)

    def run(b):
        return trotterkit.evolve(psi, t=0.7, steps=3, order=order,
                                 mode="grouped", H=H, backend=b)

    np.testing.assert_array_equal(run(backend), run("c"))


@pytest.mark.parametrize("backend", OTHERS)
def test_oscillator_chain(backend):
    fwht = pytest.importorskip("paulikit.algorithms.fwht")
    from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

    n_osc = 8
    k = {(i, j): 1.0 if j == i + 1 else 0.0
         for i in range(n_osc) for j in range(i, n_osc)}
    H, n = pad_to_power_of_two(
        build_hamiltonian(n_osc, k, [1.0] * n_osc, sparse=True),
        sparse=True)
    H = H.tocsr()
    H.eliminate_zeros()
    x, z, c = fwht.fwht_pauli_coefficients(H, sparse=True,
                                           chunk_size=256, atol=1e-12)
    psi = random_state(n, 5)
    for kw in ({"terms": (x, z, c)}, {"H": H, "mode": "grouped"}):
        got = trotterkit.evolve(psi, t=2.0, steps=10, order=4,
                                backend=backend, **kw)
        want = trotterkit.evolve(psi, t=2.0, steps=10, order=4,
                                 backend="c", **kw)
        np.testing.assert_array_equal(got, want)
