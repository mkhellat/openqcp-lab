# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Pure-Python reference of both evolution modes.

Slow and simple on purpose: every native backend is tested against
it. exp(-i theta c P) = cos(theta c) I - i sin(theta c) P because
P**2 = I; this holds for complex c, so non-Hermitian sums need no
special case.
"""

import numpy as np
import scipy.sparse as sp

from trotterkit.paulis import apply


def _apply_term(psi, x, z, c, theta):
    return (np.cos(theta * c) * psi
            - 1j * np.sin(theta * c) * apply(x, z, psi))


def evolve_terms(psi0, x, z, coeff, stage_dts, steps):
    """Per-term mode: each S2 stage applies terms in order, then
    reversed, each with angle dt/2."""
    psi = np.array(psi0, dtype=complex)
    order = list(range(len(coeff)))
    sweep = order + order[::-1]
    for _ in range(steps):
        for dt in stage_dts:
            theta = dt / 2
            for g in sweep:
                psi = _apply_term(psi, int(x[g]), int(z[g]),
                                  complex(coeff[g]), theta)
    return psi


def fragments(H):
    """XOR-offset slices of H.

    Returns (diag, frags): diag maps i -> H[i, i]; frags maps each
    nonzero offset x (ascending) to {i: (a, b)} with i < i ^ x,
    a = H[i, i ^ x] and b = H[i ^ x, i].
    """
    m = sp.csr_array(H, dtype=complex)
    m.sum_duplicates()
    coo = m.tocoo()
    diag, blocks = {}, {}
    for r, col, v in zip(coo.row.tolist(), coo.col.tolist(),
                         coo.data.tolist()):
        if v == 0:
            continue
        x = r ^ col
        if x == 0:
            diag[r] = diag.get(r, 0j) + v
            continue
        i = min(r, col)
        a, b = blocks.setdefault(x, {}).get(i, (0j, 0j))
        if r == i:
            a += v
        else:
            b += v
        blocks[x][i] = (a, b)
    return diag, {x: blocks[x] for x in sorted(blocks)}


def block_exp(a, b, theta):
    """(c, s) with exp(-i theta M) = c I - i s M, M = [[0,a],[b,0]].

    M**2 = ab I, so c = cos(theta w) and s = sin(theta w)/w with
    w**2 = ab; both are even in w. Small theta**2 ab uses the series,
    which stays exact as w -> 0.
    """
    u = theta * theta * a * b
    if abs(u) < 1e-6:
        c = 1 - u / 2 + u * u / 24
        s = theta * (1 - u / 6 + u * u / 120)
    else:
        w = np.sqrt(complex(a * b))
        c = np.cos(theta * w)
        s = np.sin(theta * w) / w
    return complex(c), complex(s)


def _apply_fragment(psi, x, blocks_or_diag, theta):
    if x == 0:
        for i, d in blocks_or_diag.items():
            psi[i] *= np.exp(-1j * theta * d)
        return
    for i, (a, b) in blocks_or_diag.items():
        c, s = block_exp(a, b, theta)
        j = i ^ x
        p, q = psi[i], psi[j]
        psi[i] = c * p - 1j * s * a * q
        psi[j] = -1j * s * b * p + c * q


def evolve_grouped(psi0, H, stage_dts, steps):
    """Grouped mode: fragments are the XOR-offset slices of H."""
    psi = np.array(psi0, dtype=complex)
    diag, frags = fragments(H)
    order = ([(0, diag)] if diag else []) + list(frags.items())
    sweep = order + order[::-1]
    for _ in range(steps):
        for dt in stage_dts:
            theta = dt / 2
            for x, part in sweep:
                _apply_fragment(psi, x, part, theta)
    return psi
