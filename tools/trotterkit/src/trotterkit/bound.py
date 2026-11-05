# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Coefficient-only upper bound on the 2nd-order Trotter error.

alpha = (1/12) sum_g ||[T_g, [T_g, H_g]]|| + (1/24) sum_g
||[H_g, [H_g, T_g]]||, T_g = sum_{g' > g} H_{g'} (Childs, Su, Tran,
Wiebe, Zhu, PRX 11, 011020 (2021)), loosened with
||[A, B]|| <= 2 ||A|| ||B|| and ||T_g|| <= sum_{g' > g} ||H_{g'}||,
so only fragment norms are needed. Error <= alpha t**3 / r**2.
Valid for Hermitian H only.
"""

import math

import numpy as np
import scipy.sparse as sp

from trotterkit.reference import fragments


def alpha_from_norms(norms):
    """alpha from fragment norms, in evolution order."""
    norms = np.asarray(norms, dtype=float)
    suffix = np.zeros_like(norms)
    if len(norms) > 1:
        suffix[:-1] = np.cumsum(norms[::-1])[::-1][1:]
    term1 = np.sum(4.0 * suffix ** 2 * norms)
    term2 = np.sum(4.0 * norms ** 2 * suffix)
    return float(term1 / 12.0 + term2 / 24.0)


def steps_for(alpha, t, eps):
    """Smallest r with alpha |t|**3 / r**2 <= eps."""
    if t == 0:
        return 1
    return max(1, math.ceil(math.sqrt(alpha * abs(t) ** 3 / eps)))


def _need_order_two(order):
    if order != 2:
        raise NotImplementedError("only the 2nd-order bound exists")


def bound_terms(coeff, t, eps, order=2, atol=1e-12):
    """Steps for per-term mode; fragment norms are |c_g|."""
    _need_order_two(order)
    coeff = np.asarray(coeff, dtype=complex)
    if np.any(np.abs(coeff.imag) > atol):
        raise ValueError("bound needs a Hermitian H (real coefficients)")
    return steps_for(alpha_from_norms(np.abs(coeff.real)), t, eps)


def bound_grouped(H, t, eps, order=2, atol=1e-12):
    """Steps for grouped mode; each slice's norm is exact: a direct
    sum of [[0, a], [b, 0]] blocks has norm max(|a|, |b|)."""
    _need_order_two(order)
    m = sp.csr_array(H, dtype=complex)
    if abs(m - m.conj().T).max() > atol:
        raise ValueError("bound needs a Hermitian H")
    diag, frags = fragments(m)
    norms = [max(abs(d) for d in diag.values())] if diag else []
    for blocks in frags.values():
        norms.append(max(max(abs(a), abs(b))
                         for a, b in blocks.values()))
    return steps_for(alpha_from_norms(norms), t, eps)
