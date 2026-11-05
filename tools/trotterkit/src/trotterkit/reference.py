# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Pure-Python reference of both evolution modes.

Slow and simple on purpose: every native backend is tested against
it. exp(-i theta c P) = cos(theta c) I - i sin(theta c) P because
P**2 = I; this holds for complex c, so non-Hermitian sums need no
special case.
"""

import numpy as np

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
