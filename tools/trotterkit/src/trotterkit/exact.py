# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Exact evolution, for validation only."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def evolve(H, psi0, t):
    """exp(-i t H) psi0 (H Hermitian or not)."""
    return expm_multiply(-1j * t * sp.csr_array(H),
                         np.asarray(psi0, dtype=complex))
