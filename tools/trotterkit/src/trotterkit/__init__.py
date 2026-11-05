# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Product-formula evolution of Pauli-sum Hamiltonians."""

import numpy as np

from trotterkit import _backend, bound, exact, reference, schedule
from trotterkit._backend import TrotterkitError

__version__ = "0.0.1"
__all__ = ["TrotterkitError", "backends", "bound", "evolve", "exact"]

_STRATEGY = {"auto": 0, "onthefly": 1, "tables": 2}


def backends():
    """Native backends that built and load, plus 'python'."""
    return _backend.available() + ["python"]


def _n_qubits(psi0):
    dim = psi0.shape[0]
    n_qubits = dim.bit_length() - 1
    if psi0.ndim != 1 or n_qubits < 1 or dim != 1 << n_qubits:
        raise ValueError("psi0 must be a 1-D vector of length 2**n")
    return n_qubits


def evolve(psi0, *, t, steps, terms=None, H=None, mode="terms",
           order=2, backend="c", strategy="auto", tile_bits=0,
           n_threads=0):
    """Evolve psi0 to time t with `steps` repetitions of the order-2
    or order-4 product formula. mode='terms' needs terms=(x, z,
    coeff); mode='grouped' needs the matrix H. Returns a new array."""
    psi0 = np.asarray(psi0, dtype=complex)
    n_qubits = _n_qubits(psi0)
    dts = schedule.stages(t, steps, order)
    if strategy not in _STRATEGY:
        raise ValueError(f"unknown strategy {strategy!r}")
    if mode == "terms":
        if terms is None:
            raise ValueError("mode='terms' needs terms=(x, z, coeff)")
        x, z, coeff = terms
        if backend == "python":
            return reference.evolve_terms(psi0, x, z, coeff, dts, steps)
        return _backend.evolve_terms(
            _backend.load(backend), psi0, n_qubits, x, z, coeff, dts,
            steps, _STRATEGY[strategy], tile_bits, n_threads)
    if mode == "grouped":
        if H is None or H.shape != (psi0.shape[0],) * 2:
            raise ValueError("mode='grouped' needs H matching psi0")
        if backend == "python":
            return reference.evolve_grouped(psi0, H, dts, steps)
        return _backend.evolve_grouped(
            _backend.load(backend), psi0, n_qubits, H, dts, steps,
            n_threads)
    raise ValueError("mode must be 'terms' or 'grouped'")
