# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Sizes that overflow size_t return TK_ENOMEM, as the header says.

n_stages = 2^62 makes the per-call constant tables 2^62 x (terms or
blocks) x (4 or 6) doubles, which wraps a 64-bit byte count. A
backend that multiplies unchecked allocates a small buffer and
writes past it, so each case runs in a subprocess: a crash fails
the test instead of the test run.
"""

import subprocess
import sys

import pytest

import trotterkit

NATIVE = [b for b in trotterkit.backends() if b != "python"]

SCRIPT = """
import ctypes, sys
import numpy as np
from trotterkit import _backend
lib = _backend.load({backend!r})
p = lambda a: a.ctypes.data_as(ctypes.c_void_p)
psi = np.array([0.6, 0, 0.8, 0], dtype=np.float64)  # 1 qubit
keep = psi.copy()
dt = np.array([0.1])
sched = _backend.Schedule(p(dt), 1 << 62, 1)
if {mode!r} == "terms":
    x = np.array([1], dtype=np.uint64); z = np.array([0], dtype=np.uint64)
    cr = np.array([0.5]); ci = np.array([0.0])
    t = _backend.Terms(p(x), p(z), p(cr), p(ci), 1)
    rc = lib.tk_evolve_terms(p(psi), 1, ctypes.byref(t),
                             ctypes.byref(sched), 0, 0, 1)
else:
    row = np.array([0, 1], dtype=np.uint64)
    col = np.array([1, 0], dtype=np.uint64)
    vr = np.array([0.5, 0.5]); vi = np.zeros(2)
    h = ctypes.c_void_p()
    assert lib.tk_groups_build(1, p(row), p(col), p(vr), p(vi), 2,
                               ctypes.byref(h)) == 0
    rc = lib.tk_evolve_grouped(p(psi), 1, h, ctypes.byref(sched), 1)
    lib.tk_groups_free(h)
assert rc == 2, rc            # TK_ENOMEM
assert (psi == keep).all()
"""


@pytest.mark.parametrize("backend", NATIVE)
@pytest.mark.parametrize("mode", ["terms", "grouped"])
def test_overflowing_size_returns_enomem(backend, mode):
    res = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(backend=backend, mode=mode)],
        capture_output=True, text=True)
    assert res.returncode == 0, (res.returncode, res.stderr[-400:])
