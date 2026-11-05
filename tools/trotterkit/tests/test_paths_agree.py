# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Every kernel path gives the same final state, bit for bit.

The grouped (SIMD) per-term path does the scalar path's arithmetic in
the same order, so the two agree exactly. TROTTERKIT_FORCE_SCALAR=1
routes every term through the scalar path; it is read once per call,
so each comparison runs in a fresh interpreter.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

TESTS = pathlib.Path(__file__).resolve().parent
SCRIPT = """
import sys
import numpy as np
import trotterkit
sys.path.insert(0, {tests!r})
from _systems import random_pauli_sum, random_state
x, z, c, _ = random_pauli_sum({n}, {n_terms}, seed={seed},
                              complex_coeffs=True)
x = np.asarray({x!r}, dtype=np.uint64) if {x!r} is not None else x
z = np.asarray({z!r}, dtype=np.uint64) if {z!r} is not None else z
c = c[:len(x)]
psi = trotterkit.evolve(random_state({n}, 1), t=0.9, steps=3, order=4,
                        terms=(x, z, c), backend="c", n_threads=1)
np.save(sys.argv[1], psi)
"""


def run(tmp_path, scalar, n, seed=0, n_terms=40, x=None, z=None):
    out = tmp_path / f"psi-{int(scalar)}.npy"
    env = {**os.environ, "TROTTERKIT_FORCE_SCALAR": "1" if scalar
           else "0"}
    code = SCRIPT.format(tests=str(TESTS), n=n, n_terms=n_terms,
                         seed=seed, x=x, z=z)
    subprocess.run([sys.executable, "-c", code, str(out)], check=True,
                   env=env)
    return np.load(out)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_vector_kernel_matches_scalar(tmp_path, n, seed):
    np.testing.assert_array_equal(run(tmp_path, False, n, seed),
                                  run(tmp_path, True, n, seed))


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_each_lane_permutation(tmp_path, p):
    # x = 4 + p, 8 + p, 16 + p (top bit >= 2, low bits p) with every
    # z below 32, plus diagonal terms: every lane permutation and sign
    # pattern of the grouped path.
    xs = [b + p for b in (4, 8, 16) for _ in range(32)] + [0] * 32
    zs = list(range(32)) * 4
    np.testing.assert_array_equal(
        run(tmp_path, False, 5, n_terms=len(xs), x=xs, z=zs),
        run(tmp_path, True, 5, n_terms=len(xs), x=xs, z=zs))
