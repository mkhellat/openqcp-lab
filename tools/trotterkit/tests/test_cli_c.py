# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""trotterkit-c reproduces the in-process C evolution exactly."""

import json
import pathlib
import subprocess

import numpy as np
import pytest
import scipy.sparse as sp

import trotterkit
from _systems import random_pauli_sum, random_state
from trotterkit import workload

CLI = (pathlib.Path(trotterkit.__file__).resolve().parent / "_lib"
       / "trotterkit-c")
pytestmark = pytest.mark.skipif(not CLI.exists(),
                                reason="trotterkit-c not built")


def _workload(tmp_path, seed=7):
    x, z, c, H = random_pauli_sum(4, 9, seed=seed, complex_coeffs=True)
    psi0 = random_state(4, seed + 1)
    path = tmp_path / "w.tkwl"
    workload.write(path, 4, x, z, c, psi0, H=sp.coo_array(H))
    return path, x, z, c, H, psi0


@pytest.mark.parametrize("mode", ["terms", "grouped"])
@pytest.mark.parametrize("order", [2, 4])
def test_cli_matches_in_process(tmp_path, mode, order):
    path, x, z, c, H, psi0 = _workload(tmp_path)
    out = tmp_path / "psi.bin"
    res = subprocess.run(
        [str(CLI), str(path), "--mode", mode, "--t", "0.7",
         "--steps", "3", "--order", str(order), "--out", str(out)],
        capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)
    got = np.fromfile(out, dtype="<f8").view(complex)
    want = trotterkit.evolve(psi0, t=0.7, steps=3, order=order,
                             mode=mode, terms=(x, z, c),
                             H=sp.csr_array(H), backend="c")
    np.testing.assert_array_equal(got, want)
    assert info["backend"] == "c" and info["mode"] == mode
    assert info["n_terms"] == 9 and info["kernel_seconds"] >= 0
    assert info["norm2"] == pytest.approx(np.vdot(want, want).real,
                                          rel=1e-14)


def test_cli_usage_error_exit_code():
    assert subprocess.run([str(CLI)], capture_output=True).returncode == 2


def test_cli_rejects_corrupt_file(tmp_path):
    path = _workload(tmp_path)[0]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    res = subprocess.run([str(CLI), str(path), "--t", "1", "--steps",
                          "1"], capture_output=True, text=True)
    assert res.returncode == 1 and "checksum" in res.stderr


def test_cli_kernel_error_exit_code(tmp_path):
    path = _workload(tmp_path)[0]
    res = subprocess.run([str(CLI), str(path), "--t", "1", "--steps",
                          "1", "--threads", "-1"],
                         capture_output=True, text=True)
    assert res.returncode == 1 and "invalid argument" in res.stderr
