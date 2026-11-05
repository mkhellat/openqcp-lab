# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Package surface and backend discovery."""

import ctypes
import subprocess
import sys

import trotterkit
from trotterkit import _backend


def test_bound_and_exact_reachable_after_plain_import():
    # A fresh interpreter: in this process other tests have already
    # imported the submodules, which would hide a missing import.
    code = ("import trotterkit; "
            "trotterkit.bound.bound_terms; trotterkit.exact.evolve; "
            "assert {'bound', 'exact'} <= set(trotterkit.__all__)")
    subprocess.run([sys.executable, "-c", code], check=True)


class _NoSymbols:
    """A library that loads but exports nothing."""

    def __getattr__(self, name):
        raise AttributeError(name)


def test_available_skips_library_missing_symbols(monkeypatch):
    real = ctypes.CDLL

    def fake_cdll(path, *args, **kwargs):
        if "libtrotterkit_cpp" in path:
            return _NoSymbols()
        if "libtrotterkit_rs" in path:
            raise OSError("not built")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(_backend, "_loaded", {})
    monkeypatch.setattr(_backend.ctypes, "CDLL", fake_cdll)
    assert _backend.available() == ["c"]
    assert trotterkit.backends() == ["c", "python"]
