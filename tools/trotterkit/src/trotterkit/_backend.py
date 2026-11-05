# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""ctypes bindings to the backend shared libraries (one C ABI)."""

import ctypes
import os
import pathlib

import numpy as np
import scipy.sparse as sp

_SUFFIX = {"c": "c", "cpp": "cpp", "rust": "rs"}
_LIB_DIR = pathlib.Path(__file__).resolve().parent / "_lib"
_loaded = {}

c_void_p = ctypes.c_void_p


class TrotterkitError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f"{message} (code {code})")
        self.code = code


class Terms(ctypes.Structure):
    _fields_ = [("x", c_void_p), ("z", c_void_p), ("c_re", c_void_p),
                ("c_im", c_void_p), ("n", ctypes.c_size_t)]


class Schedule(ctypes.Structure):
    _fields_ = [("dt", c_void_p), ("n_stages", ctypes.c_size_t),
                ("n_steps", ctypes.c_size_t)]


def _path(backend):
    base = os.environ.get("TROTTERKIT_LIB_DIR")
    directory = pathlib.Path(base) if base else _LIB_DIR
    return directory / f"libtrotterkit_{_SUFFIX[backend]}.so"


def load(backend):
    if backend not in _SUFFIX:
        raise ValueError(f"unknown backend {backend!r}")
    if backend in _loaded:
        return _loaded[backend]
    lib = ctypes.CDLL(str(_path(backend)))
    lib.tk_evolve_terms.argtypes = [
        c_void_p, ctypes.c_uint32, ctypes.POINTER(Terms),
        ctypes.POINTER(Schedule), ctypes.c_int, ctypes.c_int,
        ctypes.c_int]
    lib.tk_evolve_terms.restype = ctypes.c_int
    lib.tk_groups_build.argtypes = [
        ctypes.c_uint32, c_void_p, c_void_p, c_void_p, c_void_p,
        ctypes.c_size_t, ctypes.POINTER(c_void_p)]
    lib.tk_groups_build.restype = ctypes.c_int
    lib.tk_evolve_grouped.argtypes = [
        c_void_p, ctypes.c_uint32, c_void_p, ctypes.POINTER(Schedule),
        ctypes.c_int]
    lib.tk_evolve_grouped.restype = ctypes.c_int
    lib.tk_groups_free.argtypes = [c_void_p]
    lib.tk_groups_free.restype = None
    lib.tk_backend_name.restype = ctypes.c_char_p
    lib.tk_strerror.argtypes = [ctypes.c_int]
    lib.tk_strerror.restype = ctypes.c_char_p
    _loaded[backend] = lib
    return lib


def available():
    """Native backends whose library loads and exports the full ABI.

    A missing library raises OSError; one that loads but lacks a
    symbol raises AttributeError. Either way the backend is skipped.
    """
    found = []
    for backend in _SUFFIX:
        try:
            load(backend)
        except (OSError, AttributeError):
            continue
        found.append(backend)
    return found


def _ptr(a):
    return a.ctypes.data_as(c_void_p)


def _check(lib, rc):
    if rc != 0:
        raise TrotterkitError(rc, lib.tk_strerror(rc).decode())


def _schedule(stage_dts, steps):
    dts = np.ascontiguousarray(stage_dts, dtype=np.float64)
    return dts, Schedule(_ptr(dts), len(dts), steps)


def evolve_terms(lib, psi, n_qubits, x, z, coeff, stage_dts, steps,
                 strategy, tile_bits, n_threads):
    x = np.ascontiguousarray(x, dtype=np.uint64)
    z = np.ascontiguousarray(z, dtype=np.uint64)
    c = np.asarray(coeff, dtype=complex)
    if not (x.ndim == z.ndim == c.ndim == 1
            and len(x) == len(z) == len(c)):
        raise ValueError(
            "x, z and coeff must be 1-D arrays of equal length")
    c_re = np.ascontiguousarray(c.real)
    c_im = np.ascontiguousarray(c.imag)
    out = np.array(psi, dtype=np.complex128, order="C", copy=True)
    terms = Terms(_ptr(x), _ptr(z), _ptr(c_re), _ptr(c_im), len(c))
    dts, sched = _schedule(stage_dts, steps)
    _check(lib, lib.tk_evolve_terms(
        _ptr(out), n_qubits, ctypes.byref(terms), ctypes.byref(sched),
        strategy, tile_bits, n_threads))
    return out


def evolve_grouped(lib, psi, n_qubits, H, stage_dts, steps, n_threads):
    m = sp.csr_array(H, dtype=complex, copy=True)
    m.sum_duplicates()
    coo = m.tocoo()
    row = np.ascontiguousarray(coo.row, dtype=np.uint64)
    col = np.ascontiguousarray(coo.col, dtype=np.uint64)
    v_re = np.ascontiguousarray(coo.data.real)
    v_im = np.ascontiguousarray(coo.data.imag)
    handle = c_void_p()
    _check(lib, lib.tk_groups_build(n_qubits, _ptr(row), _ptr(col),
                                    _ptr(v_re), _ptr(v_im), len(row),
                                    ctypes.byref(handle)))
    try:
        out = np.array(psi, dtype=np.complex128, order="C", copy=True)
        dts, sched = _schedule(stage_dts, steps)
        _check(lib, lib.tk_evolve_grouped(_ptr(out), n_qubits, handle,
                                          ctypes.byref(sched),
                                          n_threads))
    finally:
        lib.tk_groups_free(handle)
    return out
