# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
"""Binary workload files read by every backend's CLI.

Little-endian. A 64-byte header (magic "TKWL", version, n_qubits,
n_terms, nnz, SHA-256 of the payload) followed by the payload: x, z
(u64), c_re, c_im (f64), the COO of H (row, col as u64, v_re, v_im as
f64) and psi0 as interleaved f64 pairs.
"""

import dataclasses
import hashlib
import struct

import numpy as np
import scipy.sparse as sp

MAGIC = b"TKWL"
VERSION = 1
_HEADER = struct.Struct("<4sIIIQQ32s")


@dataclasses.dataclass
class Workload:
    n_qubits: int
    x: np.ndarray
    z: np.ndarray
    coeff: np.ndarray
    psi0: np.ndarray
    row: np.ndarray
    col: np.ndarray
    val: np.ndarray


def write(path, n_qubits, x, z, coeff, psi0, H=None):
    x = np.ascontiguousarray(x, dtype="<u8")
    z = np.ascontiguousarray(z, dtype="<u8")
    c = np.asarray(coeff, dtype=complex)
    psi0 = np.ascontiguousarray(psi0, dtype="<c16")
    if not len(x) == len(z) == len(c):
        raise ValueError("x, z and coeff must have equal length")
    if psi0.shape != (1 << n_qubits,):
        raise ValueError("psi0 must have length 2**n_qubits")
    if H is None:
        row = col = np.zeros(0, dtype="<u8")
        val = np.zeros(0, dtype=complex)
    else:
        m = sp.coo_array(H)
        m.sum_duplicates()
        row = m.row.astype("<u8")
        col = m.col.astype("<u8")
        val = m.data.astype(complex)
    parts = [x, z, np.ascontiguousarray(c.real, dtype="<f8"),
             np.ascontiguousarray(c.imag, dtype="<f8"), row, col,
             np.ascontiguousarray(val.real, dtype="<f8"),
             np.ascontiguousarray(val.imag, dtype="<f8"), psi0]
    payload = b"".join(p.tobytes() for p in parts)
    header = _HEADER.pack(MAGIC, VERSION, n_qubits, 0, len(x),
                          len(row), hashlib.sha256(payload).digest())
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)


def read(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < _HEADER.size or data[:4] != MAGIC:
        raise ValueError(f"{path}: not a trotterkit workload")
    magic, version, n_qubits, _, n, nnz, digest = _HEADER.unpack_from(
        data)
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    payload = data[_HEADER.size:]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError(f"{path}: payload checksum mismatch")
    dim = 1 << n_qubits
    off = 0

    def take(dtype, count):
        nonlocal off
        a = np.frombuffer(payload, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a.copy()

    x, z = take("<u8", n), take("<u8", n)
    c = take("<f8", n) + 1j * take("<f8", n)
    row, col = take("<u8", nnz), take("<u8", nnz)
    val = take("<f8", nnz) + 1j * take("<f8", nnz)
    psi0 = take("<c16", dim)
    if off != len(payload):
        raise ValueError(f"{path}: payload length mismatch")
    return Workload(n_qubits, x, z, c, psi0, row, col, val)
