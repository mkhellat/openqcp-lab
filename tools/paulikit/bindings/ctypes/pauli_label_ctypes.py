"""ctypes binding for the pauli_label C kernel (Phase 3a, binding 3/4).

Stdlib-only - no compiler invoked at import time, unlike Cython/CFFI.
Loads the shared library built by ./build_pauli_label_shared (run
that first). Exposes the same pauli_label/pauli_label_batch
signatures as the other bindings for a like-for-like comparison.
"""

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "libpauli_label.so"
if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"{_LIB_PATH} not found - run ./build_pauli_label_shared first"
    )

_lib = ctypes.CDLL(str(_LIB_PATH))

_lib.pauli_label.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_int,
    ctypes.c_char_p,
]
_lib.pauli_label.restype = None

_lib.pauli_label_batch.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.c_char_p,
]
_lib.pauli_label_batch.restype = None


def pauli_label(x_mask: int, z_mask: int, n_qubits: int) -> str:
    """Single-term IXYZ label, matching fwht.pauli_label exactly."""
    buf = ctypes.create_string_buffer(n_qubits + 1)
    _lib.pauli_label(x_mask, z_mask, n_qubits, buf)
    return buf.value.decode("ascii")


def pauli_label_batch(x_masks, z_masks, n_qubits: int) -> list[str]:
    """Batch IXYZ labels for parallel arrays of (x, z) masks.

    One call into the C batch kernel regardless of term count.
    """
    x_arr = np.ascontiguousarray(x_masks, dtype=np.uint32)
    z_arr = np.ascontiguousarray(z_masks, dtype=np.uint32)
    if x_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("x_masks and z_masks must have the same length")

    n_terms = x_arr.shape[0]
    x_ptr = x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    z_ptr = z_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    out = ctypes.create_string_buffer(n_terms * n_qubits)

    _lib.pauli_label_batch(x_ptr, z_ptr, n_terms, n_qubits, out)

    raw = out.raw
    return [
        raw[i * n_qubits:(i + 1) * n_qubits].decode("ascii")
        for i in range(n_terms)
    ]
