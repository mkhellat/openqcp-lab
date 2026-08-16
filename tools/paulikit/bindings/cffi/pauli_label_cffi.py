"""Python-facing wrapper around the compiled CFFI extension
(_pauli_label_cffi, built by build_pauli_label_cffi.py).

Exposes the same two entry points as the Cython binding
(bindings/cython/pauli_label_cy.pyx) for a like-for-like comparison:
single-term ``pauli_label`` and array-batch ``pauli_label_batch``.
"""

import numpy as np

from _pauli_label_cffi import ffi, lib


def pauli_label(x_mask: int, z_mask: int, n_qubits: int) -> str:
    """Single-term IXYZ label, matching fwht.pauli_label exactly."""
    out = ffi.new("char[]", n_qubits + 1)
    lib.pauli_label(x_mask, z_mask, n_qubits, out)
    return ffi.string(out).decode("ascii")


def pauli_label_batch(x_masks, z_masks, n_qubits: int) -> list[str]:
    """Batch IXYZ labels for parallel arrays of (x, z) masks.

    One call into the C batch kernel regardless of term count.
    """
    x_arr = np.ascontiguousarray(x_masks, dtype=np.uint32)
    z_arr = np.ascontiguousarray(z_masks, dtype=np.uint32)
    if x_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("x_masks and z_masks must have the same length")

    n_terms = x_arr.shape[0]
    x_cdata = ffi.cast("uint32_t *", x_arr.ctypes.data)
    z_cdata = ffi.cast("uint32_t *", z_arr.ctypes.data)
    out = ffi.new("char[]", n_terms * n_qubits)

    lib.pauli_label_batch(x_cdata, z_cdata, n_terms, n_qubits, out)

    buf = ffi.buffer(out)
    return [
        bytes(buf[i * n_qubits:(i + 1) * n_qubits]).decode("ascii")
        for i in range(n_terms)
    ]
