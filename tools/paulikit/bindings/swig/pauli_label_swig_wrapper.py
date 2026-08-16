"""Python-facing wrapper around the SWIG-generated pauli_label_swig
module (Phase 3a, binding 4/4). Exposes the same pauli_label/
pauli_label_batch signatures as the other three bindings for a
like-for-like comparison.
"""

import numpy as np

import pauli_label_swig


def pauli_label(x_mask: int, z_mask: int, n_qubits: int) -> str:
    """Single-term IXYZ label, matching fwht.pauli_label exactly."""
    return pauli_label_swig.pauli_label_str(x_mask, z_mask, n_qubits)


def pauli_label_batch(x_masks, z_masks, n_qubits: int) -> list[str]:
    """Batch IXYZ labels for parallel arrays of (x, z) masks.

    One call into the C batch kernel regardless of term count.
    """
    x_arr = np.ascontiguousarray(x_masks, dtype=np.uint32)
    z_arr = np.ascontiguousarray(z_masks, dtype=np.uint32)
    if x_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("x_masks and z_masks must have the same length")

    n_terms = x_arr.shape[0]
    out = bytearray(n_terms * n_qubits)
    pauli_label_swig.pauli_label_batch_raw(x_arr, z_arr, n_qubits, out)

    return [
        bytes(out[i * n_qubits:(i + 1) * n_qubits]).decode("ascii")
        for i in range(n_terms)
    ]
