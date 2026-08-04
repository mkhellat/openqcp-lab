"""Minimal, dependency-free Pauli-matrix utilities shared by tests and
implementations in this module.

Deliberately does not depend on PennyLane, Qiskit, or Classiq: this
module is meant to remain usable as an independent check even if the
libraries used to *generate* the fixtures (see fixtures.py) are absent
or change behavior.
"""

import numpy as np

_PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_to_matrix(label):
    """Build the dense matrix for a Pauli string label via tensor product.

    Args:
        label: A string of Pauli letters, e.g. ``"IXZ"``. The leftmost
            character is qubit 0, matching the convention used by
            ``fixtures.pauli_word_to_label``.

    Returns:
        A ``(2**len(label), 2**len(label))`` complex ``numpy.ndarray``.
    """
    matrix = _PAULI_MATRICES[label[0]]
    for letter in label[1:]:
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def reconstruct_from_terms(terms, n_qubits):
    """Reconstruct a dense Hamiltonian from a label -> coefficient dict.

    Args:
        terms: Mapping from Pauli-string label to real coefficient,
            as produced by an implementation under test or stored in
            fixtures.py.
        n_qubits: Number of qubits; every label in ``terms`` must have
            this length.

    Returns:
        A ``(2**n_qubits, 2**n_qubits)`` complex ``numpy.ndarray``
        equal to the sum of ``coefficient * pauli_string_to_matrix(label)``
        over all terms.
    """
    dim = 2 ** n_qubits
    total = np.zeros((dim, dim), dtype=complex)
    for label, coefficient in terms.items():
        assert len(label) == n_qubits, (
            f"label {label!r} has length {len(label)}, expected {n_qubits}"
        )
        total += coefficient * pauli_string_to_matrix(label)
    return total
