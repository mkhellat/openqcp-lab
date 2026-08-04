"""Correctness tests for the original FWHT-based Pauli decomposition.

Tests at two levels:

1. Against a from-scratch, definition-level brute-force decomposition
   (Frobenius inner product against every tensor-product Pauli string,
   built independently of fwht_decompose.py's internals) on small
   random Hermitian matrices. This is the same derivation-verification
   approach used while developing the algorithm - see
   fwht_decompose.py's module docstring - re-run here as an automated
   regression test rather than a one-off manual check.
2. Against fixtures.ALL_FIXTURES (the coupled-oscillator Hamiltonians
   with PennyLane-derived expected terms), per fixtures.py's own
   instruction that implementations be tested against those fixtures
   directly rather than re-deriving expected values inline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures import ALL_FIXTURES  # noqa: E402
from fwht_decompose import (  # noqa: E402
    fwht_pauli_coefficients,
    fwht_pauli_terms,
    pauli_label,
)
from hamiltonian import pad_to_power_of_two  # noqa: E402
from pauli_utils import reconstruct_from_terms  # noqa: E402


def _popcount(value):
    return bin(value).count("1")


def _reference_pauli_matrix(x_mask, z_mask, n_qubits):
    """Definition-level Pauli string matrix, independent of fwht_decompose.py.

    Built directly from the symplectic representation
    P = kron_j( i**(x_j & z_j) * X**x_j * Z**z_j ), matching the same
    convention used to derive fwht_decompose.py's formulas (qubit j is
    bit (n_qubits - 1 - j) of the masks).
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    factors = []
    for qubit in range(n_qubits):
        bit = n_qubits - 1 - qubit
        xj = (x_mask >> bit) & 1
        zj = (z_mask >> bit) & 1
        phase = 1j ** (xj & zj)
        factor = phase * np.linalg.matrix_power(pauli_x, xj) @ np.linalg.matrix_power(
            pauli_z, zj
        )
        factors.append(factor)
    matrix = factors[0]
    for factor in factors[1:]:
        matrix = np.kron(matrix, factor)
    return matrix


def _reference_brute_force_decompose(hamiltonian, n_qubits):
    """O(4**n) definition-level decomposition: Frobenius inner product
    against every Pauli string, with no FWHT/shortcut involved."""
    dim = hamiltonian.shape[0]
    coefficients = np.zeros((dim, dim), dtype=complex)
    for x_mask in range(dim):
        for z_mask in range(dim):
            pauli = _reference_pauli_matrix(x_mask, z_mask, n_qubits)
            coefficients[x_mask, z_mask] = (
                np.trace(hamiltonian @ pauli.conj().T) / dim
            )
    return coefficients


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_fwht_matches_brute_force_on_random_hermitian(n_qubits):
    """The fast algorithm must match the definition-level brute force
    exactly (to floating-point precision) on general Hermitian input,
    not just on the structured coupled-oscillator fixtures."""
    rng = np.random.default_rng(seed=42 + n_qubits)
    dim = 2**n_qubits
    real_part = rng.random((dim, dim))
    imag_part = rng.random((dim, dim))
    matrix = real_part + 1j * imag_part
    hamiltonian = matrix + matrix.conj().T  # make Hermitian

    fast = fwht_pauli_coefficients(hamiltonian)
    reference = _reference_brute_force_decompose(hamiltonian, n_qubits)

    max_error = np.max(np.abs(fast - reference))
    assert max_error < 1e-9, f"n_qubits={n_qubits}: max error {max_error}"


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fwht_matches_fixture_expected_terms(fixture):
    """The fast algorithm's output must match fixtures.py's stored,
    independently-generated expected terms exactly."""
    padded = fixture.padded_hamiltonian()
    computed = fwht_pauli_terms(padded)

    assert set(computed) == set(fixture.expected_terms), (
        f"fixture {fixture.name!r}: label set mismatch - "
        f"missing {set(fixture.expected_terms) - set(computed)}, "
        f"extra {set(computed) - set(fixture.expected_terms)}"
    )
    for label, expected_coefficient in fixture.expected_terms.items():
        assert computed[label] == pytest.approx(expected_coefficient, abs=1e-9), (
            f"fixture {fixture.name!r}, term {label!r}: "
            f"got {computed[label]}, expected {expected_coefficient}"
        )


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fwht_terms_reconstruct_hamiltonian(fixture):
    """The fast algorithm's own output, independently reconstructed via
    pauli_utils (not fixtures.py's expected_terms), must reproduce H."""
    padded = fixture.padded_hamiltonian()
    computed = fwht_pauli_terms(padded)
    reconstructed = reconstruct_from_terms(computed, fixture.n_qubits)

    max_error = np.max(np.abs(reconstructed.real - padded))
    assert max_error < 1e-9, f"fixture {fixture.name!r}: max error {max_error}"


def test_pauli_label_round_trips_through_pauli_utils():
    """pauli_label's IXYZ strings must be understood by pauli_utils
    (shared convention across the module) for every single-term case
    at a couple of qubit counts."""
    from pauli_utils import pauli_string_to_matrix

    for n_qubits in [1, 2, 3]:
        dim = 2**n_qubits
        for x_mask in range(dim):
            for z_mask in range(dim):
                label = pauli_label(x_mask, z_mask, n_qubits)
                assert len(label) == n_qubits
                # pauli_utils's matrix, scaled by the same phase used
                # internally, should match the reference construction.
                reference = _reference_pauli_matrix(x_mask, z_mask, n_qubits)
                from_label = pauli_string_to_matrix(label)
                assert np.allclose(reference, from_label), (
                    f"label {label!r} (x={x_mask}, z={z_mask}, "
                    f"n_qubits={n_qubits}) does not match pauli_utils's "
                    "matrix for that label"
                )


def test_rejects_non_power_of_two_dimension():
    with pytest.raises(ValueError):
        fwht_pauli_coefficients(np.eye(5))


def test_rejects_non_square_input():
    with pytest.raises(ValueError):
        fwht_pauli_coefficients(np.zeros((4, 8)))


def test_padded_odd_size_hamiltonian_decomposes_correctly():
    """Exercises the pad_to_power_of_two + fwht_pauli_terms combination
    end to end on a small, non-power-of-two matrix, as used for real
    coupled-oscillator Hamiltonians."""
    rng = np.random.default_rng(7)
    raw = rng.random((5, 5))
    raw = raw + raw.T
    padded, n_qubits = pad_to_power_of_two(raw)
    assert padded.shape == (8, 8)
    assert n_qubits == 3

    terms = fwht_pauli_terms(padded)
    reconstructed = reconstruct_from_terms(terms, n_qubits)
    max_error = np.max(np.abs(reconstructed.real - padded))
    assert max_error < 1e-9
