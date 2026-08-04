"""Self-consistency tests for the correctness fixtures themselves.

These tests validate fixtures.py independently of any Pauli
decomposition implementation: they check that each fixture's stored
``expected_terms`` actually reconstructs the fixture's own Hamiltonian
matrix, using pauli_utils.py's dependency-free reconstruction (not the
library used to originally generate the fixture data). This guards
against the fixtures themselves being wrong or going stale if
hamiltonian.py's construction ever changes without regenerating them.

Any future Pauli decomposition implementation should be tested against
these same fixtures in its own test module (e.g.
test_fwht_decompose.py), asserting its output matches
fixture.expected_terms - not by re-deriving expected values inline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures import ALL_FIXTURES  # noqa: E402
from pauli_utils import reconstruct_from_terms  # noqa: E402


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fixture_reconstructs_hamiltonian(fixture):
    """The stored expected_terms must reproduce the padded Hamiltonian."""
    expected_padded = fixture.padded_hamiltonian()
    reconstructed = reconstruct_from_terms(fixture.expected_terms, fixture.n_qubits)

    max_error = np.max(np.abs(reconstructed.real - expected_padded))
    assert max_error < 1e-9, (
        f"fixture {fixture.name!r}: reconstruction from expected_terms "
        f"does not match the Hamiltonian (max error {max_error})"
    )

    max_imag = np.max(np.abs(reconstructed.imag))
    assert max_imag < 1e-9, (
        f"fixture {fixture.name!r}: reconstruction has non-negligible "
        f"imaginary part (max {max_imag}) - the Hamiltonian is real, so "
        "this would indicate mismatched or malformed Pauli terms"
    )


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fixture_term_labels_have_correct_length(fixture):
    """Every stored label must span exactly n_qubits characters."""
    for label in fixture.expected_terms:
        assert len(label) == fixture.n_qubits, (
            f"fixture {fixture.name!r}: label {label!r} has length "
            f"{len(label)}, expected {fixture.n_qubits}"
        )
        assert set(label) <= set("IXYZ"), (
            f"fixture {fixture.name!r}: label {label!r} contains "
            "characters outside I/X/Y/Z"
        )


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
def test_fixture_hamiltonian_shape(fixture):
    """The padded Hamiltonian dimension must be 2**n_qubits."""
    padded = fixture.padded_hamiltonian()
    assert padded.shape == (2 ** fixture.n_qubits, 2 ** fixture.n_qubits)
