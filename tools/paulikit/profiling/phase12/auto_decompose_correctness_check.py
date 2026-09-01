"""End-to-end (not mocked) correctness sanity check for
auto_decompose() - confirms it picks the expected path and returns
numerically correct results, matching fwht_pauli_terms as ground
truth, at real N=25/50 scale.
"""
import time

from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import auto_decompose, fwht_pauli_terms
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two


def make_padded(n):
    spring_constants = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
    masses = [1.0 + 0.05 * i for i in range(n)]
    sparse = build_hamiltonian(n, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)
    return padded, n_qubits


for n in (25, 50):
    padded, n_qubits = make_padded(n)
    dim = padded.shape[0]

    budget = autotune.available_memory_bytes()
    estimated_dense_bytes = dim * dim * 16
    expected_path = "dense" if estimated_dense_bytes <= budget * 0.5 else "streaming"

    reference = fwht_pauli_terms(padded)

    t0 = time.perf_counter()
    result = auto_decompose(padded)
    if isinstance(result, dict):
        actual_path = "dense"
        combined = result
    else:
        actual_path = "streaming"
        combined = {}
        for chunk in result:
            combined.update(chunk)
    elapsed = time.perf_counter() - t0

    assert set(combined) == set(reference), f"N={n}: label set mismatch"
    max_err = max(abs(combined[label] - reference[label]) for label in reference)

    print(f"N={n} dim={dim} estimated_dense_bytes={estimated_dense_bytes:,} "
          f"budget={budget:,}")
    print(f"  expected_path={expected_path} actual_path={actual_path} "
          f"{'OK' if expected_path == actual_path else 'MISMATCH!!'}")
    print(f"  terms={len(reference)} max_coefficient_error={max_err:.3e} "
          f"elapsed={elapsed:.4f}s")
    assert actual_path == expected_path, "auto_decompose picked an unexpected path"
    assert max_err < 1e-9, "auto_decompose result diverges from fwht_pauli_terms"

print("ALL CORRECTNESS CHECKS PASSED")
