import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_coefficients, _walsh_hadamard_transform_rows

def synth(n):
    sc = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
    masses = [1.0 + 0.05 * i for i in range(n)]
    return pad_to_power_of_two(build_hamiltonian(n, sc, masses))

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def popcount_fast(arr):
    arr = arr.astype(np.uint32)
    total = np.zeros(arr.shape, dtype=np.int64)
    for shift in (0, 8, 16, 24):
        total += _POPCOUNT8[(arr >> shift) & 0xFF]
    return total

def fwht_pauli_coefficients_v7(operator):
    """Build the (n_active, dim) gathered array directly by scattering
    the nonzero operator entries -- never touch a full (dim,dim) gather.
    Then run the existing dense WHT butterfly only on those n_active
    rows (unavoidable O(dim log dim) work per active row, but that's
    already NOT the dominant cost -- see v6 profiling)."""
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz

    unique_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(unique_x)

    gathered_active = np.zeros((n_active, dim), dtype=complex)
    gathered_active[inverse, q_nz] = vals  # scatter: O(nnz), not O(dim^2)

    transformed_active = _walsh_hadamard_transform_rows(gathered_active)

    idx = np.arange(dim)
    xz_and = unique_x[:, None] & idx[None, :]
    phase = 1j ** popcount_fast(xz_and)
    active_coeffs = transformed_active * np.conj(phase) / dim

    coefficients = np.zeros((dim, dim), dtype=complex)
    coefficients[unique_x] = active_coeffs
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50, 100]:
        padded, n_qubits = synth(n_osc)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_v7(padded)
        t1 = time.perf_counter()
        msg = f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} time={t1-t0:.4f}"
        if n_osc <= 30:
            ref = fwht_pauli_coefficients(padded)
            diff = np.max(np.abs(ref - sp))
            msg += f" max_diff={diff:.3e}"
        print(msg)
