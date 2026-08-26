import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_coefficients, _popcount_array

def synth(n):
    sc = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
    masses = [1.0 + 0.05 * i for i in range(n)]
    return pad_to_power_of_two(build_hamiltonian(n, sc, masses))

def fwht_pauli_coefficients_sparse_v2(operator):
    """Fully vectorized: no Python loop over active x rows.

    For each nonzero operator entry (p, q, v), it contributes to
    active-row x = p^q with term v*(-1)**popcount(q&z) at every z.
    Build the (n_active_x, dim) sign matrix for ALL nonzero entries at
    once via broadcasting, weight by v, then scatter-add (np.add.at)
    grouped by x into the (dim, dim) transformed-gather array. This
    keeps memory at O(nnz * dim) instead of O(dim * dim), which is the
    win when nnz << dim (true here: nnz = O(N), dim = 2**n).
    """
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz
    nnz = len(vals)

    z_all = np.arange(dim)
    # signs[i, z] = (-1)**popcount(q_nz[i] & z), shape (nnz, dim)
    and_mat = q_nz[:, None] & z_all[None, :]
    signs = 1 - 2 * (_popcount_array(and_mat, n_qubits) & 1)
    contributions = vals[:, None] * signs  # (nnz, dim)

    transformed = np.zeros((dim, dim), dtype=complex)
    np.add.at(transformed, x_nz, contributions)

    xz_and = np.arange(dim)[:, None] & z_all[None, :]
    phase = 1j ** _popcount_array(xz_and, n_qubits)
    coefficients = transformed * np.conj(phase) / dim
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50]:
        padded, n_qubits = synth(n_osc)
        ref = fwht_pauli_coefficients(padded)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_sparse_v2(padded)
        t1 = time.perf_counter()
        diff = np.max(np.abs(ref - sp))
        print(f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} max_diff={diff:.3e} time={t1-t0:.4f}")
