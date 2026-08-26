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

def fwht_pauli_coefficients_sparse_v3(operator):
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz
    nnz = len(vals)

    z_all = np.arange(dim)
    and_mat = q_nz[:, None] & z_all[None, :]
    signs = 1.0 - 2.0 * (_popcount_array(and_mat, n_qubits) & 1)  # real (+-1), shape (nnz, dim)
    contributions = vals[:, None] * signs  # (nnz, dim) complex

    order = np.argsort(x_nz, kind="stable")
    x_sorted = x_nz[order]
    contributions_sorted = contributions[order]

    unique_x, start_idx = np.unique(x_sorted, return_index=True)
    # reduceat sums each contiguous group -> (n_active_x, dim)
    grouped = np.add.reduceat(contributions_sorted, start_idx, axis=0)

    transformed = np.zeros((dim, dim), dtype=complex)
    transformed[unique_x] = grouped

    xz_and = np.arange(dim)[:, None] & z_all[None, :]
    phase = 1j ** _popcount_array(xz_and, n_qubits)
    coefficients = transformed * np.conj(phase) / dim
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50, 100]:
        padded, n_qubits = synth(n_osc)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_sparse_v3(padded)
        t1 = time.perf_counter()
        msg = f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} time={t1-t0:.4f}"
        if n_osc <= 30:
            ref = fwht_pauli_coefficients(padded)
            diff = np.max(np.abs(ref - sp))
            msg += f" max_diff={diff:.3e}"
        print(msg)
