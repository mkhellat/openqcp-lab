"""Combine: skip inactive rows (v6/v7) AND replace the O(dim log dim)
per-row butterfly with the exact O(k*dim) sparse-impulse WHT identity
(verified in sparse_wht_check.py), computed with a fast LUT popcount
instead of the bit-serial Python loop in _popcount_array. This avoids
ever touching a full (dim,dim) or (n_active,dim) all at once for the
sign computation, by doing it per-nnz-batch grouped via reduceat
(matching the working v5 grouping) but with the fast popcount.
"""
import sys
from pathlib import Path
import time
import cProfile
import pstats
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_coefficients

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

def fwht_pauli_coefficients_v8(operator):
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz

    order = np.argsort(x_nz, kind="stable")
    x_sorted = x_nz[order]
    q_sorted = q_nz[order]
    vals_sorted = vals[order]
    unique_x, start_idx = np.unique(x_sorted, return_index=True)

    z_all = np.arange(dim)
    and_mat = q_sorted[:, None] & z_all[None, :]           # (nnz, dim)
    signs = 1.0 - 2.0 * (popcount_fast(and_mat) & 1).astype(np.float64)
    contributions = vals_sorted[:, None] * signs            # (nnz, dim)
    grouped = np.add.reduceat(contributions, start_idx, axis=0)  # (n_active, dim)

    xz_and = unique_x[:, None] & z_all[None, :]
    phase = 1j ** popcount_fast(xz_and)
    active_coeffs = grouped * np.conj(phase) / dim

    coefficients = np.zeros((dim, dim), dtype=complex)
    coefficients[unique_x] = active_coeffs
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50, 100]:
        padded, n_qubits = synth(n_osc)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_v8(padded)
        t1 = time.perf_counter()
        msg = f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} time={t1-t0:.4f}"
        if n_osc <= 30:
            ref = fwht_pauli_coefficients(padded)
            diff = np.max(np.abs(ref - sp))
            msg += f" max_diff={diff:.3e}"
        print(msg)
