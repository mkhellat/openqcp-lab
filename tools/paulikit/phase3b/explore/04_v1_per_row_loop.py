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

def fwht_pauli_coefficients_sparse(operator):
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    # Nonzero entries of the operator matrix itself.
    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz  # since gathered[x,q]=operator[q^x,q] is nonzero at x = p^q

    dim_arr = np.arange(dim)
    coefficients = np.zeros((dim, dim), dtype=complex)

    # Group by x (the WHT "row" index) so each active row does ONE
    # sparse-WHT pass over all its nonzero (q, v) pairs at once.
    order = np.argsort(x_nz, kind="stable")
    x_sorted = x_nz[order]
    q_sorted = q_nz[order]
    v_sorted = vals[order]
    boundaries = np.searchsorted(x_sorted, np.arange(dim + 1))

    z_indices = dim_arr
    xz_and_all = None  # computed per-x below only for active x (still need phase per z)

    for x in range(dim):
        start, end = boundaries[x], boundaries[x + 1]
        if start == end:
            continue
        qs = q_sorted[start:end]
        vs = v_sorted[start:end]
        # WHT contribution at all z: sum_i v_i * (-1)**popcount(q_i & z)
        and_mat = qs[:, None] & z_indices[None, :]
        signs = 1 - 2 * (_popcount_array(and_mat, n_qubits) & 1)
        row_transformed = (vs[:, None] * signs).sum(axis=0)

        xz_and = x & z_indices
        phase = 1j ** _popcount_array(xz_and, n_qubits)
        coefficients[x, :] = row_transformed * np.conj(phase) / dim

    return coefficients

for n_osc in [16, 30]:
    padded, n_qubits = synth(n_osc)
    ref = fwht_pauli_coefficients(padded)
    t0 = time.perf_counter()
    sparse = fwht_pauli_coefficients_sparse(padded)
    t1 = time.perf_counter()
    diff = np.max(np.abs(ref - sparse))
    print(f"N={n_osc} dim={padded.shape[0]} max_diff={diff:.3e} sparse_time={t1-t0:.4f}")
