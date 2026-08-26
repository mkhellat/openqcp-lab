import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

def synth(n):
    sc = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
    masses = [1.0 + 0.05 * i for i in range(n)]
    return pad_to_power_of_two(build_hamiltonian(n, sc, masses))

for n_osc in [16, 30, 50]:
    padded, n_qubits = synth(n_osc)
    dim = padded.shape[0]
    nnz = np.count_nonzero(padded)
    # Count how many WHT rows (fixed x) have an all-zero gather: gathered[x,q]=op[q^x,q]
    q = np.arange(dim)
    nonzero_rows_of_op = np.any(padded != 0, axis=1)  # rows p with any nonzero entry
    # gathered[x, q] = padded[q^x, q]; nonzero only if p=q^x is among rows with nonzero entries
    # x row is entirely zero iff for every q, padded[q^x, q] == 0
    x_all = np.arange(dim)
    active_x = np.zeros(dim, dtype=bool)
    # vectorized: for each x, gathered row = padded[q^x, q] for all q
    p_indices = q[np.newaxis, :] ^ x_all[:, np.newaxis]
    gathered_all = padded[p_indices, q[np.newaxis, :]]
    nnz_per_row = np.count_nonzero(gathered_all, axis=1)
    n_nonzero_rows = np.count_nonzero(nnz_per_row)
    print(f"N={n_osc:3d} qubits={n_qubits:2d} dim={dim:5d} operator_nnz={nnz:5d} "
          f"WHT_rows_total={dim:6d} WHT_rows_with_any_nonzero={n_nonzero_rows:6d} "
          f"({100*n_nonzero_rows/dim:.2f}%) avg_nnz_per_active_row={nnz_per_row[nnz_per_row>0].mean():.2f}")
