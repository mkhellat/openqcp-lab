"""Avoid the (nnz, dim) intermediate by using the FWHT itself to compute
signs implicitly via matrix multiply against a precomputed (n_distinct_q,
dim) Hadamard-sign matrix -- but built via actual fast Hadamard transform
(apply WHT to one-hot vectors, O(dim log dim) each) instead of an explicit
popcount, OR just use scipy.linalg.hadamard-style fast matmul.

Simpler and more effective: note contributions.sum over the (nnz,dim) axis
grouped by x is a matrix product: grouped[x] = sum_q S[q,:] * V[x,q] where
V[x,q] is the (sparse!) matrix of operator entries indexed by (x=p^q, q),
and S[q,z] = (-1)**popcount(q&z) is the (dim,dim) sign matrix (same as an
unnormalized Hadamard matrix row-permuted). So: grouped = V @ S where V is
a SPARSE (dim, dim) matrix (nnz entries) and S is the dense (dim,dim) sign
matrix (this is literally the Hadamard matrix, up to row/col permutation).
Using scipy.sparse for V and a FAST WHT applied to S's columns (batched)
turns "for each active x, WHT of a k-sparse vector" into "sparse-matrix
times dense Hadamard matrix", i.e. exactly running one WHT per COLUMN of
V^T restricted to nonzero rows -- but scipy sparse @ dense uses BLAS-ish
paths and should beat our manual (nnz,dim) broadcast.

Actually cleanest: grouped[x, z] = sum_q V[x, q] * H[q, z], with H the
(dim,dim) Walsh-Hadamard "sign" matrix. This is just V @ H as a matrix
product. If we instead compute H @ (something) we can use the FAST
Hadamard transform (apply the existing _walsh_hadamard_transform_rows
butterfly to V's ROWS, since (V @ H)[x,z] = sum_q V[x,q]*(-1)**popcount(q&z)
= WHT(row x of V)[z] -- exactly what the ORIGINAL dense algorithm already
does! The dense algorithm's "gathered" matrix IS V (just built directly
without going through nonzero indices). So the real win is: don't touch
the O(dim log dim) butterfly at all (it's not the bottleneck - it was only
1.24s of the 2.2s at N=50) -- but avoid materializing/gathering the FULL
(dim,dim) `gathered` array (which needs a dim x dim advanced-index gather,
that's the 0.436s tottime) and the full (dim,dim) popcount for the phase
step (0.555s) by only doing those for necessary rows... but WHT still
needs O(dim) columns per row it processes regardless of row sparsity.

CONCLUSION: given active-row fraction is 47-86% (not <<1%), the honest
scoped fix is: (1) skip the ~15-53% all-zero rows entirely (real, modest
savings scaling with sparsity), (2) replace _popcount_array's O(n_qubits)
Python-loop bit-serial implementation with an 8-bit LUT-based popcount
(big constant-factor win, general, no sparsity assumption needed),
(3) avoid recomputing the phase array from scratch (it doesn't depend on
the operator at all, just n_qubits) - cache it. Test this "skip empty rows
+ fast popcount + cached phase" combination, likely the actually-winning,
simplest-to-verify approach given the sparse-WHT-per-row idea doesn't
asymptotically beat the dense butterfly once >~50% rows are active.
"""
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

_phase_cache = {}
def _cached_phase(dim, n_qubits):
    key = (dim, n_qubits)
    if key not in _phase_cache:
        idx = np.arange(dim)
        and_mat = idx[:, None] & idx[None, :]
        phase = 1j ** popcount_fast(and_mat)
        _phase_cache[key] = np.conj(phase) / dim
    return _phase_cache[key]

def fwht_pauli_coefficients_v6(operator):
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)
    q_indices = np.arange(dim)
    x_indices = np.arange(dim)[:, np.newaxis]
    p_indices = q_indices[np.newaxis, :] ^ x_indices
    gathered = operator[p_indices, q_indices[np.newaxis, :]]

    active_rows = np.any(gathered != 0, axis=1)
    transformed = np.zeros_like(gathered)
    transformed[active_rows] = _walsh_hadamard_transform_rows(gathered[active_rows])

    conj_phase_over_dim = _cached_phase(dim, n_qubits)
    coefficients = transformed * conj_phase_over_dim
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50, 100]:
        padded, n_qubits = synth(n_osc)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_v6(padded)
        t1 = time.perf_counter()
        # second call to see cached-phase steady-state time
        t2 = time.perf_counter()
        sp2 = fwht_pauli_coefficients_v6(padded)
        t3 = time.perf_counter()
        msg = f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} time_cold={t1-t0:.4f} time_warm={t3-t2:.4f}"
        if n_osc <= 30:
            ref = fwht_pauli_coefficients(padded)
            diff = np.max(np.abs(ref - sp))
            msg += f" max_diff={diff:.3e}"
        print(msg)
