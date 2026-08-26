"""Key insight: (-1)**popcount(q & z) = product over bits b of (-1)**(q_b & z_b)
  = product over set bits of q of (-1)**z_b (that bit of z)
This is exactly a per-bit XOR-parity dot product: sign(q, z) = (-1)**popcount(q & z).
We don't need an (nnz, dim) intermediate: build a signed lookup by precomputing,
for each nonzero (q, v), a length-dim sign vector via the SAME fast in-place
Hadamard butterfly used for a delta function -- i.e. WHT(delta_q * v) directly,
computed via the standard O(dim log dim) FWHT of a mostly-zero vector. But since
we have MANY q's, batch them: build a (nnz, dim) impulse matrix (one-hot rows
scaled by v) and run one batched row-wise WHT over it (dense butterfly is
O(dim log dim) per row - same total work as v3's explicit sign matrix, no better).

Real fix: don't materialize per-(q,z) signs at all. Instead note the phase/gather
combmbined coefficient formula factors: coefficients[x,z] = (1/dim) * conj(i**pc(x&z))
  * sum_{q: (q^x,q) nonzero} operator[q^x,q] * (-1)**popcount(q&z)

Group nonzero operator entries by x = p^q (small number of GROUPS, ~nnz total
entries across all groups, avg group size ~4). For each group we need, for ALL z,
the sum over its ~4 (q,v) pairs of v*(-1)**popcount(q&z). This is inherently a
per-x O(k*dim) computation - total work O(nnz*dim) is NOT avoidable this way
since output coefficients[x,:] is dense (dim entries) for every active x, and
there are O(dim) active x at large N (per the sparsity probe: 60-86% of rows
active). So total work is fundamentally >= O(active_x * dim) = O(dim^2) in the
worst case here - NOT the asymptotic win once density of active rows is high.

This changes the diagnosis: the win is not "avoid dim^2" (rows are too often
active for that), but "avoid the log(dim) factor of the full butterfly" per
active row, replacing O(dim log dim) with O(k*dim) where k=avg nnz/row ~ 4 << log(dim)~11.
That's the real, correctly-scoped win. Confirm the per-row-loop v1 approaches
this with less Python overhead by using a single batched (k_total, dim) matmul
instead of per-row loops AND avoiding _popcount_array's slow bit-serial loop
by precomputing an 8-bit lookup table for popcount (like population-count
tricks), which should cut the sign-matrix construction time substantially.
"""
import sys
from pathlib import Path
import time
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
    total = np.zeros(arr.shape, dtype=np.uint8)
    for shift in (0, 8, 16, 24):
        total = total + _POPCOUNT8[(arr >> shift) & 0xFF]
    return total

def fwht_pauli_coefficients_sparse_v4(operator):
    dim = operator.shape[0]
    n_qubits = int(round(np.log2(dim)))
    operator = np.asarray(operator, dtype=complex)

    p_nz, q_nz = np.nonzero(operator)
    vals = operator[p_nz, q_nz]
    x_nz = p_nz ^ q_nz

    z_all = np.arange(dim)
    and_mat = q_nz[:, None] & z_all[None, :]
    signs = 1.0 - 2.0 * (popcount_fast(and_mat) & 1).astype(np.float64)
    contributions = vals[:, None] * signs

    order = np.argsort(x_nz, kind="stable")
    x_sorted = x_nz[order]
    contributions_sorted = contributions[order]
    unique_x, start_idx = np.unique(x_sorted, return_index=True)
    grouped = np.add.reduceat(contributions_sorted, start_idx, axis=0)

    transformed = np.zeros((dim, dim), dtype=complex)
    transformed[unique_x] = grouped

    xz_and = np.arange(dim)[:, None] & z_all[None, :]
    phase = 1j ** popcount_fast(xz_and).astype(np.int64)
    coefficients = transformed * np.conj(phase) / dim
    return coefficients

if __name__ == "__main__":
    for n_osc in [16, 30, 50, 100]:
        padded, n_qubits = synth(n_osc)
        t0 = time.perf_counter()
        sp = fwht_pauli_coefficients_sparse_v4(padded)
        t1 = time.perf_counter()
        msg = f"N={n_osc} dim={padded.shape[0]} nnz={np.count_nonzero(padded)} time={t1-t0:.4f}"
        if n_osc <= 30:
            ref = fwht_pauli_coefficients(padded)
            diff = np.max(np.abs(ref - sp))
            msg += f" max_diff={diff:.3e}"
        print(msg)
