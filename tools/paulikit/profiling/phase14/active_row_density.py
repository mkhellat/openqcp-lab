"""Two questions the active-row count does not answer:
1. How many nonzeros per active row? (a row with 1 nonzero still
   costs a full dim-length WHT - that is the waste)
2. How many OUTPUT terms survive per active row, vs the dim it costs?
"""
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import _prepare_operator_for_fwht, fwht_pauli_coefficients

print(f"{'N':>5} {'dim':>6} {'active':>7} {'nnz/act row':>12} "
      f"{'WHT cost':>12} {'useful':>10} {'waste':>7} {'out terms':>11} {'out/dim^2':>10}")
for N in (20, 30, 50, 100, 150):
    H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
    Hp, nq = pad_to_power_of_two(H, sparse=True)
    dim = Hp.shape[0]
    (_o,_s,_d,_q,p_nz,q_nz,x_nz) = _prepare_operator_for_fwht(Hp)
    active_x, counts = np.unique(x_nz, return_counts=True)
    na = len(active_x)
    # cost model: each active row is a full dim-length WHT = dim*log2(dim)
    wht_cost = na * dim * np.log2(dim)
    # "useful" input: the actual nonzeros fed in
    useful = len(p_nz)
    x,z,c = fwht_pauli_coefficients(Hp, sparse=True, chunk_size=2, atol=1e-9)
    print(f"{N:>5} {dim:>6} {na:>7} {counts.mean():>12.1f} "
          f"{wht_cost:>12.3e} {useful:>10} {useful/(na*dim):>6.4f} "
          f"{len(c):>11} {len(c)/dim**2:>10.4f}")
