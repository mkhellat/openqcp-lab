"""How many x-rows are ACTIVE (have at least one nonzero gathered
entry)? That is the work paulikit skips and pauli_lcu cannot, because
its output is positional and every row must be transformed.

Reports the operator's own nnz too, since active_x is derived from the
XOR-mask partition of the nonzero entries, not from dim directly.
"""
import sys
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import _prepare_operator_for_fwht

print(f"{'N':>5} {'q':>3} {'dim':>6} {'nnz':>10} {'nnz/dim^2':>10} "
      f"{'active':>7} {'active/dim':>11} {'rows skipped':>13}")
for N in (10, 20, 30, 50, 75, 100, 125, 150):
    H = build_hamiltonian(N, _default_spring_constants(N),
                          _default_masses(N), sparse=True)
    Hp, nq = pad_to_power_of_two(H, sparse=True)
    dim = Hp.shape[0]
    (_op, _sp, _dim, _nq, p_nz, q_nz, x_nz) = _prepare_operator_for_fwht(Hp)
    active_x = np.unique(x_nz)
    na = len(active_x)
    nnz = len(p_nz)
    print(f"{N:>5} {nq:>3} {dim:>6} {nnz:>10} {nnz/dim**2:>10.5f} "
          f"{na:>7} {na/dim:>10.4f} {100*(1-na/dim):>12.1f}%")
