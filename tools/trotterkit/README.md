# trotterkit

Product-formula (Trotter-Suzuki) evolution of a state vector under an
arbitrary Pauli-sum Hamiltonian, in two modes: one exponential per
Pauli term (the product formula a quantum circuit runs), and grouped
exact exponentials of the matrix's XOR-offset slices (fast, scales
in N). Inputs are paulikit's symplectic arrays `(x, z, coeff)` or the
sparse matrix; Hermiticity is not assumed.

Status: in development inside openqcp-lab; not published.

| Backend | Per-term | Grouped | SIMD / tiles / threads |
|---|---|---|---|
| python (reference) | yes | yes | - |
| c | yes | yes | not yet |
| cpp, rust | not yet | not yet | - |

Usage:

    import trotterkit
    psi_t = trotterkit.evolve(psi0, t=1.0, steps=100,
                              terms=(x, z, coeff), backend="c")
    psi_t = trotterkit.evolve(psi0, t=1.0, steps=100, H=H,
                              mode="grouped", backend="c")

`x`, `z`, `coeff` are paulikit's symplectic arrays
(`fwht_pauli_coefficients(..., sparse=True, chunk_size=...)`).
`trotterkit.bound.bound_terms` / `bound_grouped` give a step count
for a target error (the bound requires a Hermitian H; order 2 only).

Related work: HamSim (Chundury et al., arXiv:2606.16959) also
exploits diagonal sparsity for classical Hamiltonian simulation, but
stores conventional diagonals and builds a single whole-H propagator
by Taylor expansion with scaling and squaring; it does not split H
into XOR-offset fragments or use Trotter-Suzuki product formulas.

## Build and test

    ./configure
    make build
    make check

`make check` also runs `shellcheck`, so it must be installed.
