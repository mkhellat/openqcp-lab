# trotterkit

Product-formula (Trotter-Suzuki) evolution of a state vector under an
arbitrary Pauli-sum Hamiltonian, in two modes: one exponential per
Pauli term (the product formula a quantum circuit runs), and grouped
exact exponentials of the matrix's XOR-offset slices (fast, scales
in N). Inputs are paulikit's symplectic arrays `(x, z, coeff)` or the
sparse matrix; Hermiticity is not assumed.

Status: in development inside openqcp-lab; not published.

## Build and test

    ./configure
    make build
    make check
