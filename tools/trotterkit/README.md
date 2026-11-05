# trotterkit

Product-formula (Trotter-Suzuki) evolution of a state vector under an
arbitrary Pauli-sum Hamiltonian, in two modes: one exponential per
Pauli term (the product formula a quantum circuit runs), and grouped
exact exponentials of the matrix's XOR-offset slices (fast, scales
in N). Inputs are paulikit's symplectic arrays `(x, z, coeff)` or the
sparse matrix; Hermiticity is not assumed.

Status: in development inside openqcp-lab; not published.

| Backend | Per-term | Grouped | SIMD / tiles / threads | CLI |
|---|---|---|---|---|
| python (reference) | yes | yes | - | - |
| c | yes | yes | per-term: AVX2, tiles, OpenMP; grouped: serial | `trotterkit-c` |
| cpp, rust | not yet | not yet | - | - |

Every path of the C backend (SIMD or scalar, tiles, any thread count)
gives the same final state to the bit.

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

## Command line and workload files

Each backend has a command-line program that reads a workload file
and times only the evolution kernel, so languages can be compared on
the same input:

    bench/make_workload chain 30 chain30.tkwl
    trotterkit-c chain30.tkwl --t 1 --steps 100 [--mode grouped]
        [--order 4] [--threads K] [--tile-bits K] [--out psi.bin]

A workload file (`trotterkit.workload`) holds the Pauli terms, the
sparse H and the initial state behind a SHA-256-checked header.
`bench/make_workload` builds chain or fully coupled oscillator
workloads with paulikit. `bench/quick` compares runs by `perf stat`
cycles (or wall time for threads), interleaved, with a cooldown
between runs and Welch's t-test.

Measured on an Intel i7-8550U (4 cores, AVX2); other hardware will
differ. Per-term mode, chain workloads:

| | N=30 (9 qubits) | N=50 (11) | N=80 (12) |
|---|---|---|---|
| SIMD vs scalar loop, cycles | 0.243 | 0.241 | - |
| threads (auto) vs 1 thread, time | 0.984 (serial) | 0.745 | 0.547 |

At N=30 a Strang step over the 14,080 terms takes about 11 ms.

## Build and test

    ./configure
    make build
    make check

`make build` installs the package with pip, so it needs network
access. `make check` also runs `shellcheck`, so it must be installed.
OpenMP is optional; without it the C backend is serial.
