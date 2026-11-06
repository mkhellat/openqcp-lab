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
| c (C99) | yes | yes | per-term: AVX2, tiles, OpenMP; grouped: serial | `trotterkit-c` |
| cpp (C++17) | yes | yes | same; OpenMP | `trotterkit-cpp` |
| rust | yes | yes | same; std threads, spinning barrier | `trotterkit-rs` |

All three implement one C ABI (`include/trotterkit.h`) with the same
algorithm and the same operations in the same order, without FMA
contraction, so every backend and every path (SIMD or scalar, tiles,
any thread count) gives the same final state to the bit.

**The default backend is `c`.** A comparison on an Intel i7-8550U
(cycles for one thread, wall time for several; 3-5 interleaved reps,
95% confidence intervals; method in `bench/METHODOLOGY.md`) found no
backend best everywhere. In grouped mode, the one that scales to large
N, C was fastest (C++ 10-14% and Rust 26-42% more cycles). In per-term
mode Rust was fastest: 6-7% fewer cycles on one thread at 9-11 qubits
and 7% / 18% less time with 4 threads at 11 / 12 qubits, while C++ did
not differ significantly from C. C also needs only a C compiler. For
large threaded per-term runs, `backend="rust"` is worth trying; other
hardware may order the backends differently.

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

`trotterkit-cpp` and `trotterkit-rs` take the same arguments.

A workload file (`trotterkit.workload`) holds the Pauli terms, the
sparse H and the initial state behind a SHA-256-checked header.
`bench/make_workload` builds chain or fully coupled oscillator
workloads with paulikit. `bench/quick` compares runs by `perf stat`
cycles (or wall time for threads), interleaved, with a cooldown
between runs and Welch's t-test.

Measured on an Intel i7-8550U (4 cores, AVX2); other hardware will
differ. Per-term mode, chain workloads, C backend:

| | N=30 (9 qubits) | N=40 (10) | N=50 (11) |
|---|---|---|---|
| SIMD vs scalar loop, cycles | 0.243 | - | 0.241 |
| threads (auto) vs 1 thread, time | serial | 0.765 (2) | 0.530 (4) |

At N=30 a Strang step over the 14,080 terms takes about 11 ms. How
the three languages are compared, the decisions that make the
comparison fair, and what the comparison found are in
`bench/METHODOLOGY.md`.

## Build and test

    ./configure
    make build
    make check

`make build` installs the package with pip, so it needs network
access. `make check` also runs `shellcheck`, so it must be installed.
OpenMP is optional; without it the C and C++ backends are serial.
The Rust backend is built when `cargo` is installed (no crates are
downloaded).
