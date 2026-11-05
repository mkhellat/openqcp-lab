# Benchmark methodology

How trotterkit's C, C++ and Rust backends are compared, and the
decisions taken to make the comparison about the languages. All
numbers here were measured on one machine (Intel i7-8550U, 4 cores
with 2 threads each, AVX2, Linux); other hardware will give other
numbers, and the automatic thread choices in particular are tuned to
it.

## What is compared

Each backend has a command-line program (`trotterkit-c`,
`trotterkit-cpp`, `trotterkit-rs`) that reads the same workload file
and times only the evolution call with a monotonic clock. Reading the
file, building the grouped form and writing results are outside the
timed region. All three implement one C ABI with the same algorithm,
data layout, tile sizes, flags (`-O3 -march=native`,
`-ffp-contract=off`, `opt-level = 3`, `target-cpu=native`, no
fast-math) and automatic choices, and they produce identical final
states to the bit (`tests/test_cross_backend.py`). A benchmark of
backends that disagree would compare different computations.

## Measurement protocol (`bench/quick`)

- **Metric:** `perf stat -e cycles:u,instructions:u` for one thread.
  Wall-clock time on this machine varies about 2x with turbo and
  thermal state, while cycles do not. For several threads cycles add
  up across threads, so thread runs use the kernel's wall time.
- **Harness check:** instruction counts must agree within 0.1% across
  reps of a case (single-threaded); threaded runs spin at barriers and
  are exempt. A null comparison of two identical cases gave a cycle
  ratio of 1.000, p = 0.61.
- **Replication:** at least 5 reps per case, interleaved, with the
  first case rotated each rep, so drift affects every case equally.
- **Cooldown:** before every run the CPU package must be at or below
  60 C (settled idle + 5 C). If the target is not reached within
  180 s the run stops rather than measuring a hot machine.
- **Exclusivity:** a run refuses to start while any other process
  uses more than half a CPU.
- **Statistics:** ratios are geometric means with 95% confidence
  intervals from Welch's t-test on log values (`bench/tables`). A
  difference, or its absence, is stated only at p < 0.05. Single runs
  are reported as times, never as ratios.

The CLI timing was validated against Python timing the same kernel
in-process: medians 4.298 s and 4.269 s (ratio 0.993), order
alternated with cooldown. Without alternation the second runner
looked 7% slower, which was thermal drift.

## Decisions taken before the comparison

Each was settled by measurement (chain workloads, 5 interleaved reps).

1. **One barrier algorithm in all three.** With 4 threads at 12
   qubits Rust took 0.69 s per step and C 1.00 s, though on one
   thread they were within 6%. `OMP_WAIT_POLICY=active` changed
   nothing (0.981, p = 0.53). Giving C the Rust backend's spinning
   barrier cut its time to 0.793 (p = 2e-7). All three now use that
   barrier (C through GCC's `__atomic` builtins, C++ through
   `std::atomic`, Rust through `std::sync::atomic`); OpenMP only
   creates the C and C++ thread teams. Against their OpenMP-barrier
   builds: C 0.783 / 0.701, C++ 0.740 / 0.664 at 12 / 11 qubits.
2. **Compile-time lane permutation in all three.** C++ templates and
   Rust const generics instantiate the grouped per-term loop per lane
   permutation; C now does the same with four wrappers (0.971 at 9
   qubits, 0.979 at 11 against choosing it at run time).
3. **Compiler as well as language.** Rust compiles with LLVM, C and
   C++ with GCC. A Clang-built C library separates the two: at 9 and
   11 qubits on one thread Clang's C was 0.961 / 0.976 of GCC's and
   Rust 0.923 / 0.938, so about half of Rust's lead over C there is
   the compiler. The campaign reports this ablation.
4. **Automatic thread counts** were re-measured with the new barrier:
   serial at 9 qubits (2 threads 1.07, 4 threads 1.45 of the serial
   time), 2 threads at 10 (0.765), 4 from 11 (0.530); 8 threads were
   slower than 4 (hyperthreads share the SIMD units).
5. **Sign tables dropped.** Precomputed per-term sign tables did not
   beat computing two parities per group of 4 (0.977, p = 0.06 at 9
   qubits; 1.056 slower at 11).

## The campaign (`bench/campaign`, `bench/tables`)

    bench/make_workload chain 30 WL/chain30.tkwl      # and 40, 50, 80, 100
    bench/make_workload full 30 WL/full30.tkwl
    bench/make_workload chain 100 WL/chain100g.tkwl --grouped-only
    bench/make_workload chain 150 WL/chain150g.tkwl --grouped-only
    bench/campaign WL --list
    bench/campaign WL EXPERIMENT... [--clang-cli PATH]
    bench/tables > results.md

Experiments: per-term mode on 1 thread (cycles; chain N=30, 50, 80,
fully coupled N=30); per-term mode with automatic threads (wall time;
chain N=40, 50, 80, 100); thread scaling 1/2/4 at chain N=80 per
language; grouped mode (cycles; chain N=30 to 150, fully coupled
N=30); the Clang ablation; and the N=30 notebook's full run
(t = 10.762932, 22,058 steps), one run per language. Raw results are
kept in `bench/results/campaign/`.
