# Changelog

## [Unreleased]

### Added

- Pauli convention pinned to paulikit's; order-2 and order-4 stage
  schedules.
- Pure-Python reference of per-term and grouped evolution, validated
  against exact evolution at its formal order.
- Coefficient-only 2nd-order Trotter bound (Hermitian H).
- C ABI (`include/trotterkit.h`) and a serial C99 backend for both
  modes, loaded through ctypes.
- Binary workload files (`trotterkit.workload`) and
  `bench/make_workload` for chain and fully coupled oscillators.
- `trotterkit-c` command-line program timing only the kernel.
- `bench/quick`: interleaved `perf stat` comparisons with cooldown
  and Welch's t-test.
- C++17 backend (`trotterkit-cpp`) and Rust backend (`trotterkit-rs`)
  implementing the same ABI and design; all backends agree to the
  bit, checked by `tests/test_cross_backend.py`.

### Changed

- C per-term kernel: groups of 4 amplitudes with AVX2 (scalar
  fallback), branch-free signs; 4.1x fewer cycles than the scalar
  loop. Results unchanged to the bit.
- C per-term evolution runs on OpenMP threads, with tiles for runs of
  tile-local terms; `n_threads = 0` is serial below 11 qubits.
- Grouped-mode block constants are computed from real libm functions
  instead of C99 complex.h, so other languages can match them.
