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
