# Per-stage profile, and what is worth moving to C

2026-09-09. Two questions: where does a chunk's time go now that the
butterfly is compiled, and what else belongs in C?

All figures are medians over 40-200 reps on a **real** N=150 chunk
(`chunk_size=2`, `dim=16384`, 16,384 surviving terms), not synthetic
data.

## Stage profile after the WHT kernel, before the fused kernel

| stage | µs | share |
|---|---|---|
| **phase factor** | 263.9 | **27.8%** |
| butterfly (C kernel) | 235.4 | 24.8% |
| **threshold + nonzero** | 207.1 | **21.8%** |
| gather x, coeff | 90.3 | 9.5% |
| gather / scatter (input) | 71.3 | 7.5% |
| scale by phase | 53.1 | 5.6% |
| narrow dtype | 26.6 | 2.8% |
| **total** | **947.7** | 57.8 ns/term |

The compiled butterfly had dropped to second place. Phase and
thresholding together were 49.6% - more than the transform.

Two details that pointed at the fix: of the phase step's 263.9 µs only
**11.1 µs** was the `x & z` itself (the rest being popcount and the
complex gather), and of the threshold step's 207.1 µs, **50.1 µs** was
`np.abs` alone. Both are elementwise work forced through separate
full-array passes purely because NumPy has no way to express them as
one.

## The fused kernel

Phase, scale, threshold, index gather and dtype narrowing all depend
only on one element's own transformed value and its own `(x, z)` pair.
Fusing them into a single C pass keeps each value in registers and
appends survivors directly:

**614.9 µs of NumPy → 77.6 µs fused, 7.9x.**

Projected and then confirmed end to end: sequential CPU at N=150 went
7.29 → 3.83 CPU-seconds, below pauli_lcu's 4.68 - while still emitting
the explicit `(x, z)` indices their positional output never produces.

## Stage profile now (N=100 sequential, cProfile)

Total 0.989 s, against 4.72 s at the start of this phase - **4.8x
cumulative**. The profile is flat, which is what a finished
optimisation looks like:

| function | s | share |
|---|---|---|
| butterfly (C) | 0.249 | 25% |
| fused coeffs (C) | 0.245 | 25% |
| `_GrowableArray.extend` | 0.225 | 23% |
| scipy `csr_sample_values` | 0.058 | 6% |
| `_iter_chunked_coefficients` | 0.032 | 3% |
| scipy `_validate_indices` | 0.025 | 3% |

## What else should move to C: two candidates, neither of them C

### `_GrowableArray.extend` - 23%, but the fix is not a C port

Measured directly at the N=100 shape: 0.091 s for 310 MiB = 3423
MiB/s, against a raw slice-assign ceiling of 10162 MiB/s. So **66% of
its time is growth** - reallocating and copying the accumulated array
- not the append itself.

A C port would not help, because the copying is `memcpy` either way.
**Pre-sizing would**, and the final term count is estimable from the
active-row structure before the loop starts.

It also matters less than 23% suggests: `_GrowableArray` is only used
by `fwht_pauli_coefficients`, which accumulates.
`parallel_decompose_arrays` yields each chunk and never touches it -
so this is absent from the path that carries the large workloads.

### The scipy gather - 6-9%, and it is pure per-call overhead

| operation | µs |
|---|---|
| `np.zeros((2, 16384))` | 11.6 |
| **scipy fancy index (64 values)** | **45.4** |
| scatter assign | 2.1 |

scipy takes **45.4 µs to fetch 64 values** - roughly 700 ns per value,
and more than it costs to zero the entire 512 KiB block. That is CSR
index validation and broadcasting machinery, not data movement.

Again the fix is not C: `_prepare_operator_for_fwht` has already
extracted the operator's nonzeros as `p_nz`/`q_nz`, so the values can
be pulled **once** up front and sliced per chunk, replacing a
per-chunk scipy call with an array slice.

## Conclusion

**Nothing further should move to C.** The two remaining hot spots are
both Python-level structural inefficiencies with pure-Python fixes -
pre-size the accumulator, hoist the value extraction out of the loop -
and a C port of either would be work spent on the wrong layer.

The two kernels that were written (`wht.c`, `coeffs.c`) cover the
genuinely compute-bound work: the transform, and the elementwise
phase/threshold/emit pass. Everything else in the chunk loop is now
either compiled or bounded by memory bandwidth.
