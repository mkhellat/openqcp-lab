# Phase 14 — external comparison and per-term cost reduction

Two connected questions, 2026-09-09:

1. How does paulikit compare against an independent implementation of
   the same algorithm (`pauli_lcu`, Riverlane)?
2. What does that comparison say about where our own time goes?

The answer to (1) motivated (2): three optimizations, all derived from
profiling **our** code, none ported from theirs.

## Findings

`external_comparison_findings.md` is the write-up. Headline numbers,
all N=150 (14 qubits, dim 16384, 91.65M terms):

| | wall | CPU-seconds | cores | peak RSS |
|---|---|---|---|---|
| pauli_lcu | 4.71s | 4.68 | 0.99 | 8891 MiB |
| paulikit sequential | 18.46s | 18.30 | 0.99 | ~3142 MiB |
| paulikit parallel (w4_c4) | 6.66s | 29.83 | 4.48 | **89 MiB** |

Two distinct gaps, previously conflated:

- **Per-core efficiency, ~3.9x** (18.30 vs 4.68 CPU-seconds). The
  butterfly operates on stride-2 NumPy views, which cannot vectorize
  the way a compiled contiguous loop does.
- **Parallel overhead, 11.5 CPU-seconds** (29.83 - 18.30), i.e. 63%
  overhead for a 2.77x speedup on 4.48 cores (62% efficiency).

Memory is where the design wins: 89 MiB flat against 8891 MiB, and
essentially constant from N=100 to N=150 while pauli_lcu's grows with
the dense operator it requires as input.

## Optimizations landed

| change | isolated | end-to-end |
|---|---|---|
| scratch-buffer butterfly | 1.29x | 1.10x (in situ) |
| uint8 popcount + 4-entry phase LUT | 2.65x | — |
| pre-conjugated table + reciprocal multiply | 8.2x | — |
| **combined, sequential profile** | | **1.33x** |
| **combined, replicated parallel** | | **1.07-1.12x** |

Rejected: sqrt-free thresholding (`re**2 + im**2 > atol**2`) measured
1.06x, not worth the readability cost.

The end-to-end parallel gain is smaller than the sequential profile
gain because IPC and pickling overhead is unaffected by these changes
and dilutes them - see the parallel-overhead gap above.

## Harnesses

| file | what it does |
|---|---|
| `pauli_lcu_comparison.py` | dense random Hermitian sweep, correctness-gated. **Read its docstring before quoting it** - it deliberately measures pauli_lcu's best case. |
| `replicated_head_to_head.py` | the protocol-grade one: real Hamiltonians, interleaved, n>=5, cooldowns. Use this for any quoted ratio. |
| `replicated_head_to_head_target.py` | its child process (one measurement per process, so peak RSS is that run's alone) |
| `configuration_ladder.py` | isolates dense-vs-sparse input and sequential-vs-parallel, one factor at a time |
| `butterfly_scratch_buffer_benchmark.py` | butterfly variants, correctness-checked against the current implementation |
| `phase_bit_trick_candidates.py` | phase-step candidates incl. the rejected sign/swap approach |
| `phase_uint8_lut_benchmark.py` | the winning combination, with the uint8-wrap identity asserted |
| `threshold_and_scale_candidates.py` | sqrt-free threshold (rejected) and conj/reciprocal (adopted) |

## Two measurement failures worth remembering

Both are recorded in full in `external_comparison_findings.md`.

1. **A harness that measured a configuration nobody runs.** The first
   comparison built the Hamiltonian densely and called the
   single-process path, reported 41.3s for N=150, and concluded the
   streaming memory argument was falsified. The real pipeline does it
   in ~8s at 89 MiB. A known-good number from the same day was
   available and contradicted it 5x; that contradiction should have
   halted interpretation immediately.

2. **`ps` CPU% is a lifetime average.** pauli_lcu's process showed
   122-175% CPU and 8 threads, suggesting hidden parallelism. Per-thread
   tick counts (213 on the main thread, 13 on each of seven others)
   showed a NumPy pool that built the input matrix and then idled.
   Measuring CPU-time/wall over the decomposition alone gives 0.99 -
   strictly single-core. Sample the region you care about, not the
   process.
