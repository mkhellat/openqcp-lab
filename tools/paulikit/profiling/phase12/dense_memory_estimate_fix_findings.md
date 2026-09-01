# Fixing `auto_decompose()`'s dense-path memory estimate (Bug 1 from the N=100/N=150 re-measurement)

Recorded 2026-09-01, same day as
[`n100_n150_autotuning_remeasurement_findings.md`](n100_n150_autotuning_remeasurement_findings.md)'s
Bug 1 ("dense-path memory estimate underestimates real peak usage by
~3x, in the unsafe direction"). That document deliberately reported
the bug without fixing it. This document is the fix, plus the
additional real measurement that motivated its specific constants.

## The estimate was still wrong after the first re-check — twice

Before writing a fix, the naive `dim**2 * 16` estimate's real
undercount was re-verified, since a safety-critical constant deserves
more than one data point. Isolating just `fwht_pauli_terms(padded)`
(the dense path, no `chunk_size`) at N=150 (dim=16384):

| memory cap (`ulimit -v`) | naive estimate | ratio (cap / naive) | result |
|---|---|---|---|
| ~7.6 GiB | 4.00 GiB | 1.9x | failed (`_popcount_array`) |
| ~11.4 GiB | 4.00 GiB | 2.85x | failed (`_pauli_label_batch`) |
| **~18.0 GiB** (this fix's own re-check) | 4.00 GiB | **4.5x** | **failed** (`_build_real_terms`'s `dict(zip(...))`) |

Three separate failure points across three separate runs, each at a
different stage of the pipeline — consistent with genuinely large,
distributed peak usage rather than one unlucky allocation. During the
18 GiB run, `free -h` polling showed real available memory drop as low
as 593 MiB before the process's own `MemoryError` let it exit cleanly
(no kernel OOM-kill). This means a flat multiplier anywhere up to
4.5x, by itself, would *still* have been insufficient at N=150 on this
21-run's evidence — the fix below combines a multiplier with an
independently-tightened safety fraction rather than relying on the
multiplier alone to be exactly right.

## Root cause: every concurrently-live array in the dense path, by hand

The naive estimate only accounts for
`gathered_active`/`transformed_active` (the one `(n_active, dim)`
complex128 array `fwht_pauli_coefficients`'s own docstring describes).
Tracing `fwht_pauli_coefficients`'s `sparse=True, chunk_size=None`
branch and `fwht_pauli_terms`'s own post-processing line by line finds
several more same-order-of-magnitude arrays, all `O(dim**2)`, worst
case (`n_active == dim`):

| array | dtype | bytes (worst case) |
|---|---|---|
| `gathered_active` / `transformed_active` (in-place) | complex128 | `dim**2 * 16` |
| `xz_and = active_x[:, newaxis] & z_indices` | int64 | `dim**2 * 8` |
| `_popcount_array`'s `values.astype(np.uint32)` | uint32 | `dim**2 * 4` |
| `_popcount_array`'s `count` accumulator | int64 | `dim**2 * 8` |
| `phase = 1j ** count` | complex128 | `dim**2 * 16` |
| `active_coefficients = transformed_active * conj(phase) / dim` | complex128 | `dim**2 * 16` |
| `np.abs(active_coefficients) > atol` (in `fwht_pauli_terms`'s nonzero rescan) | float64 intermediate | `dim**2 * 8` |

Summing these (excluding label-string/dict-construction overhead,
which scales with term count rather than `dim**2` and is harder to
bound the same way) gives **~17 GiB at N=150** against the naive
estimate's 4 GiB — a 4.25x ratio from hand-counting alone, before any
of the label/dict-construction overhead that the 18 GiB run's
traceback shows also matters.

## Real peak-RSS measurement: confirms the ~4-6.5x range, at a scale small enough to measure safely

Rather than trust the hand count alone (a method already proven
capable of missing an array — the first attempt's tally omitted the
`np.abs()` rescan and the label/dict-construction step, both of which
the 18 GiB run's traceback pointed to directly), a real
`resource.getrusage(RUSAGE_SELF).ru_maxrss` sweep was run at N=50, 75,
and 100 — large enough for the ratio to stabilize past small-N process
overhead, small enough to measure safely without risking the same
memory pressure the N=150 runs caused:

| N | dim | naive estimate | real peak RSS | ratio |
|---|---|---|---|---|
| 25 | 512 | 4.0 MiB | 72.8 MiB | 18.20x (excluded - see below) |
| 50 | 2048 | 64.0 MiB | 360.6 MiB | 5.63x |
| 75 | 4096 | 256.0 MiB | 1656.9 MiB | 6.47x |
| 100 | 8192 | 1024.0 MiB | 5399.5 MiB | 5.27x |

N=25's 18.20x is excluded as a small-N artifact: at 72.8 MiB total
peak RSS, the fixed baseline cost of the Python interpreter and
imported NumPy/SciPy machinery itself is a large fraction of that
figure, not representative of how the ratio behaves as `dim` grows.
N=50/75/100 cluster tightly in the **5.3-6.5x range**, consistent with
(if somewhat higher than) the ~4.25x hand-counted estimate above — the
gap is plausibly the label/dict-construction overhead the hand count
excluded. Given N=150's own real behavior (still failing at 4.5x /
18 GiB), the true ratio likely grows somewhat with scale rather than
staying flat at ~6x — the fix below does not rely on 6x alone being
sufficient at every N; the safety fraction provides independent
margin for exactly this reason.

## The fix

Two independent, additive changes in `fwht.py`:

1. **`_DENSE_MEMORY_MULTIPLIER = 6.0`** - `estimated_dense_bytes` is
   now `dim * dim * 16 * _DENSE_MEMORY_MULTIPLIER`, informed directly
   by the peak-RSS sweep above (clusters around 6x, not a round number
   picked without evidence).
2. **`_DENSE_MEMORY_SAFETY_FRACTION` lowered from `0.5` to `0.2`** -
   independent additional margin, since the multiplier alone is not
   guaranteed sufficient at N=150 scale (the true ratio there was
   never fully bounded above - measurement stopped once even 18 GiB
   failed, not because 18 GiB was known to be enough headroom).

Combined effect at N=150's real numbers (`budget ≈ 10.66 GiB` on this
machine): `estimated_dense_bytes` = 24.00 GiB, threshold = `budget *
0.2` ≈ 2.13 GiB → **`auto_decompose()` now correctly picks streaming**
at N=150 on this machine, where it previously (wrongly) picked dense.

**Deliberate over-conservatism, not a bug**: at N=100, the real
measured peak (5.27 GiB) comfortably fits under this machine's ~10.66
GiB budget - dense would in fact work and be somewhat faster. But the
new 6x-inflated estimate (6.00 GiB) exceeds the tightened 0.2-fraction
threshold (2.13 GiB), so `auto_decompose()` now streams at N=100 too.
This is intentional: for a safety-critical memory decision, choosing
streaming when dense would have worked is a far better failure mode
than choosing dense and running the process out of memory - see
`auto_decompose()`'s own updated docstring. A caller who knows their
own real memory headroom precisely and wants dense's typically-faster
performance can call `fwht_pauli_terms` directly, bypassing this
estimate entirely.

## Verification

- Regression tests added (`tests/test_autotune.py`, 2 new): one pins
  N=150's real dim/budget numbers from this measurement and asserts
  the fixed formula does NOT choose dense there; one pins the two
  constants' values directly, so an accidental relaxation back toward
  the disproven `1x`/`0.5` combination is caught without needing
  another expensive real N=150 run to notice.
- Full test suite (101 tests, `pytest -q`) passes, no regressions.
- `auto_decompose()`'s dense-path branch remains bit-identical to
  `fwht_pauli_terms` at N=25/50 (unchanged from the original
  correctness check - this fix only changes the routing decision, not
  either underlying code path).

## What this does NOT show

- Does not re-derive the multiplier from an N=150-scale real
  measurement (would require repeating the risky high-memory-pressure
  runs this fix's own re-check already did once, deliberately not
  repeated further once sufficient evidence was gathered - see the
  "still wrong after the first re-check" section above for why the
  bound is "at least 4.5x, likely more," not a precise number at
  N=150 itself).
- Does not eliminate the possibility that some larger N could still
  exceed even a 6x/0.2 combination's safety margin - no formal
  worst-case bound is derived here, only an empirically-grounded,
  deliberately conservative constant choice. A future, more precise
  fix could account for every array explicitly (the hand-counted table
  above) rather than a flat multiplier, at the cost of needing to stay
  in sync with `fwht_pauli_coefficients`'s/`fwht_pauli_terms`'s actual
  implementation as it changes.
- Does not address Bug 2 (the cache-probe non-idempotency) - tracked
  separately, see `n100_n150_autotuning_remeasurement_findings.md`.
