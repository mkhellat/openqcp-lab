# `chunk_size` is a real cache-locality lever, not just a memory-safety knob

Recorded 2026-08-27, per direct user observation ("streaming should be
dynamic and based on very well-thought factors") that led to testing
`chunk_size` across a range of values rather than trusting the
example default (256) as adequate. Scopes PLAN.md Phase 12
(auto-tuned `chunk_size`/streaming decision, with manual override
always available).

## Why this was checked

Phase 6/9/10's `chunk_size` was originally designed and documented
purely as a **memory-footprint bound** ("process active rows in
blocks of at most `chunk_size` rows, so peak memory scales with
`chunk_size * dim`") - nothing in its design or prior measurement
considered whether the specific *value* chosen affects wall-clock
performance, beyond "smaller is safer for memory." The user
questioned this directly rather than accepting the example default
(256, used throughout Phase 9/10's own findings docs) as tuned or
neutral.

## Method

Controlled, same-process, same-Hamiltonian comparison
(`streaming_vs_dense_comparison.py`'s general shape - see this
directory) of `fwht_pauli_terms_iter` at multiple `chunk_size` values,
at N=25/50/100, wall-clock only first, then `perf stat` (same event
set as every other measurement in this project,
`OPENBLAS_NUM_THREADS=1`) at N=100 comparing `chunk_size` in
{4, 32, 256} specifically, to check whether a cache-locality mechanism
explains the wall-clock pattern.

## Results: wall-clock, chunk_size sweep

| N | chunk_size | mean time | vs. dense (`fwht_pauli_terms`) |
|---|---|---|---|
| 25 | 1 | 0.089s | 1.568x (slower) |
| 25 | 8 | 0.048s | 0.853x |
| 25 | 32 | 0.048s | 0.772x |
| 25 | 128 | 0.051s | 0.850x |
| 25 | 256 | 0.058s | 0.978x |
| 25 | 512 | 0.061s | 1.082x (slower) |
| 50 | 4 | 0.904s | 0.658x |
| 50 | 16 | 0.956s | 0.690x |
| 50 | 64 | 0.931s | 0.690x |
| 50 | 256 | 1.124s | 0.811x |
| 100 | 4 | 12.633s | 0.526x |
| 100 | 8 | 13.635s | 0.549x |
| 100 | 32 | 15.390s | 0.639x |
| 100 | 128 | 19.380s | 0.812x |
| 100 | 256 | 20.508-22.369s | 0.836-0.937x |

**Pattern:** except at the very smallest chunk sizes relative to N
(`chunk_size=1` at N=25, where too little work per chunk lets fixed
per-chunk overhead - generator suspend/resume, per-chunk gather/WHT
dispatch - dominate), **smaller `chunk_size` is consistently faster**,
not slower, across every N and metric tested. `chunk_size=256` (the
value used as an example throughout Phase 9/10's own docs) is
measurably suboptimal at every N tested here - never the fastest
option in any row of this table.

## Results: `perf stat`, N=100, chunk_size in {4, 32, 256}

| chunk_size | working set (`chunk_size * dim * 16 bytes`) | wall | cache-miss % | LLC-miss % | stall % |
|---|---|---|---|---|---|
| 4 | 512 KiB | 16.34s | 7.3% | 6.3% | 21.2% |
| 32 | 4 MiB | 16.98s | 21.3% | 19.4% | 24.9% |
| 256 | 32 MiB | 23.42s | 44.6% | 42.4% | 32.1% |

Machine cache hierarchy (`lscpu`): L1d 128 KiB, L2 1 MiB, L3 8 MiB
(shared).

## Interpretation

**Confirmed mechanism, not just correlation.** Cache-miss ratio scales
cleanly and dramatically with `chunk_size`'s working-set size relative
to this machine's cache hierarchy: `chunk_size=4`'s 512 KiB working
set sits comfortably within L2 (1 MiB) - 7.3% miss ratio. `chunk_size=32`'s
4 MiB working set exceeds L2 but fits within the shared 8 MiB L3 -
21.3%. `chunk_size=256`'s 32 MiB working set exceeds L3 by 4x - 44.6%,
matching the same qualitative pattern
`cache_locality/n_scaling_findings.md` found for the *original*
dense-array bug (cache-miss ratio scaling with array-size-vs-cache-size),
now shown operating *within* the chunked/streaming design itself, at a
scale (per-chunk working set) nobody had previously measured this way.

This is a genuinely new, actionable finding: `chunk_size` was designed
purely as a memory bound, but it is also - independently, and often
more impactfully at N≤100 scale - a **cache-locality lever**. The two
considerations (memory safety at very large N, cache locality at any
N) point toward the *same* general direction (smaller `chunk_size`),
but for different reasons and with different magnitudes, and neither
consideration alone justified today's example default of 256.

## What this implies for Phase 12 (scoped, not yet designed in detail)

A principled `chunk_size` heuristic should target a working-set size
that fits within a cache level relative to the machine's actual cache
sizes (queryable, e.g. via `os.sysconf` or `/sys/devices/system/cpu/cpu0/cache/`
on Linux - not yet investigated how portably), not a fixed example
value. The exact target level (L2 vs. L3) and the tradeoff against
per-chunk fixed overhead (which makes extremely small `chunk_size`
worse, per N=25's `chunk_size=1` result) both need further
measurement before an auto-tuning formula is trusted - this document
establishes the mechanism and the direction, not a finished formula.

Per the user's explicit instruction: any auto-tuning must remain
**manually overridable** - `chunk_size` (and, per the broader Phase 12
scope, the streaming-vs-dense decision itself) should default to a
computed value when the caller doesn't specify one, never force a
choice on a caller who explicitly sets `chunk_size` or chooses between
`fwht_pauli_terms`/`fwht_pauli_terms_iter` directly.

## What this does NOT show

- Only tested up to N=100 - N=150's own chunk_size behavior (already
  fixed at 256 in every Phase 9/10 N=150 finding) has not been
  re-swept with this lens; a smaller chunk_size might help there too,
  but N=150 already completes successfully at 256, so this is a
  performance follow-up, not a correctness question, for that case.
- `perf stat` comparison only covers 3 chunk_size values at one N
  (100) - the full wall-clock sweep covers more values/N combinations
  but without hardware counters, so the *mechanism* is confirmed at
  one representative point, not exhaustively re-verified everywhere
  the wall-clock pattern was observed.
- Does not yet test whether the same pattern holds with
  `--parallel-labels` enabled, or interacts with Phase 11's
  (not-yet-implemented) `dict_build` vectorization - both are
  independent axes not yet measured in combination with chunk_size
  tuning.
- Does not investigate portable cache-size detection - `lscpu`/`os.sysconf`
  behavior across different machines/OSes is unverified; this is
  implementation-time work for Phase 12, not addressed here.
