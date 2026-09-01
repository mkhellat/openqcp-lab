# The chunk_size floor is scale-dependent: no single constant is right everywhere

Recorded 2026-09-01, prompted by direct user suspicion after the
Phase 12 re-measurement: "lots of memory stayed available throughout
the whole process. Does not it indicate our chunking is not optimal?"
That question turned out to be well-founded. This document is the
resulting investigation: real N=100/150/200 chunk_size sweeps
(single-run discovery passes, then repeated-run confirmation, then
`perf stat` mechanism confirmation), showing the current floor
constant (8, `_min_chunk_size_floor()`, unchanged since the original
N=25-only scoping) is not optimal at any of N=150 or N=200, and that
no single static floor constant can be optimal across the full N=25-200
range tested.

## Real N=150 chunk_size sweep (new - N=150 had never been swept before)

Single-run discovery pass, `n150_chunk_size_sweep.py`:

| chunk_size | elapsed |
|---|---|
| 1 | 35.91s |
| 2 | 33.82s |
| 4 | 34.73s |
| 8 (current floor) | 35.96s |
| 16 | 39.27s |
| 32 | 49.98s |
| 64 | 56.34s |

`chunk_size=2` fastest. Confirmed with repeated runs
(`n150_chunk_size_sweep_repeated.py`, 3 reps each):

| chunk_size | mean | stdev | individual |
|---|---|---|---|
| 1 | 35.17s | 3.33 (9.5% cv) | 38.91, 34.09, 32.52 |
| **2** | **33.00s** | **0.14 (0.4% cv)** | 33.11, 32.84, 33.03 |
| 4 | 34.27s | 0.20 (0.6% cv) | 34.49, 34.21, 34.10 |
| 8 (current floor) | 36.94s | 1.21 (3.3% cv) | 38.29, 36.60, 35.94 |

`chunk_size=2` is both fastest AND most stable (lowest coefficient of
variation) - a real, reproducible ~11% win over the current floor,
not single-run noise. `chunk_size=1` is clearly worse and much
noisier than 2 - consistent with per-chunk fixed-overhead dominating
at the smallest chunk size, the same mechanism the original N=25
finding (chunk_size=1 was 57% slower than dense) identified.

**`perf stat` confirms the mechanism** (`n150_perf_chunk_size.py`,
chunk_size=2 vs 8 vs 256):

| chunk_size | wall(s) | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|
| 2 | 33.39 | 4.75% | 3.45% |
| 8 | 38.93 | 12.01% | 8.65% |
| 256 | 67.84 | 46.24% | 43.33% |

Clean, monotonic - same cache-locality mechanism the original
N=100 investigation found, now confirmed independently at N=150.

## Real N=200 chunk_size sweep (new - first-ever N=200 measurement in this project)

N=200 (dim=32768, 326,139,904 terms - 3.56x N=150's term count) had
never been measured before. Single-run discovery pass across the same
range found a **different pattern than N=150**: a clean monotonic
increase from chunk_size=1 upward, no dip at 2:

| chunk_size | elapsed |
|---|---|
| 1 | 122.29s |
| 2 | 124.24s |
| 4 | 139.19s |
| 8 (current floor) | 166.99s |
| 16 | 197.89s |
| 32 | 209.31s |

Repeated runs (`n200_chunk_size_sweep_repeated.py`, 3 reps) for the
close 1-vs-2 question specifically:

| chunk_size | mean | stdev |
|---|---|---|
| **1** | **120.48s** | **0.083 (0.07% cv)** |
| 2 | 122.93s | 0.127 (0.10% cv) |
| 4 | 131.33s | ~5s (varies more) |
| 8 (current floor) | 154.91s | ~1s |

Both `chunk_size=1` and `2` are extremely stable at this scale (cv
well under 0.1%) - unlike N=150, where `chunk_size=1` was the
*noisiest* value. At N=200, `chunk_size=1` is the real, stable,
reproducible winner - a genuine ~2% edge over `chunk_size=2`, and a
~22% edge over the current floor of 8.

**`perf stat`** (chunk_size=2 vs 8, N=200):

| chunk_size | wall(s) | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|
| 2 | 127.92 | 8.96% | 6.16% |
| 8 | 162.15 | 19.85% | 17.86% |

Same mechanism as N=150, confirmed again at this larger scale.

## Real memory footprint check (answers a direct user question)

Throughout every N=150/N=200 run in this investigation, `free -h`
polling showed total system memory usage essentially flat (~2.6-2.9
GiB `used`, regardless of chunk_size or N), and direct process RSS
monitoring during the N=200 repeated-run sweep showed the Python
process itself holding **88-106 MiB RSS** throughout - including at
`chunk_size=1`, where 19,907 chunks are processed sequentially, one
at a time. This confirms Phase 9's original streaming design is
working exactly as intended: peak memory is bounded by
`O(chunk_size * dim)` per chunk, completely decoupled from the total
term count (306M+ terms at N=200 never need to be resident together) -
`chunk_size` is a per-iteration batch size in a sequential loop, not
a measure of total problem size held in memory.

## No single static floor constant is right across N=25-200

Combining this investigation's data with the small-N sweep
(`chunk_size_cache_locality_findings.md`, and a fresh direct check of
chunk_size=1/2/8 at N=25/50 run for this document):

| N | dim | best `chunk_size` found | best working set (`cs*dim*16`) |
|---|---|---|---|
| 25 | 512 | 8 | 64 KiB (fits L2) |
| 50 | 2048 | 8 | 256 KiB (fits L2) |
| 100 | 8192 | flat/noisy, no clear winner among 1/2/4/8 | - |
| 150 | 16384 | 2 | 512 KiB (fits L3, not L2) |
| 200 | 32768 | 1 | 512 KiB (fits L3, not L2) |

At N=25/50, a quick direct re-check (3 reps each, this document) found
`chunk_size=8` clearly beats `chunk_size=2` (N=25: 0.0296s vs 0.0466s,
~57% slower at cs=2; N=50: 0.4248s vs 0.4755s, ~12% slower at cs=2),
and `chunk_size=1` is worse still at both. **A static floor of 2 -
the value that wins at N=150 - would be substantially suboptimal at
N=25/50, where 8 is clearly better.** The reverse is also true: the
current floor of 8 is clearly suboptimal at N=150/200.

The "best working set" column does not resolve to one clean cache
level either: N=25/50's optimum fits comfortably in the measured
per-core L2 (256 KiB), but N=150/200's optimum (512 KiB) exceeds L2
and sits in L3 territory instead - the ideal *target cache level
itself appears to shift* as `dim` grows, not just the chunk_size
needed to hit a fixed target. This is a genuinely more complex
relationship than `_min_chunk_size_floor()`'s current single-constant
model, or a simple `chunk_size = target_cache_bytes / dim` formula
alone would capture cleanly - see "What this does NOT show" below.

## Interpretation

The user's original suspicion - that abundant unused memory during
streaming might indicate suboptimal chunking, not just "efficient
memory use" - is confirmed correct. Two things are simultaneously
true, as the user's own framing anticipated:
1. Streaming's memory footprint really is small and mostly flat
   (Phase 9's design is working as intended) - this is genuinely good,
   not wasteful.
2. The *chunk_size* that footprint is built from has never been fully
   optimized - the shipped floor (8) is a leftover guess from one old
   N=25 data point, never re-validated at N=150/200 scale until this
   investigation, and turns out to be measurably suboptimal at both
   (11-22% slower than the real per-scale optimum).

Real levers identified but not yet built, raised directly by the user
during this investigation and worth a separate scoping pass: this
whole pipeline is currently single-core/single-process throughout (no
multiprocessing/MPI anywhere in the codebase, confirmed by direct
code search) - a real, unexplored opportunity given each chunk is
already an independent sub-problem by design, distinct from and
potentially more impactful than fine-tuning the floor constant alone.

## What this does NOT show

- Does not yet fix `_min_chunk_size_floor()` - this document
  establishes the evidence base; the fix (a real dim-dependent
  formula, not a static constant) is separate, subsequent work.
- Does not fill the N=2048-16384 `dim` gap (between N=50 and N=150) -
  the crossover point where the optimal shifts from 8 toward 2 is not
  pinpointed; a real formula would benefit from at least one more data
  point in this range (e.g. N=75 or N=100 with a proper chunk_size=1/2
  repeated-run check, not just the flat/noisy single-run data already
  on hand) before being trusted.
- Does not identify a clean closed-form relationship between `dim` and
  the optimal chunk_size or working-set size - the "best working set"
  column above does not resolve to one fixed cache-level target, and
  no formula has been fit to this data yet.
- N=100's own small-value (1/2) behavior was only checked with 3
  repeats in a range where the original single-run sweep already
  showed everything roughly flat/noisy - not deeply investigated
  further here, since N=150/200's clearer signal took priority.
- Does not measure or scope multi-core/multi-process chunk
  parallelism - raised as a real, larger opportunity by direct user
  question, explicitly deferred to a separate investigation.
