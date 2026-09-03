# Properly repeated cache-miss comparison: the single-run "pinned_4 has worse cache misses" claim does NOT hold up

Recorded 2026-09-03. Direct follow-up after two things were correctly
challenged: (1) the single-run cache-miss claim in
`full_matrix_findings.md` was never statistically repeated, and (2)
the reasoning offered for it (L1/L2 dedication doesn't help against
shared L3 contention) was flawed - L3 pressure is a SHARED, constant
factor between `pinned_4` and `unpinned_4` (both have 4 physical
cores contending for the same L3 either way), so it cannot explain a
DIFFERENCE between the two conditions. Only something that actually
differs between them (pinned vs. not) can explain a difference.

## Method

`pinned4_cache_miss_welch_ttest.py`: 5 real `perf stat --no-inherit`
runs per condition (real `parallel_decompose()`, N=150, chunk_size=2,
same L3 event group as every other `perf stat` measurement in this
project), foreground, real Welch's t-test on cache-miss ratio,
LLC-miss ratio, and elapsed time together - same statistical
discipline as the wall-clock work.

Caught and fixed a real bug before trusting any data: the first
attempt's `subprocess.run` used a relative script path and replaced
the environment wholesale, causing the target script to fail
immediately - the "counters" captured were garbage from the failed
wrapper process (70-80ms task-clock, nowhere near a real ~30s run).
Fixed with an absolute path and an inherited environment
(`OPENBLAS_NUM_THREADS=1` added, everything else preserved) before
trusting any subsequent result.

## Raw data (5 reps each)

**pinned_4**: elapsed=[32.949, 31.171, 29.275, 28.519, 28.363];
cache_miss_ratio%=[35.946, 36.571, 35.086, 34.623, 34.919];
llc_miss_ratio%=[21.746, 21.386, 19.990, 19.737, 20.197]

**unpinned_4**: elapsed=[27.218, 27.440, 27.169, 27.089, 27.242];
cache_miss_ratio%=[36.474, 35.978, 35.934, 35.826, 36.101];
llc_miss_ratio%=[21.656, 21.558, 21.275, 21.449, 21.654]

## Results (Welch's t-test)

| metric | pinned_4 mean | unpinned_4 mean | diff | p-value | significant? |
|---|---|---|---|---|---|
| cache-miss ratio | 35.43% | 36.06% | +0.63pp | 0.157 | **No** |
| LLC-miss ratio | 20.61% | 21.52% | +0.91pp | 0.086 | **No** |
| elapsed | 30.06s | 27.23s | -2.82s | 0.032 | **Yes** |

## Conclusion: the single-run cache-miss claim does not survive repetition, and its explanation was never valid reasoning to begin with

Neither cache-miss ratio nor LLC-miss ratio shows a statistically
significant difference between `pinned_4` and `unpinned_4` - if
anything, `pinned_4`'s point estimates are numerically slightly LOWER
(better) than `unpinned_4`'s, the opposite direction from the earlier
single-run claim in `full_matrix_findings.md`. This is now the third
finding in this investigation where a single-run comparison did not
replicate under proper statistics (after the `pinned_2`/`unpinned_2`
wall-clock question's own multi-round reversal).

Combined with the WALL-CLOCK difference reproducing again, cleanly,
in this same run (p=0.032, consistent with every prior test in this
investigation): **cache/LLC-miss ratios are not the explanation for
the confirmed wall-clock regression** - they are statistically
indistinguishable between the two conditions, while wall-clock is not.
Whatever causes `pinned_4` to run slower than `unpinned_4` is not
visible in these L3-level cache counters at all.

## What this sharpens, not resolves

The mechanism behind the wall-clock gap is now narrower to search for:
not cache-level contention (ruled out by this document), not
throttling severity (ruled out by `pinned4_regression_discussion.md`'s
temperature-covariate check), not the two HWP-related hypotheses
(ruled out by `turbostat_verification_findings.md`). What remains
uninvestigated: anything downstream of the ONE real difference between
the two conditions - the presence or absence of a fixed CPU affinity
mask - most directly, scheduler behavior itself (migration counts,
run-queue latency, context-switch overhead) has not been measured at
all in this investigation and is the most direct remaining candidate,
not yet checked.
