# 4 physical cores vs. 2 physical cores (same 4 workers): packing onto fewer cores wins despite worse cache misses

Recorded 2026-09-03, per direct user request: compare `pinned_4`
locking 4 logical CPUs from 4 DISTINCT physical cores vs. the same 4
workers locked onto only 2 physical cores (both hyperthread siblings
of each), with wall-clock, core speeds (cycles), cache misses, and
memory footprint, all statistically verified.

## Design

Two conditions, both `n_workers=4`, both via the real
`parallel_decompose()`, differing ONLY in which logical CPUs are
pinned:
- **`pinned_4_4cores`**: cpus `[0, 1, 2, 3]` - one logical CPU from
  each of the 4 distinct physical cores (A, B, C, D) - identical to
  the standing `pinned_4` condition used throughout this
  investigation.
- **`pinned_4_2cores`**: cpus `[0, 4, 1, 5]` - BOTH hyperthread
  siblings of physical core A (0, 4) and BOTH siblings of physical
  core B (1, 5) - physical cores C and D left completely unused.

Verified directly before trusting any measurement: each of the 4 real
forked worker processes writes its own `os.sched_getaffinity(0)` to a
file on its first chunk - confirmed exactly `{0}`, `{4}`, `{1}`, `{5}`
across the 4 workers, matching intent exactly.

5 reps per condition, real `perf stat --no-inherit` (L3 event group:
`task-clock,cycles,instructions,cache-references,cache-misses,
LLC-loads,LLC-load-misses`), real Welch's t-test on wall-clock,
cycles, cache-miss ratio, LLC-miss ratio, and peak RSS (via the
existing `RssMonitor` in `full_matrix_target.py`).

## Raw data (5 reps per condition, complete and unfiltered)

**`pinned_4_4cores`**:
- elapsed (s): [27.8528, 29.1499, 28.4576, 28.7351, 28.3079]
- cycles: [70132690052, 70208822877, 69663174076, 70517949111, 68965616706]
- cache_miss_ratio (%): [33.602, 34.259, 34.912, 34.890, 34.745]
- llc_miss_ratio (%): [19.181, 19.599, 20.164, 20.406, 19.754]
- peak_rss_mib: [370.0, 368.1, 368.3, 371.2, 367.6]

**`pinned_4_2cores`**:
- elapsed (s): [26.7382, 25.9556, 26.3311, 25.9600, 26.1773]
- cycles: [62035684947, 60853113705, 61414776986, 60965723068, 60812275812]
- cache_miss_ratio (%): [37.173, 36.073, 37.091, 36.155, 37.247]
- llc_miss_ratio (%): [22.846, 22.011, 22.831, 21.906, 23.274]
- peak_rss_mib: [368.0, 368.8, 368.0, 366.8, 366.8]

## Results (Welch's t-test on the raw data above)

| metric | 4cores mean | 2cores mean | diff (2c-4c) | p-value | significant? |
|---|---|---|---|---|---|
| wall-clock | 28.50s (sd=0.48) | 26.23s (sd=0.32) | -2.27s | 0.00005 | **Yes**, 2cores faster |
| cycles | 69.90B (sd=0.60B) | 61.22B (sd=0.52B) | -8.68B | <0.000001 | **Yes**, 2cores fewer |
| cache-miss ratio | 34.48% (sd=0.56) | 36.75% (sd=0.58) | +2.27pp | 0.0002 | **Yes**, 2cores worse |
| LLC-miss ratio | 19.82% (sd=0.48) | 22.57% (sd=0.59) | +2.75pp | 0.00005 | **Yes**, 2cores worse |
| peak RSS | 369.0 MiB (sd=1.51) | 367.7 MiB (sd=0.87) | -1.4 MiB | 0.128 | No difference |

## A genuinely striking dissociation: worse cache behavior, but faster and fewer cycles

`pinned_4_2cores` shows WORSE cache-miss ratio (+2.27pp) and WORSE
LLC-miss ratio (+2.75pp) than `pinned_4_4cores` - consistent with the
textbook expectation that packing 2 pairs of hyperthread siblings onto
2 physical cores causes real L1/L2 contention between siblings that
spreading across 4 independent cores avoids.

BUT `pinned_4_2cores` is nonetheless FASTER (26.23s vs. 28.50s) AND
requires FEWER total cycles (61.22B vs. 69.90B) to complete the exact
same computation. The cycles result is important: this is not a
wall-clock/DVFS artifact (a cycle is a cycle regardless of clock
speed) - the machine genuinely does less total hardware work when the
4 workers are packed onto 2 physical cores than when spread across 4,
despite that packed configuration having measurably worse cache
behavior by the counters that specifically measure cache behavior.

## Interpretation, connecting to earlier findings

This is consistent with (not proof of, but a real, direct data point
supporting) the earlier finding in `l3_capacity_vs_bandwidth_findings.md`
that L3-CAPACITY contention is the dominant cost on this machine (not
memory-bandwidth alone, and not, as shown repeatedly in this
investigation, L1/L2 hyperthread sharing). Leaving 2 entire physical
cores completely idle (`pinned_4_2cores`) means only 2 physical cores'
worth of L2 traffic ever reaches the shared L3 at all, versus 4
physical cores' worth in `pinned_4_4cores` - even though the 2
active cores in the packed configuration are individually paying a
real, measured hyperthread-sharing tax (confirmed by the C6%/cache-
miss data), the SYSTEM-WIDE reduction in L3 pressure from having half
as many physical cores generating L3 traffic appears to more than
compensate. This is the first RESULT in this investigation (not
just measurement/elimination work) that directly and favorably
connects "use fewer physical cores" to real, statistically-confirmed
performance improvement - genuinely useful, actionable information for
`parallel_decompose`'s own worker-count defaults, distinct from the
still-unexplained `pinned_4` (4 cores) vs. `unpinned_4` puzzle from
earlier documents.

## What this does NOT show

- Does not explain WHY fewer active physical cores reduces total
  cycles required - the mechanism (plausibly: less L3 eviction
  pressure system-wide, but this is inference from the earlier L3-
  capacity finding, not directly re-measured here) is not re-verified
  in this specific experiment.
- Does not test intermediate configurations (e.g. 3 physical cores,
  or 4 workers on 3 cores with one core doubled up) - only the two
  clean endpoints (4 distinct cores vs. 2 distinct cores) were tested.
- Does not test this same comparison at `n_workers=2` (e.g. both
  workers on 1 physical core's 2 hyperthreads, vs. 2 distinct physical
  cores) - a natural, cheap follow-up not yet run.
- Does not directly reconcile this finding with the STANDING
  `pinned_4` (4 distinct cores, i.e. THIS document's `pinned_4_4cores`)
  vs. `unpinned_4` puzzle - `unpinned_4` lets the OS scheduler choose
  placement freely, which could itself be discovering something closer
  to the 2-cores configuration's efficiency dynamically. This is a
  plausible, NOT YET TESTED connection between this document's finding
  and the still-open mystery from `pinned4_regression_discussion.md`.
