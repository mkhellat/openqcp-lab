# Accounting for the extra cycles: NOT extra work, IPC (stalling) is the real source

Recorded 2026-09-03. Direct answer to the question raised after
`core_packing_series_thermal_controlled_findings.md` established a
real, substantial 5-14% extra-cycles cost when workers are spread
across more physical cores: "these are not normal amounts of extra
work, are they? Is there an overhead? How could we do some accurate
accounting to extract the sources of extra work."

## Method

`core_packing_thermal_controlled_welch_ttest.py` extended to capture
and report `instructions` (raw retired-instruction count from `perf
stat`, already being requested in every prior run but never
extracted) and IPC (`instructions / cycles`) alongside everything
already measured. Re-ran all four worker-count comparisons
(`pinned_2`, `pinned_3`, `pinned_4`, `pinned_5`), 5 reps each,
thermally controlled (cooldown to 55C before every run, same protocol
as `core_packing_series_thermal_controlled_findings.md`).

This directly distinguishes two possible sources of the extra cycles:
1. **Real extra work** - more instructions actually executed (would
   show up as a significant `instructions` difference) - e.g. retries,
   extra syscalls, scheduler/kernel bookkeeping specific to one
   configuration.
2. **Same work, worse efficiency** - identical instruction count, but
   more cycles needed to execute them (would show up as a significant
   IPC difference, with `instructions` unchanged) - consistent with
   more pipeline stalling, e.g. from cache/memory latency.

## Raw data (5 reps per condition, all four worker counts)

### pinned_2

pinned_2_2cores: instructions=[123095928102, 122818468588, 122963429032, 122577798761, 122573936856];
ipc=[1.7751, 1.8169, 1.8020, 1.8740, 1.8279]

pinned_2_1core: instructions=[122374947619, 122896004295, 122245180972, 123167775033, 122890318396];
ipc=[2.0441, 2.0668, 2.0368, 2.0903, 2.0888]

### pinned_3

pinned_3_3cores: instructions=[122456481000, 122996571057, 122860683464, 122564444722, 122591895813];
ipc=[1.8173, 1.8439, 1.8243, 1.8101, 1.8307]

pinned_3_2cores: instructions=[123006941255, 123023697335, 122813807684, 122941407613, 122453199654];
ipc=[1.9351, 1.9690, 1.9552, 1.9780, 1.9591]

### pinned_4

pinned_4_4cores: instructions=[122867614703, 122717084264, 122721064467, 122522804324, 122865764165];
ipc=[1.7588, 1.7725, 1.7803, 1.7357, 1.7528]

pinned_4_2cores: instructions=[122959400964, 123152809784, 123190591493, 122927413134, 122693818620];
ipc=[1.9889, 2.0008, 1.9809, 1.9855, 1.9772]

### pinned_5

pinned_5_4cores: instructions=[123024536201, 123022299896, 122810369134, 123392499355, 123133240971];
ipc=[1.8084, 1.8066, 1.8322, 1.7955, 1.7846]

pinned_5_3cores: instructions=[122522390167, 122888064461, 122993083206, 123073667116, 123187158313];
ipc=[1.8714, 1.8872, 1.8787, 1.9036, 1.9050]

(elapsed, cycles, cache-miss ratio, LLC-miss ratio, peak RSS, and
temperature covariates for these same runs match
`core_packing_series_thermal_controlled_findings.md`'s own numbers
within normal run-to-run variance - a second independent replication
of that series, not just an instructions/IPC addendum.)

## Results: instructions vs. IPC, all four worker counts

| n_workers | instructions p-value | instructions significant? | IPC p-value | IPC significant? |
|---|---|---|---|---|
| 2 | 0.667 | No | 0.000005 | **Yes** |
| 3 | 0.322 | No | 0.000001 | **Yes** |
| 4 | 0.058 | No (borderline, ~0.2% diff) | <0.000001 | **Yes** |
| 5 | 0.361 | No | 0.000048 | **Yes** |

Instructions means (billions), more-cores vs. fewer-cores: n=2:
122.81 vs 122.71 (diff 0.07%); n=3: 122.69 vs 122.85 (diff 0.13%);
n=4: 122.74 vs 122.98 (diff 0.20%); n=5: 123.08 vs 122.93 (diff 0.12%).
Every difference is under 0.2% of the total instruction count and
none is a large, clearly real effect - consistent with genuinely
identical work being done (the same algorithm, same data, same
correctness-verified output in every prior comparison in this
investigation).

IPC means, more-cores vs. fewer-cores: n=2: 1.82 vs 2.07; n=3: 1.83 vs
1.96; n=4: 1.76 vs 1.99; n=5: 1.81 vs 1.89. Every comparison shows a
large, highly significant (p<0.0001 in 3 of 4 cases), consistent drop
in IPC when workers are spread across more physical cores.

## Conclusion: the extra cycles are NOT extra work - they are the SAME work executing less efficiently

At all four worker counts, `instructions` is statistically
indistinguishable between the "more cores" and "fewer cores"
conditions (differences under 0.2%, never significant, or only
borderline at n=4 with a tiny effect size) - there is no real overhead
in the sense of doing MORE work. What differs, robustly and
significantly at every worker count, is IPC: the exact same
instruction stream takes MORE CYCLES to execute when workers are
spread across more physical cores.

This is now a complete, internally consistent mechanistic story,
built entirely from real measured data across this investigation:
1. Spreading workers across more physical cores means more total L3/
   memory-bandwidth pressure system-wide (more cores generating cache
   traffic) - already established in
   `l3_capacity_vs_bandwidth_findings.md` and reconfirmed as the
   likely mechanism in `pinned4_4cores_vs_2cores_findings.md`.
2. This shows up as worse cache-miss/LLC-miss ratios when spread -
   confirmed at every worker count in
   `core_packing_series_thermal_controlled_findings.md`.
3. Worse cache-miss/LLC-miss ratios mean more pipeline stalls waiting
   for data from cache/memory - this is EXACTLY what IPC measures,
   and it is exactly what this document finds: significantly lower
   IPC when spread, at every worker count, with no corresponding
   change in the actual amount of work (instructions) being done.
4. Lower IPC directly means more cycles are needed for the same
   instruction count - accounting for the 5-14% extra-cycles finding
   from the original core-packing series, without needing to invoke
   any additional "overhead" category (extra syscalls, retries,
   scheduler bookkeeping, etc.) - none of which show up in the
   instructions counter at all.

**Direct answer to "is there an overhead"**: no, not in the sense of
extra instructions/extra work being executed. The "overhead" is
entirely explained by reduced per-instruction efficiency (more
stalling) caused by worse cache locality when more physical cores are
simultaneously active and contending for the shared L3/memory
subsystem - a real, now fully quantified and mechanistically
explained effect, not a mysterious unaccounted-for cost.

## What this does NOT show

- Does not further decompose WHY IPC drops specifically - stalling
  could be memory-latency-bound (waiting for DRAM), L3-latency-bound
  (waiting for a slower cache level than L1/L2), or bandwidth-queueing-
  bound (waiting in a shared memory-controller queue) - the
  `cycle_activity.stalls_total`/`stalls_mem_any` events used earlier
  in this investigation (Phase 12 work) could distinguish these
  further but were not re-collected here.
- Does not test whether this same instructions-unchanged/IPC-degraded
  pattern holds at `n_workers=8` (logical CPU count) vs. lower counts,
  or connects to the still-open `pinned_4` vs. `unpinned_4` wall-clock
  puzzle from `pinned4_regression_discussion.md` - that comparison
  uses a different independent variable (pinned vs. unpinned, not
  more-cores vs. fewer-cores) and has not been re-examined with
  instructions/IPC accounting.
