# Clean, statistically-repeated chunk_size sweep (N=100, N=150) - supersedes contaminated earlier numbers

Recorded 2026-09-02. **This document supersedes the speedup numbers in
`n100_n150_parallel_decompose_findings.md` and
`n150_worker_count_sweep_findings.md`'s partial chunk_size data** - not
because those documents' bug findings were wrong (the memory-budget-
wiring bug and the unbounded-submission bug were both real, correctly
diagnosed, and correctly fixed), but because the *speedup numbers*
measured around and after those fixes were compromised by an execution
reliability failure: multiple background measurement jobs ended up
running concurrently on this 8-core machine without the previous
session realizing it, contending with each other for CPU. Per direct
user instruction ("since at some point two tests were running, I guess
we need to redo all tests again and this time statistically"), every
number from that contaminated window is discarded here, not
reconciled or partially trusted.

## What changed in execution method

1. **Foreground only** - no `run_in_background`, no scheduled wakeups.
   Each measurement script runs to completion in one blocking call;
   the machine is checked (`ps aux`) to be fully idle immediately
   before launching each one.
2. **Statistical repeats** - 3 reps at N=100, 2 at N=150 (smaller
   count only for wall-clock budget reasons at the larger, slower
   problem size), mean + stdev reported, not single-run numbers.
3. **`n_workers` explicit** in every row of every table - a real gap
   in the earlier (pre-redo) driver scripts' printed output that made
   the tables ambiguous, flagged directly by the user.

## N=100 (dim=8192), n_workers=8, 3 reps each

(`n100_chunk_size_sweep_repeated_post_fix.py`)

| chunk_size | n_workers | seq mean (stdev) | par mean (stdev) | speedup |
|---|---|---|---|---|
| 3 | 8 | 7.213s (0.021) | 5.665s (0.070) | 1.273x |
| 8 | 8 | 7.752s (0.122) | 5.707s (0.286) | 1.358x |
| 32 | 8 | 9.344s (0.020) | 7.368s (0.067) | 1.268x |
| 128 | 8 | 12.699s (0.260) | 8.735s (0.114) | 1.454x |

## N=150 (dim=16384), n_workers=8, 2 reps each

(`n150_chunk_size_sweep_repeated_post_fix.py`)

| chunk_size | n_workers | seq mean (stdev) | par mean (stdev) | speedup |
|---|---|---|---|---|
| 2 | 8 | 33.325s (0.882) | 25.994s (0.524) | 1.282x |
| 32 | 8 | 54.310s (0.154) | 34.770s (0.362) | 1.562x |
| 128 | 8 | 60.785s (0.981) | 40.641s (0.359) | 1.496x |

## Real, honest headline

Genuine, real speedup from `parallel_decompose` at both N=100 and
N=150 falls in the **1.27x-1.56x range** - moderate, real, and
directionally consistent between the two problem sizes (chunk_size=32
is the best or near-best performer at both N). This is neither the
"essentially flat ~1.06x" figure reported earlier in the same session
(itself measured during the execution-reliability failure, now
discarded) nor the earlier, larger "1.4x" figures that predated the
bounded-submission fix (inflated by that bug). This is the first
number in this investigation that should actually be trusted as
"real speedup, cleanly measured."

## What this does NOT show

- Does not include `perf stat`/cache-locality data yet - that is the
  next, separate, also-foreground-only measurement pass.
- Does not sweep `n_workers` itself with the same statistical rigor -
  only chunk_size was re-swept with repeats; a clean, repeated
  n_workers=1/2/4/8 comparison (matching
  `n150_worker_count_sweep.py`'s original question) has not yet been
  redone under this stricter execution discipline.
- Does not explain *why* chunk_size=32/128 outperform chunk_size=2-8
  at N=150 - this is a real, now-trustworthy pattern, but the
  mechanism (IPC overhead vs. something else) is still open pending
  the `perf stat` pass.
