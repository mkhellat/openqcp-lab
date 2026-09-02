# n_workers is not physical-core-aware: no pinning, no isolation, flat cache-miss ratio

Recorded 2026-09-02, prompted by direct, critical user questions after
the earlier clean chunk_size/perf-stat findings: "Maybe the number of
workers should be 4 (no of physical cores)... having 4 workers would
not work if they share L1/L2 caches" and, when an unverified 4-workers
claim was made, "are you sure with 4 workers, the system sets up 1
worker per physical core?!!" Both questions turned out to expose real
gaps - this document records what was actually measured to answer
them. Foreground-only, one job at a time, machine confirmed idle
before every launch (per [[feedback_foreground_only_measurement_jobs]]).

## Machine topology (checked directly, not assumed)

`lscpu`: Intel i7-8550U, **4 physical cores, 2 threads/core
(hyperthreading), 8 logical CPUs total**. `/sys/devices/system/cpu/
cpu*/topology/thread_siblings_list` confirms the physical-core pairing:
logical CPUs **(0,4), (1,5), (2,6), (3,7)** each share one physical
core's L1/L2. Every earlier "n_workers=8" measurement in this
investigation was 8 logical CPUs, i.e. 4 physical cores oversubscribed
2x - not 8 independent cores, a distinction the earlier findings docs
did not make explicit.

## Question 1: does n_workers=4 actually place one worker per physical core?

**No - checked directly, not assumed.** `check_worker_placement.py`
samples `ps -o pid,psr` (the logical CPU a process is *actually*
running on right now) every 0.15s throughout a real
`parallel_decompose(n_workers=4)` run at N=150, for all 4 worker
processes.

Result: every one of the 4 workers was observed running on **all 8**
logical CPUs (all 4 physical cores) over the run's lifetime - the
Linux scheduler freely migrates worker processes, with zero pinning
from `ProcessPoolExecutor`/`_parallel_worker_init` (neither calls
`sched_setaffinity` or anything like it). Co-residency check (two
workers sharing the same physical core at the same sampled instant):
**186 collision-events across 136 samples** - collisions in
effectively every sample, often more than one at once (3+ workers
briefly packed onto fewer than 4 physical cores).

**Conclusion**: the earlier framing of "n_workers=4 = physical cores,
n_workers=8 = logical CPUs" as if the former gives clean isolation was
wrong. Neither configuration is pinned; both are subject to the same
scheduler-driven churn.

## Question 2: does cache-miss ratio actually scale with n_workers?

Clean `perf stat` sweep, N=150, chunk_size=2 fixed, n_workers in
{1 (sequential), 2, 4, 8}:

| n_workers | wall-clock | cache-miss ratio | LLC-miss ratio | task-clock |
|---|---|---|---|---|
| 1 (sequential) | 34.42s | 5.1% | 3.8% | 34.4s |
| 2 | 24.14s | 9.2% | 8.1% | 55.6s |
| 4 | 25.48s | 9.6% | 8.2% | 60.1s |
| 8 | 25.71s | 9.7% | 8.8% | 60.9s |

**No - cache-miss ratio is essentially flat across n_workers=2, 4, 8**
(9.2% -> 9.6% -> 9.7%, within noise). The large jump happens once,
between sequential and *any* concurrent execution (5.1% -> ~9%), not
incrementally per added worker. This contradicts the natural
expectation that more concurrent workers should proportionally
increase cache pressure.

**Why, given Question 1's finding**: since no configuration achieves
clean physical-core isolation - the scheduler causes collisions at
n_workers=2 as much as at 4 or 8, given no pinning exists - there is
no clean "1-per-core" regime to compare against a "2-per-core" regime.
From the scheduler's perspective, 2, 4, and 8 concurrent processes are
all just "more than 1 process competing for 4 physical cores with
paired L1/L2," and the measured cache damage is dominated by that
basic fact (any concurrency at all triggers most of the contention
cost), not by the specific worker count.

## Why n_workers=2 still wins on wall-clock despite flat cache-miss ratio

If cache-miss ratio is flat, why did the earlier clean sweep
(`clean_chunk_size_sweep_findings.md`'s companion n_workers comparison,
same chunk_size=2) show n_workers=2 (24.14s) beating both n_workers=4
(25.48s) and n_workers=8 (25.71s) on wall-clock? Not fully explained
by cache-miss ratio alone (which is nearly identical across all
three) - the remaining, more likely explanation is `task-clock`
(total CPU-seconds consumed): 55.6s at n_workers=2 vs. 60.1s/60.9s at
4/8 - **more workers consume more aggregate CPU time for the same
wall-clock-equivalent result**, consistent with per-process overhead
(process startup/teardown, IPC/pickling, scheduler churn from more
processes being juggled across fewer physical cores) scaling with
worker count even though cache behavior itself does not. This is a
real, separate cost from cache contention, not yet independently
confirmed.

## User's hypothesis: right in direction, wrong in specific mechanism

The instinct that "n_workers should not exceed physical core count"
is **directionally correct** (2 workers beats 4 and 8 on this
4-physical-core machine) but the *reason* is not "clean 1-per-core
cache isolation at n_workers=4, degraded 2-per-core sharing at
n_workers=8" - that clean regime does not exist in the current
implementation, since nothing pins workers to cores. The real
mechanism, based on what's measured so far, looks more like: (a) any
concurrency beyond ~1 triggers most of the achievable cache-miss-ratio
increase immediately, and (b) each additional worker beyond a small
number adds real per-process overhead (task-clock) without a
corresponding cache-locality benefit, since isolation was never
achieved to lose more of as workers increase.

## What this does NOT show

- Does not yet test explicit CPU affinity pinning (one worker process
  pinned to one physical core, e.g. logical CPUs {0,1,2,3}, explicitly
  avoiding the hyperthread siblings {4,5,6,7}) - this is the direct
  next test of whether TRUE physical-core isolation (not just a
  `n_workers=4` request left to scheduler discretion) changes the
  cache-miss-ratio-vs-n_workers picture. Implementation and
  measurement of this is the immediate next step, tracked separately.
- Does not test n_workers=3 or other intermediate values - only
  1/2/4/8 were swept.
- Does not isolate whether the flat cache-miss ratio is dominated by
  L1/L2 (hyperthread-pair sharing) or L3 (shared across all 4 cores
  regardless of pinning) contention specifically - the aggregate
  cache-misses/LLC-load-misses counters used here do not distinguish
  the two, and without pinning, both are plausible simultaneously.
