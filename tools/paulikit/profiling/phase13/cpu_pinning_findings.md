# CPU affinity pinning: implemented, verified working, but does NOT improve performance

Recorded 2026-09-02, direct follow-up to
`n_workers_placement_and_cache_findings.md`'s finding that
`ProcessPoolExecutor` workers were never pinned to physical cores and
migrated freely, colliding on the same physical core in nearly every
sampled instant. This document records the pinning implementation
itself and, honestly, its real (negative) performance result.

## Implementation

`_physical_core_representative_cpus()` (new, `fwht.py`) reads
`/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list` for
every CPU this process is allowed to use (respecting
`sched_getaffinity`'s cgroup/cpuset-aware mask, same discipline as
`_detect_available_worker_count`), groups logical CPUs into physical-
core sibling sets, and returns one representative CPU id per set. On
this dev machine: `[0, 1, 2, 3]` - exactly matching the real topology
found in `n_workers_placement_and_cache_findings.md`
((0,4),(1,5),(2,6),(3,7) sibling pairs).

`_pin_current_process_to_cpu(cpu)` calls `os.sched_setaffinity(0,
{cpu})` on the CALLING process - Linux only, same caveat as
`cache_probe.c`'s own `pin_to_one_cpu`, best-effort (an unpinned
worker is still correct, just not isolated).

Wired into `_parallel_worker_init` via a `multiprocessing.Value`
shared counter (`next_pin_index`) each worker process atomically
increments on startup to claim a distinct index into `pin_cpus` -
necessary because `ProcessPoolExecutor`'s `initializer`/`initargs` are
identical for every worker; there is no built-in per-worker ordinal to
key off of otherwise.

`parallel_decompose`'s auto-detected `n_workers` default (when the
caller passes `None`) was ALSO changed to use the physical-core count
(`len(_physical_core_representative_cpus())`) rather than the logical
CPU count from `_detect_available_worker_count()` - the evidence-based
choice from the n_workers sweep (n_workers=2 beating both 4 and 8 on
this 4-physical-core machine), falling back to the logical-CPU count
only if the physical-core probe itself is unavailable (non-Linux).
Explicit `n_workers=` from a caller is never overridden.

## Verification: pinning actually works (direct measurement, not assumed)

`check_worker_placement.py` re-run after the fix: all 4 real worker
processes each observed on exactly ONE logical CPU throughout the
entire run (pid->CPU: worker1->0, worker2->1, worker3->2, worker4->3),
zero migration - a clean, verified fix for the isolation gap. (A
follow-up variant script attempting to isolate worker-only collision
counts had its own measurement bug - the monitoring script's own
`ps`/`subprocess.run` calls are themselves transient children of the
root process and polluted the "descendant" list - but the unambiguous
per-worker CPU list above already proves the fix works; this
measurement-script bug was not pursued further since the answer was
already clear from the raw placement data.)

## The honest result: pinning does NOT help wall-clock or cache-miss ratio

Direct `perf stat` comparison, N=150, chunk_size=2, same event set as
every other measurement in this project:

| n_workers | pinning | wall-clock | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|---|
| 2 | unpinned | 24.14s | 9.2% | 8.1% |
| 2 | **pinned** | **25.45s** | **9.7%** | **8.3%** |
| 4 | unpinned | 25.48s | 9.6% | 8.2% |
| 4 | **pinned (run 1)** | **27.18s** | **9.5%** | **8.0%** |
| 4 | **pinned (run 2)** | **26.54s** | **9.5%** | **8.5%** |

Pinning made wall-clock **slightly worse** at both n_workers=2 (24.14s
-> 25.45s) and n_workers=4 (25.48s -> ~26.9s mean of 2 runs) - a small
but consistent, reproduced-twice regression, not noise. Cache-miss
ratio is essentially UNCHANGED by pinning (9.2%->9.7% at n_workers=2,
9.6%->9.5% at n_workers=4) - within the same noise band as run-to-run
variance seen throughout this whole investigation.

## Why pinning didn't help - critical discussion

This is a genuinely useful negative result, not just a failed
experiment. It falsifies a specific, plausible hypothesis: that the
*measured* cache contention (Question 2's flat ~9-10% cache-miss ratio
across all n_workers, found in `n_workers_placement_and_cache_findings.md`)
was primarily caused by hyperthread-sibling collisions from
un-pinned scheduler placement. If that were true, forcing clean
one-worker-per-physical-core placement should have visibly reduced
the cache-miss ratio. It did not.

This points toward the contention being dominated by something
pinning cannot fix: **L3 is shared across ALL 4 physical cores
regardless of which specific cores workers run on** - pinning controls
L1/L2 sharing (hyperthread siblings) but has no effect on L3
contention between *different* physical cores, which was already the
primary suspect from the original sequential-vs-parallel `perf stat`
comparison (`n150_perf_cache_locality_findings.md`, where cache-miss
ratio jumped from 5.1% sequential to ~9-10% with ANY concurrent
execution). The small wall-clock regression from pinning is also
plausible: removing the scheduler's freedom to migrate workers away
from momentarily-busy cores (e.g. if the OS itself needs a cycle on a
given core) can occasionally cost more than the isolation gains back.

**Conclusion**: L1/L2 hyperthread-sibling contention, while real and
now directly observable via the placement data, is NOT the dominant
cost in this workload's parallel slowdown - L3 sharing across all
physical cores is the more likely primary limiter, and no
per-process CPU pinning strategy can address that on this
single-socket, shared-L3 machine. The fix is kept (it is a genuine
correctness improvement - true isolation is now verified, not just
requested - and does not meaningfully regress performance), but it
should not be presented as a performance win.

## What this does NOT show

- Does not test pinning combined with a smaller n_workers count that
  might avoid L3 saturation differently (e.g. n_workers=1 "pinned" is
  trivially just the sequential path already measured).
- Does not test on a multi-socket or larger-L3-per-core machine, where
  the L1/L2-vs-L3 balance could be different - this conclusion is
  specific to this dev machine's single-socket, shared-8MB(ish)-L3
  topology.
- Does not separately measure L1/L2-specific hit/miss counters
  (only L1-dcache-loads/L1-dcache-load-misses, which count ALL L1
  misses including those that hit L2/L3, not L1-vs-L2-specific
  contention) - a more surgical counter set could still reveal an L1/
  L2 effect too small to show up in the aggregate ratios used here.
