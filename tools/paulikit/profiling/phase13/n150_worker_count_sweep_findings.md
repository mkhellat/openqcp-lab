# N=150 n_workers sweep: found and fixed a second real bug, then real speedup collapses to ~1.06x

Recorded 2026-09-02, directly requested by the user (combined
memory-footprint + n_workers analysis in one pass, after the
memory-budget-wiring fix earlier the same day). Driver:
`n150_worker_count_sweep.py` - wall-clock and total process-tree RSS
(`/proc/<pid>/status` `VmRSS`, main process + all descendants,
POSIX-only, no third-party dependency) at `n_workers` in {1, 2, 4, 8},
`chunk_size` auto (`_recommended_parallel_chunk_size`).

## First run: found a second real bug (unbounded task submission)

| n_workers | elapsed | peak RSS | speedup |
|---|---|---|---|
| 1 | 39.86s | 5137 MiB | 1.000x |
| 2 | 30.38s | 5235 MiB | 1.312x |
| 4 | 28.16s | 14802 MiB | 1.415x |
| 8 | 27.88s | 24793 MiB | 1.429x |

RSS scaling from ~5 GiB to ~25 GiB across the sweep could not be
explained by `chunk_size`'s working set (confirmed directly:
`chunk_size=2` at `dim=16384` is under 1 MiB per chunk, and the
shared setup arrays - operator nonzeros, sorted index arrays,
`active_x` - are under 2 MiB total, checked directly). Root cause,
found by reading `parallel_decompose`'s own submission code: **every
chunk was submitted as a task to the pool up front**
(`futures = [pool.submit(...) for ... in pending]` - at N=150 with
`chunk_size=2`, that's 5,595 tasks submitted simultaneously). Workers
complete chunks faster than the single-threaded `as_completed` loop in
the main process can drain them (label generation + dict construction
per chunk), so completed-but-unconsumed `(x, z, coeff)` result arrays
pile up in the pool's IPC/result queue - a backlog that grows with
`n_workers` (more workers finish work faster, but the drain rate does
not increase to match), completely unbounded by
`per_worker_memory_budget_bytes` (that fix, from earlier the same
day, only bounds each *task's* working set, not how many completed
results are allowed to queue up before being consumed).

## Fix: bounded submission

Rewrote the pool-driving loop to keep at most `2 * n_workers` tasks
in flight at once (`concurrent.futures.wait(..., return_when=
FIRST_COMPLETED)`, submitting a replacement task each time one
completes and is consumed), instead of submitting the whole `pending`
list at once. Bounds the result backlog to `O(n_workers)`, matching
the `O(chunk_size * dim)` per-task bound the memory-budget-division
fix already provides for a single task.

## Second run, post-fix: memory collapses to expected scale, but so does the "speedup"

| n_workers | elapsed | peak RSS | speedup |
|---|---|---|---|
| 1 | 28.31s | 208 MiB | 1.000x |
| 2 | 27.18s | 269 MiB | 1.041x |
| 4 | 26.50s | 385 MiB | 1.068x |
| 8 | 26.64s | 627 MiB | 1.063x |

**Memory**: fixed dramatically - 208 MiB to 627 MiB across the sweep
(roughly linear in `n_workers`, consistent with `O(n_workers)`
in-flight task working sets, exactly as designed), versus 5 GiB to 25
GiB before the fix. Confirms the unbounded-submission bug, not
`chunk_size`, was the dominant memory cost.

**Speedup**: collapses from the pre-fix 1.00x/1.31x/1.42x/1.43x to
1.00x/1.04x/1.07x/1.06x - essentially flat, barely above 1x, and
*degrading slightly* from 4 to 8 workers. This means most of the
earlier "1.1-1.2x/1.4x" speedup measurements
(`n100_n150_parallel_decompose_findings.md`, and this document's own
first run above) were largely an artifact of the unbounded-submission
bug letting workers race far ahead of the consumer and buffer results
speculatively - not genuine parallel throughput. The real, honest
number for chunk-level parallelism at N=150 on this 8-core machine, as
currently implemented, is **essentially no speedup** (~1.06x at best).

## What this means for the contention-vs-dispatch-overhead question

The speedup curve's shape (flat from the very first additional worker,
not rising-then-saturating) is more consistent with **per-task
dispatch/IPC overhead dominating almost entirely**, not
L3/memory-bandwidth contention: if cache/bandwidth contention were the
main limiter, adding a second worker (going from fully-serial to
lightly-contended) should still show a real, non-trivial speedup
before contention catches up at higher worker counts - instead
speedup is already pinned near 1x at `n_workers=2`. This is consistent
with (not yet conclusive proof of) the earlier chunk_size-sweep
finding (`n100_n150_parallel_decompose_findings.md`'s "Quick
chunk_size sweep") that a >40x range of chunk_size did not change the
roughly-flat speedup - both point toward the same suspect: the fixed
cost of pickling a task's arguments and a chunk's result across the
process boundary is comparable to or larger than the actual per-chunk
compute time at this chunk_size, for this problem size.

## What this does NOT show

- Does not run `perf stat` yet - the scoping doc's own
  verification-plan item (LLC-specific counters under concurrent
  load) is still open, and would help distinguish "IPC/pickling
  overhead" from "real compute-level contention" more directly than
  wall-clock alone.
- Does not measure with a much larger `chunk_size` under the bounded-
  submission fix specifically (the earlier chunk_size sweep predates
  this fix, run under the unbounded-submission bug) - larger chunks
  mean fewer, bigger tasks, which should reduce the *relative* share
  of per-task IPC overhead if that is indeed the dominant cost; this
  is the most direct next test of the dispatch-overhead hypothesis.
- Does not test N=100 or N=200 under the fixed bounded-submission
  code - this sweep is N=150 only.
- Does not measure a genuinely large problem (larger `n_active`,
  where each chunk's compute cost is bigger relative to fixed
  per-task overhead) - all evidence so far is at scales where
  `chunk_size` stays very small (1-8), which may simply not be a
  favorable regime for chunk-level parallelism regardless of the
  bounded-submission fix.
