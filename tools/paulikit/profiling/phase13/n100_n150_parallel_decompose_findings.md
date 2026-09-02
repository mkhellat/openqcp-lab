# 13a first real measurement: correctness confirmed, speedup real but far from linear

Recorded 2026-09-02, right after `parallel_decompose` was first
implemented. Confirms the scoping doc's own caution
(`scoping.md`'s "Correctness verification plan": "the real speedup
will be well under `n_workers`x") empirically, rather than leaving it
as an unverified hedge.

## Correctness

`parallel_decompose` output (order-combined via `dict.update` across
whatever order chunks complete in) matches `fwht_pauli_terms` exactly
on every fixture in `ALL_FIXTURES`, at multiple `chunk_size` values
(1, 2, 4) and worker counts (1, 2, 3, 64-clamped-down) - see
`tests/test_parallel_decompose.py`, 16 tests, all passing. Checkpoint/
resume (the new set-based format) verified directly: interrupting
after exactly one completed chunk, resuming, and confirming the final
combined result still matches the reference exactly, and that resuming
an already-complete checkpoint submits no new work.

## Real wall-clock speedup: N=100 and N=150, this machine (8 cores)

`n100_parallel_decompose_comparison.py` / `n150_parallel_decompose_comparison.py`,
both using `autotune.recommended_chunk_size(dim)` (the Phase 12
auto-tuned value, unchanged - Phase 13a does not yet re-tune
chunk_size for concurrent workers, see "What this does NOT show"),
`n_workers=8` (`_detect_available_worker_count()` on this 8-core dev
machine), `OPENBLAS_NUM_THREADS=1` (standard confound control for this
directory - see `profiling/README.md`):

| N | dim | auto chunk_size | sequential (1 proc) | parallel (8 workers) | speedup |
|---|---|---|---|---|---|
| 100 | 8192 | 3 | 7.98s (mean of 3) | 7.10s (mean of 3) | **1.12x** |
| 150 | 16384 | 2 | 33.18s (mean of 2) | 26.92s (mean of 2) | **1.23x** |

Both real, reproducible, directionally consistent (larger N -> larger
speedup) - not noise. But both are dramatically below the 8.0x ideal
linear speedup 8 workers would suggest if this scaled cleanly.

## Why the speedup is small: two real, distinct causes, not yet disentangled

1. **Chunk_size is tiny at these N** (Phase 12's own finding: 3 at
   N=100, 2 at N=150 - see `chunk_size_floor_scale_dependence_findings.md`).
   Each task therefore does very little actual work
   (`O(chunk_size * dim * log dim)`) relative to the fixed
   per-task overhead of a `ProcessPoolExecutor` dispatch (pickling the
   task's small arguments, IPC round-trip, result marshaling back to
   the main process). With `n_active / chunk_size` in the tens of
   thousands of tasks at these N, that per-task overhead - not the
   per-chunk compute itself - plausibly dominates total wall-clock.
   This is a genuinely different mechanism from #2 below, and the two
   were not separated in this first measurement.
2. **Shared LLC/memory-bandwidth contention** - flagged as a real,
   unresolved tension in `scoping.md` before any code existed: 8
   processes now compete for the one shared L3 this machine has
   (`cache_probe_extension_findings.md`), where Phase 12's own
   chunk_size tuning assumed a single lone process's undisputed cache.

## Quick chunk_size sweep at N=100 (partial evidence toward cause #1)

Same-process quick check, `n_workers=8`, `chunk_size` in {3 (auto), 32,
128}:

| chunk_size | sequential | parallel | speedup |
|---|---|---|---|
| 3 (auto) | 7.27s | 5.45s | 1.33x |
| 32 | 9.00s | 6.93s | 1.30x |
| 128 | 12.01s | 8.78s | 1.37x |

Speedup stays roughly flat (1.3-1.4x) across a >40x range of
chunk_size, not climbing toward linear as chunk_size grows - weak
evidence *against* cause #1 (dispatch overhead) being the dominant
factor alone, since a bigger chunk_size should shrink dispatch
overhead's relative share if it were the main limiter. This is
consistent with cache/memory-bandwidth contention (cause #2) being a
real, independent limiter regardless of chunk_size - but this sweep is
too small (3 points, single run each, no repeats) to be conclusive on
its own; flagged as a lead for the next session, not a closed
question.

**Correction**: L1/L2 are per-core private caches, and
`recommended_chunk_size` already targets the L1/L2 boundary
specifically (see `cache_probe_extension_findings.md`) - so if that
formula is working, 8 concurrent workers should NOT contend with each
other at L1/L2 at all; each core serves its own worker from its own
private cache. If cache/bandwidth contention (cause #2) is real, it is
therefore an **L3/memory-bandwidth** story, not an L1/L2 one - the
flat speedup across the chunk_size sweep is consistent with L3/
bandwidth/process-pool-IPC overhead, not with the specific
cache level `chunk_size` was tuned to target. This sharpens the
follow-up `perf stat` pass to watch LLC-load/LLC-miss counters
specifically. A cheap, more direct discriminator not yet run: compare
speedup at `n_workers` in {2, 4, 8} - roughly linear scaling at low
worker counts that degrades toward 8 would support L3/bandwidth
saturation; flat speedup even at 2 workers would point instead toward
per-task dispatch/IPC overhead as the dominant cost, independent of
cache contention entirely.

## What this does NOT show

- Does not disentangle cause #1 (per-task dispatch overhead at tiny
  chunk_size) from cause #2 (shared-cache contention) - both are
  plausible and neither has been isolated yet. A next step: sweep
  `chunk_size` explicitly under `parallel_decompose` (larger chunks =
  fewer, bigger tasks = less relative dispatch overhead, but a bigger
  per-worker working set = more real cache contention if #2 dominates)
  and see whether speedup improves, worsens, or is flat - would
  distinguish the two mechanisms.
- Does not re-tune `chunk_size` for concurrent-worker use - both runs
  above used the Phase 12 single-process-derived
  `recommended_chunk_size(dim)` unchanged, exactly the open question
  `scoping.md` flagged rather than resolved.
- Does not run `perf stat` under concurrent load yet - the
  cache-miss-ratio confirmation `scoping.md`'s own verification plan
  calls for is still open.
- Does not test N=200 or larger worker counts (this machine only has
  8 cores) - no data yet on whether the speedup trend continues to
  improve, plateaus, or reverses at larger scale.
- Does not measure a multi-node (13b) scenario - single machine only.
