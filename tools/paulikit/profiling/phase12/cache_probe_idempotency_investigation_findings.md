# Investigating Bug 2 (cache-probe non-idempotency): root cause, and why the real fix was a thread lock, not a bigger warm-up

Recorded 2026-09-01, following up on Bug 2 from
[`n100_n150_autotuning_remeasurement_findings.md`](n100_n150_autotuning_remeasurement_findings.md)
("the cache probe is not idempotent when called repeatedly in the same
process"). That document characterized the bug probabilistically (~1
in 5 trials) without root-causing it precisely. This document finds
the exact mechanism, tries and rejects a fix aimed at the wrong
target, and lands on the actual, correctness-relevant fix.

## Root cause, precisely: large-buffer cache/TLB pollution, not random noise

Reproduced deterministically (not probabilistically) by controlling
which buffer sizes each call measures:

```
call 1: probe_cache_boundaries()                      # full 13 sizes, up to 32 MiB
call 2: probe_cache_boundaries(min_size_bytes=8192, n_sizes=3, ...)  # small only

call1[0:3]: [(8192, 5.68), (16384, 5.53), (32768, 5.56)]
call2:      [(8192, 8.19), (16384, 8.18), (32768, 8.45)]   # ~1.4-1.6x inflated
```

The second call's small-buffer readings are inflated **only when a
prior call already walked up through the large (multi-MiB) sizes** -
confirmed by isolating a large-only first call (8/16/32 MiB) followed
by a fresh small-only second call, which reproduces the same
inflation. Each buffer size already gets its own dedicated warm-up
walk (`3x` its own element count) before being timed - but that
warm-up is evidently not enough to fully re-normalize cache/TLB state
if a much larger working set was recently walked (in the same *or* a
prior call - the effect is about wall-clock-recent memory-hierarchy
state, not which Python call boundary it happened in).

## First fix attempt: raise the warm-up floor - worked, but on the wrong target

Tried making every buffer size's warm-up **at least** `N` accesses
regardless of its own size (previously purely `3x` the buffer's own
element count, meaning a tiny buffer got a tiny warm-up). Measured
directly:

| warm-up floor | small-buffer reading stability (repeated 2nd-call trials) |
|---|---|
| none (original, `3x` only) | ~1-in-5 wrong (per the original findings doc) |
| 4,194,304 accesses | improved to a consistent but still-wrong ~1.4-1.6x inflation |
| 16,777,216 accesses | ~1-in-8 wrong, each probe call now noticeably slower |

Diminishing returns, rising cost, never fully eliminated even at a
16M-access floor (13 sizes x up to 16M accesses each meaningfully
lengthens every `probe_cache_boundaries()` call). Before pushing the
floor even higher chasing full elimination, the actual question was
raised: **does this scenario ever happen in shipped code at all?**

## The scenario this was "fixing" cannot occur in production

`autotune.recommended_chunk_size()` (the only shipped caller of
`probe_cache_boundaries()`) caches its result at the top:

```python
global _cached_chunk_size
if _cached_chunk_size is not None:
    return _cached_chunk_size
```

So the probe is called **at most once per process**, full stop - every
real invocation is a first, isolated call. The original findings doc's
own data already showed first/single calls are reliable: **10/10
fresh-process trials returned the correct value.** The
"call-probe-twice-in-one-process" scenario the warm-up-floor fix was
targeting is not a scenario `auto_decompose()` or
`fwht_pauli_terms_iter` can ever trigger through the shipped public
API - it only manifests if a caller reaches past `autotune` and calls
`paulikit._native.cache_probe.probe_cache_boundaries()` directly,
repeatedly, themselves, which nothing in this package does.

**Decision: reverted the warm-up-floor change.** It was real runtime
cost paid for a scenario that cannot happen in production, chasing a
symptom of an artificial test rather than a real bug.

## The real gap: no protection against CONCURRENT (not sequential) calls

Reverting the warm-up change does not mean there was nothing to fix.
Explicitly checked, per direct instruction to verify this holds "even
using supercomputers or heavy/aggressive parallelism" before accepting
the per-process cache as sufficient:

- **Multiple separate OS processes** (the real HPC pattern - Slurm
  `--ntasks-per-node`, MPI ranks, or a user's own
  `multiprocessing.Pool`): each process gets its own fresh
  `_cached_chunk_size = None` and calls the probe at most once. Safe -
  this is exactly the "first, isolated call" pattern already confirmed
  reliable, repeated N times independently across N processes, not a
  repeated-call-in-one-process pattern at all.
- **`os.fork()` after a parent already primed its cache**: the child
  inherits the parent's already-populated `_cached_chunk_size` via
  copy-on-write memory, so it does not re-call the probe either.
- **Multiple THREADS in one process calling `auto_decompose()`
  concurrently** (a real pattern under heavy/aggressive parallelism,
  not covered by the process-level reasoning above): found and
  confirmed as a genuine gap. `_cached_chunk_size`/
  `_cached_memory_budget_bytes` were plain module-level globals with
  no synchronization - two threads could both observe an empty cache
  and both invoke the underlying probe/detection concurrently. This is
  a *different* failure mode than the sequential-pollution bug above
  (simultaneous CPU/cache contention between two probes racing against
  each other, not one call's stale state read by a later call) and was
  never tested by anything in this investigation until checked
  directly.

## The actual fix: a lock around cache population

Added `_cache_lock` (`threading.Lock`) in `autotune.py`, guarding both
`recommended_chunk_size()`'s and `available_memory_bytes()`'s
population logic with a double-checked-locking pattern (the common,
already-cached case stays lock-free; only first-population contends
for the lock). This closes the concurrent-call race by construction -
it is impossible for two threads to both invoke the underlying
detection, regardless of timing.

**Verified via mutation testing** (the same discipline used elsewhere
in this project - see Phase 11's tolerance-formula test): temporarily
replaced `with _cache_lock:` with a no-op `if True:` and re-ran the
new regression tests - both failed cleanly (8 concurrent threads all
invoking the underlying detection, `call_count == 8` instead of `1`),
confirming the tests actually exercise the race rather than passing
vacuously. Restored the real fix and re-ran - both pass.

Two new tests (`tests/test_autotune.py`):
`test_recommended_chunk_size_thread_safe_single_underlying_call` and
`test_available_memory_bytes_thread_safe_single_underlying_call` -
each spins up 8 threads calling the function concurrently, with an
artificial `time.sleep(0.05)` inside a mocked detection function to
reliably widen the race window (rather than relying on timing luck to
occasionally expose it), and asserts the underlying detection ran
exactly once.

## What this does NOT show

- Does not benchmark the lock's overhead - expected negligible (one
  `threading.Lock` acquisition per process for each of two functions,
  on the common already-cached fast path there is no lock contention
  at all since the `if _cached_... is not None` check happens before
  the `with _cache_lock:` block), not independently measured here.
- Does not address the underlying sequential-pollution mechanism
  itself (large buffer walk leaves state a small-buffer warm-up
  doesn't fully clear) - this remains true of the raw
  `cache_probe.probe_cache_boundaries()` extension if called directly,
  repeatedly, by a future caller that bypasses `autotune`'s cache. Not
  fixed, since no such caller exists today and chasing it further (per
  the warm-up-floor experiment above) has real cost and diminishing
  returns - flagged here for whoever adds such a caller in the future
  to be aware of, not silently forgotten.
- The two module functions currently share **one** lock
  (`_cache_lock`), not one lock each - a concurrent call to
  `recommended_chunk_size()` and `available_memory_bytes()` from
  different threads will serialize against each other even though
  they are logically independent. Accepted as a negligible one-time
  cost (both are cheap relative to the actual Pauli decomposition
  work they gate) rather than adding a second lock for a currently
  theoretical throughput concern.
- Does not test real multi-process HPC behavior end-to-end (no live
  Slurm/MPI job was run) - the "separate processes are safe" reasoning
  above is architectural (each process's own independent global state)
  and grounded in the already-confirmed first-call reliability data,
  not newly re-verified under a real multi-node job in this document.
