# Phase 13 scoping: multi-core / multi-node chunk parallelism

Scoped 2026-09-02. Raised directly by the user during Phase 12's
chunk_size-floor investigation, via a real architectural question: if
this pipeline cannot use more than one core, what is the point of
running it on a supercomputer node or a machine with a lot of memory?
Confirmed at the time via direct code search: **zero**
multiprocessing/MPI/thread-pool execution exists anywhere in
`paulikit` today. Every measured speedup so far (Phases 3, 9, 10, 11,
12) has come from making one core's work cheaper or its memory
footprint smaller - never from using a second core. This document
scopes the design; nothing here is implemented yet.

## Why this is real, and why it's genuinely available

`_iter_chunked_coefficients` (`fwht.py`) already states the key
property in its own docstring: **each chunk is a fully independent
sub-problem** - no cross-chunk combination step exists in the
underlying math (unlike, say, tiled matmul's block-sum reduction).
Concretely, per chunk `c`:

```
gathered_chunk = gather(operator, active_x[c], ...)      # chunk-local
transformed_chunk = WHT(gathered_chunk)                   # chunk-local
chunk_coefficients = transformed_chunk * phase(active_x[c])  # chunk-local
chunk_x_out, z_out, coeff_out = threshold(chunk_coefficients)  # chunk-local
```

Nothing in this reads or writes state belonging to another chunk. The
final result is the **union** of all chunks' `(x, z, coefficient)`
triples - concatenation, not reduction. This is about as clean a
data-parallel decomposition as exists: no shared mutable state, no
ordering dependency, no partial-sum accumulation across workers. Nodes
in an HPC allocation, or extra cores on a workstation, are sitting
idle against exactly the axis (`n_active` chunks) this pipeline
already iterates over sequentially.

## Two distinct axes, deliberately scoped separately

Per the user's own question - "what would be the use of running on
supercomputers with nodes or machines with large memories" - the
motivating question mixes two different resources (cores within a
node, and multiple nodes). They need different mechanisms and are
scoped as two sub-phases, not one:

**13a - Multi-core, single-node, single-process-tree.** Distribute
chunks across the cores available *to this process* (respecting
`sched_getaffinity`/cgroup cpuset restrictions - the same discipline
`cache_probe.c`'s `pin_to_one_cpu` and `autotune.py`'s cgroup-memory
check already apply, extended here to core *count* rather than just
memory). No cluster/job-launcher dependency; works identically on a
laptop and inside one Slurm task with `--cpus-per-task`. This is the
higher-value, lower-risk piece and should be built and measured first.

**13b - Multi-node.** Distributing chunks across a Slurm/PBS
allocation's multiple *nodes*, not just multiple cores on one node -
needs a process-launch mechanism that crosses node boundaries (MPI, or
a Slurm-array/job-step approach where each node processes a
pre-partitioned slice of chunks and results are collected separately,
e.g. one output file per node). Explicitly deferred until 13a is built
and measured - see "Sequencing" below for why.

## 13a design: process pool, not threads - and why

Python's GIL is held during the WHT butterfly stage's own Python-level
loop control (`_walsh_hadamard_transform_rows`'s `while span < dim`
loop, `log2(dim)` reshape/slice/assign operations per row-block) even
though each individual NumPy op releases the GIL internally - the
per-stage Python overhead does not disappear just because the array
ops are vectorized. More fundamentally: `threading.Lock`-guarded
caches in `autotune.py` already establish that *this specific
codebase's* concurrency primitive of choice for HPC/parallel safety is
processes-as-the-real-parallelism-unit, threads-only-for-in-process
coordination - worker chunks should follow the same pattern, using
`multiprocessing`/`concurrent.futures.ProcessPoolExecutor`, not
threads, to get real parallel CPU-bound execution.

**Sketch** (not final API, see "Open design questions" below):

```
def _process_chunk(operator_bytes, chunk_range, ...) -> (x, z, coeff):
    ...  # exactly _iter_chunked_coefficients's per-chunk body

def parallel_decompose(operator, chunk_size, n_workers=None, ...):
    chunk_ranges = _partition_into_chunks(n_active, chunk_size)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_process_chunk, chunk_ranges, ...):
            yield result  # or accumulate, depending on API shape - see below
```

`n_workers` auto-detection should use `len(os.sched_getaffinity(0))`
(Linux; already the pattern `cache_probe.c`'s CPU-pinning code
respects) rather than `os.cpu_count()`, which reports the *node's*
total core count even inside a cgroup/cpuset-restricted job - the same
correctness bug class Phase 12 fixed for memory
(`available_memory_bytes` vs. raw `/proc/meminfo` `MemTotal`), now for
core count. `multiprocessing.cpu_count()` has the identical flaw (it
is a thin wrapper over `os.cpu_count()`).

## Interaction with Phase 12's memory-budget/chunk_size auto-tuning (a real, unresolved tension)

This is the most important open question, not a footnote. Phase 12's
`chunk_size` floor formula and `auto_decompose`'s dense-vs-streaming
memory budget were both measured and tuned **on a single core, single
process, no contention from sibling workers**. Multi-core execution
directly perturbs both:

- **Memory budget**: `available_memory_bytes()` returns a single
  process's view of available memory. If `N` worker processes each
  independently query and act on that same budget, an
  `auto_decompose` call that correctly decided "stream, budget is
  tight" for one process could still OOM the node if `N` processes
  each hold their own `O(chunk_size * dim)` working set
  simultaneously - the *node's* effective budget for this job must be
  divided by (at least) the worker count, not read once and reused
  unchanged per worker. Concretely: `available_memory_bytes() /
  n_workers`, not `available_memory_bytes()`, is each worker's real
  per-process ceiling once running in parallel - the current function
  is correct for the single-process code path but would be silently
  wrong if reused verbatim inside a multi-process one.
- **Chunk-size floor / cache locality**: Phase 12's whole floor-formula
  investigation (`chunk_size_floor_scale_dependence_findings.md`)
  measured cache-miss ratios and optimal working-set sizes assuming
  this process has a core's full L2/L3 to itself. With `N` worker
  processes running concurrently, they compete for the *shared* LLC
  (this machine's L3 is shared across cores - see
  `cache_probe_extension_findings.md`) and for total memory bandwidth.
  The chunk_size that was cache-optimal for one lone process may no
  longer be optimal once several processes are hammering the same L3
  concurrently - this needs its own real measurement (`perf stat`
  again, same discipline as Phase 12) once a working implementation
  exists, not an assumption either way. Flagged directly per the
  user's own question about "performance competition between... size
  of normal memory... vs the smallness of cache" - multi-core
  execution is exactly where that competition becomes real for the
  first time in this project (a single process today never contends
  with itself for cache).

Neither of these blocks starting 13a's implementation, but both mean
Phase 12's existing formulas must be revisited (not assumed to
transfer unchanged) as part of 13a's own measurement pass, not
deferred to some later phase.

## API shape: open design question, not yet decided

Three distinct places parallelism could plug in, each with different
tradeoffs:

1. **A new top-level function** (`parallel_decompose` / `auto_decompose(..., n_workers=...)`) -
   keeps `fwht_pauli_terms`/`fwht_pauli_terms_iter`'s existing
   contracts untouched (same "don't change an existing function's
   return-type/behavior contract based on runtime state" discipline
   `auto_decompose` itself already established in Phase 12 - see its
   own docstring's nondeterminism-hazard rationale). Cleanest
   backward-compatibility story; likely the right default choice.
2. **A `n_workers` parameter added to `fwht_pauli_terms_iter`** -
   changes iteration order (chunks would complete out of submission
   order under a process pool, unless explicitly serialized) and the
   generator's own resource lifetime (a pool must stay alive for the
   duration of iteration, unlike today's plain generator) - a more
   invasive contract change to an existing, documented function.
   Disfavored unless option 1 turns out to be awkward in practice.
3. **A separate, standalone parallel-map utility** callers combine
   with the existing chunked generator themselves - maximum
   flexibility, but pushes the memory-budget-division and
   worker-count-detection correctness work (see above) onto every
   caller instead of solving it once. Disfavored - repeats Phase 12's
   own reasoning for why `auto_decompose` exists as a single
   correctness-critical decision point rather than leaving every
   caller to reinvent it.

**Leaning toward option 1**, consistent with Phase 12's own precedent,
but not finalized - worth a real AskUserQuestion round before writing
code, same as Phase 12's own design phase.

## Checkpoint/resume interaction

`_iter_chunked_coefficients`'s checkpoint file (Phase 9) assumes
strictly sequential chunk processing - `next_chunk` is a single
monotonic index, and resuming re-processes from that index onward.
Parallel workers complete chunks out of order, so either:
- checkpointing is disabled/unsupported in the parallel path
  initially (simplest, and likely fine - checkpointing exists for
  single-process crash recovery on very large N, a scenario this
  phase's own point is to make less necessary by finishing faster), or
- the checkpoint format is redesigned to record a *set* of completed
  chunk indices rather than one monotonic marker, allowing resume to
  skip only the specific chunks already done, regardless of order.

Leaning toward "disabled initially, revisit if a real need surfaces" -
consistent with not building speculative machinery ahead of a real
requirement, and avoids blocking 13a's actual goal (wall-clock
speedup) on solving a problem (parallel-safe resumability) nobody has
asked for yet.

## Correctness verification plan (once built)

Same discipline as every other phase in this project - not measured
via mocks alone:
1. Unit-level: parallel path's combined output must exactly match
   `fwht_pauli_terms`'s single-process output on every existing
   fixture (`ALL_FIXTURES`), same pattern as Phase 12's own
   `test_auto_decompose_*_matches_fwht_pauli_terms` tests.
2. Real-world: re-run the N=100/150/200 measurements already on file
   (`n100_n150_autotuning_remeasurement_findings.md`,
   `chunk_size_floor_scale_dependence_findings.md`) with the parallel
   path enabled, confirming identical term counts/values and measuring
   the real wall-clock speedup (not assumed from core count - Amdahl's
   law and cache contention both mean the real speedup will be well
   under `n_workers`x).
3. `perf stat` re-measurement of cache-miss ratio under concurrent
   worker load specifically, per the memory-budget/chunk-locality
   tension flagged above - this is new territory, not a re-check of
   an existing single-process number.

## Sequencing recommendation

1. **13a first** (multi-core, single-node) - self-contained, no
   cluster dependency, directly testable on this dev machine,
   addresses the more common real-world case (a single workstation or
   a single HPC node/job-step) before the less common one
   (multi-node).
2. **13b second** (multi-node), only after 13a's real speedup and
   cache-contention findings are in hand - multi-node adds an entire
   new failure/coordination surface (network partition, node failure
   mid-job, result-collection across nodes) on top of whatever 13a's
   own findings suggest is or isn't worth pursuing at the single-node
   level first. Building 13b before 13a's real numbers exist would
   risk over-engineering a distributed-systems layer on top of an
   unverified core assumption (that per-core parallel throughput
   scales usefully here at all, given the cache-contention question
   above).

## What this scoping does NOT do

- Does not implement anything - `fwht.py`/`autotune.py` are unchanged.
- Does not measure real multi-core speedup, cache contention under
  concurrent workers, or memory-budget-division correctness - all
  flagged above as needing real measurement once 13a exists, not
  assumed.
- Does not finalize the API shape (leaning toward a new top-level
  function, per Phase 12's own precedent, but not decided) or the
  worker-count auto-detection's exact fallback chain for non-Linux
  platforms (macOS/BSD equivalents of `sched_getaffinity` not yet
  researched - `os.cpu_count()` may be the only portable option there,
  a real gap parallel to `cache_probe`'s own existing Linux-only cpu-
  pinning limitation).
- Does not scope 13b (multi-node) in implementation detail - only
  names it as a distinct, deliberately-deferred second sub-phase.
