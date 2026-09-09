# Parallelism no longer pays for this workload

2026-09-09, after the Phase 14 kernels landed. An uncomfortable result
that should not be buried: **`parallel_decompose_arrays` is now slower
in wall clock than the sequential path, and costs several times the
CPU.**

## The measurement

Same process, same operator, `chunk_size=2`, `getrusage` deltas around
each call (self + children):

| N | sequential wall | sequential CPU | parallel wall | parallel CPU | wall speedup | CPU cost |
|---|---|---|---|---|---|---|
| 100 | 0.606s | 0.604 | 0.642s | 2.028 | **0.94x** | 3.36x |
| 150 | 2.525s | 2.516 | 3.136s | 13.882 | **0.81x** | 5.52x |

Parallelism makes the wall clock *worse* and burns 3-5x the CPU doing
it.

## Why, and why it is a consequence of success

Nothing regressed in the pool. The compute it wraps got cheap:

| | per chunk |
|---|---|
| useful compute, after both C kernels | ~0.5 ms |
| ProcessPoolExecutor round trip | ~1.33 ms |

At the start of this phase the ratio was the other way round (1.216 ms
of compute against 0.843 ms of machinery), and parallelism bought a
real 2.95x. The kernels cut per-chunk compute roughly 2.5x while the
round trip - pickling the result arrays, the pipe write and read,
unpickling in the parent - is untouched by anything done to the
arithmetic. Amdahl's law then does the rest.

This is the same arithmetic recorded in
`parallel_overhead_attribution.md`, arriving at its conclusion: the
overhead was never fixable by tuning the pool, and the only thing that
moved it (the cache-tiled kernel, cutting 1-to-4-worker CPU growth from
+77% to +30%) moved it by reducing contention, not machinery.

## What this does NOT mean

- **Not** that the parallel path is broken. It is correct, its output
  is bit-identical to sequential, and its streaming peak RSS is still
  ~67 MiB.
- **Not** that parallelism is useless in general. The crossover
  depends on per-chunk compute cost against a roughly fixed ~1.33 ms
  round trip. A workload with more expensive chunks - a much larger
  `dim`, or a future per-chunk step that is genuinely heavy - would
  cross back.
- **Not** an argument for removing it. Phase 13 built the
  chunk-independence property deliberately, and it is what makes
  multi-node possible later. Independence costs nothing when unused.

## What it does mean

The library's own advice has to change. Recommending
`parallel_decompose_arrays` for speed at these sizes is now wrong on
this machine's evidence, and `docs/tutorial.md` currently frames it as
the fast path.

The honest recommendation for N in the range measured here:

- Use `fwht_pauli_coefficients(..., sparse=True, chunk_size=...)` for
  throughput. It is faster in wall clock and 3-5x cheaper in CPU.
- Use `parallel_decompose_arrays` when the streaming, chunk-at-a-time
  contract is what you want - bounded memory regardless of result
  size, resumable checkpoints, or a consumer that processes chunks as
  they arrive - not because it is faster, because it is not.

## Open

Whether shared memory closes the gap enough to make parallelism pay
again is untested. It was probed at 1.96x on raw transfer, which
against a ~1.33 ms round trip and ~0.5 ms of compute would still leave
parallelism roughly break-even at best. That is worth measuring before
either fixing or documenting around it.

The alternative worth considering is a threaded drain. Both kernels
already release the GIL, so threads would run the compute genuinely
concurrently while avoiding pickling entirely - the chunk arrays stay
in one address space.

A feasibility probe (64 pre-built blocks at the N=150 shape, running
`wht_rows_inplace` then `coeffs_from_transformed` in a
`ThreadPoolExecutor`):

| threads | serial | threaded | speedup |
|---|---|---|---|
| 2 | 0.091s | 0.030s | 3.04x |
| 4 | 0.091s | 0.017s | 5.42x |

**Treat the magnitude with suspicion.** Speedup above the thread count
is not physical for compute-bound work; reusing the same 64 blocks
across reps almost certainly leaves them cache-warm in a way the real
pipeline's freshly-gathered chunks would not be. What the probe does
establish is the direction and the mechanism: the GIL is genuinely
released, threads do run the kernels concurrently, and none of the
~1.33 ms round trip is paid.

Scoping a threaded path properly - real gathers, real memory traffic,
thermal control, and the question of whether the sparse gather (still
Python/scipy, still holding the GIL) becomes the new serial fraction -
is the natural next piece of work, and is not started.
