# The batched-refill drain loop: falsified by instrumentation

2026-09-09. Revisits `phase13/batched_resubmit_prototype.py`, a fix
that was designed and written but never measured. **It does not help,
and the reasoning behind it no longer applies.** Not implemented.

## The hypothesis, from Phase 13

`parallel_decompose`'s drain loop calls `_submit_next()` once per
completed future, *inside* the `for future in done:` loop - after that
result is unpickled, and (in the array API) immediately before a
`yield` that suspends the generator until the consumer returns. Since
`wait(..., FIRST_COMPLETED)` can return several futures at once, the
argument was that the pool is fed one replacement at a time,
interleaved with slow drain-side work, and therefore idles.

Phase 13 recorded this as a defect found "by direct code review (not
measurement)". That distinction turned out to be the whole story.

## Why it looked more urgent after Phase 14

The Phase 13 reasoning assumed drain-side work is expensive - at the
time, labeling and dict construction measured ~82% of runtime
(`phase13/perf_deep_dive_correlation.md`). Phase 14's C kernel then cut
sequential CPU 2.4x, so pool machinery became the largest remaining
cost (0.843 ms/chunk against 1.216 ms/chunk of useful work). A
starving pool would have been the obvious explanation.

## Implemented, measured, reverted

The change: on each `wait()` return, refill one slot per completed
future *before* any drain-side work for the batch. The in-flight bound
is untouched (still one replacement per completion), so the
`O(n_workers)` backlog bound protecting peak RSS - the fix from
`phase13/n150_worker_count_sweep_findings.md`, where unbounded
submission cost 25 GiB - still holds.

Interleaved A/B, separate processes, 5 reps, cooldown to 55 C, N=150,
chunk_size=2, both variants producing 91,652,096 terms:

| variant | mean | sd | peak RSS |
|---|---|---|---|
| old (one-at-a-time) | 4.603s | 0.234 | 64 MiB |
| new (batched refill) | 4.823s | 0.291 | 65 MiB |

new/old = 1.048 - 4.8% in the **wrong** direction. Welch t = -1.315,
df = 7.6: not significant. The honest reading is no detectable effect,
and certainly not the improvement predicted.

## Why: the pool was never starving

Direct instrumentation of the real loop, counting what `wait()`
actually returns and how full the pool stays:

```
n_workers=4, max_in_flight=8
futures returned per wait():
   len(done)=1:   5592 times (100.0%)
   len(done)=3:      1 times (0.0%)
in-flight remaining after wait():
   7:   5585   (of 5593 iterations)
```

`len(done)` is **1 in 100.0% of cases**, so the "batch" refill is
byte-for-byte the same operation as the old code in every iteration
but one. And in-flight sits at 7 of a maximum 8 for 5585 of 5593
iterations - the pool is saturated the entire run, never idle.

The premise was false. Workers are the bottleneck, so completions
arrive one at a time and a replacement is submitted immediately; there
is no backlog of simultaneous completions to batch, and no idle slot
to fill sooner.

## The lesson, which is the reusable part

The Phase 13 prototype was sound reasoning about the code *as it
stood*, when drain-side work was ~82% of runtime. It was written,
correctly flagged as unmeasured, and then never re-validated after
`parallel_decompose_arrays` removed that work. A fix designed against
one bottleneck was inherited as a to-do item after the bottleneck it
targeted had already been eliminated.

Two rules this supports:

1. **A code-review defect is a hypothesis, not a finding.** Phase 13
   labelled it exactly that way. The label should have been carried
   forward with it.
2. **Instrument before optimizing a queue.** Counting `len(done)` and
   in-flight occupancy took one run and settled the question outright,
   where the A/B alone would only have said "no effect" without
   explaining why.

## What this leaves

The ~0.84 ms/chunk of pool machinery is real but is **not** dispatch
latency or pool starvation. It is the per-task cost of the
ProcessPoolExecutor round trip itself: pickling the result arrays,
the pipe write and read, unpickling in the parent. Reducing it means
moving fewer bytes per chunk or making fewer round trips - and the
`chunk_size` sweep in `parallel_overhead_attribution.md` already shows
fewer, larger chunks makes things worse, because per-chunk cost scales
with `chunk_size * dim` data volume rather than with task count.

Shared memory remains the only untested lever here (probed at 1.96x on
raw transfer), and on these numbers it addresses at most a third of
0.84 ms/chunk against 1.216 ms/chunk of useful work - worth measuring,
but not before something with more headroom.
