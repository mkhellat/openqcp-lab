# Phase 13: multi-core / multi-node chunk parallelism

Scoped 2026-09-02. See `../../PLAN.md`'s Phase 13 section for the
short version; this directory holds the supporting design,
measurement, and root-cause work behind it - the largest single
profiling effort in this project to date, spanning `perf stat`/`perf
record`, `strace`, `py-spy`, thermal-controlled Welch's t-tests, and a
full DAG re-derivation.

## Scoping

[`scoping.md`](scoping.md) - the full architectural design pass:
why this is real and available (each chunk is an independent
sub-problem, confirmed via `_iter_chunked_coefficients`'s own
docstring), the 13a (multi-core, single-node)/13b (multi-node) split
and why 13a comes first, a process-pool sketch and why processes over
threads, the real unresolved tension with Phase 12's memory-budget and
chunk_size-floor formulas (both measured on a single lone process;
multi-core execution means shared-cache contention and a memory budget
that must be divided across workers, not reused unchanged), three API
shape options with a lean toward a new top-level function, the
checkpoint/resume interaction, and the verification plan for once
something is actually built.

## What was built (13a)

`fwht.parallel_decompose` (implemented 2026-09-02): a
`ProcessPoolExecutor`-based worker pool over
`_iter_chunked_coefficients`'s per-chunk body, with per-worker memory
budgeting, bounded in-flight submission, and CPU pinning, all found
necessary by direct measurement and documented in this directory's
`*_findings.md` files. Real speedup on this 8-core dev machine
plateaued near 1.0-1.28x regardless of chunk_size, worker count, or
pinning strategy - flat rather than rising-then-saturating, which
pointed away from per-task dispatch overhead or cache/memory-bandwidth
contention as the dominant limiter and toward something structural.

## Root cause: a serial bottleneck, not a contention problem

`dag_extraction_and_parallelism.md` and the deep, non-truncated
`perf.data` correlation pass in `perf_deep_dive_correlation.md` traced
the flat speedup to its real source: at N=150, building the ~91.6
million Pauli label strings and dict entries that
`parallel_decompose` returns happens entirely in the single parent
process's drain loop, and is measured at **~82% of the function's
total runtime** (serial fraction 0.824). Amdahl's law caps the
achievable multi-core speedup at **~1.21x with infinitely many
cores** for a workload with that serial fraction - which matches every
number measured above (best-ever 1.284x; 8 workers measured 1.100x).
`dag_extraction_with_labeling_moved.md` explored moving the labeling
step into the worker processes instead (parallelism 3.0 -> 41.9 in the
re-derived DAG); implemented as commit `9c5f1c6` and reverted the same
day as `9224b41` once real measurement showed the IPC cost of shipping
~91.6M label strings back across the process boundary exceeded the
compute it saved.

## The fix: let callers skip labeling instead of relocating it

`drain_gil_backpressure_sweep.py` /
`drain_gil_backpressure_results.jsonl` isolated the drain loop's cost
in a controlled, single-variable experiment: full label+dict work
measured 0.865x, an arrays-only drain loop measured 2.191x, and a
control doing no per-chunk work at all was statistically
indistinguishable from the arrays-only result. That finding motivated
`fwht.parallel_decompose_arrays` and `fwht.terms_from_arrays`
(2026-09-07, see `PLAN.md`'s Phase 13 section and `docs/tutorial.md`):
the array-yielding function keeps the
label/dict-construction cost out of the drain loop entirely, and
`terms_from_arrays` lets a caller render labels for only the terms
they actually need. The 2.191x figure is the controlled drain-loop
experiment's result, not yet an end-to-end measurement of the shipped
`parallel_decompose_arrays` with checkpointing enabled - that sweep is
a follow-up task and has not run yet.

## Status

13a (`parallel_decompose`) and the array-yielding follow-up
(`parallel_decompose_arrays`, `terms_from_arrays`) are both
implemented, tested against `ALL_FIXTURES`, and documented. 13b
(multi-node) remains unscoped in implementation detail, deliberately
deferred as originally planned. An end-to-end, thermal-controlled
speedup measurement of `parallel_decompose_arrays` itself (as opposed
to the controlled drain-loop experiment above) is the next open item.
