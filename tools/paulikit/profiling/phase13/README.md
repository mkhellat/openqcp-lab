# Phase 13: multi-core / multi-node chunk parallelism

Scoped 2026-09-02, not yet implemented. See `../../PLAN.md`'s Phase 13
section for the short version; this directory holds the supporting
design work.

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

## Status

Design/scoping only - `fwht.py`/`autotune.py` are unchanged.
Implementation, correctness verification against `ALL_FIXTURES`, and
real wall-clock/cache-contention measurement (N=100/150/200, `perf
stat`) all remain to be done for 13a; 13b remains unscoped in
implementation detail pending 13a's real findings.
