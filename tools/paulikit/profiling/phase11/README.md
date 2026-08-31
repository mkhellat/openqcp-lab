# Phase 11: `dict_build` optimization scoping

Scoped 2026-08-27, implemented 2026-08-31. See `../../PLAN.md` Phase
11 for the full design/implementation narrative (including a formula
bug this scoping's own microbenchmark had, caught and fixed before
production code was touched); this directory holds the supporting
measurement.

This phase was scoped by a finding inside Phase 10's own investigation
(`../phase10/full_pipeline_n150_findings.md`): a per-stage wall-clock
breakdown of the real N=150 streaming pipeline found that `dict_build`
— the per-chunk Python loop converting `(label, coefficient)` pairs
into a `dict` — dominates at ~60% of total pipeline time, well above
labeling (~7%) or the WHT butterfly (~21%).

## Finding

[`phase11_dict_build_scoping_findings.md`](phase11_dict_build_scoping_findings.md) -
breaks `dict_build` down further via a standalone microbenchmark
(`dict_build_microbenchmark.py`): the per-term Hermiticity check
(`abs(c.imag) > max(atol, 1e-6 * abs(c))`, evaluated one Python object
at a time) is the single largest sub-cost — a fully vectorizable NumPy
operation currently paid for one term at a time. Vectorizing it (NumPy
`abs`/`maximum` instead of a per-term Python loop) produced a 3.76x
speedup at 1M terms and 2.58x at 10M terms in isolation. A secondary
win: `dict(zip(...))`'s C-level constructor beats an explicit per-item
insert loop by a further ~30-40%.

**Implemented 2026-08-31** — see `../../PLAN.md` Phase 11 for how the
open design questions were resolved: a shared `_build_real_terms`
helper now backs both `fwht_pauli_terms` and `fwht_pauli_terms_iter`
(including the non-streaming/dense path), preserving per-term
error-message specificity via a rare-path `np.nonzero` re-scan on
violation only.

## Takeaway if you only read one thing

Phase 11 was the highest-leverage remaining optimization target in
this whole profiling trail: a scoped fix with a measured 2.7-3.2x
isolated-benchmark upside, now applied to the real pipeline. It is a
pure performance improvement, not a correctness fix — Phase 10's
streaming result (N=150 completing successfully) never depended on it
either way.
