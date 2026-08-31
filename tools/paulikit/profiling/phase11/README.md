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

## Post-implementation re-measurement

[`n150_post_implementation_findings.md`](n150_post_implementation_findings.md) -
re-ran the same real-N=150 per-stage breakdown Phase 11 was scoped
from, against the shipped fix. Dict construction dropped 64.64s ->
27.64s (2.34x, short of the synthetic 2.7-3.2x estimate), cutting
total pipeline time 34.2% (107.48s -> 70.77s). Because dict_build
shrank, every other stage's *relative* share rose - most notably the
WHT butterfly, from 21.1% to **31.9%** of total time, now close to
dict_build (39.1%) rather than dwarfed by it. This was the direct
trigger for a live GPU-worth-it question the user asked mid-session:
the answer moved from "clearly not" to "genuinely marginal" -
recommended next step is scoping dict_build's own remaining internals
further (lower-risk, higher-certainty) before committing to any GPU
work.

## Is there a further cheap win left in dict_build?

[`phase11b_remaining_dict_build_scoping.md`](phase11b_remaining_dict_build_scoping.md) -
answers the recommendation above: **no.** Breaking `_build_real_terms`
down further shows the vectorized Hermiticity check Phase 11 added is
now under 2% of the function's own cost (a complete reversal from
before Phase 11); **`dict(zip(labels, real_list))` itself is now
~90%** of what remains, and roughly 56% of *that* is specifically
Python string-hashing + dict-entry insertion - CPython-object-level
work with no NumPy-vectorizable equivalent, unlike the check Phase 11
fixed. Confirmed `dict(zip(...))` is already at or near CPython's own
practical floor (a dict comprehension is statistically
indistinguishable; an explicit insert loop is ~1.9x worse). Any
further cut would require a breaking API change (a different return
container, e.g.) or a C/Cython dict-construction kernel of unclear
realistic upside - both meaningfully bigger undertakings than a
Phase-11-shaped micro-fix, not scoped here. This closes the "check
dict_build first" step and leaves a real GPU-port cost/benefit
estimate as the actual next decision on the WHT-butterfly question.

## Takeaway if you only read one thing

Phase 11 was the highest-leverage remaining optimization target in
this whole profiling trail: a scoped fix with a measured 2.7-3.2x
isolated-benchmark upside, now applied to the real pipeline (2.34x
real-world, 34.2% total pipeline speedup). It is a pure performance
improvement, not a correctness fix — Phase 10's streaming result
(N=150 completing successfully) never depended on it either way. Its
side effect — the WHT butterfly's relative share nearly doubling — is
what makes GPU acceleration worth a further look now; a follow-up
check confirmed dict_build itself has no comparably cheap further fix
available, so that decision now rests entirely on a real GPU-port
cost/benefit estimate, not yet done.
