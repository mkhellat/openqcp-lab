# Correction: three real overreach claims in the IPC-accounting conclusion, and what's actually still open

Recorded 2026-09-03. Direct correction after the user challenged
`instructions_ipc_accounting_findings.md`'s conclusion with three
separate, valid points: "are you 100% sure?!!! ... Is this a property/
limitation of our code or parallel python codes in general?!! ...
There is nothing specific to our code here!!!!" Each point is real and
is addressed here rather than left standing.

## Replication of the pinned_5 comparison first (per direct instruction)

A fresh, independent 5-rep run of `pinned_5_4cores` vs
`pinned_5_3cores` (thermally controlled, same protocol):

- wall-clock: 28.75s vs 23.65s, diff=-5.10s, p=0.0378 - **significant**,
  same direction as both prior runs.
- instructions: 122.94B vs 122.96B, diff=+14.5M (0.01%), p=0.929 -
  **not significant**, even more clearly than the first run.
- IPC: 1.807 vs 1.875, p=0.000449 - **significant**, same direction.

This specific numerical finding (instructions unchanged, IPC lower
when spread across more cores, at `n_workers=5`) replicates cleanly a
second time. The three challenges below are about the CONCLUSIONS
drawn from this kind of data, not about whether the measurements
themselves are real - they are real and reproducing.

## Overreach 1: "best speedup is 2-core hyperthreading" was never established

**What was actually tested**: for each of `n_workers` in {2,3,4,5},
ONE pair of configurations - the "most spread" arrangement (as many
distinct physical cores as possible) vs. ONE "more packed"
arrangement (exactly one fewer physical core, via one additional
hyperthread pair). In every pair, "more packed" won.

**What this does NOT establish**: that packing further (e.g. all 4
workers on ONE physical core's 2 hyperthreads plus somehow more, or
comparing n_workers=2-on-1-core against n_workers=4-on-2-cores against
n_workers=8-on-4-cores as a genuine SEARCH for a minimum) finds an
actual optimum at "2 cores" or anywhere else. No such search was ever
run. The correct, narrower claim supported by the data is: "within
each tested pair, packing onto one fewer physical core was faster" -
not "2-core hyperthreading is the best achievable configuration." This
was stated too strongly in the prior document's own framing ("the best
speedup achieved using multiple cores" was the USER'S question, and it
was not actually answered with data - it should have been flagged as
unanswered, not implicitly agreed with).

## Overreach 2: "there is nothing we can do about it" was never tested

**What was actually shown**: that contention (worse cache-miss/LLC-
miss ratios, lower IPC) exists and correlates with more active
physical cores, using the EXISTING `chunk_size`, the EXISTING gather/
scatter access pattern, and the EXISTING shared setup-array design -
none of which were varied in this investigation.

**What this does NOT establish**: that the contention is unavoidable.
Real, untested levers exist that could plausibly reduce it without
reducing core count:
- A smaller `chunk_size` under multi-core contention specifically
  (Phase 12's `recommended_chunk_size` was tuned for a SINGLE lone
  process's cache - PLAN.md Phase 13's own scoping doc flagged this
  exact gap and it was never revisited).
- Restructuring the per-chunk gather to be more sequential/cache-
  friendly (the current gather is a scatter into a zeroed dense array
  at arbitrary column positions - `sorted_q_nz[lo:hi]` - not a
  sequential access pattern).
- NUMA/memory-controller-aware data placement (not applicable on this
  single-socket machine, but would matter on a real multi-socket HPC
  node - untested either way).
- Reducing the shared setup arrays' total footprint (identified as
  the "real source of the small residual L1-missing slice" back in
  `n_workers_placement_and_cache_findings.md`'s own "how to apply"
  section - flagged then, never pursued).

None of these were tried. The correct claim is "contention exists and
correlates with core count, GIVEN the current implementation" - not
"contention is an inherent, unfixable property of this problem."

## Overreach 3: the whole analysis is hardware-generic, not code-specific

This is the sharpest and most important of the three challenges. Every
measurement in this entire IPC/cache-miss/LLC-miss investigation was
taken with `perf stat` at the WHOLE-PROCESS level - it answers "does
this process, running SOME code, experience more stalling when more
cores are active" but says NOTHING about:
- WHERE in `paulikit`'s own code the stalling actually happens (the
  WHT butterfly stage's reshape/slice/add loop? the gather/scatter
  step? `_pauli_label_batch`'s label construction? `_build_real_terms`'s
  dict construction, already identified in Phase 11 as ~60% of
  single-process time?).
- Whether `paulikit`'s SPECIFIC access patterns (a scattered gather
  into a dense zeroed buffer, per-chunk generator overhead, the
  shared setup arrays re-read by every chunk) make this WORSE than a
  more cache-friendly implementation would experience under the same
  hardware contention.
- Whether this is a property of "any CPU-bound multi-process Python
  workload on this specific hardware" (plausible, given the mechanism
  - shared L3 contention is a hardware property, not a Python-specific
  one) or whether `paulikit`'s own implementation choices are making
  it measurably worse than necessary (equally plausible, completely
  untested).

**The honest, corrected claim**: this investigation has established a
REAL, REPRODUCING hardware mechanism (shared L3/memory-bandwidth
contention causing measurable IPC degradation as more physical cores
become active) that CURRENTLY manifests in `paulikit`'s
`parallel_decompose` under its EXISTING implementation choices. It has
NOT established that this mechanism is (a) an optimum/ceiling, (b)
unavoidable, or (c) independent of `paulikit`'s own code - all three
of those are open questions this investigation's data cannot answer,
and should not have been implied to answer.

## What would actually be needed to answer the user's real questions

1. **Optimum search**: sweep `n_workers` from 1 to 8 at a FIXED core-
   packing strategy (or vice versa) with proper statistics, not just
   paired comparisons, to find where wall-clock is actually minimized -
   not yet done.
2. **Avoidability**: re-run the SAME core-packing comparison with a
   deliberately different `chunk_size` (smaller, targeting a tighter
   cache-locality budget under multi-core contention specifically) to
   see if the gap between "more cores" and "fewer cores" shrinks -
   would directly test whether the effect is fixable via tuning, not
   just observable.
3. **Code-specificity**: profile WITHIN one real run (e.g. `perf
   record`/`perf annotate`, or Python-level `cProfile`/`py-spy`
   sampling of a single worker process under contention vs. without)
   to identify which specific functions/lines account for the
   stalling - would distinguish "generic hardware limit" from
   "paulikit-specific inefficiency" directly, rather than by
   inference.

None of these three have been done. They are the correct next steps
if the user wants an actual answer to "is this fixable and is it
specific to our code," rather than continuing to extrapolate from the
process-level IPC/cache-miss data already in hand.
