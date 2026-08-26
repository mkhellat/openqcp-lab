# Phase 6 sparse output: negligible cache-locality effect at N≤100

Recorded 2026-08-26. Phase 6 (see `PLAN.md`) removed the O(dim²)
dense-array materialization from `fwht_pauli_coefficients`/
`fwht_pauli_terms`'s hot path, motivated by
[`n150_oom_finding.md`](n150_oom_finding.md) and
[`steady_state_scaling_findings.md`](steady_state_scaling_findings.md)'s
cache-miss-vs-N scaling data. This measurement was the actual A/B
comparison that hypothesis needed before claiming the fix helps at
N≤100, per PLAN.md Phase 6's plan (step 2/4) and the explicit
instruction not to assume "sparse is obviously better."

## Method

`steady_state_decompose_dense.py` replicates `fwht_pauli_terms`'s
entire pre-Phase-6 body (dense `fwht_pauli_coefficients(sparse=False)`
call, full-array `np.nonzero` re-scan, label generation, dict
construction), so it is a fair, complete comparison against
`steady_state_decompose.py` (the real, current `fwht_pauli_terms`,
which now always uses the sparse path internally). Both scripts share
the same warm-up/steady-state timing protocol and the same `perf stat`
event set used throughout this investigation, with
`OPENBLAS_NUM_THREADS=1` set for both (per
[`stall_floor_mystery_solved.md`](stall_floor_mystery_solved.md)).
Correctness was verified before measuring: term counts and coefficient
values match exactly between the two paths at every N tested (see
`fwht.py`'s own bit-for-bit verification, and the label/coefficient
counts printed in the raw output below).

3 runs each at N=25/50/100. N=150 excluded from this comparison and
measured separately (dense leg reliably OOMs even under the two
memory-footprint fixes made later the same day - see PLAN.md's Phase 6
"Update 2026-08-26" for that investigation and its outcome). Script:
`run_phase6_comparison.sh`. Raw data:
`phase6_comparison_25_50_100_20260826T042348Z.txt`.

## Results (means over 3 runs)

| N   | path   | mean time | cache-miss % | stall-mem % |
|-----|--------|-----------|---------------|-------------|
| 25  | dense  | 0.0619s   | 19.43%        | 18.99%      |
| 25  | sparse | 0.0619s   | 19.07%        | 20.06%      |
| 50  | dense  | 1.3651s   | 57.39%        | 28.06%      |
| 50  | sparse | 1.3446s   | 57.26%        | 28.32%      |
| 100 | dense  | 25.5166s  | 59.83%        | 30.16%      |
| 100 | sparse | 24.3785s  | 59.68%        | 31.09%      |

Wall-clock speedup (sparse vs. dense): 1.000x at N=25, 1.015x at
N=50, 1.047x at N=100 - a 0-4.5% difference, the same order of
magnitude as the run-to-run variance already documented elsewhere in
this investigation (see e.g. `tbb_evaluation_findings.md`'s
discussion of noise at this measurement scale).

## Interpretation

**The original hypothesis - that densification is the dominant driver
of cache-miss ratio and wall-clock time at these N - does not hold at
N≤100.** Cache-miss ratio and memory-stall percentage are
statistically indistinguishable between the dense and sparse paths at
every N tested; sparse is not consistently better on either metric
(sparse's stall-mem % is actually marginally *higher* than dense's at
all three N, though within noise). Wall-clock time favors sparse by an
amount that grows with N (0% → 1.5% → 4.5%) but stays small in
absolute terms through N=100.

This does not contradict `steady_state_scaling_findings.md`'s earlier
finding that cache-miss ratio scales with how far the dense array
exceeds L3 cache size (23% at N=25 up to 58.5% at N=100) - that
scaling is real and reproducible, and this table's numbers land close
to those same values. What it corrects is the *causal* story: the
dense array's size does correlate with cache-miss ratio as N grows,
but the sparse path pays nearly the same cache-miss cost, because the
Pauli-term dictionary construction downstream of the WHT step (label
generation, Python dict building) is identical in both paths and,
per `perf_record_n50_findings.md`'s cProfile breakdown, dominates
wall-clock time at these N - roughly 70% of total time at N=100 is
Python-level label/dict construction, not array-scale-dependent NumPy
work. Densifying an already-small active-row set to `(dim, dim)`
apparently does not push the *working set* meaningfully further outside
cache than the sparse active-row array already is, at these problem
sizes - both are large enough relative to L3 that neither fits, and
the re-scan/gather patterns end up similarly cache-unfriendly either
way.

**Phase 6's real, measured benefit through N=100 is memory footprint
and crash-avoidance, not cache-locality or wall-clock time.** The
dense path's O(dim²) allocation is the direct cause of
`n150_oom_finding.md`'s OOM - that risk is real and is what Phase 6
fixes - but "removes a robustness/crash risk at large N" and "improves
cache locality at N≤100" are different claims, and only the first is
supported by this data. Framing Phase 6 as a cache-locality
optimization for N≤100 would be overselling it; framing it as
"prevents the dense-array OOM that would otherwise block N=150
entirely" is what the evidence actually supports.

## Does this change Phase 6's status?

No - `sparse=True` is correct, tested, and worth keeping regardless of
this finding, since the crash-avoidance case stands on its own. What
this finding does change is the *narrative*: PLAN.md and README.md
should not claim a cache-locality or wall-clock win at N≤100 from
Phase 6 alone (see the corrected framing added to both). The
cache-locality story for N≤100 remains what
`perf_record_n50_findings.md` already found: dominated by Python-level
label/dict construction, not by dense-vs-sparse array representation.

**Recommendation**: no further action needed on the N≤100 cache-locality
question - it's answered, and the answer is "look at label/dict
construction, not array density, if N≤100 wall-clock time needs to
improve further." N=150 remains open and is tracked separately (see
PLAN.md's Phase 6 "Update 2026-08-26" and its two follow-up items).
