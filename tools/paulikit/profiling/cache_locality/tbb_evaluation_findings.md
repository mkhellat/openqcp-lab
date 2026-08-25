# TBB-parallel label kernel: no measurable cache-locality or performance effect

Recorded 2026-08-25, per explicit instruction: before starting Phase
6's sparse-representation work, test the TBB-parallel label-generation
kernel (`pauli_label_batch_parallel`) properly with the same
cache-locality methodology used throughout this investigation, rather
than rely on Phase 3a's wall-clock-only "barely helps" finding (see
[`tbb_not_actually_used_finding.md`](tbb_not_actually_used_finding.md))
for a question - cache locality - it was never measuring.

## Method

`steady_state_decompose_tbb.py` monkeypatches
`paulikit.algorithms.fwht._pauli_label_batch` at runtime, in-process
only, to call `_native.pauli_label_batch_parallel` instead of the
serial `_native.pauli_label_batch` that `fwht_pauli_terms` actually
calls in production. Correctness was verified before measuring: at
N=50, the TBB path produces identical term counts, identical label
sets, and identical coefficient values (to 1e-9) versus the serial
path.

Compared against `steady_state_decompose.py` (serial, the real
production path) under the same `perf stat` event set, same
warm-up/steady-state timing protocol, `OPENBLAS_NUM_THREADS=1` set for
both (per
[`stall_floor_mystery_solved.md`](stall_floor_mystery_solved.md)'s
noise-control finding), at N=25/50/100, 3 runs each. N=150 excluded -
[`n150_oom_finding.md`](n150_oom_finding.md) already established the
unmodified code OOMs at that size regardless of which label kernel
runs (the OOM happens upstream, in `fwht_pauli_coefficients`'s
dense-array allocation, before label generation is ever reached).
Script: `run_tbb_comparison.sh`. Raw data:
`tbb_comparison_20260825T155051Z.txt`.

## Results (means over 3 runs)

| N   | kernel | mean time | cycles   | cache-miss % | LLC-miss % | stall-total % | stall-mem % |
|-----|--------|-----------|----------|---------------|------------|----------------|-------------|
| 25  | serial | 0.0858s   | 2.134e9  | 18.7%         | 14.3%      | 23.6%          | 16.1%       |
| 25  | TBB    | 0.0869s   | 2.332e9  | 18.8%         | 13.5%      | 27.6%          | 15.6%       |
| 50  | serial | 1.9655s   | 3.285e10 | 57.5%         | 44.3%      | 27.8%          | 22.5%       |
| 50  | TBB    | 1.9019s   | 3.356e10 | 57.4%         | 44.5%      | 27.8%          | 22.5%       |
| 100 | serial | 36.4836s  | 6.225e11 | 57.6%         | 56.5%      | 31.4%          | 24.6%       |
| 100 | TBB    | 36.1464s  | 6.356e11 | 58.2%         | 57.4%      | 31.8%          | 24.9%       |

## Interpretation

No measurable effect, on anything, at any N. Every metric - wall
time, cycle count, cache-miss ratio, LLC-miss ratio, stall
percentages - differs between serial and TBB by at most a few percent
relative, well within the run-to-run variance already seen elsewhere
in this investigation (compare e.g. the individual-rep spread within
a single `steady_state_decompose.py` invocation in any of the earlier
findings). There is no consistent direction of effect either: TBB is
marginally faster at N=50/100 and marginally slower at N=25, which is
noise, not a trend.

This is not surprising given `perf_record_n50_findings.md`'s original
localization: the cache misses live in NumPy's dense-array
construction and re-scan (`fwht_pauli_coefficients`,
`fwht_pauli_terms`), not in the label-generation loop that
`pauli_label_batch`/`pauli_label_batch_parallel` implement. TBB
parallelizes label-string construction; it has no path through the
code that touches the dense coefficients array at all. Parallelizing
a part of the pipeline that isn't where the cache misses happen
cannot fix - or worsen - the cache-miss behavior measured throughout
this investigation.

This closes the empirical half of the question the user raised
("are u sure loop unrolling, branch predictions, and TBB have nothing
to do with cache locality?"): for TBB specifically, now measured, not
just inferred from source reading. Confirms
[`tbb_not_actually_used_finding.md`](tbb_not_actually_used_finding.md)'s
correction was safe to rely on, and that
[`compiler_flags_findings.md`](compiler_flags_findings.md)'s null
result for `-march=native` etc. is not masking a TBB effect underneath
it.

## Does this decide Phase 6?

No, and it doesn't need to for the dense-array fix itself - that
root cause is unrelated to TBB either way, per
`perf_record_n50_findings.md`. What this does resolve is the *open
question* `tbb_not_actually_used_finding.md` raised: whether TBB might
become relevant if Phase 6 restructures the pipeline. This finding
doesn't fully close that (a restructured pipeline could route work
through label generation differently), but it does establish the
correct baseline to compare any future TBB-in-a-redesigned-pipeline
experiment against - a flat, no-effect null result, not the "barely
helps" wall-clock-only heuristic from Phase 3a.

**Recommendation**: proceed to Phase 6's sparse-representation design
work. TBB is not a lever for the cache-locality problem this
investigation identified, at the current pipeline structure. Revisit
only if Phase 6's own prototyping surfaces a new hot loop that TBB
could plausibly parallelize - not as a default assumption.
