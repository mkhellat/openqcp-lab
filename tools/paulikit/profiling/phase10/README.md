# Phase 10: streaming output and its cache-locality follow-up

Map for this directory - read this first, then follow the findings
below in order. See `PLAN.md` Phase 10 for the design/implementation
narrative; this directory holds the supporting measurements.

## Findings, in order

1. [`tbb_labeling_n150_findings.md`](tbb_labeling_n150_findings.md) -
   re-measures the existing oneTBB-parallel label kernel
   (`pauli_label_batch_parallel`) at N=150-representative scale (a
   synthetic 40M-pair benchmark), since the earlier
   `../cache_locality/tbb_evaluation_findings.md` only tested N≤100,
   before label generation was ever the real bottleneck. Finds a real
   1.1-1.4x wall-clock win at the cost of a modest cache-locality
   regression - measured **in isolation**. Scripts:
   `tbb_label_40m_serial.py` / `tbb_label_40m_parallel.py`, raw `perf
   stat` output in the two `tbb_label_40m_perf_*.txt` files.

2. [`phase10_streaming_findings.md`](phase10_streaming_findings.md) -
   the core Phase 10 result: `fwht_pauli_terms_iter` (the streaming
   generator) completes N=150's full ~91.6M-term decomposition under
   both a 4 GB and a 2 GB `ulimit -v` cap, down from Phase 9's
   non-streaming accumulator needing 10+ GB and still failing past
   13.5 GB. Script: `n150_streaming_test.py`.

3. [`full_pipeline_n150_findings.md`](full_pipeline_n150_findings.md) -
   **corrects finding 1's practical takeaway.** Re-measures TBB
   labeling embedded in the real streaming pipeline (not isolated) and
   finds it delivers **no measurable wall-clock or cache-locality
   benefit** at N=150's real proportions - because a per-stage
   wall-clock breakdown reveals dict construction, not labeling,
   dominates (~60% of total time, vs. ~7% for labeling). This is the
   new highest-leverage optimization target for this pipeline, not yet
   scoped as a phase. Scripts: `n150_stage_breakdown_driver.py` (the
   per-stage breakdown; requires a manually-instrumented scratch copy
   of `fwht.py`, see the script's own docstring for the exact
   reproduction steps) and `n150_pipeline_perf_serial.py` /
   `n150_pipeline_perf_parallel.py` (whole-pipeline `perf stat`), raw
   output in the two `n150_pipeline_perf_*.txt` files.

4. [`n_scaling_streaming_findings.md`](n_scaling_streaming_findings.md) -
   a steady-state N=25/50/100/150 timing table using the real
   streaming path uniformly at every N (not dense-then-sparse like the
   historical `../cache_locality/steady_state_decompose.py`). The
   first successful N=150 data point in any such table in this
   project's history - every earlier attempt OOM-killed or failed to
   complete before Phase 10. Mean time: 0.069s (N=25) -> 1.329s (N=50)
   -> 22.064s (N=100) -> 101.310s (N=150, single rep). Script:
   `steady_state_streaming_sweep.py`.

## Takeaway if you only read one thing

`--parallel-labels` is not a meaningful lever for this pipeline's
performance at N=150 scale, despite winning clearly in isolation
(finding 1) - it is swamped by dict construction, which finding 3
identifies as the real bottleneck and is not yet addressed by any
phase. Finding 2's N=150-solved result does not depend on
`--parallel-labels` either way.
