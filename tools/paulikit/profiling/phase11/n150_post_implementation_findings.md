# Post-implementation re-measurement: Phase 11's real N=150 effect

Recorded 2026-08-31, immediately after Phase 11 (`_build_real_terms`
vectorization, see `phase11_dict_build_scoping_findings.md` and
`../../PLAN.md`) shipped, prompted by the user asking directly whether
GPU-accelerating the WHT butterfly stage was now worth it. Answering
that required real numbers, not the pre-implementation scoping's
synthetic 1M/10M-term estimate - this re-runs
`full_pipeline_n150_findings.md`'s exact methodology against the
post-Phase-11 code, on the real N=150 Hamiltonian. Machine: same as
that document (15.7 GiB RAM, 8 cores, 8 MiB shared L3).

## Method

Identical to `full_pipeline_n150_findings.md`: a scratch-copied,
timing-instrumented variant of `fwht.py` (not committed - same
"document exploration scripts" convention as before; only the driver
and this findings doc are checked in) wraps the same five stages
(gather, WHT, phase/threshold, label, dict_build) using
`n150_stage_breakdown_driver.py` (`profiling/phase10/`, unchanged),
plus a whole-pipeline `perf stat` run via the existing, unmodified
`n150_pipeline_perf_serial.py` (`profiling/phase10/`) against the real
installed (post-Phase-11) package. Both runs used the standard
`ulimit -v 4000000` + `free -h` polling safety harness; memory stayed
flat before/after both runs (no leak).

One methodology difference from the original: `perf_event_paranoid=2`
on this run (vs. whatever value was active in the original
2026-08-27 session) restricted `perf stat` to userspace-only counting
(`:u`-suffixed events) rather than full system-wide counting - flagged
here rather than silently compared as if identical. This does not
affect the wall-clock breakdown (unrelated tool), only the cache/stall
percentages below.

## Results: per-stage wall-clock breakdown

| stage | before (2026-08-27) | after (2026-08-31) |
|---|---|---|
| gather | 0.46s (0.4%) | 0.54s (0.8%) |
| WHT butterfly | 22.68s (21.1%) | 22.56s (31.9%) |
| phase + threshold | 9.91s (9.2%) | 9.07s (12.8%) |
| label generation | 7.36s (6.8%) | 8.26s (11.7%) |
| **dict construction** | **64.64s (60.1%)** | **27.64s (39.1%)** |
| unaccounted | 2.42s (2.3%) | 2.70s (3.8%) |
| **total** | **107.48s (100%)** | **70.77s (100%)** |

(91,652,096 terms, 44 chunks, both runs - term count matches exactly,
confirming correctness is unaffected.)

**Dict construction dropped from 64.64s to 27.64s - a 2.34x wall-clock
reduction**, short of the scoping microbenchmark's synthetic 2.7-3.2x
(`phase11_dict_build_scoping_findings.md`), plausibly because the real
coefficient array's `.tolist()`/memory-layout costs differ from the
synthetic benchmark's. **Total pipeline time dropped 34.2%** (107.48s
to 70.77s). As a direct consequence, every other stage's *relative*
share rose even though their *absolute* times were flat-to-slightly-
higher (noise-level, not a regression) - most notably **WHT butterfly
rose from 21.1% to 31.9%**, now the largest non-dict_build stage by a
wide margin, and dict_build/WHT moved from roughly 3:1 to roughly
1.2:1.

## Results: whole-pipeline `perf stat` (userspace-only counters)

| run | wall | cycles | cache-miss % | LLC-miss % | stall % | stall-mem % |
|---|---|---|---|---|---|---|
| before (2026-08-27, system-wide) | 96.96s | 253.15e9 | 45.7% | 45.6% | 33.9% | 30.8% |
| after (2026-08-31, userspace-only) | 71.18s | 159.55e9 | 46.5% | 44.3% | 42.8% | 39.2% |

Raw output: `n150_pipeline_perf_serial_post_phase11.txt` (this
directory).

Cache-miss ratio is essentially unchanged (45.7% -> 46.5%, within
noise), consistent with Phase 11 being a pure Python-level
vectorization with no new large-array allocation pattern. Stall
percentage rose notably (33.9% -> 42.8%), which is expected and not
concerning: with dict_build's Python-object-heavy loop shortened, the
pipeline's *remaining* time is now proportionally dominated by the
already-memory-bound WHT/gather stages (large `(chunk_size, dim)`
complex128 arrays scanned beyond L3, per `cache_locality/README.md`'s
findings 2/4/12) - a higher share of *remaining* cycles being
memory-stalled is exactly what removing a CPU-bound Python bottleneck
should produce, not a new problem.

## Interpretation: does this change the GPU-worth-it answer?

**Before Phase 11**: WHT butterfly was 21% of total time, dict_build
60% - GPU-accelerating a 21% slice of a Python-object-loop-dominated
pipeline was clearly not worth pursuing (the win would be capped
around 21% even with a large per-stage speedup, and dict_build would
remain the obvious next target regardless).

**After Phase 11**: WHT butterfly is 31.9% of total time, now larger
than every stage except dict_build (39.1%) and much closer to it in
absolute terms (22.56s vs 27.64s) than before (22.68s vs 64.64s).
This moves GPU work from "clearly not worth it" to **genuinely
marginal** - a real, not dismissible, fraction of pipeline time, but
not yet the dominant cost either.

**Recommendation (not yet acted on): scope a further dict_build
sub-cost breakdown before committing to GPU work.** If dict_build has
more easily-reachable headroom (this document's own "what this does
NOT show" section, and `full_pipeline_n150_findings.md`'s equivalent
section, both flagged that neither prior measurement broke dict_build
down more granularly than "the per-term Hermiticity check dominates
over dict construction itself" - worth re-checking post-implementation
whether that internal split still holds, or whether `.tolist()`/label
lookup now dominates differently), that remains the lower-risk,
higher-certainty target. GPU work carries real costs this analysis has
not yet estimated (host<->device transfer overhead for the coefficient
arrays, implementation/maintenance complexity, portability - this
project has no existing GPU toolchain or dependency) that could easily
exceed a ~22s slice's available upside; that estimate has not been
done and should precede any commitment to a GPU phase.

## What this does NOT show

- Same caveat as the original: the timing-instrumented `fwht.py`
  variant is a scratch copy, not committed - only the (unchanged)
  driver script and this findings doc are checked in.
- `perf stat` here is userspace-only (`perf_event_paranoid=2` on this
  run), not directly comparable counter-for-counter to the original's
  presumed system-wide counting - the wall-clock comparison (the
  primary evidence for the GPU question) is unaffected, but the
  cache/stall percentage *deltas* between before/after should be read
  with that caveat in mind, not treated as perfectly apples-to-apples.
- Does not measure `--parallel-labels` post-Phase-11 (only serial
  labels re-measured here) - `full_pipeline_n150_findings.md` already
  found it delivers no measurable full-pipeline benefit at N=150's
  proportions; nothing about Phase 11 changes that mechanism (labeling
  is still ~12% of a now-smaller total, an even smaller absolute
  target than before), so it was not re-verified here.
- Does not itself scope or estimate a GPU port's real cost/benefit -
  only establishes that the question is now worth that further
  scoping work, which is the recommended next step, not done in this
  document.
