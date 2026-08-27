# Full-pipeline cache-locality re-investigation at N=150: dict construction dominates, not labeling

Recorded 2026-08-27, per direct user request to re-investigate
cache-locality on the full streaming pipeline (not just the label
kernel in isolation, as `tbb_labeling_n150_findings.md` did) once
Phase 10 was implemented. Machine: 15.7 GiB RAM, 8 cores, 8 MiB
shared L3.

## Why this measurement was needed

`tbb_labeling_n150_findings.md` isolated `pauli_label_batch_parallel`
against a *synthetic* 40M `(x, z)` pair array and found a real
1.1-1.4x wall-clock win at the cost of a modest cache-locality
regression. That measurement never ran inside the real
`fwht_pauli_terms_iter` pipeline, so it could not show what fraction
of *total* pipeline time labeling actually represents, or whether the
isolated kernel's cache-locality tradeoff still holds once embedded
in the full gather → WHT → threshold → label → dict-build sequence.

## Method

**Per-stage wall-clock breakdown**: a scratch-copied, timing-
instrumented variant of `fwht.py` (not committed - see "What this
does NOT show") wraps five stages with `time.perf_counter()`: gather
(XOR-index scatter into the chunk buffer), WHT (the butterfly
transform), phase/threshold (phase-factor multiply + `atol` filtering
+ `np.nonzero`), label (`_pauli_label_batch` call), and dict_build
(the Python loop constructing each chunk's `dict`, including the
Hermiticity check). Run via `n150_stage_breakdown_driver.py`
(loads the instrumented module directly by file path, alongside the
real installed `paulikit.hamiltonian`), driving the real N=150
Hamiltonian through `fwht_pauli_terms_iter(chunk_size=256)`, with and
without `parallel_labels=True`.

**Whole-pipeline `perf stat`**: two standalone scripts
(`n150_pipeline_perf_serial.py` / `n150_pipeline_perf_parallel.py`,
this directory) run the same real N=150 workload through the
unmodified, production `fwht_pauli_terms_iter`, under the same event
set and `OPENBLAS_NUM_THREADS=1` control used throughout this
project. Both runs used the standard `ulimit -v 4000000` + `free -m`
polling safety harness (2s interval, kill below 1500 MiB available).

## Results: per-stage wall-clock breakdown

| stage | serial labels | parallel labels |
|---|---|---|
| gather | 0.46s (0.4%) | 0.46s (0.5%) |
| WHT butterfly | 22.68s (21.1%) | 21.19s (21.3%) |
| phase + threshold | 9.91s (9.2%) | 9.02s (9.1%) |
| label generation | 7.36s (6.8%) | 5.72s (5.8%) |
| **dict construction** | **64.64s (60.1%)** | **60.67s (61.0%)** |
| unaccounted | 2.42s (2.3%) | 2.45s (2.5%) |
| **total** | **107.48s** | **99.51s** |

(91,652,096 terms, 44 chunks, both runs.)

**Dict construction dominates at ~60% of total time** - roughly 3x
the WHT butterfly's share and nearly 9x label generation's share.
This was not visible in any prior measurement: Phase 3's original
`perf_record_n50_findings.md` found `pauli_label` (not dict
construction specifically) dominant at N=50 pre-Phase-9/10, and no
prior study isolated the streaming path's own per-chunk
`dict[label] = coefficient` loop (with its per-term Hermiticity
`abs()`/`float()` check) as a distinct cost.

## Results: whole-pipeline `perf stat`

| labels | wall | cycles | cache-miss % | LLC-miss % | stall % | stall-mem % |
|---|---|---|---|---|---|---|
| serial | 96.96s | 253.15e9 | 45.7% | 45.6% | 33.9% | 30.8% |
| parallel (TBB) | 97.97s | 262.41e9 | 46.5% | 46.7% | 34.1% | 31.1% |

Raw output: `n150_pipeline_perf_serial.txt`, `n150_pipeline_perf_parallel.txt`.

**Parallel labeling is essentially flat-to-slightly-worse at the full-
pipeline level** - wall time is *slower* by ~1s (97.97s vs 96.96s,
opposite direction from the isolated kernel's 1.1-1.4x win), and every
cache/stall percentage moves by well under 1.5 points, consistent with
noise rather than a real effect. This **does not reproduce** the
isolated 40M-synthetic-benchmark finding's meaningful cache-locality
degradation (there: cache-miss% +1.5pts, stall% +2.8pts, and a real
wall-clock win) - because in the real pipeline, label generation is
only ~6-7% of total time, so saving ~1.6s there against a 97s pipeline
is within the run-to-run noise floor already documented elsewhere in
this project (`stall_floor_mystery_solved.md`, `tbb_evaluation_findings.md`).

## Interpretation

1. **The isolated TBB-labeling measurement's conclusion does not hold
   at the real pipeline's scale.** Not because the isolated measurement
   was wrong (it correctly characterized the label kernel by itself),
   but because label generation is a *small* fraction of real
   end-to-end time once dict construction is accounted for - so a real
   per-kernel effect gets swamped to noise level in the full pipeline.
   This is the same lesson `tbb_evaluation_findings.md` already drew
   once before (measuring the wrong stage in isolation can mislead
   about the *system*), now confirmed a second time in a different
   part of this same pipeline.

2. **Dict construction, not labeling or the WHT butterfly, is the
   real remaining hot spot** in the streaming pipeline at N=150 scale.
   This directly parallels Phase 3's original finding at N=50
   (`pauli_label`'s per-term Python loop dominating over the
   vectorized FWHT core) - the specific culprit has moved (dict
   construction, not label-string formatting, now that labels are
   TBB/Cython-fast), but the *shape* of the problem (pure-Python
   per-term bookkeeping dominating a vectorized numeric core) has not.

3. **`--parallel-labels` should not be recommended as a default
   optimization** based on this finding - it does not deliver its
   isolated-benchmark's win once embedded in the real pipeline, and
   the earlier decision to make it opt-in (not default) turns out to
   have been the right call for a different reason than originally
   measured: not "real win, modest cost" but "no measurable win either
   way at this pipeline's actual proportions."

## Follow-up implied, not yet scoped as a phase

Optimizing `dict_build` (currently a pure-Python per-term loop with a
per-term Hermiticity check) is now the highest-leverage remaining
target for this pipeline's performance, ahead of anything touching
labeling or the WHT step - this was not previously visible because no
prior measurement isolated it. Not yet scoped as a PLAN.md phase;
recorded here so the next investigation starts from this finding
rather than re-deriving it.

## What this does NOT show

- The per-stage timing-instrumented `fwht.py` variant used for the
  breakdown is a **scratch copy**, not committed to the repository -
  only the driver script (`n150_stage_breakdown_driver.py`) and this
  findings document are committed, per this project's "document
  exploration scripts" convention; the instrumented module itself
  lives only in the session's scratchpad and is not reproducible from
  the repo alone (a future session wanting to re-run this exact
  breakdown would need to re-create the instrumentation, following
  this document's description of where the five timing wraps go).
- Wall-clock instrumentation only for the per-stage breakdown, not
  hardware performance counters per stage - `perf stat` was only run
  on the whole pipeline, not per-stage, since `perf stat` cannot
  isolate sub-regions of one process without heavier tooling
  (`perf record` + annotation) not used here.
- Does not identify *why* dict construction is 60% of the time in
  more granular detail (e.g. whether it's the `dict.__setitem__` calls,
  the `.tolist()` conversions, or the per-term Hermiticity check that
  dominates within that 60%) - a natural next step if this is pursued
  further, not done here.
- `--parallel-labels`'s isolated-kernel finding
  (`tbb_labeling_n150_findings.md`) is not retracted - it remains an
  accurate characterization of that kernel in isolation. This document
  adds the system-level context that the isolated effect does not
  propagate to a measurable full-pipeline difference at N=150's real
  proportions, not that the isolated measurement itself was flawed.
