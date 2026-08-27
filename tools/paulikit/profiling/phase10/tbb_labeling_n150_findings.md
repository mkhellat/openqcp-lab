# Re-measuring TBB-parallel label generation at N=150-representative scale

Recorded 2026-08-27, at the start of PLAN.md Phase 10 (streaming
output) design. Prompted directly by the user, applying
[[feedback_divide_and_conquer_strategy]]'s discipline of reasoning
about parallelism explicitly rather than defaulting to serial or
parallel without measurement.

**Superseded in practice by [`full_pipeline_n150_findings.md`](full_pipeline_n150_findings.md):**
this document's measurement of the label kernel *in isolation* is
still accurate, but once embedded in the real streaming pipeline, the
1.1-1.4x win and cache-locality cost found here both wash out to
noise level - label generation turns out to be only ~7% of total
pipeline time (dict construction is ~60%). Read this document for the
isolated-kernel numbers; read the full-pipeline doc for what actually
matters in production.

## Why re-measure at all

`cache_locality/tbb_evaluation_findings.md` (2026-08-25) already
measured `pauli_label_batch_parallel` (the oneTBB-parallel label
kernel, built in Phase 3a) against the serial `pauli_label_batch`, at
N=25/50/100, and found **no measurable effect on anything** - wall
time, cache-miss ratio, LLC-miss ratio, stall percentages all agreed
with the serial path to within run-to-run noise. That finding's own
explicit caveat: *"Revisit only if Phase 6's own prototyping surfaces
a new hot loop that TBB could plausibly parallelize - not as a
default assumption."*

Phase 8/9 have since done exactly that: N=150's real bottleneck moved
from dense-Hamiltonian storage (Phase 8, fixed) to the coefficient
accumulator (Phase 9, fixed) to - now - label generation and dict
construction at ~134M terms, a scale the original TBB measurement
never tested (it excluded N=150 entirely, citing the then-unresolved
OOM upstream of label generation - see that finding's Method
section). The 2026-08-25 null result was measured where TBB
*couldn't* matter yet; it says nothing about whether it matters now
that label generation is reached at real scale.

## Method

Two standalone scripts, `tbb_label_40m_serial.py` /
`tbb_label_40m_parallel.py` (this directory), each: generate 40M
random `(x, z)` `uint32` pairs at `n_qubits=14` (`dim=16384`, matching
N=150's real qubit count), then call `pauli_label_batch` /
`pauli_label_batch_parallel` respectively and print the result count.

Scale note: the real N=150 result is ~134M terms (see
`../phase9/phase9_findings.md`), not 40M. 134M was not used directly
here because comparing two label lists in memory simultaneously (as
the initial wall-clock-only sanity check did, before switching to
separate `perf stat` processes) risked exceeding this machine's ~11
GiB available RAM at that size - 40M was chosen as the largest size
comfortably measurable standalone under the same `ulimit -v`
safety-monitored harness used throughout this project, while still
being 4x larger than the previous study's largest case (N=100, whose
term count is far below 40M). Correctness was verified first (see the
sanity-check run below) - both kernels produce byte-identical label
lists at every size tested.

Same `perf stat` event set as `cache_locality/tbb_evaluation_findings.md`,
same `OPENBLAS_NUM_THREADS=1` noise control, run as two separate
processes (not two calls in one process) so `perf stat` measures each
kernel in isolation:

```bash
OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python tbb_label_40m_serial.py
# (same command for tbb_label_40m_parallel.py)
```

Both runs used the same `ulimit -v 9000000` + `free -m` polling
safety harness (2s interval, kill below 1000 MiB available) as every
other N=150-scale measurement in this project. Raw output:
`tbb_label_40m_perf_serial.txt`, `tbb_label_40m_perf_parallel.txt`.
Machine: 8 cores (`nproc`), 15.7 GiB RAM.

A smaller preliminary sanity check (1M and 10M terms, wall-clock only,
no `perf stat`, both kernels called in the same process to also verify
correctness) was run first and is not separately recorded, beyond
establishing that the parallel path's advantage grows with scale
(0.75x at 1M i.e. parallel *slower* - dispatch overhead dominates at
that size - rising to 1.37x at 10M), motivating going straight to 40M
for the real measurement rather than trusting the small-scale number.

## Results

| kernel | wall time | cycles | cache-miss % | LLC-miss % | stall-total % | stall-mem % |
|---|---|---|---|---|---|---|
| serial | 4.590s | 10.94e9 | 58.4% | 43.8% | 15.7% | 12.3% |
| parallel (TBB) | 4.026s | 13.80e9 | 59.9% | 44.3% | 18.5% | 15.6% |

(cache-miss % = cache-misses / cache-references; LLC-miss % =
LLC-load-misses / LLC-loads; stall % = cycle_activity.stalls_total /
cycles or stalls_mem_any / cycles - same derivation as
`tbb_evaluation_findings.md`.)

**Wall-clock speedup: 1.14x** in this `perf stat`-instrumented run
(1.22x-1.37x in the earlier uninstrumented sanity checks at 40M/10M -
`perf stat`'s own overhead plausibly narrows the margin slightly, not
investigated further).

**Cache-locality cost, real but modest:** cache-miss % rises 58.4% →
59.9% (+1.5 points), LLC-miss % rises 43.8% → 44.3% (+0.5 points),
stall-total % rises 15.7% → 18.5% (+2.8 points), stall-mem % rises
12.3% → 15.6% (+3.3 points). Total CPU-cycles consumed rises ~26%
(10.94e9 → 13.80e9) - expected: 8 threads doing the same total work
plus per-thread cache contention costs more aggregate cycles even
when wall-clock time drops, since work is spread across cores rather
than done more efficiently per core.

## Interpretation

**This is a genuinely different result from the 2026-08-25 null
finding** - at this larger scale, TBB-parallel labeling produces a
real (not noise-level) wall-clock win, unlike at N≤100. But it is not
a free win: it is a real trade of cache efficiency for multi-core
throughput, consistent with the general pattern that parallelizing
memory-bound work often increases aggregate cache pressure (more
concurrent working sets competing for the same shared LLC) even as it
reduces wall-clock time. The magnitude here is modest (single-digit
percentage-point degradation in miss/stall ratios, not an
order-of-magnitude regression) against a real 1.1-1.4x wall-clock
gain.

## Decision

Per direct discussion with the user (2026-08-27): use
`pauli_label_batch_parallel` in Phase 10's streaming design now, with
this tradeoff explicitly recorded (not silently absorbed) rather than
either defaulting to serial out of caution or adopting parallel
without measuring the cost. Cache-locality behavior at real N=150+
scale (not just this 40M synthetic proxy) is to be **re-investigated
iteratively** as Phase 10 and later phases proceed - this finding is
not treated as a final word on the tradeoff, only as the basis for
today's decision.

## What this does NOT show

- Not measured at the real N=150 scale (~134M terms) directly, only a
  40M-term synthetic proxy with the same `n_qubits`/`dim` - chosen for
  memory safety, not because 134M was expected to behave differently
  in kind (see Method's scale note).
- Does not measure the *numeric* (WHT/threshold) per-chunk computation
  possibly running in parallel across chunks - a separate question
  raised alongside this one, deferred: chunks are numerically
  independent (no cross-chunk combination needed), but that is an
  orthogonal parallelism opportunity from label-generation
  parallelism, and was not the subject of this measurement.
- Does not yet measure the *combined* streaming pipeline (chunked WHT
  + per-chunk parallel labeling + accumulation/yield) as one integrated
  system - only isolates the label-generation kernel itself, matching
  the original 2026-08-25 study's own scope.
