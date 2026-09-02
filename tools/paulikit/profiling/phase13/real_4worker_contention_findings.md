# Contention findings validated at the REAL 4-worker scale, not just the synthetic proxy

Recorded 2026-09-02, direct follow-up requested after
`l3_capacity_vs_bandwidth_findings.md`'s own open item: all earlier
contention evidence (that document, `l3_contention_direct_evidence_findings.md`)
used a controlled proxy - 1 real worker + 3 SYNTHETIC noise processes
touching arbitrary buffers, not 4 real workers doing real divided
work. This document closes that gap.

## Design: 4 real workers, real work, individually timed

`real_4worker_contention_test.py` uses the EXACT SAME production
functions `parallel_decompose` itself calls
(`_parallel_worker_init`/`_parallel_worker_chunk`, not a
reimplementation) - 4 processes, each pinned to a distinct physical
core, each processing its own real, disjoint slice of N=150's actual
chunk list (striped assignment: worker `i` gets chunks
`i, i+4, i+8, ...`, matching how `parallel_decompose`'s pool
distributes work in practice), each independently timed via a shared
`multiprocessing.Barrier` so all 4 start their measured work at
literally the same instant.

Verified before the `perf stat` run: term counts per worker (~22.9M
each) match the earlier synthetic-proxy's per-worker figure exactly,
and per-worker elapsed times cluster tightly (5.0-5.3s, not
staircased) - confirming genuine concurrent execution, not accidental
serialization (an earlier draft of this script had exactly that bug -
caught before the real measurement by checking the raw numbers rather
than trusting the script blindly).

## Result: real 4-worker case vs. the synthetic proxy conditions

| condition | wall-clock (per worker or overall) | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|
| synthetic: alone (1 real worker, no noise) | 4.62s | 0.41% | 0.40% |
| synthetic: L2-bound noise (3 fake, own-L2-only) | 5.48s | 0.88% | 0.79% |
| synthetic: L3-exceeding noise (3 fake, 64MB/core) | 14.78s | 2.87% | 2.20% |
| **REAL: 4 real workers, real work** | **~5.0-5.3s per worker** | **2.90%** | **2.51%** |

## Interpretation: wall-clock tracks the L2-bound proxy, cache-miss ratio tracks the L3-exceeding proxy - both are real and consistent, not contradictory

At first glance this looks odd - real wall-clock is close to the mild
L2-bound-noise condition, but the cache-miss ratio matches the
aggressive L3-exceeding-noise condition almost exactly (2.90% vs.
2.87%). This is explainable, not contradictory: the synthetic
L3-exceeding noise processes did PURE throwaway work (touching a 64
MB buffer with no useful output) on 3 of the 4 cores, so all of that
condition's wall-clock cost came from raw contention with nothing
productive happening on those cores. The real 4-worker case instead
has all 4 cores doing GENUINELY USEFUL, individually light work (each
chunk's own working set is tiny, ~512 KiB at chunk_size=2, chosen to
fit L1/L2 by Phase 12's own tuning) - so the L3-capacity-level
contention shows up clearly and comparably at the cache-counter level
(validating that the synthetic proxy was not overstating the
mechanism's real magnitude), but each real worker's own compute is
light enough that the wall-clock penalty does not compound the way it
does against a bandwidth-saturating synthetic load with nothing else
to do.

**Practical, evidence-based conclusion**: the real 4-worker
`parallel_decompose` case genuinely experiences L3-capacity-level
contention of a magnitude consistent with (not smaller than) the
synthetic L3-exceeding proxy - this is real, validated at production
scale, not just a proxy artifact. But because the real workload's own
per-chunk footprint is small and well-tuned to L1/L2 (Phase 12's own
result, re-confirmed here), the real wall-clock cost of that
contention stays closer to the milder end of what was measured in the
controlled experiments, not the worst case.

## What this does NOT show

- Does not re-test at `n_workers=8` (logical CPUs, not physical
  cores) at this real-workload granularity - only the pinned
  4-physical-core case was tested here, matching the auto-detected
  default `parallel_decompose` now uses.
- Does not re-test with `perf stat --no-inherit` per-worker (each
  worker's own individual counters, not the aggregate across all 4) -
  the aggregate figures reported here answer "how contended is the
  whole job," not "which specific worker suffered most," which could
  differ if chunk assignment or scheduling created any asymmetry
  (the near-identical per-worker wall-clock times suggest this is not
  a large effect, but it was not directly measured at the counter
  level).
- Does not test other N (this is N=150 only, matching every other
  measurement in this specific investigation thread).
