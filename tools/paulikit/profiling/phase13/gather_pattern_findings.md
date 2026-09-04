# Gather-pattern isolation experiment: mixed result, task-granularity confound identified (2026-09-04)

The fourth control from `traffic_intensity_findings.md`'s "Actual next
isolation step" (dense-traffic-volume already refuted as sufficient)
and `dag_gst_master_analysis.md` section 3e: does reproducing
paulikit's real IRREGULAR gather/scatter access pattern - not just its
dense buffer size/stage-touch count - reproduce the 2-physical-core
ceiling?

## Method

`gather_pattern_precompute.py` builds the REAL N=150/chunk_size=2
gather-index pattern once (`gather_pattern_chunks.npz`): for every one
of the real `n_chunks=5595` chunks, the exact `(row_offset, column)`
positions `_parallel_worker_chunk`'s own gather/scatter
(`fwht.py:1246-1255`) writes into its `(chunk_size, dim)` buffer -
`nnz` per chunk ranges 2-64, mean 8.04 (confirms this is a genuinely
irregular, non-uniform pattern, nothing a dense-random-buffer control
could reproduce). `n_chunks=5595` matches every prior measurement in
this investigation exactly - a real consistency check.

`gather_pattern_target.py`: same `ProcessPoolExecutor` shape as every
other control in this investigation (pinned workers, bounded
in-flight, pickle-over-pipe IPC). Two workload variants:
- `gather_only` - zero a buffer, scatter synthetic values at the REAL
  positions, return 64 floats. Isolates the irregular-write cost
  alone.
- `gather_and_wht` - same scatter, then the real
  `_walsh_hadamard_transform_rows`, return 64 floats. Isolates
  gather+WHT together.

The precomputed index arrays are passed to every worker explicitly via
the pool's `initializer`/`initargs` (matching how paulikit's own
`_parallel_worker_init` actually receives its operator/setup arrays) -
**a real methodology bug was found and fixed while building this**:
this machine's default `multiprocessing` start method is
`forkserver`, not `fork` (confirmed directly via
`multiprocessing.get_start_method()`) - workers do NOT inherit
already-loaded module-level globals via copy-on-write under
`forkserver`. An earlier draft relied on module-level `np.load()` of
the 3 MiB `gather_pattern_chunks.npz` file, which silently re-read
that file from disk fresh in EVERY worker at pool startup - a
per-worker disk-I/O cost that scaled with `n_workers` and contended
under core-packing, contaminating `gather_only`'s first (discarded)
sweep with noise unrelated to the gather pattern itself (w8_c4 stdev
0.57 vs w2_c1's 0.024 in that run). Fixed by passing the pre-loaded
arrays via `initargs`; re-run cleanly after the fix (results below are
from the corrected run only - the confounded run's
`gather_pattern_results.jsonl` was deleted, not reconciled, per this
project's established practice for contaminated measurement windows).

Same protocol as every prior measurement: `w2_c1` vs `w8_c4`, thermal
cooldown to <=55C before every run, 5 reps/cell, Welch's t-test.

## Results

| workload | w2_c1 mean (s) | w8_c4 mean (s) | speedup (w2/w8) | Welch p |
|---|---|---|---|---|
| `gather_only` | 0.8229 (sd 0.0515) | 1.0013 (sd 0.0135) | **0.822x** (w8 SLOWER) | 1.61e-03 |
| `gather_and_wht` | 9.0241 (sd 0.0804) | 7.0548 (sd 0.0726) | **1.279x** (w8 faster) | 4.28e-10 |
| paulikit (prior, contrast) | 22.068 | 24.801 | 0.89x (w8 slower) | 8.0e-3 |
| traffic-intensity controls (prior, contrast) | - | - | 2.19-2.69x (w8 faster) | <1.6e-8 |

**A genuinely mixed result, not a clean confirm/refute.**
`gather_only` DOES reverse (w8 slower than w2, tight and significant) -
the first isolated control in this whole investigation to reproduce
paulikit's DIRECTION of ceiling, and the magnitude (0.822x) is even
slightly stronger than paulikit's own (0.89x). `gather_and_wht` does
NOT reverse - it scales normally (1.279x), same direction and similar
magnitude to every dense-traffic control already tested.

## Interpretation: a task-granularity confound, not (yet) a clean answer

Per-chunk timing breakdown (mean wall-clock / 5595 chunks):

| workload | w2_c1 per-chunk | w8_c4 per-chunk |
|---|---|---|
| `gather_only` | 147.1 us | 179.0 us |
| `gather_and_wht` | 1612.9 us | 1260.9 us |

`gather_only`'s per-chunk cost (~147-179 us) is almost entirely
`ProcessPoolExecutor` submit/pickle/schedule/collect overhead - the
actual gather work (scattering a MEAN of 8 values into a zeroed
buffer) is computationally tiny. Subtracting `gather_only` from
`gather_and_wht` gives the WHT's own per-chunk cost: ~1466 us -
roughly 10x the gather step's own cost, and in the right ballpark
versus real paulikit's own measured `W_chunk ~= 4.71 ms` full-pipeline
average (this synthetic omits the phase-multiply/threshold steps, so
a smaller-but-comparable number is expected, not a mismatch).

**This means `gather_only`'s reversal is plausibly a DIFFERENT known
failure mode - task granularity too fine relative to
`ProcessPoolExecutor`'s own per-task dispatch overhead - not
necessarily evidence that the irregular ACCESS PATTERN itself is the
trigger.** At ~150-180 us/task, 8 workers contending to dispatch and
collect ~5595 very cheap tasks through the pool's shared queue/lock
machinery plausibly costs more coordination overhead per unit of real
work than 2 workers would, since the tasks are too cheap to amortize
that fixed per-task cost - a granularity problem, independent of
whether the buffer touched is dense or sparse/irregular.

`gather_and_wht`, at ~1.3-1.6 ms/task (an order of magnitude coarser),
scales normally - consistent with granularity being the dominant
effect at the `gather_only` scale, and with the irregular access
pattern NOT being sufficient on its own once task granularity is
closer to paulikit's real chunk cost.

## What this does and does not show

**Does NOT show**: that the irregular gather/scatter pattern is
sufficient to explain paulikit's ceiling - `gather_and_wht`, the
closer proxy to paulikit's real per-chunk cost scale, scales normally
(1.28x), unlike paulikit (0.89x).

**Does NOT rule out** gather/access-pattern as a contributing factor
either - `gather_only`'s reversal, even if partly/mostly a granularity
artifact, is still the first control result in this whole
investigation that reproduces paulikit's DIRECTION at a real,
statistically tight result. It is not yet possible to say how much of
paulikit's real ceiling is (a) task-granularity/IPC-dispatch overhead,
(b) irregular access pattern, or (c) something not yet isolated
(operator/setup-array resident footprint - `traffic_intensity_findings.md`'s
still-untested item 2) - `gather_and_wht` conflates gather AND WHT
into one measurement and cannot separate their individual
contributions from each other or from paulikit's real full pipeline
(which also includes the operator/setup-array resident footprint,
phase multiply, and threshold/filter step, none present here).

**Does show**: paulikit's real chunk granularity (~4.7ms full pipeline,
~1.5ms for gather+WHT alone as measured here) sits well above the
region where pure `ProcessPoolExecutor` dispatch overhead would
dominate (the `gather_only` regime, ~150-180 us/task) - so paulikit's
own ceiling is NOT simply a task-granularity artifact of the same kind
found in `gather_only`. Something else, still not isolated, explains
why paulikit's real (coarser-grained) chunks still fail to scale while
`gather_and_wht`'s comparably-coarse-grained chunks (irregular access
+ real WHT, no operator footprint) do scale.

## Next isolation step (not yet built)

Per `traffic_intensity_findings.md`'s own decision tree (now
reachable, since both dense-traffic-volume AND gather+WHT-without-
operator-footprint are refuted or inconclusive as sufficient causes):
isolate the **operator/setup-array resident footprint** each real
worker holds for its whole lifetime, independent of any one chunk -
a workload that keeps `gather_and_wht`'s access pattern and task
granularity but ALSO holds a paulikit-scale resident operator array
(and the `nnz`-length setup arrays) in each worker, to test whether
that fixed per-worker memory footprint (not any one chunk's own
traffic) is what tips the shared-memory-subsystem contention into
paulikit's observed ceiling. This is the one item from
`traffic_intensity_findings.md`'s original decision tree not yet
tested by either this experiment or the dense-traffic one.

## Artifacts

- `gather_pattern_precompute.py`
- `gather_pattern_target.py`
- `gather_pattern_sweep.py`
- `gather_pattern_chunks.npz` (precomputed real N=150 index pattern, 3.0 MiB)
- `gather_pattern_results.jsonl` (20 runs, the corrected/final sweep only)
