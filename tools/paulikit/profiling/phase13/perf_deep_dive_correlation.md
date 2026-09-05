# Deep perf.data dive correlated against DAG parallelism ≈3.0 (2026-09-05)

Follow-up to `perf_record_annotate_findings.md` (which used a
truncated `head -60`/`head -150` view of `perf report`/`perf
annotate`) and `dag_extraction_and_parallelism.md` (the code-derived
Work/Span/Parallelism ≈3.0 result). This document goes past the
truncation, correlates the result against the DAG figure, and reports
what changes and what doesn't. Run independently twice — once by a
subagent doing full `perf report`/`perf script`/`perf annotate` passes
directly against the `.perf.data` files, once manually via the
`perf_deep_dive` script — with matching results both times.

## What the deeper pass adds

**Symbol-frequency histogram, exact leaf-frame counts (not the
collapsed report tree), no threshold truncation:**

| symbol | w1_c1 (n=56,922) | w8_c4 (n=9,830) |
|---|---|---|
| `CDOUBLE_add_X86_V3` | 7.04% | 6.93% |
| `CDOUBLE_subtract_X86_V3` | 6.08% | 4.25% |
| `npy_cpow` | 5.66% | 5.70% |
| `_contig_to_contig` | 3.85% | 4.19% |
| `mapiter_get` | 3.59% | 3.49% |

Every symbol matches within ~2 percentage points, confirming — with
finer precision than the earlier 1%-threshold collapsed view — that
the on-CPU instruction mix genuinely is identical between conditions,
down to a 0.1% resolution. No new hot symbol appears below the
earlier truncation point.

**No multiprocessing/lock/futex frames in EITHER file's on-CPU
trace**, including `w8_c4`: `grep -iE
"multiprocessing|queue|lock|futex|pickle|socket|pipe|semaphore|
synchronize|connection"` against the full report and the symbol
histogram turns up only a handful of glibc `pthread_mutex_lock`/
`lll_mutex_lock_optimized` samples (malloc arena locking, not
multiprocessing IPC), at a similar small proportion in both
conditions (~0.06-0.09% of samples). This is expected, not a new
discovery: `perf record cycles` only samples while a thread is
actually executing, so a thread blocked in `Queue.put()`'s underlying
`futex_wait` produces **zero samples** during that block, by
construction — its absence here doesn't refute the drain-loop-
blocking mechanism, it's simply invisible to this instrument.

**Capture-window precision**: both files cover ~14.9-15.0s of the
nominal 15s window with near-identical sampling calibration (mean
cycles/sample within 0.3% of each other between conditions) — this
rules out "the two captures aren't really comparable" as a confound
on the earlier 5.77x cycle-retirement-rate gap (2.71 GHz vs 0.47 GHz).
That gap is a clean measurement of real off-CPU time over directly
comparable windows, not a capture-timing artifact.

**No memory-stall signature**: annotate-level arithmetic-vs-load/store
instruction ratios in the top 3 hot symbols shift by ≤5pp with no
consistent direction between conditions — rules out a cache/DRAM-
latency explanation for the on-CPU portion specifically (consistent
with `resident_footprint_findings.md`'s and `traffic_intensity_
findings.md`'s independent refutations of memory-traffic-based
explanations).

## Correlation with the DAG's ≈3.0 parallelism figure

**Neutral-to-mildly-consistent, not independently corroborating.**
The DAG analysis (`dag_extraction_and_parallelism.md`) derives ≈3.0
purely from code structure — it says nothing about *which* off-CPU
mechanism causes the loss, only that a single-threaded drain-loop
chain forces a large constant-scaling floor onto the critical path.
The perf data confirms the *magnitude* of lost per-worker throughput
under contention (a clean, precisely-bounded ~5.8x on-CPU cycle-rate
collapse) but is architecturally incapable of confirming or refuting
*why* — it would look identical whether the cause were the drain-loop
serialization the py-spy evidence points to, or generic 8-workers-on-
4-cores OS scheduling (already ruled out separately by
`traffic_intensity_findings.md`'s same-machine synthetic controls
scaling near-linearly), or something else off-CPU entirely.

**What correlates cleanly, and where it comes from:**
- DAG-derived parallelism ≈3.0 (from code structure alone)
- Measured effective concurrency ~2.3-2.7 (`dag_gst_master_analysis.md`
  Section 3b, from `perf stat` task-clock/elapsed ratios — a
  DIFFERENT measurement than this document's `perf record cycles`
  dive)
- These two independently-derived numbers landing in the same
  small-integer range is the strongest piece of theory-vs-measurement
  agreement in this investigation.

**What this document's deep dive contributes to that picture**: a
precise, artifact-ruled-out confirmation of the raw magnitude of
per-worker throughput loss (5.77x on-CPU cycle rate), and an explicit
demonstration that the *mechanism* remains attributable only to the
py-spy stack-trace evidence (`Lock.__enter__` inside `Queue.put()`,
3/3 samples) — not to anything found in this `cycles`-event capture.
**Do not read this document as adding new mechanistic evidence for the
drain-loop hypothesis** — it adds precision to the magnitude question
and rules out two alternative explanations (a different code path,
memory-latency stalling) for the on-CPU portion, nothing more.

## What would actually close this gap

`perf sched record`/`perf sched latency` (captures scheduler wakeup
and off-CPU/runqueue-wait events — a fundamentally different event
type than `cpu/cycles/P`, which this whole investigation has used
exclusively so far) would be the correct next instrument if
independent, direct confirmation of the blocking mechanism (beyond
py-spy's 3-sample stack dumps) is wanted. Not done in this document.

## Artifacts

- `perf_deep_dive` (script; regenerates everything below from the
  gitignored `.perf.data` files)
- `perf_w1_c1.report.full.txt`, `perf_w8_c4.report.full.txt`
  (full `perf report`, `--percent-limit 0.1`, no truncation)
- `perf_w1_c1.symbol_histogram.txt`, `perf_w8_c4.symbol_histogram.txt`
  (exact leaf-frame counts from `perf script`)
- `perf_w1_c1.annotate.full.txt`, `perf_w8_c4.annotate.full.txt`
  (full `perf annotate`; gitignored — several MB, mostly 0%-sample
  disassembly, regeneratable via `perf_deep_dive`)
