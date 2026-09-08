# The progress marker is ~91% of what checkpointing still costs

Recorded 2026-09-08, after the binary chunk-framed format replaced
JSONL. Measured by `progress_marker_cost.py`.

## Why this was measured

The redesign made the checkpoint *payload* write ~500x cheaper
(`arrays_vs_dict_findings.md`, "The redesign, measured"). It left the
*progress marker* untouched: `_append_parallel_checkpoint_chunk` still
does

```python
json.dump({"completed_chunk_indices": sorted(completed)}, f)
```

on every completed chunk, rewriting the whole set each time - O(n) work
per chunk, O(n^2) across a run. When a frame write cost ~4.2 us/term
that was noise. At ~0.009 us/term it might not be.

This was sized before touching checkpoint *granularity* - the question
the redesign left open - because granularity is the wrong lever if the
marker is the cost.

## Scale of a real N=150 run

N=150 is `n_qubits=14`, `dim=16384`, `n_active=11,189` distinct x
values, so at the sweep's `chunk_size=2` a run completes **5,595
chunks** and rewrites the marker 5,595 times.

## Result

Both halves of one chunk's checkpoint write, swept over how many chunks
have already completed. The marker's cost depends on the set size; the
frame's does not.

| completed set size | frame | marker | marker/frame | marker bytes |
|---|---|---|---|---|
| 1 | 0.118 ms | 0.094 ms | 0.79x | 32 |
| 100 | 0.131 ms | 0.150 ms | 1.14x | 419 |
| 1,000 | 0.435 ms | 2.357 ms | 5.42x | 4,919 |
| 2,797 | 0.152 ms | 1.746 ms | 11.46x | 15,701 |
| 5,595 | 0.199 ms | 3.011 ms | **15.13x** | 32,489 |

Run at 7 and at 25 reps; the marker share came out 90.9% and 90.8%.

The frame row is flat in set size, as designed. The marker row is not.
Isolated separately at 40 reps per point, with no frame writes at all,
the growth is **clean and linear at ~0.55 us per set element**:

| set size | 1 | 100 | 1,000 | 2,000 | 3,000 | 4,000 | 5,595 |
|---|---|---|---|---|---|---|---|
| mean | 0.070 ms | 0.122 ms | 0.786 ms | 1.280 ms | 1.835 ms | 2.272 ms | 3.088 ms |

So the per-chunk cost is O(n) in chunks completed, and the run total is
O(n^2). Confirmed, not inferred.

## Extrapolated cost of a full N=150 run

| component | cost |
|---|---|
| frames (5,595 x 0.207 ms, flat) | **1.16 s** |
| markers (trapezoid over the growing set) | **11.48 s** |
| total | **12.6 s** |

Against the 8.729 s that `arrays_no_checkpoint` takes at `w8_c4`, that
is **145% overhead** - checkpointing an N=150 run still costs more than
recomputing it from scratch.

For contrast, the same figure in the JSONL era was ~4,422%. The
redesign removed the bulk of it; the marker is what remains.

**If the marker write were O(1), overhead would fall to ~13%** - the
point where checkpointing becomes a reasonable default rather than
something to avoid.

## What this settles

**Checkpoint granularity is the wrong lever.** Batching frames every N
chunks would divide the marker cost by the batching factor while losing
up to N chunks of work on a crash. Making the marker O(1) removes ~91%
of the cost and weakens no durability guarantee. The open question from
the redesign ("is per-chunk still right?") is answered: per-chunk is
fine, the marker is not.

## Direction, not yet designed

Every frame already carries its own `chunk_index` in its header, so the
set of completed chunks is **derivable from the payload file itself** -
the separate marker may be redundant rather than merely slow. Failing
that, an append-only marker (one record per completed chunk) makes each
write O(1) and keeps recovery a scan.

Either option interacts with the crash-safety ordering (frame first,
marker second) that the current design relies on, so this needs
designing rather than patching - see the next spec.

---

# Measured after the change

Recorded 2026-09-08, after the append-only record format replaced the
rewrite on both checkpoint paths
(`docs/superpowers/specs/2026-09-08-append-only-progress-marker-design.md`).
Measured by the same `progress_marker_cost.py`, now carrying the
append writer as a fourth arm.

Everything above this line is the motivation and still stands. The
numbers below are a fresh run; they are NOT the same run as the table
above, so the `frame` and `marker` columns differ slightly from it.

## Result

`OPENBLAS_NUM_THREADS=1 python profiling/phase13/progress_marker_cost.py 25`

| completed set size | frame | marker (old) | append (new) | marker/frame | marker/append | marker bytes |
|---|---|---|---|---|---|---|
| 1 | 0.137 ms | 0.076 ms | 0.026 ms | 0.55x | 2.9x | 32 |
| 100 | 0.145 ms | 0.175 ms | 0.029 ms | 1.21x | 6.0x | 419 |
| 1,000 | 0.146 ms | 0.617 ms | 0.036 ms | 4.21x | 17.1x | 4,919 |
| 2,797 | 0.169 ms | 1.320 ms | 0.044 ms | 7.82x | 29.8x | 15,701 |
| 5,595 | 0.186 ms | 2.438 ms | 0.049 ms | 13.08x | **49.9x** | 32,489 |

The record file is built once per `k` **outside** the timed region;
each rep times exactly one 8-byte append onto it. This is the harness
detail that matters: rebuilding it per rep dirties page cache in
proportion to `k` right before the timed write, and that artifact alone
produced a false "rising, O(1) falsified" verdict during design.

**Append flatness, in the sweep: 26.1 us .. 48.9 us, spread 1.87x**
across a 5,595x range of `k` - under the 3x bar, so the stop condition
did not trigger.

## The residual drift is the harness, not the format

The append column still drifts monotonically (26 -> 49 us). So does the
`frame` column over the same rows (0.137 -> 0.186 ms), and the frame
write is k-independent by construction - so the drift is shared, and
belongs to the interleaved sweep (three arms hitting the same disk in
one loop), not to `k`.

Confirmed by re-measuring the append **in isolation** - no frame or
rewrite arm in the loop, 40 reps x 3 passes per `k`, with the `k` order
randomized so a monotone drift in *time* cannot masquerade as one in
`k`:

| k | 1 | 100 | 1,000 | 2,797 | 5,595 |
|---|---|---|---|---|---|
| mean | 11.8 us | 12.3 us | 13.5 us | 12.0 us | 12.6 us |
| median | 11.5 us | 11.5 us | 11.8 us | 11.5 us | 11.8 us |

**Mean spread 1.14x, median spread 1.03x.** Flat. O(1) confirmed, and
the sweep's 1.87x is an upper bound inflated by its own interleaving.

## Recomputed cost of a full N=150 run

*Extrapolation.* The per-chunk costs above are measured; multiplying
them by 5,595 chunks is a projection, not an observed full run. No
checkpointed N=150 run has been executed end to end.

| component | before | after |
|---|---|---|
| frames (5,595 x 0.157 ms, flat) | 0.88 s | 0.88 s |
| marker (trapezoid over the growing set) | 7.37 s | - |
| append (5,595 x 36.9 us, sweep figure) | - | 0.21 s |
| **checkpoint total** | **8.24 s** | **1.08 s** |

That is **86.9% off the checkpoint total** and **97.2% off the marker
itself**. Against the 8.729 s `arrays_no_checkpoint` run at `w8_c4`,
checkpoint overhead falls from **94.5% to 12.4%** - past the point
where checkpointing costs less than the work it protects, which was the
whole objective.

Using the isolated 12.6 us instead of the sweep's contaminated 36.9 us,
the append total is 0.07 s, the checkpoint total 0.95 s, and overhead
**10.9%**. The 12.4% figure is the conservative one and is the one to
quote.

The marker file also stops growing quadratically: 44.8 kB written once,
against 0.09 GB rewritten across the run before.

## What this closes

The design's premise holds as measured: the marker write is O(1), and
checkpoint granularity stays a closed question. Per-chunk checkpointing
is now cheap enough to be a reasonable default.

Still unmeasured: an integrated checkpointed run at N=150. Both halves
are sized in isolation only. That remains the plan's deferred item.
