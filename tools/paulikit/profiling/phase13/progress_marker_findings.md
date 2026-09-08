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
