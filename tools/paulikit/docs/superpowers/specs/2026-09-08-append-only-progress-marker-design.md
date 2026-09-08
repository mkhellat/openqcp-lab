# Append-only progress marker

Design, 2026-09-08. Replaces the rewrite-everything progress marker
with a fixed-width append-only one, on both the sequential and
parallel checkpoint paths.

## Why

The binary chunk-framed format made the checkpoint *payload* ~500x
cheaper but left the *progress marker* untouched. Every completed
chunk still runs

```python
json.dump({"completed_chunk_indices": sorted(completed)}, f)
```

rewriting the whole set: O(n) per chunk, O(n^2) across a run.

Measured (`profiling/phase13/progress_marker_findings.md`): at N=150's
real shape - 5,595 chunks - the marker costs **15.13x the frame write**
by the final chunk and **~91% of the per-chunk checkpoint total**.
Extrapolated over a full run: frames 0.88s, markers 11.48s. Total
checkpoint overhead is still **142%** of the 8.729s run it protects,
so checkpointing an N=150 decomposition remains more expensive than
recomputing it from scratch.

An 8-byte append is O(1). Isolated across a 20,000x range of completed
counts, it is flat at ~13-21 us, spread 1.62x. Projected over a full
run the marker falls **11.48s -> 0.09s (99.2%)**, the checkpoint total
falls **92.2%**, and overhead against the run drops **142% -> 11%** -
the first point at which checkpointing costs less than the work it
protects.

## What was rejected, and why it matters

**Deleting the marker entirely** (deriving completion from the frames,
which already carry `chunk_index` in their headers) was the initial
recommendation and is **unsafe**. It rests on replayed chunks being
idempotent. They are not, for one of the two consumers:

- `fwht_pauli_terms_iter` yields dicts - `update()` is idempotent.
- `fwht_pauli_coefficients` calls `_GrowableArray.extend()` - **append,
  not merge.** Verified directly: extending the same chunk twice
  yields `[1 2 3 1 2 3]`. A replayed chunk would produce a COO triple
  with duplicated `(x, z)` entries carrying the same coefficient,
  silently double-counting on densification.

The earlier "duplicate replay is harmless" finding (Task 4's rollback
investigation) was established against dict-accumulating consumers
only, and does not generalize. The marker's real job is to distinguish
"frame complete and counted" from "frame complete but written by a run
that died before recording it" - a distinction the payload cannot make
on its own, and which one consumer genuinely needs.

**Checkpointing every K chunks** was rejected earlier: it divides the
marker cost by K while losing up to K chunks of work on a crash. The
measurement showed granularity was never the lever - the marker's
per-write cost was.

## Format

One file per path, same filenames and suffixes as today
(`.progress.json`, `.parallel_progress.json` - see Migration).
Each completed chunk appends one fixed-width record:

```
record := u64 chunk_index, little-endian   (8 bytes)
```

`struct.Struct("<Q")`. `u64` matches the width the frame header
already uses for `chunk_index`, so the two cannot disagree about
range.

No magic bytes and no per-record checksum: this file is written only
by the process that owns the checkpoint, is never partially rewritten,
and every record is validated against the payload anyway (see
Recovery). A record is 8 bytes, so it is smaller than the 16-byte
minimum a magic-plus-index scheme would need, and the recovery rule
below already rejects anything that does not correspond to a real
frame.

The **sequential** path currently stores a single monotonic
`next_chunk` rather than a set. It moves to the same append-only
record format: chunks complete in order there, so the recovered set is
contiguous and `next_chunk` is `max(recovered) + 1`. Using one format
for both paths is what keeps a single writer and a single reader.

## Recovery

`_read_completed_indices(progress_path) -> set[int]`:

1. Read the file. If absent, return the empty set.
2. Take `n = len(data) // 8` whole records; **discard any trailing
   partial record.** A crash mid-append can only ever leave a torn
   *last* record, because appends are 8 bytes written in one call and
   never rewritten.
3. Unpack the `n` indices into a set.

A torn tail is therefore not an error - it is the expected shape of a
crash - and it degrades to "that chunk was not recorded", which is
exactly the pre-existing contract: the chunk gets resubmitted and
recomputed.

**Ordering is unchanged and still crash-safe:** the frame is appended
first, the marker record second. Every intermediate state is handled:

| crash point | file state | recovery |
|---|---|---|
| mid-frame | torn frame, no record | frame reader stops at torn frame; chunk recomputed |
| after frame, before record | complete frame, no record | frame not in `valid_indices`; dropped and recomputed |
| mid-record | complete frame, torn record | partial record discarded; as above |
| after record | both complete | chunk replayed from the frame |

The third row is the one the new format adds, and it collapses onto
the second - which the current design already handles correctly.

## Duplicate records

A rollback-resume can record the same `chunk_index` twice (the file is
append-only and never compacted), exactly as frames can already be
duplicated - see the "Known consequence" section of the binary
chunk-framed checkpoint design. Recovery reads into a **set**, so
duplicate records collapse with no special handling. Growth is 8 bytes
per duplicate, against a frame's tens of kilobytes; it is not worth
compacting.

## Components

Three private functions in `fwht.py`, replacing the inline `json.dump`
calls at four sites:

- `_append_progress_record(progress_path, chunk_index) -> None` -
  one `open(..., "ab")`, one 8-byte write. Shared by both paths.
- `_read_completed_indices(progress_path) -> set[int]` - the recovery
  rule above. Shared by both paths.
- `_PROGRESS_RECORD: struct.Struct` - the format's single source of
  truth.

Call sites: `_append_parallel_checkpoint_chunk` and
`_append_checkpoint_chunk` (writes); `_load_parallel_checkpoint` and
`_load_checkpoint` (reads). The sequential loader derives its
`next_chunk` as `max(recovered) + 1`, or `0` when the set is empty.

Public signatures are unchanged: `checkpoint_path` on all five public
functions behaves identically.

## Testing

The 14 existing checkpoint tests are the regression suite and must
keep passing. Note that two of them currently hand-write a JSON
progress file (`tests/test_chunked_accumulator.py` writes
`json.dump({"next_chunk": 1}, f)` to force a partial-progress resume);
those fixtures must be rewritten to emit binary records, the same way
the JSONL payload fixtures were rewritten during the format change.

New tests:

- **Record round-trip**: append k indices, read back exactly that set.
- **Torn tail**, parameterized over truncation offsets 1-7 within the
  final record: every complete record recovered, the partial one
  dropped.
- **Empty and absent file**: both yield the empty set, neither raises.
- **Duplicate records collapse**: appending the same index twice
  recovers a single-element set.
- **Out-of-order records**: parallel workers complete out of order, so
  records are not sorted; recovery must not assume they are.
- **Sequential `next_chunk` derivation**: a set `{0,1,2}` yields
  `next_chunk == 3`; the empty set yields `0`.
- **Crash-point matrix**: the four rows of the ordering table above,
  each constructed by truncating a real checkpoint at that boundary.

Cost is verified by extending `progress_marker_cost.py` with the
append writer as a fourth arm, against the same N=150 shape; the
target is the measured ~16 us flat, versus the rewrite's rising
0.09-3.0 ms.

## Migration

None, and the filenames deliberately do not change. paulikit is
v0.1.0 and unreleased; no checkpoint files exist outside this repo's
own test runs, and the payload format they pair with was already
replaced in the same unreleased window. A stale JSON progress file
from an intermediate working tree would fail `len(data) // 8`
recovery into a nonsense index set, but every such index is validated
against the frames it must correspond to, so the run degrades to
"recompute everything" rather than producing wrong output.

## Scope

In scope: the three new functions, four call sites, the two
JSON-writing test fixtures, and the tests above.

Not in scope: compaction of duplicate records or frames (noted above
as not worth it); any change to the frame format itself; any change to
checkpoint granularity, which the measurement settled.
