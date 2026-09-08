# Binary chunk-framed checkpoint format

Design, 2026-09-08. Supersedes the JSONL checkpoint format used by
both the sequential and parallel decomposition paths.

## Why

Checkpointing is the last remaining place where per-term Python object
construction sits on a hot path. Phase 13 removed it from the drain
loop and recovered 2.058x multi-core scaling
(`profiling/phase13/arrays_vs_dict_findings.md`); the checkpoint
writer puts it straight back.

Three independent defects, all traceable to "one JSON object per
term":

**1. The writer is CPU-bound, not I/O-bound.**
`profiling/phase13/checkpoint_cost_attribution.py` falsified each
candidate cause separately, holding byte volume identical: removing
the disk entirely leaves **94.8%** of the cost, removing the
formatting leaves **1.1%**. Writing 41.7 MB costs 24 ms; producing
those bytes costs 2130 ms. The cost is `.tolist()` + a dict literal +
`json.dumps` per term, GIL-held in the parent between
`future.result()` and `yield` (`fwht.py:1991-1995`) - structurally the
same `d5` dict-build Phase 13 removed, in the same position.
Extrapolated to N=150: **~386s of single-threaded parent CPU** to
protect a 17.9s computation, with parallel scaling re-capped to ~1.0x
(measured 0.973x).

**2. Resume cannot run at N=150 at all.** `_read_checkpoint_triples`
(`fwht.py:248-272`) calls `f.readlines()` (~7.6 GB of Python `str`),
builds three full Python lists (~10.3 GB of `int`/`complex` objects),
then a `last_by_key` dedup dict (~11 GB more): **over 20 GB peak on a
15 GB machine.** The streaming memory contract (PLAN.md Phases 9/10/12,
208-627 MiB peak RSS) is violated by the *resume* path independently
of the writer's cost. Checkpointing at N=150 is not merely expensive
today - it is non-functional.

**3. It affects both execution modes.** `_append_checkpoint_chunk`
(`fwht.py:399-401`) carries the identical per-term loop as its
parallel counterpart and shares the same reader. The sequential path
is the constant-memory mode paulikit deliberately keeps offering, so
leaving it on JSONL would leave that mode broken at exactly the sizes
where resumability matters most.

None of this was caught earlier because **checkpointing had never been
exercised above n_qubits=4**: all 14 checkpoint tests run on
`ALL_FIXTURES` (dim=8 and dim=16), and no profiling script ever passed
`checkpoint_path` at N=150 - `drain_loop_dag_d_benchmark.py` documents
the exclusion in its header. The tests validate correctness (resume,
crash truncation, dedup, format non-collision) and continue to; they
were never designed to measure cost.

## What this guarantees

Checkpointing holds **bounded memory and small cost on both write and
resume, at every N the library supports** - the same contract the
streaming and parallel paths already meet. Checkpointing stops being a
scale cliff.

## Format

One file, appended per completed chunk. Each chunk is a self-
describing frame:

```
frame := header || x_bytes || z_bytes || coeff_bytes
header (fixed width, little-endian):
    magic       4s   b"PKCP"
    version     u16  format version, starts at 1
    idx_dtype   u8   code: 0=uint16, 1=uint32, 2=intp/int64
    reserved    u8   zero, alignment
    chunk_index u64  which chunk this frame records
    n_terms     u64  terms in this frame
```

Payload is three raw `ndarray.tobytes()` blocks: `x` and `z` at
`idx_dtype`, `coeff` at `complex128`. Sizes follow from `n_terms` and
`idx_dtype`, so a reader needs no separator scanning.

**Index dtype** comes from the existing `_index_dtype_for_dim(dim)`
(`fwht.py:1306`), reused rather than duplicated. Recording the code in
the header is a correctness requirement, not a size optimization: that
helper's own docstring notes `uint16` silently *wraps* above 16 qubits,
so a reader must never assume a width. A frame whose `idx_dtype` code
is unknown is a hard error.

**Coefficients stay `complex128`.** Narrowing them to `complex64` would
save 8 B/term but breaks the Hermiticity check, whose tolerance
(`max(atol, 1e-6*|c|)`, `fwht.py`'s `_check_hermitian_violation`)
assumes double precision. This trade was considered and rejected on
the same grounds during Phase 13's IPC narrowing.

Size at N=150: **2.20 GB** at `uint32` (24 B/term) versus 7.64 GB for
JSONL (83.4 B/term) - 3.5x smaller, and 2.93 GB in the `intp` fallback.

Progress marker keeps its current shape and filenames: the parallel
path records the *set* of completed chunk indices (workers finish out
of order), the sequential path one monotonic `next_chunk`. Both keep
their distinct suffixes so the two formats never collide.

## Crash safety, and why dedup disappears

Ordering is unchanged and still crash-safe: frames are appended first,
the progress marker updated only after. A crash mid-write leaves a
chunk unmarked, so it is recomputed on resume.

What changes is the recovery. Today a crash between the triples-write
and the marker update leaves duplicate lines for the same `(x, z)`,
which `_read_checkpoint_triples` repairs with the `last_by_key` dict -
the 11 GB structure, added for the "over-record resume" bug
(REVIEW_NOTES.md 2026-09-04). Framing makes that class of bug
**structurally impossible**: a reader validates each frame's magic and
declared length, and **truncates the file at the first frame that is
incomplete or not listed in the progress marker.** No duplicate can
survive to be read, so `last_by_key` is deleted outright rather than
ported.

This is strictly stronger than the current behaviour: JSONL can only
detect a truncated *final* line, and any earlier corruption raises.
Frame length + magic validation detects corruption anywhere.

### Known consequence: duplicate frames after a rollback-resume

Removing `last_by_key` removes read-time *repair*, not the condition
that created duplicates. Both writers append (`"a"`/`"ab"`), so a
resume whose progress marker has been rolled back below what the file
already holds recomputes those chunks and appends them again, leaving
duplicate chunk indices in non-monotonic order. Measured on a 4-qubit
fixture: 1216 bytes after a full run, then +872 bytes per rollback
cycle, without bound.

**This is inherited, not introduced** - the JSONL writer opened in
append mode too, so the same growth always occurred; only the reader's
dedup pass hid its effect on the replayed values. Results stay correct
either way, now via idempotence rather than deduplication: every
duplicate frame holds the same deterministic values for the same
`(x, z)`, so an accumulator absorbing one twice is unaffected. This
was verified directly - three consecutive rollback-resume cycles each
reproduced the reference decomposition exactly.

The cost is disk, not correctness, and it is bounded by how many times
a caller rolls a marker back - normally zero. Compaction on resume
(rewriting the file with only the frames being kept) is the obvious
fix if this ever matters; it is deliberately not in scope here, since
it trades a simple append for a read-modify-write on the path this
design exists to keep cheap.

## Components

Five private functions in `fwht.py`, replacing five existing ones:

- `_checkpoint_frame_header(...) -> bytes` / `_parse_checkpoint_frame_header(bytes)`
  - pure, total, independently testable; the format's single source of truth.
- `_append_checkpoint_frame(path, chunk_index, x, z, coeff, idx_dtype)`
  - one `open(..., "ab")`, one header write, three `.tobytes()` writes.
    No per-term Python. Shared by both paths - which is what preserves
    the documented interchangeability guarantee (`docs/tutorial.md:212`).
- `_iter_checkpoint_frames(path, valid_indices) -> Iterator[(idx, x, z, coeff)]`
  - `np.frombuffer` per frame, one frame live at a time. Stops at the
    first invalid/unmarked frame and truncates.
- `_load_parallel_checkpoint` / `_load_checkpoint` keep their names and
  signatures, reimplemented over `_iter_checkpoint_frames`.

The two public resume call sites (`fwht.py:1719-1727` parallel,
sequential equivalent) currently materialize the whole replay in one
`yield`. They become a loop over frames, yielding per frame - which is
what keeps resume inside the memory contract.

## Testing

The existing 14 checkpoint tests stay as-is and must keep passing:
they pin resume, crash truncation, dedup, format non-collision, and
cross-path interchangeability. That is the regression suite for this
change.

New tests:

- **Header round-trip** for every `idx_dtype` code, including the
  `intp` fallback; unknown code raises.
- **Frame round-trip**: arrays in, identical arrays and dtypes out.
- **Truncation recovery**, parameterized over truncation points: mid-
  header, mid-payload, and exactly at a frame boundary. Each must
  recover every complete marked frame and no partial one.
- **Corruption anywhere**: a byte flipped in an early frame's magic is
  detected (JSONL could not do this).
- **Unmarked-tail truncation**: a frame present in the file but absent
  from the progress marker is discarded, not replayed.
- **Bounded resume**: at a size large enough to matter, peak RSS during
  resume stays within the streaming contract rather than scaling with
  file size. This is the test that would have caught defect 2.
- **Cross-path interchange**: unchanged in intent, re-verified against
  the new format.

Cost is verified by extending `checkpoint_cost_attribution.py` with the
new writer as a fourth variant; its `no_format` row (0.024s/500K terms)
is the target, ~100x below the current writer.

## Scope

In scope: both writers, the shared reader, both resume call sites, and
the tests above. The public signature `checkpoint_path` is unchanged on
all five public functions.

Not in scope: whether per-chunk is the right checkpoint *granularity*
when a chunk is ~2s of work. Worth revisiting once the per-chunk cost
is ~100x lower, since the answer likely changes.

**Migration: none.** paulikit is v0.1.0 and unreleased; no checkpoint
files exist outside this repo's test runs. A `version` field is in the
header so a future change has somewhere to go, but v1 readers reject
anything else rather than guessing. Old JSONL checkpoints are not
readable by the new code and are not migrated - deliberately, since
any that exist are ephemeral test artifacts.
