# Append-Only Progress Marker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rewrite-everything progress marker with a fixed-width append-only one, so a checkpoint write costs O(1) per chunk instead of O(n), taking N=150 checkpoint overhead from 142% of runtime to ~11%.

**Architecture:** Each completed chunk appends one 8-byte little-endian `u64` record holding its chunk index. Recovery reads whole records into a set and discards any trailing partial record, which is the expected shape of a crash rather than corruption. Both the sequential and parallel paths use the same record format, one shared writer and one shared reader; the sequential path derives its `next_chunk` as `max(recovered) + 1`.

**Tech Stack:** Python 3.12, NumPy, `struct` (stdlib), pytest. Build via the project's own `make build` — never a bare `pip install -e .`, which bakes in an ephemeral build-isolation NumPy include path and breaks rebuilds.

**Spec:** `docs/superpowers/specs/2026-09-08-append-only-progress-marker-design.md`

## Global Constraints

- All work happens in `tools/paulikit/`. Paths below are relative to it.
- Branch is `checkpoint-marker-removal` (a misnomer kept for continuity — the marker stays, it just becomes cheap). Do not merge to `main`.
- `make test` takes NO arguments. Targeted runs: `make build && /home/desadm/.venvs/paulikit/bin/pytest <args>`. Full suite: `make test`.
- Record format is `struct.Struct("<Q")`, 8 bytes, little-endian `u64`. It is persisted on disk — never change its width or endianness.
- Recovery reads into a **set**. Records may be duplicated (rollback-resume) and are NOT sorted (parallel workers complete out of order). Never assume either.
- A trailing partial record is discarded, not an error. Appends are 8 bytes written in one call and never rewritten, so only the final record can ever be torn.
- Crash-safety ordering is unchanged: the payload frame is appended FIRST, the marker record SECOND.
- The 14 existing checkpoint tests must keep passing. Two of them hand-write a JSON progress file and will need their fixtures rewritten — see Task 4.
- No migration. paulikit is v0.1.0, unreleased; progress filenames and suffixes are unchanged.
- Commit after each task. Do not batch. Omit any `Co-Authored-By` or attribution trailer.
- Keep new code lines ≤ ~79 chars while writing, not as a retrofit pass.

---

## File Structure

- **`src/paulikit/algorithms/fwht.py`** (modify) — the marker code lives beside the checkpoint functions that use it, matching where the frame format already lives.
  - Add: `_PROGRESS_RECORD`, `_append_progress_record`, `_read_completed_indices`.
  - Modify: `_load_parallel_checkpoint` (`fwht.py:386`), `_append_parallel_checkpoint_chunk` (`fwht.py:426`), `_load_checkpoint` (`fwht.py:458`), `_append_checkpoint_chunk` (`fwht.py:494`).
  - Remove: the `json.dump` at `fwht.py:455` and `fwht.py:515`, and the `json.load` at `fwht.py:412` and `fwht.py:480`.
- **`tests/test_progress_marker.py`** (create) — unit tests for the record format and recovery rule, kept separate from the four decomposition-behaviour checkpoint test files.
- **`tests/test_chunked_accumulator.py`** (modify) — one fixture writes a JSON progress file directly.
- **`profiling/phase13/progress_marker_cost.py`** (modify) — add the append writer as a fourth arm to confirm the measured saving.

---

## Task 1: Record format and recovery

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py` (add after `_parallel_checkpoint_progress_path`, ends `fwht.py:216`)
- Test: `tests/test_progress_marker.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks. `struct` is not yet imported at module level for this purpose — it IS already imported (added with the frame header), so no new import is needed.
- Produces:
  - `_PROGRESS_RECORD: struct.Struct` with format `"<Q"`, size 8
  - `_append_progress_record(progress_path: str | Path, chunk_index: int) -> None`
  - `_read_completed_indices(progress_path: str | Path) -> set[int]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_progress_marker.py`:

```python
"""Tests for the append-only progress marker
(docs/superpowers/specs/2026-09-08-append-only-progress-marker-design.md).

These pin the record format and the recovery rule directly, separately
from the decomposition behaviour the four existing checkpoint test
files already cover.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _PROGRESS_RECORD,
    _append_progress_record,
    _read_completed_indices,
)


def test_record_is_8_bytes():
    assert _PROGRESS_RECORD.size == 8


def test_round_trip_recovers_exactly_the_appended_set(tmp_path):
    path = tmp_path / "p.bin"
    for i in (0, 1, 2, 7):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 2, 7}


def test_absent_file_is_the_empty_set(tmp_path):
    assert _read_completed_indices(tmp_path / "absent.bin") == set()


def test_empty_file_is_the_empty_set(tmp_path):
    path = tmp_path / "p.bin"
    path.write_bytes(b"")
    assert _read_completed_indices(path) == set()


def test_records_need_not_be_sorted(tmp_path):
    # Parallel workers complete out of order, so the file is not
    # sorted. Recovery must not assume it is.
    path = tmp_path / "p.bin"
    for i in (5, 0, 3, 1):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 3, 5}


def test_duplicate_records_collapse(tmp_path):
    # A rollback-resume can record the same chunk twice; the file is
    # append-only and never compacted. Reading into a set handles it.
    path = tmp_path / "p.bin"
    for i in (2, 2, 2):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {2}


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 7])
def test_torn_final_record_is_discarded(tmp_path, cut):
    # A crash mid-append can only ever tear the LAST record, because
    # appends are 8 bytes in one call and never rewritten. Every
    # complete record must survive; the partial one must not appear.
    path = tmp_path / "p.bin"
    for i in (0, 1, 2):
        _append_progress_record(path, i)
    raw = path.read_bytes()
    path.write_bytes(raw[:-cut])
    assert _read_completed_indices(path) == {0, 1}


def test_large_chunk_index_survives(tmp_path):
    # u64, matching the width the frame header uses for chunk_index,
    # so the two cannot disagree about range.
    path = tmp_path / "p.bin"
    _append_progress_record(path, 2**40)
    assert _read_completed_indices(path) == {2**40}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_progress_marker.py -v`
Expected: FAIL at import — `cannot import name '_PROGRESS_RECORD'`

- [ ] **Step 3: Write the implementation**

Add below `_parallel_checkpoint_progress_path` (ends `fwht.py:216`):

```python
# Persisted on disk: one record per completed chunk. u64 matches the
# width the frame header already uses for chunk_index, so the two
# cannot disagree about range. Never change the width or endianness.
_PROGRESS_RECORD = struct.Struct("<Q")


def _append_progress_record(
    progress_path: str | Path, chunk_index: int
) -> None:
    """Record one completed chunk by appending a fixed-width record.

    This replaced a ``json.dump`` of the whole completed set on every
    chunk, which was O(n) per chunk and so O(n^2) across a run. At
    N=150's 5,595 chunks that marker cost 15.13x the payload frame
    write by the final chunk and ~91% of the per-chunk checkpoint
    total (profiling/phase13/progress_marker_findings.md). An 8-byte
    append is O(1) - measured flat at ~13-21us across a 20,000x range
    of completed counts.
    """
    with open(progress_path, "ab") as f:
        f.write(_PROGRESS_RECORD.pack(chunk_index))


def _read_completed_indices(progress_path: str | Path) -> set[int]:
    """Recover the set of completed chunk indices.

    Reads whole records only: a trailing partial record is DISCARDED
    rather than treated as corruption. Appends are 8 bytes written in
    one call and never rewritten, so only the final record can ever be
    torn, and a torn tail means exactly "that chunk was not recorded" -
    the chunk is resubmitted and recomputed, which is the pre-existing
    contract.

    Returns a set, not a sequence: records are not sorted (parallel
    workers complete out of order) and may be duplicated (a
    rollback-resume re-records chunks it recomputes).
    """
    progress_path = Path(progress_path)
    if not progress_path.exists():
        return set()
    data = progress_path.read_bytes()
    size = _PROGRESS_RECORD.size
    n_whole = len(data) // size
    return {
        _PROGRESS_RECORD.unpack_from(data, i * size)[0]
        for i in range(n_whole)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_progress_marker.py -v`
Expected: PASS, 14 tests (the torn-record test parametrizes to 7)

- [ ] **Step 5: Commit**

```bash
git add tests/test_progress_marker.py src/paulikit/algorithms/fwht.py
git commit -m "feat(paulikit): add append-only progress marker format and recovery"
```

---

## Task 2: Port the parallel path

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py:386-424` (`_load_parallel_checkpoint`), `fwht.py:426-456` (`_append_parallel_checkpoint_chunk`)
- Test: `tests/test_parallel_decompose.py`, `tests/test_array_yielding.py` (existing, unchanged)

**Interfaces:**
- Consumes: `_append_progress_record`, `_read_completed_indices` (Task 1).
- Produces: both function signatures are UNCHANGED. Only their bodies change. `_load_parallel_checkpoint` still returns `(set[int], Iterator[tuple[NDArray, NDArray, NDArray]] | None)`; `_append_parallel_checkpoint_chunk` still takes `(checkpoint_path, completed_chunk_indices, chunk_index, x_out, z_out, coeff_out, idx_dtype)`.

- [ ] **Step 1: Run the existing parallel checkpoint tests to record the starting state**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_parallel_decompose.py tests/test_array_yielding.py -v -k checkpoint`
Expected: PASS. These exercise the JSON marker about to be replaced and must still pass afterward.

- [ ] **Step 2: Replace the marker read in `_load_parallel_checkpoint`**

Replace the block at `fwht.py:410-414` — currently:

```python
    with open(progress_path) as f:
        progress = json.load(f)
    completed = set(progress["completed_chunk_indices"])
```

with:

```python
    completed = _read_completed_indices(progress_path)
```

Leave the surrounding `if not checkpoint_path.exists() or not progress_path.exists(): return set(), None` guard and the `if not completed: return completed, None` guard exactly as they are.

- [ ] **Step 3: Replace the marker write in `_append_parallel_checkpoint_chunk`**

Replace the tail of the function at `fwht.py:453-455` — currently:

```python
    progress_path = _parallel_checkpoint_progress_path(checkpoint_path)
    with open(progress_path, "w") as f:
        json.dump({"completed_chunk_indices": sorted(completed_chunk_indices)}, f)
```

with:

```python
    progress_path = _parallel_checkpoint_progress_path(checkpoint_path)
    _append_progress_record(progress_path, chunk_index)
```

Keep `completed_chunk_indices.add(chunk_index)` immediately before it: the in-memory set is still the live view used by the caller within a single run.

- [ ] **Step 4: Run the parallel checkpoint tests**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_parallel_decompose.py tests/test_array_yielding.py -v`
Expected: PASS — including `test_checkpoint_written_by_dict_path_resumes_under_array_path` and its mirror, which pin cross-path interchangeability.

- [ ] **Step 5: Commit**

```bash
git add src/paulikit/algorithms/fwht.py
git commit -m "feat(paulikit): port the parallel progress marker to append-only records"
```

---

## Task 3: Port the sequential path

**Files:**
- Modify: `src/paulikit/algorithms/fwht.py:458-492` (`_load_checkpoint`), `fwht.py:494-516` (`_append_checkpoint_chunk`)
- Test: `tests/test_progress_marker.py` (extend), `tests/test_streaming.py` (existing, unchanged)

**Interfaces:**
- Consumes: `_append_progress_record`, `_read_completed_indices` (Task 1).
- Produces: both signatures UNCHANGED. `_load_checkpoint` still returns `(int, Iterator[...] | None)`; `_append_checkpoint_chunk` still takes `(checkpoint_path, next_chunk, x_out, z_out, coeff_out, idx_dtype)`.

The sequential path stored a single monotonic `next_chunk`. It now uses the same records as the parallel path and derives `next_chunk` from them.

- [ ] **Step 1: Write the failing test for next_chunk derivation**

Append to `tests/test_progress_marker.py`:

```python
def test_sequential_next_chunk_is_max_plus_one(tmp_path):
    # The sequential path completes chunks in order, so the recovered
    # set is contiguous and next_chunk is max + 1. It shares the
    # parallel path's record format so both use one writer/reader.
    from paulikit.algorithms.fwht import _load_checkpoint

    ckpt = tmp_path / "c.bin"
    prog = tmp_path / "c.bin.progress.json"
    # A payload frame must exist for the checkpoint to be considered
    # present; its contents do not matter to next_chunk derivation.
    from paulikit.algorithms.fwht import _append_checkpoint_frame

    for i in (0, 1, 2):
        _append_checkpoint_frame(
            ckpt, i,
            np.array([i], dtype=np.uint16),
            np.array([i], dtype=np.uint16),
            np.array([1 + 0j], dtype=complex),
            np.dtype(np.uint16),
        )
        _append_progress_record(prog, i)

    next_chunk, frames = _load_checkpoint(ckpt)
    assert next_chunk == 3
    assert len(list(frames)) == 3


def test_sequential_next_chunk_is_zero_when_nothing_recorded(tmp_path):
    from paulikit.algorithms.fwht import _load_checkpoint

    ckpt = tmp_path / "c.bin"
    ckpt.write_bytes(b"")
    (tmp_path / "c.bin.progress.json").write_bytes(b"")
    next_chunk, frames = _load_checkpoint(ckpt)
    assert next_chunk == 0
    assert frames is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_progress_marker.py -k next_chunk -v`
Expected: FAIL — `_load_checkpoint` still calls `json.load` on a binary file, raising `UnicodeDecodeError` or `JSONDecodeError`.

- [ ] **Step 3: Replace the marker read in `_load_checkpoint`**

Replace the block at `fwht.py:478-486` — currently:

```python
    with open(progress_path) as f:
        progress = json.load(f)
    next_chunk = progress["next_chunk"]
    if next_chunk <= 0:
        return 0, None
```

with:

```python
    completed = _read_completed_indices(progress_path)
    if not completed:
        return 0, None
    # Sequential chunks complete strictly in order, so the recovered
    # set is contiguous and the resume point is one past its maximum.
    next_chunk = max(completed) + 1
```

The `_frames()` closure below it already filters on `valid_indices=set(range(next_chunk))`; leave it unchanged.

- [ ] **Step 4: Replace the marker write in `_append_checkpoint_chunk`**

Replace the tail at `fwht.py:513-515` — currently:

```python
    progress_path = _checkpoint_progress_path(checkpoint_path)
    with open(progress_path, "w") as f:
        json.dump({"next_chunk": next_chunk}, f)
```

with:

```python
    progress_path = _checkpoint_progress_path(checkpoint_path)
    # next_chunk is the COUNT of completed chunks, so the chunk just
    # written carries index next_chunk - 1 - matching the index passed
    # to _append_checkpoint_frame immediately above.
    _append_progress_record(progress_path, next_chunk - 1)
```

- [ ] **Step 5: Run the sequential tests**

Run: `make build && /home/desadm/.venvs/paulikit/bin/pytest tests/test_progress_marker.py tests/test_streaming.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/paulikit/algorithms/fwht.py tests/test_progress_marker.py
git commit -m "feat(paulikit): port the sequential progress marker to append-only records"
```

---

## Task 4: Fix the JSON-writing test fixture and clear the full suite

> **COMPLETED DURING TASKS 2-3, no separate commit.** This task was
> under-scoped: it named only `test_chunked_accumulator.py`, but FOUR
> fixtures were coupled to the marker's on-disk format. Each was fixed
> in the task whose change broke it, so the suite stayed green at every
> boundary:
> - `test_parallel_decompose.py::test_parallel_decompose_checkpoint_resume` (Task 2)
> - `test_array_yielding.py::test_resume_replay_still_checks_hermiticity` (Task 2) - had failed SILENTLY: hand-written JSON parsed as zero valid records, so the code path under test never ran and it reported "DID NOT RAISE" rather than a parse error
> - `test_chunked_accumulator.py::test_checkpoint_resume_from_partial_progress_file` (Task 3)
> - `test_checkpoint_format.py::test_sequential_resume_replays_per_chunk_not_one_combined_tile` (Task 3)
>
> Verified after Task 3: no `json.dump`/`json.load`/hand-written
> progress file remains anywhere in `tests/`, no `import json` remains
> in any test file, and the suite is green at 212.


**Files:**
- Modify: `tests/test_chunked_accumulator.py:107-128`
- Test: full suite

**Interfaces:**
- Consumes: `_append_progress_record` (Task 1), and the ported paths (Tasks 2-3).
- Produces: nothing new.

`tests/test_chunked_accumulator.py::test_checkpoint_resume_from_partial_progress_file` hand-writes `json.dump({"next_chunk": 1}, f)` to force a rollback-resume. That format no longer exists.

- [ ] **Step 1: Run the full suite to see exactly what breaks**

Run: `make test`
Expected: failures confined to fixtures that hand-write a JSON progress file. Read each failure rather than assuming which; the plan's list may be incomplete, as it proved to be during the frame-format change.

- [ ] **Step 2: Rewrite the fixture**

In `test_checkpoint_resume_from_partial_progress_file`, replace:

```python
    with open(progress_path, "w") as f:
        json.dump({"next_chunk": 1}, f)
```

with:

```python
    # Roll the marker back to "only chunk 0 completed" by rewriting the
    # append-only record file with a single record. The payload frames
    # are deliberately left intact: over-recording is safe, and the
    # resumed run recomputes and re-appends the later chunks on top.
    from paulikit.algorithms.fwht import _PROGRESS_RECORD

    progress_path.write_bytes(_PROGRESS_RECORD.pack(0))
```

Keep the surrounding comment about on-crash behaviour and the assertions unchanged — the test's intent is unaltered.

- [ ] **Step 3: Fix any other fixture the run surfaced**

Apply the same treatment to any additional test the Step 1 run reported: replace a hand-written JSON progress file with `_PROGRESS_RECORD.pack(...)` records, preserving the test's original intent and assertions. If a test cannot be adapted because the behaviour it pins no longer exists, STOP and report rather than deleting it.

- [ ] **Step 4: Run the full suite**

Run: `make test`
Expected: PASS. The suite was at 196 before this plan; expect 196 plus Task 1's 14 and Task 3's 2 new tests.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(paulikit): rewrite JSON progress-file fixtures for the record format"
```

---

## Task 5: Confirm the measured saving

**Files:**
- Modify: `profiling/phase13/progress_marker_cost.py`
- Modify: `profiling/phase13/progress_marker_findings.md`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: the evidence that the change delivered what the design predicted.

- [ ] **Step 1: Add the append arm to the benchmark**

In `progress_marker_cost.py`, add beside `time_marker`:

```python
def time_marker_append(path, _x, _z, _coeff, _idx_dtype, completed):
    """The append-only marker that replaced the rewrite."""
    progress_path = _parallel_checkpoint_progress_path(path)
    t0 = time.perf_counter()
    _append_progress_record(progress_path, len(completed))
    return time.perf_counter() - t0
```

Import `_append_progress_record` alongside the existing imports, and add an `append` column to the sweep table beside `marker`, reporting `marker/append` per row.

**Measurement integrity — do not skip:** build the record file ONCE per `k`, outside the timed region. Rebuilding it per rep dirties page cache immediately before the timed write and produced a false "rising, O(1) falsified" result during design (`feedback_isolate_before_falsifying`). The append must come out flat in `k`.

- [ ] **Step 2: Run it**

Run: `OPENBLAS_NUM_THREADS=1 python profiling/phase13/progress_marker_cost.py 25`

Expected: the `append` column flat across every `k` (spread under ~3x), around 13-21 us, against the rewrite column's rising 0.09-3.0 ms.

**Stop condition:** if `append` rises materially with `k`, do not record a number. First move all per-rep setup outside the timer and re-measure; only if it still rises is the premise wrong, and then STOP and report.

- [ ] **Step 3: Record the result**

Append a "Measured after the change" section to `progress_marker_findings.md` with the actual table, the flatness check, and the recomputed full-run projection. Report only numbers the run produced; label any extrapolation as such. Leave the existing sections intact — they are the motivation and remain correct.

- [ ] **Step 4: Commit**

```bash
git add profiling/phase13/progress_marker_cost.py profiling/phase13/progress_marker_findings.md
git commit -m "profiling(paulikit): confirm the append-only marker is O(1) in practice"
```

---

## Task 6: Documentation sweep

**Files:**
- Modify: docstrings of `_load_parallel_checkpoint`, `_append_parallel_checkpoint_chunk`, `_load_checkpoint`, `_append_checkpoint_chunk` in `src/paulikit/algorithms/fwht.py`
- Modify: `parallel_decompose`'s `checkpoint_path` docstring
- Modify: `PLAN.md`

- [ ] **Step 1: Update the four private docstrings**

Each currently describes a JSON progress file. State the record format instead: one 8-byte `u64` per completed chunk, appended; recovery reads whole records into a set and drops a torn tail.

- [ ] **Step 2: Correct `parallel_decompose`'s parameter docstring**

It says the parallel path records "the *set* of completed chunk indices rather than one monotonic marker". After Task 3 both paths use the same record format; the difference is only that the sequential path derives `next_chunk = max(recovered) + 1` because its chunks complete in order. Say that.

- [ ] **Step 3: Add the PLAN.md entry**

Add to the end of the Phase 13 material (before section 6, "Explicitly out of scope"): the motivation (marker was ~91% of per-chunk checkpoint cost, 15.13x the frame write at N=150's 5,595 chunks), the change, and the measured outcome from Task 5. Note that checkpoint granularity is now a closed question.

- [ ] **Step 4: Verify no stale JSON references remain**

Run: `grep -rn --include="*.py" --include="*.md" -i "completed_chunk_indices\|next_chunk" src/ docs/tutorial.md README.md PLAN.md`
Expected: matches only in past-tense design/plan documents under `docs/superpowers/`, and in code where `next_chunk` remains a live local variable in `_load_checkpoint`/`_append_checkpoint_chunk`. No claim that either is stored as JSON.

- [ ] **Step 5: Build the docs**

Run: `make docs`
Expected: builds with no new warnings (4 pre-existing toctree notices). Run it in the FOREGROUND and let it finish.

- [ ] **Step 6: Commit**

```bash
git add src/paulikit/algorithms/fwht.py PLAN.md
git commit -m "docs(paulikit): document the append-only progress marker"
```

---

## Task 7: Final verification

**Files:** none modified unless a defect is found.

- [ ] **Step 1: Full suite from a clean build**

Run: `make build && make test`
Expected: PASS

- [ ] **Step 2: Confirm no JSON marker writes remain**

Run: `grep -n "json.dump\|json.load" src/paulikit/algorithms/fwht.py`
Expected: no matches in checkpoint code. `json` may remain imported only if something else uses it — if nothing does, remove the import.

- [ ] **Step 3: Confirm one shared writer and one shared reader**

Run: `grep -n "_append_progress_record\|_read_completed_indices" src/paulikit/algorithms/fwht.py`
Expected: one definition each, two call sites each — parallel and sequential.

- [ ] **Step 4: Report for the merge decision**

Do NOT merge or push. Summarize: final test count, the measured append flatness, the recomputed N=150 overhead, and anything that deviated from this plan.

---

## Deferred

**Compaction of duplicate records or frames.** A rollback-resume appends 8 bytes per duplicate record, against a frame's tens of kilobytes. Not worth a read-modify-write on the path this design exists to keep cheap.

**End-to-end checkpointed run above n_qubits=4.** Checkpointing has never been exercised above 4 qubits: all 14 checkpoint tests use `ALL_FIXTURES` (dim=8, dim=16), and no profiling script has passed `checkpoint_path` at N=150. Both halves are now measured in isolation, but the integrated path is untested at scale — which is how the >20 GB resume defect survived. Worth its own task after this plan lands.
