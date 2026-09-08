# Array-yielding vs dict-yielding parallel decomposition at N=150

Recorded 2026-09-07. Measures whether the shipped
`parallel_decompose_arrays` actually delivers the multi-core scaling
that a controlled drain-loop experiment predicted
(`drain_gil_backpressure_results.jsonl`: 2.191x for an arrays-only
drain loop versus 0.865x with label+dict work).

Protocol: N=150, `chunk_size=2`, cooldown to 55C package temp before
every timed run, 5 reps per cell, Welch's unequal-variance t-test on
the `w2_c1` vs `w8_c4` contrast within each cell. Raw data:
`arrays_vs_dict_results.jsonl` (27 runs).

## Result: the fix works end to end

| variant | w2_c1 | w8_c4 | speedup | Welch p |
|---|---|---|---|---|
| `dict` (existing `parallel_decompose`) | 20.711s | 23.540s | **0.880x** | 7.8e-04 |
| `arrays_no_checkpoint` | 17.967s | **8.729s** | **2.058x** | 3.1e-05 |
| `arrays_with_checkpoint` | 298.174s | 306.317s | 0.973x | 1.8e-02 |

**The controlled experiment's 2.191x survived in the real shipped
pipeline: 2.058x measured, highly significant.** The existing dict
path reproduced its known anti-scaling (0.880x - adding workers makes
it *slower*), so both the problem and the fix reproduce on the real
API, not just in a synthetic control.

In absolute terms the array path finishes N=150 at `w8_c4` in
**8.729s versus the dict path's 23.540s - 2.70x faster wall-clock**,
with a standard deviation of 0.059s across 5 reps.

Peak RSS is unchanged in kind (252-261 MiB at `w2_c1`, 571-588 MiB at
`w8_c4` across all variants), so the streaming memory contract is
preserved - the array path does not trade memory for speed.

## Correctness

**27 of 27 completed runs produced exactly 91,652,096 terms**, across
every variant and both conditions. The array path loses no terms,
duplicates none, and agrees with the dict path's own count. This gate
was checked before any timing was interpreted.

## The checkpoint row: read the absolute numbers with care

The `arrays_with_checkpoint` cell was run with the checkpoint file
under `tempfile.mkdtemp()`, i.e. `/tmp`, which on this machine is a
**7.7 GB RAM-backed tmpfs**, not disk. At ~83.4 bytes per surviving
term the N=150 payload is ~7.6 GB, so these runs pushed gigabytes into
RAM; a *separate* smoke-test run failed outright with `OSError:
[Errno 122] Disk quota exceeded`. The absolute timings are therefore
perturbed by tmpfs memory pressure and should not be quoted to three
digits.

They are not, however, garbage. All nine completed runs produced
exactly 91,652,096 terms, and the spread is tight (297.3-309.2s,
sd ~1s at `w2_c1`). An earlier retraction of this row as "measuring
memory exhaustion, not checkpoint cost" was **too strong**: the
mechanism was misattributed, not the magnitude. See below.

The cell is incomplete (5 reps at `w2_c1`, 4 at `w8_c4`) because the
sweep was stopped mid-run.

## What the checkpoint cost actually is: CPU, not I/O

Settled 2026-09-08 by `checkpoint_cost_attribution.py`, which
falsifies each candidate cause independently at small N (the writer is
shared, unmodified code, so this needs no thermal sweep). Byte volume
is held identical across variants by assertion.

| variant | 500K terms | 2M terms | removes |
|---|---|---|---|
| `full` (real writer, real disk) | 2.247s | 8.644s | - (control) |
| `no_io` (same serialization -> `/dev/null`) | 2.130s | 8.423s | **I/O** |
| `no_format` (pre-serialized bytes -> disk) | **0.024s** | **0.063s** | **CPU** |
| `raw_bytes` (both removed) | 0.008s | 0.027s | both (floor) |

**Removing the disk entirely leaves 94.8% of the cost. Removing the
formatting leaves 1.1%.** Writing 41.7 MB costs 24 ms; *producing*
those same bytes costs 2130 ms - 89x more. Linear in term count
(4.259 -> 4.211 us/term at 4x), so extrapolation is sound.

The cause is `_append_parallel_checkpoint_chunk` (`fwht.py:337-339`):
per surviving term it does three `.tolist()` materializations into
Python objects, a dict literal, and a `json.dumps` call - all CPython
bytecode, **GIL-held, in the parent's drain loop between
`future.result()` and `yield`** (`fwht.py:1991-1995`). That is
structurally the same `d5` dict-build, in the same position, that this
phase removed to get 2.058x. Predicted parent-side serialization CPU
at N=150 is **386s**, which brackets the observed +280s gap from
above - as it should, since a real run overlaps some of it with worker
compute.

This also settles **inherited vs introduced without running the
`dict_with_checkpoint` control**: the cost lives entirely in the
shared, unmodified writer that both paths call, so it is
**pre-existing**, not introduced by the array path.

## The redesign, measured

Recorded 2026-09-08, after JSONL was replaced by the binary
chunk-framed format (24-byte header + three `tobytes()` blocks per
completed chunk; see
`docs/superpowers/specs/2026-09-08-binary-chunk-framed-checkpoint-design.md`).
Same script, same machine, same `~/.paulikit_ckpt_bench` real-disk
directory, with a fifth `frames` variant added.

| variant | 500K terms (5 reps) | 2M terms (3 reps) |
|---|---|---|
| `full` (original JSONL writer, real disk) | 2.043s (sd 0.071) | 8.651s (sd 0.199) |
| `no_io` (same serialization -> `/dev/null`) | 2.035s (sd 0.082) | 8.553s (sd 0.203) |
| `no_format` (pre-serialized bytes -> disk) | 0.025s (sd 0.012) | 0.053s (sd 0.002) |
| `raw_bytes` (both removed) | 0.008s (sd 0.001) | 0.028s (sd 0.001) |
| **`frames` (binary writer, real disk)** | **0.004s (sd 0.001)** | **0.018s (sd 0.002)** |

**The binary writer is 511x cheaper than JSONL at 500K terms and 481x
cheaper at 2M terms.** It lands *below* `no_format`, the pre-serialized
lower bound for the JSONL byte volume, which is expected: it writes
24 bytes/term against JSONL's measured 83.4, so it moves 3.5x fewer
bytes as well as doing no per-term Python work at all. Per-term cost is
0.0080 us at 500K and 0.0090 us at 2M - flat enough that the cost is
linear in term count, so the per-term figure may be extrapolated. The
control row is the *original* JSONL body, inlined into the script,
because the shared writer it used to call is the thing that changed;
inlining is what keeps `full` a control rather than a second copy of
the treatment.

The `full` control reproduced its earlier attribution unchanged:
removing the disk still leaves 98.9-99.6% of the cost, removing the
formatting still leaves 0.6-1.2%. The mechanism diagnosis above stands;
this section only records what removing that mechanism bought.

*Extrapolation, labelled as such:* at the measured 0.0090 us/term, the
N=150 checkpoint write would be **~0.8s** of parent CPU against the
JSONL format's ~392s extrapolated from the same run. That figure has
not been measured at N=150 and item 2 below still applies - no
checkpoint path has been exercised above n_qubits=4 in the test suite.

**Regression guard:**
`tests/test_checkpoint_format.py::test_resume_memory_is_bounded_by_chunk_not_file`
writes 200 frames and asserts the reader's `tracemalloc` peak stays
under 25% of the file size, which fails loudly for any reader that
materializes more than one frame at a time - the defect the JSONL
`readlines()` reader had.

## What is still open

1. ~~The checkpoint format needs redesign.~~ **Done - the format was
   replaced and the redesign measured.** See "The redesign, measured"
   below.
2. **Checkpointing has never been exercised above n_qubits=4.** All 14
   checkpoint tests across `test_parallel_decompose.py`,
   `test_streaming.py`, `test_chunked_accumulator.py` and
   `test_array_yielding.py` run on `ALL_FIXTURES`, which is dim=8
   (3 qubits) and dim=16 (4 qubits). No profiling script ever passed
   `checkpoint_path` at N=150 either - `drain_loop_dag_d_benchmark.py`
   documents the exclusion in its header. The tests validate
   correctness (resume, crash truncation, dedup, format
   non-collision); they were never designed to say anything about
   cost, and nothing else did. This sweep was the first time the path
   met 91.6M terms.
3. If the absolute N=150 checkpoint number is ever wanted, re-run to
   real disk (`/home` had 29 GB free), never `/tmp`, budgeting ~7.6 GB
   for the current format.

## Scope

This measures the drain-side change only. It does not evaluate
`terms_from_arrays`'s own cost when a caller renders every term - that
is by construction the same work the dict path already does, and the
saving comes from rendering a subset. See the design doc,
`docs/superpowers/specs/2026-09-07-array-yielding-parallel-decompose-design.md`.
