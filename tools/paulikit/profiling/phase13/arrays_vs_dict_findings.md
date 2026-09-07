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
| `arrays_with_checkpoint` | 298.174s | 305.359s | 0.976x | 7.5e-02 |

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

## The checkpoint row is NOT a valid measurement - do not cite it

The `arrays_with_checkpoint` numbers above are recorded for
completeness but **must not be read as the cost of checkpointing.**

`_append_parallel_checkpoint_chunk` writes one JSON line per surviving
term - ~91.6M lines at N=150. The sweep's `tempfile.mkdtemp()` places
that under `/tmp`, which on this machine is a **7.7 GB RAM-backed
tmpfs**, not disk. The runs were therefore writing multiple gigabytes
into memory: a partial run was measured at 4.55 GB before a subsequent
attempt failed outright with `OSError: [Errno 122] Disk quota
exceeded`. Those timings measure memory exhaustion and tmpfs pressure
at least as much as checkpoint I/O.

An earlier reading of this data as "~280s of checkpoint cost, 50
ms/chunk" was stated before the tmpfs issue was noticed, and is
**retracted**. The real cost of checkpointing is currently UNMEASURED.

The cell is also incomplete (5 reps at `w2_c1`, 2 at `w8_c4`) because
the sweep was stopped mid-run.

## What is still open

1. **The cost of per-chunk checkpointing is unmeasured.** Re-running
   must direct the checkpoint to real disk (`/home` had 29 GB free),
   never `/tmp`, and should budget for a file of ~91.6M JSON lines.
2. **Inherited or introduced?** A `dict_with_checkpoint` control was
   added to `arrays_vs_dict_target.py` and `arrays_vs_dict_sweep.py`
   for exactly this question but has NOT been run. Both variants call
   the same unmodified `_append_parallel_checkpoint_chunk`, so a
   similar magnitude there would show the cost is pre-existing and
   unrelated to this phase's change. Until that runs, no claim should
   be made either way.
3. A cheaper route to (2) than a 30-run campaign: measure the
   checkpoint writer directly at small N. It is shared, unmodified
   code, so inherited-vs-introduced can be settled without a full
   thermal-controlled sweep.

## Scope

This measures the drain-side change only. It does not evaluate
`terms_from_arrays`'s own cost when a caller renders every term - that
is by construction the same work the dict path already does, and the
saving comes from rendering a subset. See the design doc,
`docs/superpowers/specs/2026-09-07-array-yielding-parallel-decompose-design.md`.
