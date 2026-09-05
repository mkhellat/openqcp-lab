# DAG D identified: the main-process drain loop is a real, previously-unmodeled node — it explains part, not all, of the multi-worker slowdown (2026-09-05)

**Relationship to the DAG analysis, stated precisely.**
`dag_gst_master_analysis.md`'s original "DAG C" (written 2026-09-04)
named this class of mechanism in prose — a bulleted list including
"main-process drain loop... serializes result handoff" — but that
was never actually built as a DAG: no nodes, no edges, no Work(C), no
Span(C), no parallelism number. The GST bound violation in that
document's Section 3a was an *inference* that something outside
DAG A/B must be responsible, not a *derivation* from a modeled graph.
That gap was directly flagged by the user and has now been fixed: see
`dag_gst_master_analysis.md` **Section 3f (added 2026-09-05)**, which
builds the drain loop as an actual DAG node ("DAG D") with measured
Work, derived from the evidence in this document (py-spy stacks,
strace syscall summary) plus a direct microbenchmark of
`_pauli_label_batch` at the real per-chunk term count.

**Section 3f's own finding, restated here so this document does not
overclaim on its own**: DAG D's measured cost (~5.0 s total serial
floor across all 5595 chunks) explains only **part** of the gap —
at `w8_c4`, the corrected GST bound (8.28 s) still falls 15.7 s
(65%) short of the observed 24.0 s. The remainder is real but NOT
yet reduced to a modeled DAG — see Section 3f's "DAG C residual" for
the honest accounting. **This document's own conclusion below has
been corrected to match that** (see "Conclusion", not "the ceiling is
fully explained by the drain loop").

That prior document also explicitly and correctly ruled out a race
(see its own Section 0: "What was actually found is NOT a race - it
is slow-but-correct serialization... No data corruption, no
nondeterministic wrong answers - a performance pathology, not a
correctness bug" — nothing in this session's work changes that
conclusion).

**On the earlier "ideal parallelism ~5595 (outer DAG A) / ~1170 per
row (inner DAG B)" figures**: these were never a claim that the real
run achieves that parallelism. `dag_gst_master_analysis.md` Section 2
states outright that these are "what Master + GST *would* predict
**if the model held**," presented in the same section as "Observed:
speedup T1/T8 ≈ 1.10." Those figures are the theoretical ceiling of
the math, not a description of the real run, and are not contradicted
by anything here.

**This document's own analysis went through three revisions before
reaching this framing**, each caught by direct user pushback rather
than self-corrected in advance — preserved below for the
investigation's own audit trail:
1. "Hardware core/SMT oversubscription" — wrong, contradicted by
   `traffic_intensity_findings.md`'s same-machine near-linear
   synthetic-control results.
2. "Off-CPU time, paulikit-specific, cause unidentified" — correct as
   far as it went, but stopped short of naming the cause despite the
   needed evidence (py-spy + strace) already sitting in this
   directory.
3. "The drain loop fully explains the ceiling" — the mechanism is
   real and now has a genuine DAG-node model (Section 3f), but it was
   overclaimed as complete; the corrected accounting shows it
   explains ~35% of the `w8_c4` gap, not all of it.

## The direct evidence: py-spy caught the worker mid-block, in the act

`code_specificity_capture` (an earlier session, this same directory)
already ran `py-spy dump --pid <worker>` three times, 3s apart, for
both `w1_c1` and `w8_c4`, on the real target. This was never
previously cross-referenced against the perf capture. It should have
been the first thing checked before drawing any conclusion from
`perf record cycles`, since py-spy directly names what a thread is
doing (or blocked on) at the moment of the sample — it isn't limited
to on-CPU time the way `perf record cycles` is.

**w1_c1 (`pyspy_w1_c1.txt`), all 3 dumps: thread state `(active)` /
`(active+gil)`** — doing real work every time it was sampled:
`_walsh_hadamard_transform_rows` (paulikit/algorithms/fwht.py:162/164)
twice, and `dumps`/`put`/`_sendback_result` (serializing a finished
result) once.

**w8_c4 (`pyspy_w8_c4.txt`), all 3 dumps: thread state `(idle)`,
every single time**, inside the identical call stack:

```
__enter__ (multiprocessing/synchronize.py:95)
put (multiprocessing/queues.py:398)
_sendback_result (concurrent/futures/process.py:220)
_process_worker (concurrent/futures/process.py:270)
```

`multiprocessing/synchronize.py:95` is `Lock.__enter__` —
`multiprocessing.Queue.put()` acquires an internal write lock before
pushing onto the queue's pipe. The `w8_c4` worker was caught, 3 times
out of 3, **blocked trying to acquire that lock to hand its finished
chunk back to the main process** — not computing anything.

## Corroborating quantitative evidence: strace syscall summary

`strace_w1_c1.txt` / `strace_w8_c4.txt` (same earlier capture,
`strace -f -c` over the whole process tree):

| syscall | w1_c1 (% time) | w8_c4 (% time) |
|---|---|---|
| `futex` | 44.83% | 49.32% |
| `poll` | 8.21% | 16.81% |
| `wait4` | 21.68% | 23.73% |

`futex` (the syscall behind `multiprocessing.Lock`/pipe synchronization)
is already the dominant syscall even at `w1_c1` (bounded submission's
own `wait(FIRST_COMPLETED)` plus checkpoint I/O use futexes too), but
`poll` **doubles** from w1_c1 to w8_c4 — consistent with more
processes contending to service the same result pipe/queue.

## Why the earlier perf-only conclusions were incomplete

`perf record cycles` only samples while a PID is actually executing
on a CPU; by construction it cannot see time spent blocked in a
`futex_wait` inside `Lock.__enter__`. Result 2 of the previous version
(worker's on-CPU cycle-retirement rate collapsing ~5.8x under `w8_c4`)
was real and reproducible, but the tool used could only show that the
worker was off-CPU more, not why. The "why" was already sitting in
`pyspy_w8_c4.txt` and `strace_w8_c4.txt` from a prior capture in this
same directory — it did not require a new experiment, only reading
data already collected.

## The mechanism, confirmed by reading the real drain loop

`fwht.py`'s `parallel_decompose` (~line 1636-1666) uses a bounded
in-flight window (`max_in_flight = 2 * n_workers`, itself a real fix
from an earlier session for a genuinely different bug — unbounded
result-queue backlog causing RSS blowup, see the comment at
`fwht.py:~1260`), but the **drain loop that consumes completed
futures is single-threaded, in the main process**:

```python
while in_flight:
    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
    for future in done:
        chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()
        _submit_next()
        if checkpoint_path is not None:
            _append_parallel_checkpoint_chunk(...)   # file I/O, main process
        labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)  # main process
        ... build dict / yield ...
```

Every completed chunk's checkpoint write AND `_pauli_label_batch` call
(confirmed via direct check that the native C extension IS loaded for
this run — `fwht._native is not None` — so this is not even the slow
pure-Python fallback) run **one at a time, in the single main
process**, regardless of how many workers finished chunks
concurrently. With `w1_c1` there is exactly one producer, so the
consumer never falls behind. With `w8_c4`, 8 producers can finish
chunks near-simultaneously (all doing the same-cost WHT+gather work,
confirmed identical by Result 1's `perf report`/`perf annotate`), but
there is still only one consumer draining them — so workers queue up
waiting to `put()` their result through the `ProcessPoolExecutor`'s
shared result pipe/lock while the main process works through its
single-threaded per-chunk consumption cost. This is exactly what
py-spy caught w8_c4's worker doing 3/3 times.

## Why the earlier synthetic controls did NOT reproduce this

`traffic_intensity_findings.md`'s `wht_small`/`touch_small`/
`wht_large` controls (and `resident_footprint`'s extension of them)
all used a **plain `ProcessPoolExecutor` submit-and-drain loop with no
consumer-side per-chunk work** — no `_pauli_label_batch` call, no
checkpoint file write, just receiving a small/large array back. Their
main-process consumer cost per completed task is close to zero, so it
never becomes the bottleneck no matter how many workers finish
concurrently — hence their near-linear 5.8-7.6x effective concurrency
at `w8_c4`. Paulikit's real drain loop has non-trivial main-process
work per chunk (label construction + checkpoint I/O), which the
synthetic controls never included. That is the specific, concrete
difference this whole investigation was looking for.

## Conclusion (corrected — see `dag_gst_master_analysis.md` Section 3f for the quantified version)

**Part of the ceiling**, not all of it, is Amdahl's-law-style
serialization in `parallel_decompose`'s own single-threaded
result-drain loop — a real, now DAG-modeled node (Section 3f's "DAG
D"), not a hardware limit and not an SMT/core-oversubscription effect
(`traffic_intensity_findings.md` already proves this exact machine
handles 8-way CPU-bound work near-linearly when the consumer side is
cheap). The more workers submit chunks concurrently, the more they
contend for the one main-process thread's attention to drain
`_pauli_label_batch` + checkpoint-append before it can call `wait()`
again and free up queue capacity — directly observed via py-spy
catching the worker blocked on the exact lock (`Queue.put()`'s
`Lock.__enter__`) that this predicts, and corroborated by `poll`
syscall time doubling under contention.

**But DAG D's own measured cost (~5.0 s total, section 3f) accounts
for only ~35% of the observed `w8_c4` slowdown gap** (corrected bound
8.28 s vs. observed 23.96 s — a 15.7 s residual). The remaining
majority of the gap is real but not yet identified as a specific,
modeled mechanism: candidates include actual time-blocked-in-lock
(queueing delay, distinct from the ~0.89 ms of USEFUL label-building
work DAG D measures), the checkpoint-I/O and dict-construction cost
this benchmark excluded, and/or genuine shared-L3/memory-bus
contention as 8 workers' bursts overlap (the original DAG C's other,
still-unmodeled bullet). **Do not treat this document as having found
the complete explanation** — it identifies one real, quantified
contributor and leaves the majority of the gap explicitly open.

## Actionable implication (addresses ~35% of the measured gap, not all of it)

The label-construction and checkpoint-I/O work currently done
per-chunk in the single main-process drain loop is a serialization
point (DAG D) that scales AGAINST worker count instead of with it,
and removing it would only close part of the observed slowdown per
Section 3f's own accounting. Candidate fixes (not yet implemented or
evaluated here): move label construction into the worker itself (each
worker returns finished labels, not raw `(x, z, coeff)` triples, so
the main process only aggregates); batch multiple completed chunks'
checkpoint writes instead of one syscall per chunk; or increase
`max_in_flight` further so workers have more slack to keep producing
while the drain loop catches up (does not fix the underlying
single-consumer limit, only defers its visibility). None of these
would be expected to close the full gap on their own — the residual
65% (queueing delay and/or memory-bus contention, still unmodeled)
would need to be separately measured and addressed. This is a design
question for a future phase, not resolved here.

## Correction record (kept for the investigation's own audit trail)

1. First conclusion ("hardware core/SMT oversubscription") — wrong,
   contradicted by `traffic_intensity_findings.md`'s same-machine
   near-linear synthetic-control results. Retracted.
2. Second conclusion ("off-CPU time, paulikit-specific, cause
   unidentified") — correct as far as it went, but stopped short of
   actually naming the cause despite the needed evidence (py-spy +
   strace) already sitting in this directory from an earlier capture.
3. Third conclusion ("the drain loop fully explains the ceiling") —
   named a real mechanism but overclaimed its scope. Corrected by
   building it as an actual DAG node with measured Work
   (`dag_gst_master_analysis.md` Section 3f), which shows it accounts
   for ~35% of the observed `w8_c4` gap, with the remaining ~65% left
   explicitly open rather than attributed to this mechanism.

## Artifacts

- `perf_record_annotate_capture` (script)
- `perf_w1_c1.perf.data` (449 MB, gitignored), `perf_w8_c4.perf.data` (78 MB, gitignored)
- `perf_w1_c1.report.txt`, `perf_w1_c1.annotate.txt`, `perf_w8_c4.report.txt`, `perf_w8_c4.annotate.txt` (worker on-CPU instruction mix — identical between conditions, see Result 1 in git history of this file)
- `pyspy_w1_c1.txt`, `pyspy_w8_c4.txt` (pre-existing, from `code_specificity_capture` — the decisive evidence, previously uncross-referenced)
- `strace_w1_c1.txt`, `strace_w8_c4.txt` (pre-existing, from `code_specificity_capture` — corroborating syscall-time evidence)
- `src/paulikit/algorithms/fwht.py:1636-1666` (`parallel_decompose`'s drain loop — the actual code under discussion)
