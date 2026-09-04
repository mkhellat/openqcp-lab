# DAG / GST / Master analysis for Phase 13 parallelism (2026-09-04)

Careful two-way analysis: (1) forward from the shipped code’s
computational DAG + Master/GST; (2) reverse from measured wall-clock
and effective concurrency. Numbers below are for the real N=150
workload used throughout Phase 13 (`chunk_size=2`, dim=16384).

This does **not** implement a fix. It decides what the theory+data
jointly say we are allowed to conclude.

---

## 0. Determinacy races: none identified in paulikit's own code

Before any Work/Span/Parallelism analysis, the lec7-precise question:
does shipped code have a determinacy race (two logically parallel
instructions touch the same location, at least one a write)? **No.**
Walked structurally per DAG (no race detector exists for this
Python/NumPy/multiprocess stack - Cilksan is C/Cilk-specific and does
not apply):

- **DAG A (outer chunks)**: each chunk owns its own `gathered_chunk`
  buffer and reads a disjoint slice of `operator`'s nonzero entries
  (`sorted_p_nz[lo:hi]`/`sorted_q_nz[lo:hi]`, contiguous per chunk via
  the stable sort in `parallel_decompose`) - zero shared mutable state
  between chunks, race-free by construction.
- **DAG B (inner WHT)**: not currently parallelized at all - NumPy
  vectorizes the whole `(chunk_size, dim)` array as one op per stage
  on one core, so there is no concurrent access to race on. (If the
  butterfly-network-level parallelism named in section 4b below were
  ever implemented, writing overlapping array slices across
  `log2(dim)` stages without a barrier is exactly where a real race
  could be introduced - flagged for that future work, not present
  today.)
- **DAG C (runtime layer)**: the one place with genuine shared mutable
  state across workers is `next_pin_index`
  (`multiprocessing.Value` in `_parallel_worker_init`), and it is
  correctly lock-guarded (`with next_pin_index.get_lock()`).
  `autotune.py`'s module-level caches (`_cached_l2_bytes`,
  `_cached_memory_budget_bytes`) are the one within-process
  shared-across-threads state and are also lock-guarded (double-
  checked locking), with a dedicated regression test
  (`test_recommended_chunk_size_thread_safe_single_underlying_call`)
  pinned to catch a reintroduced race. `ProcessPoolExecutor` workers
  are separate OS processes with no shared address space, which rules
  out classic shared-memory races across them structurally.

**What was actually found is NOT a race** - it is slow-but-correct
serialization: workers blocked in `ProcessPoolExecutor`'s own
lock/pipe coordination (py-spy evidence, section 3b). No data
corruption, no nondeterministic wrong answers - a performance
pathology, not a correctness bug.

---

## 1. Forward: the computational DAG in code

There are **three nested DAGs**. Conflating them is how you get the
wrong “ideal parallelism.”

### DAG A — Outer: independent chunks (`parallel_decompose`)

Per chunk (`_parallel_worker_chunk`):

```text
gather/scatter into dense (cs, dim)
        ↓
_walsh_hadamard_transform_rows   # log2(dim) stages
        ↓
phase × / dim
        ↓
threshold → (x,z,coeff) triples
        ↓
pickle/IPC back to main  (runtime, not math)
```

Chunks do **not** depend on each other mathematically (no cross-chunk
reduction). Measured for this workload:

| quantity | value |
|---|---|
| dim | 16384 |
| n_active | 11189 |
| chunk_size | 2 |
| **n_chunks** | **5595** |

**Outer work / span (pure compute model):**

- \(T_1^{\mathrm{outer}} \approx C \cdot W_{\mathrm{chunk}}\) with \(C=5595\)
- \(T_\infty^{\mathrm{outer}} \approx W_{\mathrm{chunk}}\) (plus negligible serial setup)
- **Ideal outer parallelism** \(T_1/T_\infty \approx C \approx 5595\)

With measured \(T_1 \approx 26.4\,\mathrm{s}\) (sweep `w1_c1`):

- \(W_{\mathrm{chunk}} \approx T_1/C \approx 4.7\,\mathrm{ms}\)
- GST with \(P=8\), this span:  
  \(T_8 \le T_1/8 + T_\infty \approx 3.30 + 0.005 \approx \mathbf{3.3\,\mathrm{s}}\)
- **Observed** `w8_c4` ≈ **24.0 s** (sweep) / **24.8 s** (bandwidth set)

So under a pure outer-chunk DAG + greedy scheduler, we are ~**7×**
slower than GST allows. That is not “sublinear speedup”; it is
incompatible with the assumptions of the model.

### DAG B — Inner: WHT butterfly (`_walsh_hadamard_transform_rows`)

```120:167:tools/paulikit/src/paulikit/algorithms/fwht.py
def _walsh_hadamard_transform_rows(...):
    ...
    span = 1
    while span < dim:
        transformed = transformed.reshape(..., dim // (2 * span), 2, span)
        left, right = left + right, left - right
        ...
        span *= 2
```

Per row of length \(d=\mathrm{dim}\):

- **Stages:** \(\log_2 d = 14\), **sequential** (each stage reads the
  previous stage’s writes).
- **Within a stage:** \(d/2\) independent butterfly pairs → theoretically
  fully parallel.

**Master theorem (work), classic divide-and-conquer form of the same
algorithm:**

\[
W(d) = 2\,W(d/2) + \Theta(d)
\]

- \(a=2\), \(b=2\), \(f(d)=\Theta(d)=\Theta\!\left(d^{\log_b a}\right)\)
- Master **case 2** → \(W(d)=\Theta(d\log d)\)

**Span** if pairs within a stage have unbounded processors:

\[
S(d) = S(d/2) + \Theta(1) = \Theta(\log d)
\]

- Ideal **inner** parallelism \(W/S = \Theta(d/\log d) \approx 1170\)

**What the code actually exposes to the OS:** neither stage-internal
pairs nor rows are scheduled as tasks. NumPy vectorizes the whole
`(chunk_size, dim)` array on **one worker process / one (logical) CPU**.
So DAG B’s huge inner parallelism is **not available** to
`ProcessPoolExecutor`. It only shows up as SIMD within a core.

Per-chunk working set: `chunk_size * dim * 16 = 512 KiB` = **2× L2**
(256 KiB/core), **~6% of shared L3** (8 MiB). Every one of 14 stages
touches that full footprint again → high **traffic intensity**, not
just capacity.

### DAG C — Runtime / resource DAG (not in the math)

Edges that exist in wall-clock reality but **not** in A/B:

- ProcessPoolExecutor queue lock + pipe send/recv (py-spy under
  contention: workers stacked in `synchronize.__enter__` / `_send`)
- Shared L3 / memory-controller serialization when several workers’
  512 KiB×14-stage bursts overlap
- Main-process drain loop (`wait`/`as_completed`) — already bounded by
  `max_in_flight`, but still serializes result handoff

GST **assumes** a greedy/work-stealing scheduler over a pure compute
DAG with unit-cost nodes. DAG C violates that: ready work can sit
blocked on IPC/memory without being “span” in the math sense.

---

## 2. Forward: what Master + GST *would* predict if the model held

| Model | \(T_1/T_\infty\) | GST \(T_8\) upper bound | Meaning |
|---|---|---|---|
| Outer chunks only | ~5595 | ~3.3 s | Massive outer parallelism available in the math |
| Inner WHT only (unlimited procs) | ~1170 / row | not how we parallelize today | Not exposed to the pool |
| Observed | speedup \(T_1/T_8\approx 1.10\) | — | Almost no useful multi-core gain |

**Master’s role here:** it does **not** say “flat WHT cannot parallelize
across chunks” (that’s already ideal). It says: if we **restructure**
the inner butterfly into a cache-fitting D&C (lec8 matmul lesson),
**arithmetic work stays \(\Theta(d\log d)\)** but **memory transfers**
can drop (classic blocked-FFT / cache-oblivious FFT territory:
fewer compulsory trips through shared L3/DRAM per unit work). That
shrinks DAG-C costs, which is the only way GST’s ~3 s class of bound
could become approachable on this machine.

Leaf tuning (`chunk_size`) was already ruled out experimentally —
consistent with Master: changing the leaf size does not change the
recurrence structure or the \(\Theta(d\log d)\) traffic pattern enough
under multi-core contention.

---

## 3. Reverse: what the data imply for GST + Master + DAG

### 3a. Implied span lower bounds (GST rearranged)

GST: \(T_P \le T_1/P + T_\infty\)  
⇒ any observed run forces \(T_\infty \ge T_P - T_1/P\).

Using sweep means, \(T_1=26.366\,\mathrm{s}\), and \(P=n_{\mathrm{workers}}\):

| config | \(T_P\) | \(T_\infty\) lower bound |
|---|---|---|
| w2_c1 | 20.537 | **7.35 s** |
| w4_c2 | 21.395 | 14.80 s |
| w4_c4 | 23.850 | 17.26 s |
| w8_c4 | 23.964 | **20.67 s** |

**\(T_\infty\) implied by the data grows with \(P\).** In a true fixed
DAG, span is a constant. Growing implied span means: either (i) we are
not measuring a pure DAG schedule, or (ii) node costs inflate with \(P\)
(contention) so the unit-cost DAG model is false.

Fixing \(T_\infty = 7.35\,\mathrm{s}\) from w2_c1 and predicting:

| P | GST upper bound | Observed (best at that width) | Obs / GST |
|---|---|---|---|
| 2 | 20.54 s | 20.54 s | 1.00× (calibrated) |
| 4 | 13.95 s | ~21.4–23.9 s | **1.5–1.7× worse** |
| 8 | 10.65 s | 23.96 s | **2.25× worse** |

Same pattern on the bandwidth-hypothesis means (\(T_8/T_{\mathrm{GST}}\approx 2.33\)).

**Conclusion (reverse → GST):** the measurements are **incompatible**
with GST on the computational DAG with constant node costs. Calling
this a “bound violation” is fair as a red flag; the precise statement
is: **ProcessPoolExecutor under this workload does not implement a
greedy schedule of DAG A/B** — DAG C dominates.

### 3b. Effective concurrency (task_clock / elapsed)

Child-inheriting `perf` (bandwidth hypothesis):

| workload | condition | elapsed | eff. \(P\) | cache-miss % |
|---|---|---|---|---|
| paulikit | w2_c1 | 22.07 s | **2.69** | 9.37 |
| paulikit | w8_c4 | 24.80 s | **2.33** | 9.18 |
| synthetic | w2_c1 | 12.36 s | 2.18 | 0.16 |
| synthetic | w8_c4 | 4.09 s | **7.58** | 0.44 |

Paulikit: adding cores **does not raise** useful concurrent CPU time
(eff. \(P\) stuck ~2.3–2.7). Synthetic: eff. \(P\) scales to ~7.6.

Instructions stay ~constant across paulikit core-packing (prior IPC
study) while wall-clock worsens → same work, worse concurrency /
stalling — matches DAG C, not “more arithmetic span.”

### 3c. What reverse implies for Master

Master on the **arithmetic** recurrence is fine: work is still
\(\Theta(d\log d)\) per row; synthetic vs paulikit shows the difference
is **memory traffic**, not flop count. So:

- Master does **not** predict the observed multi-core collapse by itself.
- Master **does** motivate a D&C / cache-blocked WHT: same work class,
  fewer transfers → DAG C edges get cheaper → GST may start to apply.

### 3d. What reverse implies for “which DAG is the critical path?”

| Candidate critical path | Supported by data? |
|---|---|
| Outer chunk chain (math span ~1 chunk) | **No** — ideal \(T_\infty\sim5\,\mathrm{ms}\); implied \(T_\infty\sim7\)–\(21\,\mathrm{s}\) |
| Inner WHT stage chain (\(\Theta(\log d)\) on one core) | Partially — each worker is serial through 14 stages, but that alone doesn’t explain multi-core slowdown |
| **Resource critical path: memory + IPC** | **Yes** — eff. \(P\) capped; py-spy in locks; synthetic (low traffic) scales; miss *ratio* flat while concurrency stuck |

So the **critical path that matters for parallelism is not the math
DAG’s longest dependence chain**; it is the **resource-serialized
path** (shared memory subsystem + result IPC) that the unit-cost DAG
omits.

### 3e. IPC-blocking explains the SYMPTOM, not the root cause (added after traffic_intensity_findings.md)

The py-spy/IPC-blocking evidence above (3b, 3d) answers *where the
missing time goes* under contention - correctly. It does **not** by
itself answer *why paulikit specifically* triggers that blocking. The
traffic-intensity control experiment
(`traffic_intensity_findings.md`) tested this directly: three
controls (`wht_small`, `touch_small`, `wht_large`) use the byte-for-
byte SAME `ProcessPoolExecutor` mechanism - same pool shape, same
pickle-over-pipe IPC, same lock/queue machinery paulikit uses - while
matching or exceeding paulikit's own dense per-chunk buffer traffic
(512 KiB, 14 full-array stage touches, even a 512 KiB IPC payload for
`wht_large`). **All three scale normally** (2.2-2.7x at w8_c4 vs
w2_c1, eff. \(P\) 5.8-7.6) - essentially identical to the synthetic
control's own scaling in 3b. If "workers block in
`ProcessPoolExecutor`'s IPC machinery" were sufficient on its own to
explain the ceiling, these controls - which exercise that exact same
machinery under an equivalent traffic load - should show it too. They
do not.

**Conclusion**: IPC-blocking is the correct description of the
*mechanism* by which lost time manifests (confirmed, real,
measured), but it is a **downstream symptom**, not the *trigger*.
Something specific to paulikit's own per-chunk work - not shared by
any traffic-volume-matched control - is what causes that IPC path to
serialize in the first place. The leading remaining suspect (not yet
tested) is the **irregular gather/scatter access pattern** (sparse
`operator` index lookups scattered into a dense buffer at arbitrary
positions) and/or the resident `operator`/setup-array footprint each
worker carries - both untouched by any control tested so far. This is
exactly what the gather-pattern isolation experiment (see this
project's own "Actual next isolation step") is designed to test.

---

## 4. Joint conclusions (careful)

1. **Ideal parallelism from the code DAG is huge** (~5600 outer, ~1170
   inner). GST then predicts multi-core should crush the sequential
   time (~3 s class at P=8). **We do not get that.**

2. **Measured parallelism is ~1.1× vs sequential** and **~2.3 effective
   CPUs** even with 8 workers. Implied span grows with P → **GST’s
   hypotheses fail** for this runtime, not because the chunk DAG is
   sequential.

3. **Master theorem on today’s flat butterfly** explains work/span of
   the *math*, and why leaf `chunk_size` retuning was never going to
   restore multi-core scaling. It points at **restructuring the
   butterfly (cache-blocked / recursive D&C)** as the lever that can
   change *transfer complexity*, which is what the reverse path says
   is binding.

4. **Next design question (precise):**  
   Can a cache-blocked WHT cut per-chunk memory traffic enough that
   effective \(P\) under w8_c4 rises toward the synthetic’s ~7–8, so
   that observed \(T_P\) approaches GST’s outer-DAG bound?  
   That is algorithm design + measurement — not another pairwise
   pinning study.

5. **What not to claim yet:** that cache-blocking *will* work; that
   GST is “wrong” as a theorem; or that the math critical path is long.
   The honest claim: **we are bandwidth/IPC limited; the math DAG is
   wide; Master tells us how to redesign the inner kernel; GST tells
   us how good life could be if DAG C stops dominating.**

6. **Master's "restructure the inner kernel" prescription now has a
   hard, falsifiable bound**, not just a qualitative direction: see
   `roofline_analysis.md` (Roofline model, Williams/Waterman/Patterson
   2009 - the standard performance-engineering tool for exactly this
   compute-bound-vs-memory-bound question, complementary to but
   distinct from DAG+GST+Master, which was built for Cilk-style
   shared-memory fork-join programs, not for bounding a single
   kernel's memory-traffic ceiling). Paulikit's WHT kernel today runs
   at arithmetic intensity 0.0625 FLOPs/byte, 64x below this
   machine's measured ridge point (4.01 FLOPs/byte, from a real
   thermal-controlled DRAM-bandwidth probe, not a spec-sheet guess) -
   severely memory-bound. Even a THEORETICALLY PERFECT cache-blocking
   (zero re-traffic across all 14 butterfly stages) only reaches AI
   0.875 FLOPs/byte - still 4.6x below the ridge point, so the kernel
   never becomes compute-bound - bounding the maximum possible
   single-chunk/single-core wall-clock speedup at 14.0x (exactly
   `log2(dim)`, the stage count). This bounds only DAG B in isolation
   (single-core); it does NOT bound or predict whether cache-blocking
   relieves DAG C's multi-worker contention (section 3e) - that
   remains a separate, untested question, most directly addressed by
   the gather-pattern isolation experiment, not by this calculation.

---

## 4b. Consolidated Work/Span/Parallelism/critical-path answer

The three DAGs above give three different answers to "what is our
work / span / parallelism" - conflating them is the error to avoid.
This table is the single place all three are stated side by side.

| | Work \(T_1\) | Span \(T_\infty\) | Parallelism \(T_1/T_\infty\) | Critical path is... |
|---|---|---|---|---|
| DAG A (outer chunks, ideal) | 26.366 s | \(\approx\) 4.7 ms | \(\approx\) 5595 | one chunk's own compute |
| DAG B (inner WHT, ideal, **not scheduled today**) | \(\Theta(d\log d)\) per row | \(\Theta(\log d)\approx\)14 hops | \(\Theta(d/\log d)\approx\)1170 (dim=16384) | the 14 sequential butterfly stages |
| DAG C (resource layer, what actually binds) | - | **implied span grows with \(P\)**: 7.35 s (\(P{=}2\)) \(\to\) 20.67 s (\(P{=}8\)) | - | shared memory bus + `ProcessPoolExecutor` lock/pipe contention |

A growing implied span (row DAG C) is impossible for a true DAG - span
is a fixed structural property. That growth is the mathematical
signature that **the real critical path lives in DAG C, not in DAG A
or DAG B** - a resource-contention path whose cost scales with how
many workers compete for it simultaneously, unlike a genuine data
dependency edge.

**Direct answer to "what is our parallelism, and why":** ideal
parallelism from the code's own DAG structure is enormous (~5595
outer x ~1170 inner, and these compose since DAG B is *within* one
DAG-A chunk). **Measured parallelism is ~1.1x** (\(T_1/T_8 \approx
26.366/24.0\)). The gap is not a span problem in the DAG-theoretic
sense - it is DAG C (memory-bus + IPC contention) sitting outside both
theorems' unit-cost-node assumptions.

### Why this is the same failure mode as naive parallel merge sort (lec8)

The MIT 6.172 lec8 material's parallel-merge-sort example is the
clean textbook case of an algorithm whose recursive *structure* looks
parallel while a piece of its actual execution is not: merge sort's
divide/conquer recursion parallelizes fine, but a NAIVE sequential
merge step has span \(\Theta(n)\) - the lecture's own "PUNY!"
verdict - because walking two sorted arrays one comparison at a time
is an inherently serial process, regardless of how parallel the
surrounding recursion is. Restoring real parallelism required a
genuinely different merge algorithm (binary-search-based parallel
merge, recursively splitting the LARGER array so the bigger
recursive sub-merge is bounded to \(\le(3/4)n\) elements), achieving
span \(\Theta(\log^2 n)\) instead - a change to the algorithm's
recursive decomposition, not a tuned constant.

The direct analogy here: DAG B's flat `_walsh_hadamard_transform_rows`
is architecturally the "naive sequential merge" of this codebase -
mathematically it has a fine ideal span (\(\Theta(\log d)\)), but the
CODE never exposes the within-stage pairs as schedulable units, so
that ideal span is invisible to any scheduler. Just like the merge-sort
fix, restoring it requires restructuring the recursive
decomposition itself (cache-blocked/recursive D&C, mirroring lec8's
D&C matrix-multiply example, pages 31-40) - not a leaf-level
parameter. This is precisely why `chunk_size` retuning (a leaf
parameter, already tested and ruled out, see
`contended_chunk_size_screen_results.jsonl`) could never have closed
the gap: it operates one level too shallow to change either DAG B's
scheduling granularity or DAG C's memory-traffic pattern.

**What Master did and did not tell us, stated precisely:**
- **Did tell us**: DAG B's work class (\(\Theta(d\log d)\), fixed -
  no restructuring changes this asymptotically) and its ideal span IF
  parallelized to the pair level (\(\Theta(\log d)\)). This is
  prescriptive - it bounds what a redesign *could* achieve.
- **Did NOT tell us**: why the current code collapses under multi-core
  contention. That is a DAG C phenomenon, invisible to a theorem
  about the arithmetic recurrence's asymptotic work/span.

---

## 5. Numbers quick-reference

```text
T_1 (sweep w1_c1)     = 26.366 s
Best T_P (w2_c1)      = 20.537 s   (S ≈ 1.28×)
Wide T_P (w8_c4)      = 23.964 s   (S ≈ 1.10×)
Outer C               = 5595 chunks
Ideal outer T_∞       ≈ 4.7 ms
GST T_8 (ideal outer) ≈ 3.3 s
Obs/GST at P=8        ≈ 2.25–2.3×
Paulikit eff_P w8     ≈ 2.33
Synthetic eff_P w8    ≈ 7.58
Per-chunk footprint   = 512 KiB (2× L2, 14 full-array stage touches)
```
