# DAG extraction and Work/Span/Parallelism — derived from code alone (2026-09-05)

**Purpose of this document, stated precisely up front.** Work, Span,
and Parallelism (T1, T-infinity, T1/T-infinity) are properties of a
**dependency graph extracted from the algorithm as written** — nodes
are units of computation the code actually performs, edges are real
data/control dependencies between them, and each node's cost is its
own **algorithmic complexity**, not a measured wall-clock number.
Only after the graph is built and Work/Span are computed from it does
a SEPARATE step compare that theoretical parallelism against
measurement. Blending measured timings INTO the DAG's own node costs
— which `dag_gst_master_analysis.md`'s later sections drifted into
doing (calibrating a node's "Work" from a microbenchmark, then folding
that back into a GST bound) — inverts the method: it uses the
empirical result to manufacture the theoretical model that is
supposed to be judged against that same result. That is circular, and
this document does not do it. `dag_gst_master_analysis.md` remains as
the historical record of that lineage (including its own
self-corrections); this document supersedes it as the DAG/Work/Span
answer, built the right way once.

## Step 1: extract the DAG from the actual code

The full computation, as written, in call order (`fwht.py`):

```
parallel_decompose(operator, chunk_size, n_workers):
    S1  _prepare_operator_for_fwht(operator)       # validate, CSR convert, XOR-gather setup
    S2  np.unique(x_nz, return_inverse=True)        # active_x, inverse  (sort/dedupe)
    S3  np.argsort(inverse, kind="stable")          # order
    S4  sorted_inverse, sorted_p_nz, sorted_q_nz = ...[order]   # 3 gathers by `order`
    S5  chunk_starts = range(0, n_active, chunk_size)            # partition into C chunks
    for each chunk i in 0..C-1:                      # <-- DAG A: parallel region starts here
        A_i:  _parallel_worker_chunk(i):
            a1  searchsorted(sorted_inverse, chunk_start/chunk_end)   # locate this chunk's rows
            a2  gathered_chunk = zeros((cs, dim))                     # allocate
            a3  gathered_values = operator[sorted_p_nz[lo:hi], sorted_q_nz[lo:hi]]  # sparse gather
            a4  gathered_chunk[...] = gathered_values                 # scatter into dense buffer
            a5  _walsh_hadamard_transform_rows(gathered_chunk):        # DAG B, nested inside A_i
                    for stage in 0..log2(dim)-1:        # SEQUENTIAL stages (each reads prev stage's write)
                        b_stage:  left, right = left+right, left-right   # dim/2 independent pairs per stage
            a6  phase = 1j ** popcount(chunk_x & z_indices)           # elementwise, vectorized
            a7  chunk_coefficients = transformed * conj(phase) / dim  # elementwise, vectorized
            a8  nonzero(abs(chunk_coefficients) > atol)               # threshold filter
            return (i, chunk_x_out, z_idx, chunk_coeff_out)
        D_i:  drain-loop body for chunk i (runs in the SINGLE main process, after A_i completes):
            d1  future.result()                        # receive A_i's output (IPC boundary)
            d2  _submit_next()                          # submit chunk i+max_in_flight (bookkeeping)
            d3  [optional] _append_parallel_checkpoint_chunk(...)   # file I/O
            d4  labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)   # per-surviving-term label build
            d5  dict/yield construction
```

This is the honest extraction: `S1`-`S5` are a serial prefix,
`{A_0..A_{C-1}}` is the parallel region (one node-tree per chunk,
each containing the nested `a1..a8`/DAG-B sub-DAG), and `{D_0..D_{C-1}}`
is **not part of the same parallel region** — each `D_i` has a real
data dependency edge from `A_i` (it consumes `A_i`'s return value),
and, because it is a single Python `while` loop with no threading,
every `D_i` and `D_j` (i != j) are also totally ordered with respect
to each other **by construction of the code**, not by measurement.
This ordering is a real structural fact about the code (one `for
future in done:` loop, one thread) — it belongs in the DAG whether or
not anyone ever measures it.

## Step 2: node costs, from algorithmic complexity only

| node | operation | cost (Big-Theta, from the code's own structure) |
|---|---|---|
| S1-S5 (serial prefix) | validate, CSR convert, sort/dedupe/argsort over `n_active` entries | Θ(n_active log n_active) |
| a1 (`searchsorted` x2) | binary search over `n_active` | Θ(log n_active) |
| a2 (`zeros`) | allocate `(chunk_size, dim)` | Θ(chunk_size · dim) |
| a3 (sparse gather) | `hi - lo` nonzero lookups | Θ(nnz_chunk) |
| a4 (scatter) | write `hi - lo` entries | Θ(nnz_chunk) |
| a5 (WHT, per row) | `log2(dim)` sequential stages, `dim/2` independent pairs each | Work Θ(dim log dim), Span Θ(log dim) per row; × `chunk_size` rows (independent across rows) |
| a6-a7 (phase/coeff) | elementwise over `(chunk_size, dim)` | Θ(chunk_size · dim) |
| a8 (threshold) | elementwise over `(chunk_size, dim)` | Θ(chunk_size · dim) |
| d1 (IPC receive) | one message | Θ(1) message, Θ(surviving_terms) payload size |
| d2 (bookkeeping) | O(1) set op | Θ(1) |
| d3 (checkpoint, optional) | file append | Θ(surviving_terms) |
| d4 (`_pauli_label_batch`) | one label per surviving term | Θ(surviving_terms · n_qubits) (each label is `n_qubits` characters) |
| d5 (dict/yield) | one entry per surviving term | Θ(surviving_terms) |

No entry above is a measured number — every cost is read directly off
a loop bound, an array shape, or a stated algorithmic bound in the
code's own docstrings (e.g. `a5`'s Θ(d log d) is this module's own
documented complexity, matching the classical WHT recurrence).

## Step 3: Work and Span of the true DAG

Let `C` = number of chunks, `d` = dim, `cs` = chunk_size,
`T_x` = total surviving terms across the whole run (Σ over chunks of
that chunk's surviving-term count).

**Work** (total cost if the whole DAG ran on ONE processor, summing
every node — this is a pure counting exercise over the graph, not a
timing):

\[
\mathrm{Work} = \Theta(n_{\mathrm{active}} \log n_{\mathrm{active}})
\;+\; \sum_{i=1}^{C}\Big[\Theta(\mathrm{nnz}_i) + \Theta(cs \cdot d \log d) + \Theta(cs \cdot d)\Big]
\;+\; \sum_{i=1}^{C}\Big[\Theta(1) + \Theta(t_i \cdot n_{\mathrm{qubits}})\Big]
\]

where the first big sum is every `A_i` and the second is every `D_i`
(`t_i` = chunk i's surviving-term count, Σt_i = T_x). Both sums are
part of Work regardless of parallelism, since Work counts EVERY node
once, on any schedule.

**Span** (longest dependency chain — the part that actually
determines ideal parallelism):

The `A_i` chunks have **no edges between each other** (confirmed
race-free / dependency-free in `dag_gst_master_analysis.md` Section
0 — disjoint `sorted_p_nz[lo:hi]` slices, independent output buffers).
So the `A_i` subgraphs contribute, to the critical path, only the
LONGEST single `A_i` chain: `Θ(log n_active)` (a1) `+` `Θ(1)` (a2-a4,
constant-depth vector ops) `+` **`Θ(log d)`** (a5's sequential
butterfly stages — this is the dominant depth-contributing term
inside one chunk) `+` `Θ(1)` (a6-a8). So one `A_i`'s own span is
`Θ(log d)` (with `d=16384`, `log2(d)=14` — the WHT's 14 sequential
stages are the deepest chain inside a single chunk, matching
`dag_gst_master_analysis.md`'s original DAG-B span calculation
exactly — that part of the earlier analysis was correct and is not
being revised here).

BUT: because every `D_i` is **totally ordered with every other `D_j`**
(one thread, one loop, by construction of the code — this is the part
the earlier "DAG C" prose named but never put into the graph), the
critical path through the WHOLE computation is not "the longest
single chunk's chain" — it must pass through **all C** of the
`D_i` nodes in sequence, because a schedule cannot execute `D_2`
before `D_1` finishes (they are the same thread), regardless of how
many processors are available for the `A_i` region. So:

\[
\mathrm{Span} = \Theta(\log d) \;+\; \sum_{i=1}^{C} \Big[\Theta(1) + \Theta(t_i \cdot n_{\mathrm{qubits}})\Big]
= \Theta(\log d) + \Theta(T_x \cdot n_{\mathrm{qubits}})
\]

**This is the actual, code-derived Span** — not a number pulled from
a microbenchmark, but the direct consequence of `D_i`'s nodes being
totally ordered by the single-threaded `while in_flight:` loop
structure, which is a fact about the code, extractable by reading it,
independent of how fast any one operation happens to run on this or
any other machine.

## Step 4: Parallelism, from the DAG alone

\[
\text{Parallelism} = \frac{\mathrm{Work}}{\mathrm{Span}}
= \frac{\Theta(C \cdot cs \cdot d \log d) \;+\; \Theta(T_x \cdot n_{\mathrm{qubits}})}
       {\Theta(\log d) \;+\; \Theta(T_x \cdot n_{\mathrm{qubits}})}
\]

The Work term is dominated by the `A_i` chunks' WHT cost
(`C · cs · d log d`, growing with total problem size), but the
**Span term is ALSO dominated by `Θ(T_x · n_qubits)` once `T_x` is
large** — because the drain loop's label-construction work, which is
Θ(1) per node in isolation, appears in Span **C times over** (once
per chunk, since the `D_i` nodes are chained), not just once. This is
the key structural fact the earlier "DAG A/B/C" framing missed
entirely: **the drain loop is not a small constant added to an
otherwise-huge-Span-denominator; because it is repeated once per
chunk and chunks are totally ordered, its total cost enters Span
linearly in `C`, the SAME `C` that makes Work large.**

Concretely, with real N=150 numbers (`C = 5595`, `d = 16384`,
`n_qubits = 14`, `T_x = 91,652,096`, `cs = 2`):

\[
\text{Work} \approx \Theta(5595 \times 2 \times 16384 \times 14) + \Theta(91{,}652{,}096 \times 14)
\]
\[
\text{Span} \approx \Theta(14) + \Theta(91{,}652{,}096 \times 14)
\]

The `Θ(T_x · n_qubits)` term appears in **both** Work and Span with
the **same asymptotic weight** — and at these real problem-size
numbers this is not a loose order-of-magnitude gesture: computing
both raw operation-count terms directly, `C · cs · d · log2(d) =
5595 × 2 × 16384 × 14 ≈ 2.567 × 10^9` versus `T_x · n_qubits =
91{,}652{,}096 × 14 ≈ 1.283 × 10^9` — the drain-loop term is **exactly
half** the WHT term's magnitude at this actual workload, i.e. the same
order of magnitude, not a negligible correction. This means,
structurally, **this DAG's parallelism is bounded by a constant as
`T_x` grows relative to the WHT term**, not by the enormous
`C ≈ 5595`-chunk-wide parallelism the earlier document's DAG A claimed
in isolation. That earlier claim
(`Work(A)/Span(A) ≈ 5595`) was not wrong as a statement about the
`A_i` subgraph alone — but it was never the parallelism of the WHOLE
DAG, because it silently dropped the `D_i` chain from the graph
entirely. Once `D_i` is included (as it must be — it is real code
that really runs, on the real critical path, once per chunk), the
asymptotic parallelism bound collapses toward a small constant
whenever `T_x · n_qubits` is comparable to or larger than
`C · cs · d log d` — which, at the real measured problem sizes, it is.

### The actual number, computed directly from this DAG

Plugging the real problem's own structural sizes (n_active=11189,
C=5595, cs=2, d=16384, log2(d)=14, n_qubits=14, T_x=91,652,096 — every
one of these is a shape/count read off the workload, not a timing)
into Work and Span exactly as derived above:

\[
\mathrm{Work} \approx \underbrace{n_{\mathrm{active}}\log_2 n_{\mathrm{active}}}_{\approx 1.51\times 10^5}
+ \underbrace{C\cdot cs\cdot d\cdot\log_2 d}_{\approx 2.567\times10^9}
+ \underbrace{T_x\cdot n_{\mathrm{qubits}}}_{\approx 1.283\times10^9}
\approx 3.850\times10^9
\]
\[
\mathrm{Span} \approx \underbrace{\log_2 d}_{14} + \underbrace{T_x\cdot n_{\mathrm{qubits}}}_{\approx 1.283\times10^9}
\approx 1.283\times10^9
\]
\[
\text{Parallelism} = \mathrm{Work}/\mathrm{Span} \approx \mathbf{3.0}
\]

**This is the honest, code-derived theoretical parallelism of the
whole DAG: approximately 3, not ~5595 (DAG A's subgraph number, which
silently omitted the drain-loop chain) and not 1 (which would mean the
code has no genuine independent work at all).** The per-chunk
gather/WHT/threshold computation IS real, independent parallel work —
confirmed dependency-free across chunks — so it is not correct to say
parallelism is 1. But the single-threaded drain loop's per-chunk label
construction, forced onto the critical path once per chunk by the
code's own `while` loop, is large enough at this problem's real size
to cap the WHOLE DAG's achievable speedup at roughly 3x, no matter how
many workers are thrown at the independent chunk region. Adding more
processors beyond that point cannot help, because the bottleneck is
not "not enough processors for the parallel region" — it is "the
serial region is nearly as large as the parallel one."

## Step 5: how this differs from the earlier (retracted) approach

This document's Span for `D_i` is **not** a number taken from a
timing run — it is `Θ(T_x · n_qubits)`, a pure function of the
algorithm's own loop structure (`_pauli_label_batch` visits every
surviving term once, building an `n_qubits`-character label; this
happens once per chunk, and chunks are chained). The EARLIER
(now-retracted) Section 3f of `dag_gst_master_analysis.md` instead
took a *microbenchmark's wall-clock number* (0.89 ms at "16,381
terms," itself an empirically-measured average) and called that
"Work(D)" — mixing a measured constant into what should have been a
symbolic complexity term. That was the user-identified error: DAG
extraction must stop at the algorithm's structure; comparing the
resulting Θ-bound against real measurements is a legitimate and
separate next step, but the DAG itself must not be built FROM the
measurement.

## Step 6: comparing this DAG's prediction against measurement (a separate, honest step)

Only now, having built Work/Span from the code alone, is it valid to
ask whether the DAG's predicted behavior matches observation. The
qualitative prediction is stark and directly testable: **parallelism
should degrade specifically as `T_x` (total surviving terms) grows
relative to the per-chunk WHT cost** — i.e., problems with a HIGHER
term-survival rate per chunk (more nonzero output per chunk) should
show WORSE multi-worker scaling than problems with the same chunk
count but sparser output, even at identical `n_workers`/`chunk_size`.
This is a falsifiable, structural prediction from the DAG itself, not
from any timing number:

- **Consistent with observation so far**: paulikit's real workload's
  measured effective concurrency (~2.3 at w8_c4,
  `dag_gst_master_analysis.md` Section 3b) is far below the
  synthetic controls' (~5.8-7.6, `traffic_intensity_findings.md`) —
  and the synthetic controls' drain loops do NOT call
  `_pauli_label_batch` per completed task at all (Section 3f/this
  investigation's own earlier finding), i.e. they have `T_x = 0` in
  this DAG's Span term, exactly consistent with this DAG predicting
  their Span stays `Θ(log d)`-only and their parallelism stays high.
- **Not yet directly tested**: whether varying `T_x` alone (holding
  `C`, `cs`, `d`, `n_workers` fixed) produces the MONOTONIC
  degradation in effective concurrency this DAG predicts. This is the
  correct next experiment — a controlled sweep over surviving-term
  density, not another synthetic proxy that omits the drain loop
  entirely (as every synthetic control built so far, including
  `resident_footprint`, has done).

## Summary — the corrected, code-only answer

- **DAG A alone** (chunks, ignoring the drain loop): Work Θ(C·cs·d·log d),
  Span Θ(log d), parallelism ≈ C ≈ 5595. **True, but incomplete** — it
  is the parallelism of a subgraph, not the whole computation.
- **The whole DAG** (chunks + the single-threaded drain loop that
  really consumes their output, as the code actually runs): Span
  gains a Θ(T_x · n_qubits) term that Work also contains at the same
  weight, so **true whole-program parallelism is NOT Θ(C)** — it is
  bounded by how large `T_x · n_qubits` is relative to
  `C · cs · d · log d`, purely as a consequence of the drain loop
  being one totally-ordered chain over all C chunks in the code as
  written.
- **At the real N=150 workload's own structural sizes (Step 4), this
  works out to Work/Span ≈ 3.0** — a small constant, NOT ~5595
  (DAG A's subgraph number) and NOT 1 (the code does contain genuine,
  confirmed-independent parallel work across chunks; it is not purely
  serial). The bottleneck is that the single-threaded drain loop's
  total cost is comparable in magnitude to the parallel region's own
  work, at this problem size — so no amount of added workers can push
  achievable speedup much past ~3x while the drain loop stays
  single-threaded.
- **This ≈3.0 theoretical figure is directly consistent with the
  independently measured effective concurrency (~2.3-2.7 at `w8_c4`,
  `dag_gst_master_analysis.md` Section 3b)** — the code-derived DAG
  bound and the empirical measurement land in the same small-integer
  range, which is the correct kind of theory-vs-measurement agreement
  (computed separately, then compared — not the earlier circular
  approach of computing one from the other).
- **This was derivable from the code alone**, without running
  anything — the earlier documents' repeated need to fall back on
  measurement to explain the gap was itself a symptom of having built
  an incomplete DAG (one that dropped the drain loop's real, chunked,
  totally-ordered chain), not evidence that DAG-only analysis is
  insufficient in principle.

## Artifacts

- `src/paulikit/algorithms/fwht.py:1146-1636` (`_parallel_worker_chunk`,
  `_walsh_hadamard_transform_rows`, `parallel_decompose`'s drain loop —
  the code this whole extraction is read from)
- `dag_gst_master_analysis.md` (prior lineage, including two
  self-corrections; superseded by this document for the Work/Span/
  Parallelism question specifically)
