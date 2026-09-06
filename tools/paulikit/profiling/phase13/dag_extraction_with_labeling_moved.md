# DAG re-extraction: parallelism if `_pauli_label_batch` moves into the worker (2026-09-06)

**Purpose.** `dag_extraction_and_parallelism.md` computed the current
code's whole-DAG parallelism at ≈3.0, and identified the cause: the
single-threaded drain loop's `d4` step
(`labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)`) runs
once per chunk, in the one thread that also has to service every other
chunk, so its Θ(t_i · n_qubits) cost is forced onto the critical path
**C times over** (once per chunk), not once overall. This document
answers a direct follow-up question: **if `_pauli_label_batch` is
moved into `_parallel_worker_chunk` (so each worker labels its own
chunk before returning, and the drain loop never calls it), what does
the DAG say the new theoretical parallelism ceiling is?**

Same method as the superseding document: every node cost below is read
directly off a loop bound, array shape, or the code's own documented
complexity — nothing here is a measured wall-clock number. Only the
final section compares the result to measurement, as a separate step.

## The one structural change

Currently (`fwht.py:1651-1660`):

```python
chunk_index, chunk_x_out, z_idx, chunk_coeff_out = future.result()   # d1
_submit_next()                                                        # d2
if checkpoint_path is not None:
    _append_parallel_checkpoint_chunk(...)                            # d3
labels = _pauli_label_batch(chunk_x_out, z_idx, n_qubits)              # d4 <- MOVES
yield _build_real_terms(labels, chunk_coeff_out, atol)                 # d5
```

`_parallel_worker_chunk` (`fwht.py:1224-1268`) already computes
everything `_pauli_label_batch` needs (`chunk_x_out`, `z_idx`,
`n_qubits` is a closure/init constant) before it returns — nothing
about `d4` depends on any other chunk's output. The proposed fix has
the worker call `_pauli_label_batch` itself and return `labels`
instead of raw `(chunk_x_out, z_idx)`; the drain loop then only does
`d1, d2, d3, d5`.

## Step 1: revised DAG

```
A_i (revised):  _parallel_worker_chunk(i), now includes labeling:
    a1  searchsorted (locate chunk rows)              Θ(log n_active)
    a2  zeros((cs, dim))                              Θ(cs · d)
    a3  sparse gather                                 Θ(nnz_chunk)
    a4  scatter into dense buffer                      Θ(nnz_chunk)
    a5  WHT (nested DAG B): Work Θ(d log d), Span Θ(log d), per row; × cs rows (independent)
    a6  phase = 1j ** popcount(...)                   Θ(cs · d)
    a7  chunk_coefficients = ...                       Θ(cs · d)
    a8  threshold filter                               Θ(cs · d)
    a9  labels = _pauli_label_batch(...)   <- MOVED HERE   Θ(t_i · n_qubits)
    return (i, labels, chunk_coeff_out)

D_i (revised, drain loop body — d4 removed):
    d1  future.result()                    IPC receive          Θ(1) message, Θ(t_i) payload
    d2  _submit_next()                     bookkeeping           Θ(1)
    d3  [optional] checkpoint append       file I/O              Θ(t_i)
    d5  dict/yield construction            one entry per term    Θ(t_i)
```

`D_i` nodes remain **totally ordered** with each other — this is
unchanged: it is still one Python `while in_flight:` loop, one thread,
by construction of the code, regardless of what got moved out of it.
What changes is only the PER-LINK COST of that chain: it drops from
Θ(1) + Θ(t_i · n_qubits) to Θ(1) + Θ(t_i) — the `n_qubits` multiplier
leaves the chained region entirely, because it now lives inside the
independent, per-chunk `A_i` node instead.

## Step 2: Span of the revised DAG

**A-region Span** (chunks remain independent of each other — this
does not change): the critical path through the parallel region is
still the single longest `A_i` chain, but that chain is now longer by
one more sequential step (`a9`, labeling), since `a9` runs after `a8`
inside the same chunk:

\[
\text{Span}(A) = \Theta(\log n_{\mathrm{active}}) + \Theta(1) + \Theta(\log d) + \Theta(1) + \Theta(t_{\max}\cdot n_{\mathrm{qubits}})
= \Theta(\log d) + \Theta(t_{\max}\cdot n_{\mathrm{qubits}})
\]

where `t_max` = the largest single chunk's surviving-term count (Span
is a max over independent branches, not a sum — this is the key
change from the D-chain, which sums because it is a **chain**, not
independent branches).

**D-region Span** (still a chain, but each link is now cheaper):

\[
\text{Span}(D) = \sum_{i=1}^{C}\big[\Theta(1) + \Theta(t_i)\big] = \Theta(C) + \Theta(T_x)
\]

**Total Span** (the A-region's longest branch, plus the D-chain that
consumes its output — same edge structure as before, `D_i` still
depends on `A_i`'s result):

\[
\text{Span} = \Theta(\log d) + \Theta(t_{\max}\cdot n_{\mathrm{qubits}}) + \Theta(C) + \Theta(T_x)
\]

## Step 3: Work of the revised DAG

Total operation count is unchanged from the current DAG — moving `a9`
does not add or remove any operations, only relocates which node
performs them:

\[
\text{Work} = \Theta(n_{\mathrm{active}}\log n_{\mathrm{active}}) + \Theta(C\cdot cs\cdot d\log d) + \Theta(T_x\cdot n_{\mathrm{qubits}})
\]

## Step 4: plugging in the real N=150 numbers

Same workload shapes as the superseding document — n_active=11189,
C=5595, cs=2, d=16384, log2(d)=14, n_qubits=14, T_x=91,652,096. Chunk
sizes are close to uniform at this workload (cs=2 fixed, dedupe/sort
already balances chunk boundaries — the only place this matters here
is `t_max`, taken as `T_x / C` since no single-chunk outlier has been
measured or claimed): `t_max ≈ T_x / C ≈ 91,652,096 / 5595 ≈ 16,382`.

\[
\text{Work} \approx \underbrace{1.51\times10^5}_{n_{\mathrm{active}}\log n_{\mathrm{active}}}
+ \underbrace{2.567\times10^9}_{C\cdot cs\cdot d\log d}
+ \underbrace{1.283\times10^9}_{T_x\cdot n_{\mathrm{qubits}}}
\approx 3.850\times10^9
\]

(Work is identical to the current DAG's Work, as expected — see Step 3.)

\[
\text{Span} \approx \underbrace{14}_{\log d}
+ \underbrace{16{,}382\times14}_{t_{\max}\cdot n_{\mathrm{qubits}}\approx 2.29\times10^5}
+ \underbrace{5{,}595}_{C}
+ \underbrace{91{,}652{,}096}_{T_x}
\approx 9.191\times10^7
\]

\[
\text{Parallelism} = \text{Work}/\text{Span} \approx \frac{3.850\times10^9}{9.191\times10^7} \approx \mathbf{41.9}
\]

## Step 5: what this means, precisely

- **Current code (labeling in the drain loop): parallelism ≈ 3.0.**
- **Fixed code (labeling moved into the worker): parallelism ≈ 41.9**
  — roughly a **14x increase in the theoretical ceiling**, purely from
  relocating one function call to run inside the already-independent
  worker instead of the shared drain thread.
- **The new ceiling is not "8x" and not "~5595"** — it is bounded by a
  *different* remaining bottleneck: once `n_qubits`'s multiplier
  leaves the chained D-region, the dominant term left in the D-chain
  is `Θ(T_x)` — the checkpoint-append (`d3`, if checkpointing is
  enabled) and dict/yield construction (`d5`), both still summed once
  per chunk across a single thread. This is consistent with
  independent, already-collected evidence: `full_pipeline_n150_
  findings.md` (cited in `_pauli_label_batch`'s own docstring, Phase
  10) measured dict construction at ~60% of total pipeline time with
  labeling at only ~7% — i.e., the same bottleneck this DAG now
  predicts structurally was already observed empirically, in a
  different investigation, before this DAG calculation existed. That
  is a real point of independent corroboration, not manufactured after
  the fact.
- **41.9 is a ceiling on the DAG's own structure, not a promise of
  41.9x measured speedup.** Realized speedup on real hardware is also
  capped by `n_workers` (physical cores actually available, currently
  auto-tuned to 4 on the dev machine — see `fwht.py:1536-1553`) and by
  whatever overheads (IPC, scheduling, memory bandwidth) the DAG
  method does not model at all, per this whole investigation's
  standing distinction between a code-derived theoretical bound and a
  measured result (`feedback_dag_theory_no_empirical_mixing.md`).
  Comparing 41.9 against measurement after the fix is implemented is
  the correct next step — not yet done, since the fix itself has not
  been applied yet as of this document.
- **If checkpointing is disabled** (`checkpoint_path=None`, the
  default), `d3`'s Θ(T_x) term drops out of the D-chain entirely,
  leaving only `d5`'s Θ(T_x) (dict/yield) as the dominant remaining
  term — Span would be smaller still and parallelism higher than 41.9,
  but `d5` (building the actual output dict) cannot be moved into the
  worker the way labeling can, since the drain loop's whole job is to
  `yield` results back to the caller one at a time and yielding is
  fundamentally a single-thread operation on a generator. Not
  separately computed here since it was not asked for.

## Correction to a prior answer in this same investigation

Before this document was written, a claim was made that moving
`_pauli_label_batch` into the worker "should recover most of the gap
between measured ~2.3–2.7x and the theoretical ceiling," and a
follow-up question — "would this let us expect close to 8x speedup
with 8 cores" — was answered by asserting 8x was the wrong target,
**without first computing the revised DAG.** That dismissal was wrong
to make before doing this calculation: the actual revised ceiling
(≈41.9) is higher than 8x, not lower or comparable to the pre-fix
≈3.0 as was implied. The correct standing lesson, consistent with
`feedback_verify_before_fixing.md`: a claim about a hypothetical
code change's effect must be computed from the DAG before being
confirmed or dismissed, not estimated from intuition about Amdahl's
law or n_workers defaults.

## Artifacts

- `src/paulikit/algorithms/fwht.py:1224-1268` (`_parallel_worker_chunk`,
  the function `_pauli_label_batch` would move into)
- `src/paulikit/algorithms/fwht.py:1648-1666` (drain loop, `d4`'s
  current call site, to be removed)
- `dag_extraction_and_parallelism.md` (baseline DAG this document
  revises one node relocation from; current-code parallelism ≈3.0)
- `profiling/phase10/full_pipeline_n150_findings.md` (independent,
  prior-session empirical measurement: dict construction ~60% of
  pipeline time vs. labeling ~7% — corroborates this document's
  Θ(T_x) dict/yield bottleneck prediction for the post-fix code)
