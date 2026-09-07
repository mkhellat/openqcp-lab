# DAG extraction v2: node costs separated by EXECUTION TIER — corrects the ≈3.0 figure

**Supersedes `dag_extraction_and_parallelism.md` for the Work/Span/
Parallelism question.** That document's method (DAG built from code
alone, no measured timings) was right and is kept here unchanged. Its
*arithmetic* was also right for the graph it modeled. What was wrong
was a modeling choice one level below the arithmetic, and it inverted
the conclusion.

## The defect in v1, stated precisely

`dag_extraction_and_parallelism.md` Step 2 assigned:

| node | v1 cost | what the code actually is |
|---|---|---|
| `d4` `_pauli_label_batch` | Θ(t_i · n_qubits) | a **compiled C** loop (`_native.pauli_label_batch`) |
| `d5` dict/yield | Θ(t_i) | `t_i` **CPython dict inserts** with string hashing |

Both entries are *asymptotically correct*. Summed as if they were the
same kind of unit, they say labeling costs `n_qubits`=14× more than
dict-building, which is what drives v1's Span to
`Θ(T_x · n_qubits) ≈ 1.283×10⁹` and yields **parallelism ≈ 3.0**.

But a `Θ(1)` step of `d4` is one C-level byte store into a
preallocated buffer (`pauli_label.c:28-33`: shift, mask, table index,
store — a handful of machine instructions, no allocation, no
refcount), while a `Θ(1)` step of `d5` is a CPython dict insertion:
hash a str object, probe the table, incref key and value, box a float,
and possibly resize. These differ by a large constant factor, and
**Work/Span is a ratio — a constant factor that applies to one node
and not another does not cancel.** v1 summed across a boundary where
the unit cost changes by more than an order of magnitude, which is
exactly the situation Θ-notation is designed to let you ignore and
where you cannot afford to.

This is not a rounding error. It put the 14× weight on the wrong node.

### The independent cross-check that was already in the repo

Kept strictly separate from the derivation below (per this
investigation's own rule — measurement never enters DAG construction),
but recorded because it was available at the time v1 was written and
would have caught this immediately:

`profiling/phase10/full_pipeline_n150_findings.md:51` — measured at
this exact N=150 workload: **dict construction 60.1% of pipeline time,
labeling ~7%**. v1 assigned the ~7% node 14× the cost of the ~60% node.

## Step 1: the DAG (unchanged from v1 — the graph itself was correct)

```
parallel_decompose(operator, chunk_size, n_workers):
    S1..S5   serial prefix (validate, CSR, unique/argsort, partition)
    for each chunk i in 0..C-1:
        A_i:  _parallel_worker_chunk(i)      # in a WORKER process
            a1 searchsorted x2               a5 WHT butterfly (log2(dim) stages)
            a2 zeros((cs, dim))              a6 phase = 1j ** popcount(...)
            a3 sparse gather                 a7 coeff = transformed * conj(phase) / dim
            a4 scatter into dense            a8 nonzero(abs(coeff) > atol)
        D_i:  drain-loop body for chunk i    # in the SINGLE main process
            d1 future.result()               # IPC receive + UNPICKLE
            d2 _submit_next()                # bookkeeping, + PICKLE of task args
            d3 [optional] checkpoint append
            d4 _pauli_label_batch(...)       # C kernel + list materialization
            d5 dict build / yield            # CPython dict inserts
```

Structure is identical to v1: `A_i` are mutually independent; every
`D_i` is totally ordered with every `D_j` by the single-threaded
`while in_flight:` loop (`fwht.py:1666-1684`). That ordering is a fact
about the code, and belongs in the graph. v1 was right to put it there
— that was its real contribution and it stands.

## Step 2: node costs, separated by execution tier

Costs are still read off loop bounds and array shapes only. The change
is that each node is now tagged with the tier it executes in, and
tiers are **not summed together** until Step 3 converts them with an
explicit, declared factor.

Tiers:
- **V** — vectorized/native: NumPy ufunc or compiled C kernel. One
  unit = one element-op inside a tight native loop.
- **P** — CPython interpreter: one unit = one bytecode-level object
  operation (hash, incref, box, dict probe).

| node | tier | cost (from code structure) | source |
|---|---|---|---|
| S1–S5 | V | Θ(n_active log n_active) | `np.unique`/`argsort` |
| a1 | V | Θ(log n_active) | 2× `searchsorted` |
| a2 | V | Θ(cs · dim) | `np.zeros((cs,dim))` |
| a3, a4 | V | Θ(nnz_i) | gather/scatter, `hi-lo` entries |
| a5 | V | Work Θ(cs · dim · log dim), Span Θ(log dim) | `fwht.py:156-166`, `log2(dim)` sequential stages |
| a6–a8 | V | Θ(cs · dim) | elementwise over the chunk |
| d1 | P | Θ(t_i) unpickle + Θ(1) msg | array payload rebuild |
| d2 | P | Θ(1) | set add + one `pool.submit` |
| d3 | P | Θ(t_i) (optional, off by default) | file append |
| **d4** | **V** | Θ(t_i · n_qubits) C bytes | `pauli_label.c:24-34` |
| **d4′** | **P** | **Θ(t_i)** str objects | `pauli_label_native.pyx:94-98` — the `for i in range(n_terms)` loop building `t_i` Python str objects |
| **d5** | **P** | **Θ(t_i)** dict inserts | `_build_real_terms` → `dict(zip(...))`, `fwht.py:783` |

**`d4′` is a node v1 did not have at all.** `_pauli_label_batch` is
not purely a C kernel: the Cython wrapper's *own* Python loop
(`pauli_label_native.pyx:94-98`) materializes `t_i` individual Python
`str` objects — one allocation, one ASCII decode, one refcount each.
That loop is tier-P and is proportional to `t_i`, not `t_i · n_qubits`.
So even the labeling node's real dominant cost is `Θ(t_i)` in the
interpreter, not `Θ(t_i · n_qubits)` in C.

## Step 3: Work and Span, with the tier factor made explicit

Let `κ` = cost of one tier-P unit ÷ cost of one tier-V unit. `κ` is a
property of CPython, not of this workload or this machine's clock
speed; it is not a measurement of this program. It is well above 1 —
a dict insert with string hashing versus a shift-mask-store is
conservatively κ ≳ 20, plausibly κ ≳ 50. **Crucially, the conclusion
below does not depend on κ's exact value** — only on κ being large
enough that tier-P terms dominate, which Step 4 shows happens for any
κ ≳ 2.

With `C`=chunk count, `d`=dim, `cs`=chunk_size, `T_x = Σt_i`:

**Work** (every node once):

  Work_V ≈ C·cs·d·log₂d  +  T_x·n_qubits
  Work_P ≈ κ · (T_x  +  T_x  +  T_x) = κ · 3·T_x
           └ d1 unpickle, d4′ str build, d5 dict insert

**Span** (longest chain). The `A_i` are independent, so they
contribute only one chunk's own depth: Θ(log d). Every `D_i` is
chained, so the *entire* `D` chain enters Span:

  Span ≈ Θ(log d)  +  T_x·n_qubits  +  κ · 3·T_x
         └ tier-V   └ d4 tier-V      └ d1+d4′+d5 tier-P

## Step 4: the number, at the real N=150 workload

Structural constants, extracted directly from the code
(`scratchpad/structural_constants.py`, shapes and counts only — no
timings): `dim=16384`, `n_qubits=14`, `log₂d=14`, `n_active=11189`,
`chunk_size=2`, `C=5595`, `nnz=45000`, `T_x=91,652,096`.

Tier-V terms:
- WHT work:  C·cs·d·log₂d = 5595 × 2 × 16384 × 14 ≈ **2.567×10⁹**
- labeling C loop: T_x·n_qubits ≈ **1.283×10⁹**

Tier-P term (the one v1 omitted entirely):
- 3·T_x ≈ 2.750×10⁸ **interpreter-level object operations**

Now the ratio, as a function of κ:

  Work ≈ 2.567×10⁹ + 1.283×10⁹ + κ·2.750×10⁸ = 3.850×10⁹ + κ·2.750×10⁸
  Span ≈ 14        + 1.283×10⁹ + κ·2.750×10⁸ ≈ 1.283×10⁹ + κ·2.750×10⁸

| κ | Work | Span | **Parallelism** |
|---|---|---|---|
| 1 (v1's implicit assumption, tiers equal) | 4.13×10⁹ | 1.56×10⁹ | **2.6** |
| 10 | 6.60×10⁹ | 4.03×10⁹ | **1.64** |
| 20 | 9.35×10⁹ | 6.78×10⁹ | **1.38** |
| 50 | 1.76×10¹⁰ | 1.50×10¹⁰ | **1.17** |
| 100 | 3.14×10¹⁰ | 2.88×10¹⁰ | **1.09** |

**Parallelism does not rise when the model is corrected — it falls,
monotonically, toward 1.**

## Step 5: what this changes, and why both fixes failed

v1 said parallelism ≈ 3.0 and identified `d4` (labeling) as the node
holding Span. Two fixes followed from that and both failed:

1. **Labeling relocation** (commit `9c5f1c6`, reverted `9224b41`):
   moved `d4` into the worker. Measured 18–23% *slower* at every
   condition including w1. Under v2 this is expected, not surprising:
   `d4` is tier-V and was never the dominant Span term. Moving it
   removes `1.283×10⁹` tier-V units from Span while leaving all
   `κ·2.750×10⁸` tier-P units in place — and pays new IPC cost to do
   it. At κ=20 that predicts Span drops only ~19%, easily erased by
   the pickling cost the graph has no node for.

2. **Batched pull/submit**: restructured *when* `d2` fires. `d2` is
   Θ(1); it is not on the Span-dominating path under v2 at all. A
   1.02× result at chunk_size=2 is consistent with reordering a
   negligible node.

**Both fixes targeted nodes that v2 says were never the bottleneck.**
The bottleneck is the tier-P chain `d1 + d4′ + d5` — unpickling, Python
str materialization, and dict insertion — all of which run in the
single main process, once per chunk, totally ordered, and *none of
which either fix moved.*

## Step 6: the hard structural constraint v1 never surfaced

`d5` cannot be relocated the way `d4` was. It builds the `dict` that
**is** the generator's yielded value (`fwht.py:1680-1684`). A generator
yields from the thread that runs it. So `d5`'s cost is pinned to the
consuming process by the public API's own contract, not by an
implementation detail.

That means "fix the drain loop" has a much narrower option space than
v1 implied:
- moving `d5` to workers requires changing **what crosses the yield
  boundary** (yield arrays, let callers build dicts) — an API change;
- keeping the dict but overlapping its construction requires
  **releasing the GIL**, i.e. threads, which works only if the tier-P
  work actually drops the GIL (dict inserts do **not**);
- or the ceiling is accepted and documented.

None of these is a drain-loop scheduling tweak. v1's framing —
"the drain loop schedules badly" — pointed at `d2`, and that framing
is what produced two failed fixes.

## Falsifiable predictions (for the separate measurement step)

v2 is only worth acting on if it survives a check it could fail:

- **P1**: In the real drain loop, `d5` (dict build) must cost
  **more** than `d4`+`d4′` (labeling). v1 predicts the opposite
  (14× the other way). *These are directly opposed — one measurement
  decides it.*
- **P2**: `d1` (unpickle) + `d4′` + `d5` together must account for
  the large majority of drain-loop time.
- **P3**: Total drain-loop time must be comparable to or larger than
  total worker compute at w8_c4 — i.e. the serial region really is
  the ceiling.
- **P4**: `d2` must be negligible (<1%), explaining the batching
  result directly.

If P1 fails, v2 is wrong and must be retracted the same way v1 is
being retracted here.

## Artifacts

- `src/paulikit/algorithms/fwht.py:1666-1684` (drain loop),
  `:783` (`_build_real_terms`), `:1726-1731` (`_pauli_label_batch`)
- `src/paulikit/_native/pauli_label.c:24-34` (the C kernel — tier V)
- `src/paulikit/_native/pauli_label_native.pyx:94-98` (**`d4′`** — the
  tier-P str-materialization loop v1 had no node for)
- `dag_extraction_and_parallelism.md` (v1 — superseded by this
  document for Work/Span/Parallelism; its DAG *structure*, and its
  identification of the drain loop's total ordering, both stand)
