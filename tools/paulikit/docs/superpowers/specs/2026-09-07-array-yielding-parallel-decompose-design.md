# Array-yielding parallel decomposition — design

**Date:** 2026-09-07
**Status:** approved design, not yet implemented
**Phase:** PLAN.md Phase 13 (multi-core chunk parallelism)

## Problem

`parallel_decompose` is effectively not parallel. Measured drain-side
serial work is 21.73s of a 26.37s sequential runtime at N=150 — a
**serial fraction of 0.824**, which caps Amdahl's law at **1.21× with
infinite cores**. The best speedup ever observed is 1.284× (`w2_c1`,
two workers on *one* physical core); eight workers gives 1.100×. The
implementation is already at its theoretical ceiling, which is why
three successive fixes (labeling relocation `9c5f1c6`, batched
pull/submit, IPC dtype narrowing `6f40d98`) each failed to move
multi-core scaling: they optimized inside the parallel 17.6% or moved
work around within the serial 82.4%, without removing serial work.

The serial 82.4% is per-term Python object construction in the parent
process: ~91.6M `str` labels (`d4`, 7.07s) and ~91.6M dict insertions
(`d5`, 14.47s). Both are pinned to the parent because they build the
`dict` that *is* the generator's yielded value.

### Root cause, confirmed by single-variable experiment

`drain_gil_backpressure_target.py` (commit `d3ed339`, extended in
`6e03f00`) takes a workload already known to scale and changes exactly
one thing — the drain-loop body:

| mode | w2_c1 | w8_c4 | speedup | Welch p |
|---|---|---|---|---|
| `bare` (no drain work) | 17.755s | 7.983s | 2.224× | 6.5e-07 |
| `drain_work` (labels + dict) | 18.677s | 21.583s | **0.865×** | 1.6e-05 |
| `arrays_only` (no labels/dict) | 17.411s | 7.946s | **2.191×** | 2.8e-08 |

`arrays_only` recovers full scaling — statistically indistinguishable
from the no-drain-work control at `w8_c4` (p=0.84) and 2.72× faster
than `drain_work` (p=2.3e-11). The 1.8× decision threshold was fixed
in the script before the run.

Mechanism: the parent must run CPython bytecode to service the pool's
result pipe. While it holds the GIL building dicts it is not draining
that pipe, so workers block in `multiprocessing.Queue.put()` — one
shared semaphore, lock and feeder thread for every worker. Identical
drain work costs +0.92s at `w2_c1` but +13.60s at `w8_c4`: **14.8×
more for the same work**, worsening monotonically with core count.

## Why the output format is the right lever

The output format has always been this project's performance lever,
and this project has already moved it once, deliberately.

| format | per-term cost | parallelizable |
|---|---|---|
| PennyLane `LinearCombination` (an operator object per term) | very high | no |
| `dict[str, complex]` — paulikit today | ~0.18 µs | no (Amdahl 1.21×) |
| `(x, z, coeff)` arrays | ~0.0035 µs | **yes** (measured 2.19×) |

paulikit's 115-120× advantage over PennyLane (PLAN.md §3.4) comes
partly from returning `str` + dict entries instead of full operator
objects. Arrays are the next step down that same axis — and they are
not an exotic format: `(x, z)` is the symplectic bitmask
representation the algorithm computes natively. The `str` labels are a
*rendering* of it.

**Precedent already in this codebase:**
`fwht_pauli_coefficients(sparse=True, chunk_size=...)` already returns
`(x, z, coeff)` triple arrays as a public contract, documented with
the same rationale ("the fix for callers that only need the nonzero
terms"). This design extends an established pattern to the parallel
path rather than introducing a new concept.

### What is genuinely irreducible

If a caller demands a `dict[str, complex]` of all 91.6M terms, then
91.6M `str` objects and 91.6M dict insertions must exist in one
process. That is CPython semantics, not a fixable design flaw, and it
is why the dict path stays capped at 1.21×.

A "ship labels from the workers as packed bytes" variant was
investigated and **rejected on measurement**: packed buffers pickle
35× faster than `list[str]` (0.031ms vs 1.086ms, solving the cost that
killed `9c5f1c6`), but the parent then pays **3.870ms** to slice and
decode 16,381 `str` objects out of the buffer — making the total
*worse* than building them from scratch (5.093ms vs 2.916ms). Moving
label bytes relocates only the cheap C part and leaves the expensive
Python-object part exactly where it was.

## Design

Two new public functions in `paulikit.algorithms.fwht`:

```python
parallel_decompose_arrays(
    operator, chunk_size=None, n_workers=None, atol=1e-10,
    assume_hermitian=True, checkpoint_path=None,
) -> Iterator[tuple[NDArray, NDArray, NDArray]]   # (x, z, coeff) per chunk

terms_from_arrays(
    x, z, coeff, n_qubits, assume_hermitian=True, atol=1e-10,
) -> dict[str, complex] | dict[str, float]
```

`parallel_decompose` is **unchanged** — same signature, same dict
contract, same behavior. No existing caller is affected.

### Why two functions, not a flag

`auto_decompose`'s docstring already states this codebase's rule:
making an existing function's return type depend on runtime state or
arguments is "a hidden-nondeterminism hazard." `parallel_decompose`
was itself created as a separate function rather than a flag on
`fwht_pauli_terms_iter`, precisely because it changes the output
contract (results are not in chunk order). A `return_arrays=True` flag
would violate the rule this codebase already wrote down, and static
type checkers handle argument-dependent return types poorly.

A result-object approach (yield a `ChunkResult` with a lazy `.labels`
/ `.to_dict()`) was considered and **deferred**, not rejected: it adds
a class to a package that currently returns only plain NumPy and dict
types, and a lazy property that looks cheap but costs ~3ms is a
footgun. It can wrap these functions later without breaking anything.

### Composition contract

```python
for x, z, c in parallel_decompose_arrays(op):
    terms = terms_from_arrays(x, z, c, n_qubits)   # == parallel_decompose's chunk
```

This equivalence is the correctness anchor and is a test, not a claim.

### Hermiticity checking — in both paths

`assume_hermitian=True` currently raises via `_build_real_terms`,
which runs in the parent. If `terms_from_arrays` were the only place
that check happened, an array-only caller would silently lose it.
Therefore:

- `parallel_decompose_arrays` runs the vectorized check itself in the
  drain loop **and on checkpoint-resume replay** (a separate code path,
  easy to miss). This is the ~0.058 ms/chunk of GIL-releasing NumPy the
  `arrays_only` control measured at 2.191×.
- `terms_from_arrays` re-runs it independently, so it is safe
  standalone.

Coefficients therefore stay `complex128` across IPC. Narrowing them to
their real part would cut payload a further ~1.7× and was rejected:
it would delete the evidence the check exists to find, converting a
raised `ValueError` into a wrong answer.

Per-chunk semantics are unchanged from today: `assume_hermitian` is
checked per chunk, so a violation raises *after* earlier chunks have
been yielded — the existing documented partial-yield-then-error
contract. This wording is carried into the new docstrings rather than
left for callers to infer.

**Error message parity:** on violation the array path labels only the
single offending term (via `np.nonzero(violation)[0][0]`), producing a
byte-identical message to `_build_real_terms` at O(1) cost rather than
labeling all 91.6M terms.

### Checkpointing — reuses existing machinery unchanged

The checkpoint layer is already array-native end to end:

- `_append_parallel_checkpoint_chunk` already takes the three arrays
  and writes `{"x", "z", "re", "im"}` per term via `.tolist()` —
  dtype-agnostic, so the `uint16` narrowing from `6f40d98` flows
  through untouched.
- `_load_parallel_checkpoint` already **returns `(x, z, coeff)`
  arrays**; `parallel_decompose` converts them to a dict only
  afterward.

So the array path uses the same on-disk format, the same progress
file, and the same resume semantics — and its replay is strictly less
work than the dict path's. A checkpoint written by either function is
readable by the other; this is asserted by test, not assumed.

### Invariants preserved

Both constraints established by PLAN.md Phases 9/10 and 12 hold:

- **Cache-bound working set** — `chunk_size` is untouched, still
  auto-tuned so `chunk_size * dim * 16B` fits L2.
- **Bounded streaming memory** — one chunk live at a time. Peak RSS
  should *drop*, since 91.6M dict entries are never built.

### Other behavior

Pool exceptions propagate as today (untouched). A chunk with zero
surviving terms yields three empty arrays rather than being skipped,
so chunk count stays stable and callers can rely on it.
`terms_from_arrays` accepts any integer dtype, so both narrowed
(`uint16`) and legacy (`intp`) arrays work.

## Testing plan

TDD: tests written first, each able to fail for the right reason.

**Correctness**
1. Round-trip equivalence — `terms_from_arrays(*chunk)` over
   `parallel_decompose_arrays` equals `parallel_decompose`'s chunks
   exactly.
2. Cross-validation against `fwht_pauli_terms` (the independent
   oracle), not just against `parallel_decompose` — guards against
   both paths sharing a bug.
3. Term count and no duplication — union of chunks equals the
   reference term set exactly.
4. Empty-chunk handling — three empty arrays, not a skip.

**Hermiticity** (5-6 mutation-checked: confirmed to fail if the check
is removed, rather than assumed to guard anything)

5. Non-Hermitian input raises from `parallel_decompose_arrays`.
6. Non-Hermitian input raises from `terms_from_arrays` standalone.
7. Error message names the offending term, byte-identical to
   `_build_real_terms`'s.
8. `assume_hermitian=False` yields complex coefficients without
   raising.

**Checkpointing**

9. Resume from a partial checkpoint yields the correct complete result.
10. Cross-function interop — a checkpoint written by
    `parallel_decompose` resumes under `parallel_decompose_arrays` and
    vice versa.
11. Resume replay still runs the Hermiticity check.

**Dtype**

12. Works with both narrowed (`uint16`) and legacy (`intp`) arrays.

**Performance — separate from the unit suite.** Unit tests prove
correctness; they cannot prove the 2.19× survives. That needs a real
N=150 sweep under the standing protocol (thermal cooldown to 55°C,
5 reps, Welch's t-test, `w2_c1` vs `w8_c4`, committed `.jsonl`).

Two questions to be reported honestly whichever way they land:
- Does the 2.19× hold with **real checkpointing enabled**? The control
  had none, and checkpoint writing is per-chunk parent-side I/O — a
  plausible new serial term.
- Does peak RSS drop as predicted?

If the measured speedup falls well short of 2.19×, that is reported as
such and the ship/no-ship decision is the user's — not quietly
reframed as a win.

## Documentation and delivery

Per this project's standing practice, each of these is part of the
work, not follow-up:

- **Atomic commits** — one logical change each, full multi-paragraph
  bodies matching this repo's `git log` style. Expected sequence:
  (1) tests, (2) implementation, (3) docs, (4) performance findings +
  raw `.jsonl`.
- **Dual push** — every commit pushed to both remotes via
  `proxychains4 -q` (github *and* origin; github proxied as of
  2026-09-07).
- **Doc sweep at phase end** — not just docstrings: `PLAN.md` Phase 13
  status, `docs/` Sphinx tree (`api/`, `tutorial.md`), the package
  `__init__.py` Public API list, and `profiling/phase13/README.md`.
- **Memory checkpoint** — update the Phase 13 resume checkpoint with
  the outcome, including any failed predictions.

## What this design does not claim

- That a real implementation achieves 2.191×. The control measured a
  drain loop, not the shipped pipeline with checkpointing, resume, and
  real operator data.
- That the dict path can be made parallel. It cannot, for the reason
  given above; it stays capped at ~1.21× and remains correct and
  supported.
- That ~5595-way parallelism is reachable. That figure is the
  chunk-region parallelism for N=150 *given enough processors*; it is
  workload-specific and hardware-independent, and this machine has
  four physical cores.
