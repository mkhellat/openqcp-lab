# Scoping a further dict_build cut: what's left after Phase 11, and why it's a much harder problem

Recorded 2026-08-31, the recommended follow-up from
`n150_post_implementation_findings.md`: before scoping any GPU work
for the WHT butterfly (now 31.9% of total N=150 pipeline time, up from
21.1% pre-Phase-11), check whether `_build_real_terms`'s own remaining
39.1% has more easily-reachable headroom - the lower-risk option if
so.

## Method

`dict_build_phase11b_microbenchmark.py` (this directory) - a synthetic
10M-term array (matching the scale used throughout Phase 11's own
scoping) times every sub-step inside `_build_real_terms` individually:
the two `np.abs` calls, the vectorized comparison, `.real.tolist()`,
and `dict(zip(labels, real_list))` itself. `dict_vs_alternatives.py`
(this directory) then isolates *why* the dict-construction step costs
what it does, by comparing it against a hash-free alternative
(`list(zip(...))`) and the label list's own materialization cost.

## Results

| sub-step | time (10M terms) | % of `_build_real_terms`'s cost |
|---|---|---|
| `np.abs(coefficient_values)` | 0.030s | 1.0% |
| `np.abs(coefficient_values.imag)` | 0.015s | 0.5% |
| vectorized comparison + `.any()` | 0.034s | 1.1% |
| `.real.tolist()` | 0.22-0.25s | ~7% |
| **`dict(zip(labels, real_list))`** | **2.86-2.92s** | **~90%** |

(Consistent across repeated runs - two shown, both landing at
89.7-90.8%.)

**The vectorized Hermiticity check Phase 11 added (the three lines
above `.tolist()`) is now essentially free** - under 2% of the
function's own cost combined, a complete reversal from before Phase
11, when the equivalent per-term check was the *dominant* cost
(`phase11_dict_build_scoping_findings.md`). Phase 11 correctly
targeted the actual bottleneck at the time; **the dict construction
itself, previously assumed secondary (~30-40% of the old combined
cost per that same document), is now the overwhelming majority of
what remains** once the check stopped competing for time.

## Why this dict(zip(...)) cost, specifically

`dict_vs_alternatives.py` breaks `dict(zip(...))`'s 2.98s further:
`list(zip(...))` (pairing labels with values into tuples, no hash
table at all) alone costs 1.32s; `dict(zip(...))`'s remaining 1.66s
(**~56% of its own total**) is specifically the cost of hashing 10M
Python `str` objects and inserting them into a hash table. Neither
component is a NumPy-expressible operation - both are irreducibly
CPython-object-level work (string hashing, tuple/PyObject allocation,
dict-entry insertion), the same category of cost Phase 3's original
`pauli_label` C-porting fixed for *label string construction itself*,
but here applied to what happens *after* the labels already exist as
Python strings.

Also checked whether `dict(zip(...))` itself is suboptimal versus
alternatives: a dict comprehension is statistically indistinguishable
(2.79s vs 2.86-2.98s, within run-to-run noise); an explicit
`dict.fromkeys(...)` + loop-insert is measurably *worse* (5.42s, ~1.9x
slower) - confirming `dict(zip(...))` (already in production since
Phase 11) is at or near CPython's own practical floor for this
construction pattern, not a further-optimizable choice.

## Interpretation: is there a further cheap win here?

**No - not without changing the API's return-type contract.** Every
component measured is either already at its practical floor
(`dict(zip(...))` vs. alternatives) or fundamentally not
NumPy-vectorizable (string hashing and Python dict-entry construction
happen one object at a time by CPython's own implementation, with no
bulk/array-level equivalent the way `np.abs`/`np.maximum` have). This
is a materially different situation from Phase 11's own target: the
Hermiticity check was Python-level arithmetic accidentally paid for
one term at a time, which NumPy could subsume wholesale; dict
construction from string keys is Python-level *bookkeeping*
(hashing/allocation) with no equivalent bulk primitive to subsume it
into.

**The only paths that could meaningfully cut this further are all
substantially bigger changes than a Phase-11-style micro-fix:**
- Return a different container (e.g., parallel NumPy arrays of labels
  and values, or a structured array) instead of `dict[str, float]` -
  a real, breaking API change affecting every caller, not an internal
  implementation swap.
- Avoid materializing per-term Python label strings at all for callers
  that don't need dict-style lookup - conflicts directly with this
  package's core label-string contract (`pauli_label`,
  `paulikit.testing.fixtures`'s format) used throughout.
- A C/Cython-level dict-construction kernel (mirroring Phase 3's
  `pauli_label` C-port) - plausible in principle, but CPython's `dict`
  builtin is already a highly-optimized C implementation; the
  realistic ceiling for a hand-written kernel doing the same
  hash-table insertion is unclear and would need its own scoping
  before assuming there's real headroom, not obviously a win the way
  Phase 3's label-string port was (that replaced a slow *Python*
  string-formatting loop, not a call into an already-C-level builtin).

None of these are being recommended or scoped as a phase here - they
are meaningfully larger design decisions than anything implemented so
far in Phases 3/6/9/10/11, and this document's purpose is only to
establish that the "easy, Phase-11-shaped win" does not exist here,
answering the question that motivated this investigation.

## Bearing on the GPU question

This closes the "check dict_build for more headroom first" step
`n150_post_implementation_findings.md` recommended before considering
GPU work on the WHT butterfly. Since dict_build has no comparably
cheap further fix available, **the WHT butterfly (31.9% of total
pipeline time) and dict_build (39.1%, now at its practical floor for a
Python API returning `dict[str, float]`) are both, independently,
about as optimized as they can get without either a GPU port or a
breaking API-contract change.** This does not itself resolve whether
GPU work is worth it - it only removes the "wait, check dict_build
first" objection, leaving a real GPU-port cost/benefit estimate
(host<->device transfer overhead, toolchain/dependency cost,
portability - none yet estimated) as the actual next decision, if this
line of optimization is pursued further at all.

## What this does NOT show

- Real N=150 Hamiltonian coefficients were not used here (synthetic
  10M-term data, matching Phase 11's own scoping convention) - a real
  run would confirm the same sub-step proportions hold, not just the
  synthetic proxy; given how large and consistent the ~90% figure is
  across two runs, this is not expected to change materially, but is
  not independently confirmed at real N=150 scale in this document.
- Does not estimate what a C/Cython dict-construction kernel would
  actually achieve - flagged above as a real, larger option, but no
  prototype or measurement was attempted here.
- Does not estimate GPU-port cost/benefit for the WHT butterfly stage
  - this document only closes the "is there an easier win elsewhere
  first" question, per `n150_post_implementation_findings.md`'s
  recommendation; the GPU question itself remains open.
