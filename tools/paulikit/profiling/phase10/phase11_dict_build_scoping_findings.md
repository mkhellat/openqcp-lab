# Scoping Phase 11: what's actually inside `dict_build`'s ~60%

Recorded 2026-08-27, scoping PLAN.md Phase 11 (`dict_build`
optimization) after `full_pipeline_n150_findings.md` identified
`dict_build` as ~60% of total pipeline time at N=150, and
`n_scaling_streaming_findings.md` confirmed the resulting time-vs-
term-count scaling is linear (consistent with a per-term-dominated
cost). This document breaks `dict_build`'s own internals down further
before any implementation is attempted, per this project's own
established discipline (measure before optimizing, per-stage, not by
assumption).

## Method

`dict_build_microbenchmark.py` (this directory) - a synthetic array
(not a real Hamiltonian; large enough to see a real signal at 1M and
10M terms, small enough to iterate on quickly and safely) compares
two variants of the exact loop body found in `fwht_pauli_terms`/
`fwht_pauli_terms_iter`'s `assume_hermitian=True` branch:

- **Variant A (today's code)**: `chunk_coeff.tolist()`, then a Python
  `for` loop doing a per-term Hermiticity check
  (`abs(c.imag) > max(atol, 1e-6 * abs(c))`) and a per-term dict
  insert (`real_terms[label] = float(c.real)`).
- **Variant B (candidate fix)**: one vectorized NumPy check
  (`np.abs`, `np.maximum`, one array comparison, one `.any()`) before
  any per-term work, then `dict(zip(labels, coeffs.real.tolist()))`
  instead of an explicit loop.

Correctness is asserted (`real_terms_a == real_terms_b`) before either
timing number is trusted - not assumed equivalent from reading the
formulas.

## Results

| n_terms | A (today) | B (vectorized + dict(zip)) | speedup |
|---|---|---|---|
| 1,000,000 | 0.772s | 0.243s | 3.18x |
| 10,000,000 | 8.339s | 3.098s | 2.69x |

(An earlier interactive exploration, not separately committed, found
similar numbers - 3.76x/2.58x at the same two sizes - the exact
multiplier varies run to run within a normal range but the effect is
consistently large and in the same direction.)

## Interpretation

**The per-term Hermiticity check dominates within `dict_build`**, not
the dict construction itself, though the dict-construction method
also matters on its own (a separate, smaller effect - see PLAN.md
Phase 11's write-up for the isolated `dict(zip(...))`-vs-explicit-loop
comparison from the original scoping session, not re-run as its own
variant here to keep this script focused on the one change actually
proposed). Vectorizing the check moves an O(n_terms) Python-level
`abs()`/`max()`/comparison sequence into a handful of O(n_terms)
NumPy array operations - the same category of fix as every prior
phase in this project (Phase 3's `pauli_label` C-porting, Phase
6/9's dense-to-sparse accumulator, Phase 10's per-chunk streaming):
moving per-item Python work into vectorized/batched form wherever the
math allows it.

The speedup shrinking somewhat from 1M to 10M terms (3.18x -> 2.69x)
is expected and not a red flag: `.tolist()`'s own cost (present in
both variants) becomes proportionally larger as n grows, so the
*relative* win from removing the per-term check narrows even though
the *absolute* time saved keeps growing.

## What this does NOT show

- Not yet measured against a real N=150 Hamiltonian's actual
  coefficient distribution (all-real per this synthetic setup,
  matching the Hermitian case) - a real run would confirm the same
  effect holds, not just the synthetic proxy.
- Does not yet address design question 1 from PLAN.md Phase 11 (error-
  message specificity when the vectorized check fails) - this
  microbenchmark's synthetic data never triggers the violation branch.
- Does not measure the `assume_hermitian=False` branch (a plain dict
  comprehension today, no per-term check) - expected to benefit only
  from the `dict(zip(...))`-vs-comprehension distinction, not the
  vectorized-check fix, since it has no check to vectorize.
