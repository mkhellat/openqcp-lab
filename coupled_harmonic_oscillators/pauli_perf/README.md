# Pauli Decomposition — Performance Engineering

This module is a performance engineering project, not a tutorial. It
exists to build an **original** fast Pauli decomposition
implementation for the coupled-oscillator Hamiltonian used elsewhere
in `coupled_harmonic_oscillators/`, so that Hamiltonian simulation can
scale to larger N (target N=30, stretch N=100+).


## Problem

`N_coupled_harmonic_oscillators_1_D.ipynb`'s `decompose_to_pauli_terms`
is a dense, symbolic (SymPy), brute-force decomposition: it loops over
all $4^n$ Pauli strings and computes a symbolic trace for each one.
This does not scale past roughly N=4 in practice, which is why the
draft N=30 notebook currently has unfinished (`## to be revised`)
placeholders at exactly this step.


## Approach

This is being built as an original implementation, informed by (but
not copied from) prior work:

- Published algorithms for fast Pauli decomposition, particularly the
  **Fast Walsh-Hadamard Transform (FWHT)** method described in
  [Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d)
  (O(N² log N) for an N×N matrix, vs. the naive O(4ⁿ) approach), and
  related work (Tensorized Pauli Decomposition, PHASE).
- Lessons learned from a private prior C project (allocator design
  pitfalls, autoconf-style build scaffolding) — not its code.
- Performance-engineering methodology from MIT 6.172 course notes
  (Bentley's Rules, profiling-driven optimization, parallelism
  options).

Phasing:

1. **Pure-Python/NumPy implementation first.** Get an original,
   correct FWHT-based decomposition working and validated before
   reaching for C. Cheaper to iterate on, and profiling it tells us
   which specific loop (if any) actually needs a native port, instead
   of guessing.
2. **Profile with real data.** cProfile + snakeviz for an initial
   flame-graph pass, `line_profiler` for granular per-line hot-spot
   drilling, `py-spy` as a no-instrumentation sampling cross-check.
3. **Port only confirmed hot loops to C**, and use that port as a
   structured comparison of binding techniques: Cython first, then
   CFFI, then ctypes, then SWIG — measuring both raw performance and
   binding/build complexity for each.
4. **Parallelize** the confirmed-hot, ported kernel (oneTBB is
   installed on this machine and is a good fit for the
   embarrassingly-parallel per-term/per-row structure of FWHT-based
   decomposition; published results report ~7x speedup on 8 cores,
   matching this machine's core count).


## Reference baseline (not the deliverable)

PennyLane's built-in `qml.pauli_decompose` (already a pinned
dependency of this repo) was used this session purely as a
correctness and performance **reference point** — not as part of the
final implementation. Measured against a `scipy.sparse`-native
encoding of the actual coupled-oscillator Hamiltonian's sparsity
pattern:

| N   | qubits | matrix dim | time (sparse) |
|-----|--------|------------|----------------|
| 16  | 8      | 256        | 0.37s          |
| 30  | 9      | 512        | 1.29s          |
| 50  | 11     | 2048       | 9.37s          |
| 100 | 13     | 8192       | 94.8s          |

Correctness was cross-checked against the repo's own hand-built N=2
Hamiltonian (`prepare_hmatrix` in `N_coupled_harmonic_oscillators_1_D.ipynb`):
reconstructing H from the decomposition matched to machine epsilon
(~1e-16), and the nonzero term count (12) matched the notebook's own
documented claim for N=2.

This module's own implementation will be benchmarked against this
table as it develops.


## Files

- `PLAN.md` — full research trail, phased plan, and rationale.
- `hamiltonian.py` — independent NumPy reimplementation of the
  notebook's `prepare_hmatrix(N)`, used to build test Hamiltonians
  without symbolic (SymPy) overhead. Cross-checked against the
  notebook's SymPy version to machine epsilon at N=2 and N=4.
- `pauli_utils.py` — minimal, dependency-free Pauli-matrix helpers
  (string label <-> matrix, reconstruction from a term dict).
  Deliberately does not depend on PennyLane/Qiskit/Classiq, so it
  stays usable as an independent check regardless of which library
  (if any) an implementation under test happens to use internally.
- `fixtures.py` — correctness fixtures: known-good Hamiltonians and
  their expected Pauli decompositions (generated once via PennyLane's
  `qml.pauli_decompose` as an independent oracle, then stored as
  plain constants — not regenerated automatically). See the module
  docstring for how to regenerate if `hamiltonian.py` ever changes.
- `tests/test_fixtures.py` — validates the fixtures are internally
  consistent (the stored terms actually reconstruct the Hamiltonian),
  independent of any decomposition implementation. Future
  implementations should be tested against `fixtures.ALL_FIXTURES` in
  their own test module, not by re-deriving expected values inline.

Run the fixture tests:

```bash
pytest coupled_harmonic_oscillators/pauli_perf/tests/ -v
```


## Status

Correctness fixtures (N=2, N=4) are in place and self-validated. Next:
the original pure-Python FWHT implementation itself (see PLAN.md
Phase 1). See the repository's task list for current progress.


## Software Requirements

Runtime: `numpy`, `scipy` (already in the top-level `requirements.txt`).

Development/profiling only (not needed to use the final
implementation): see `requirements-dev.txt` in the repository root.
