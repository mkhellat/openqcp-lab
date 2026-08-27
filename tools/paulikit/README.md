# paulikit

Performance-engineering tools for Pauli decomposition of Hermitian
operators. This package is not a tutorial — it exists to build
**original**, fast Pauli decomposition implementations to scale the
`coupled_harmonic_oscillators` Hamiltonian-simulation tutorial (in the
parent `openqcp-lab` repository) to larger N than its original
symbolic brute-force approach allows.

See [`PLAN.md`](PLAN.md) for the full research background, phased
plan, and design rationale.

To use the library, start with Installation and Usage below, or the
step-by-step tutorial ([`docs/tutorial.md`](docs/tutorial.md)). For
the physical motivation and the algorithm's mathematical derivation,
see [`docs/background.md`](docs/background.md) and
[`docs/theory.md`](docs/theory.md); for non-Hermitian operators
specifically, see [`docs/non_hermitian.md`](docs/non_hermitian.md).


## Problem

The tutorial notebook's Pauli decomposition is a dense, symbolic
(SymPy), brute-force approach: it loops over all $4^n$ Pauli strings
and computes a symbolic trace for each one. This does not scale past
roughly N=4 oscillators in practice.

`paulikit` implements faster, original algorithms as a standalone
Python package, kept separate from the tutorial notebooks so it can be
developed, tested, and (potentially) published independently.


## Installation

paulikit is built with [meson-python](https://mesonbuild.com/meson-python/)
(the same build backend NumPy and SciPy use), and optionally compiles a
native (Cython/C++) `pauli_label` kernel for a ~2.5-2.9x end-to-end
speedup — see [Native extension](#native-extension) below.

From this directory (editable install, recommended for development):

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required for editable installs: without it,
NumPy's include path gets baked in from a throwaway build-isolation
environment that goes stale on later rebuilds (this is NumPy's own
documented practice for meson-python editable installs, not a
paulikit-specific quirk). A regular, non-editable `pip install .` does
not need the flag.

With test/profiling dependencies:

```bash
pip install -e ".[test]" --no-build-isolation   # pytest, PennyLane (for fixture regeneration/reference checks), scipy
pip install -e ".[dev]" --no-build-isolation    # the above, plus snakeviz, line_profiler, py-spy
pip install -e ".[sparse]" --no-build-isolation # just scipy, for build_hamiltonian(sparse=True) / pad_to_power_of_two(sparse=True)
```

Requires Python >= 3.10. Runtime dependencies are just `numpy` — the
core algorithms have no dependency on PennyLane, Qiskit, or Classiq;
those are only used in the `test`/`dev` extras, for generating and
cross-checking correctness fixtures. `scipy` is likewise optional:
only needed for the `sparse=True` path on `build_hamiltonian`,
`pad_to_power_of_two`, and (as an input type)
`fwht_pauli_coefficients`/`fwht_pauli_terms` (see `PLAN.md` Phase 8) -
calling any of those with `sparse=True` (or passing a
`scipy.sparse` operator directly) without `scipy` installed raises a
clear `ImportError` naming the `sparse` extra, rather than silently
falling back to the dense path. Building from source always
requires `meson-python`, `Cython`, and `numpy` (PEP 517/518's
`[build-system] requires` has no conditional mechanism), but both are
pure-Python-installable — no C toolchain is needed just to build the
pure-Python parts of the package.


### Native extension

By default (`-Dnative=auto`) the build compiles
`paulikit._native.pauli_label_native`, a Cython/C++ port of the
per-term label-generation kernel, if a C++ toolchain and
[oneTBB](https://github.com/oneapi-src/oneTBB) are available. If they
aren't, the build falls back to pure Python automatically — no error,
just slower label generation, with a one-time `UserWarning` the first
time the fallback path actually runs.

To force the behavior explicitly:

```bash
# Fail the build if the native extension can't be compiled:
pip install -e . --no-build-isolation --config-settings=setup-args="-Dnative=enabled"

# Force pure-Python-only, even if a toolchain is available:
pip install -e . --no-build-isolation --config-settings=setup-args="-Dnative=disabled"
```

The native extension is currently an optional, best-effort
accelerator, not a hard requirement — paulikit has no prebuilt-wheel
CI yet, so requiring a C++ toolchain for every `pip install` would be
too heavy a default. This is a deliberate, temporary trade-off, not
a permanent architecture decision — see `PLAN.md` Phase 3c for the
full rationale. Migrating to prebuilt wheels (so the extension can
become a hard requirement, matching the NumPy/SciPy model) is tracked
as a near-term goal, not indefinitely deferred.


## Usage

### Command line

Once installed, the `paulikit` console script is available:

```bash
paulikit --help
paulikit decompose --n-oscillators 4 --show-terms
paulikit benchmark --n-oscillators 2 4 8 16 30
paulikit regenerate-fixtures
```

Run `paulikit <subcommand> --help` for full details on each.

### As a library

```python
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
H_padded, n_qubits = pad_to_power_of_two(H)

terms = fwht_pauli_terms(H_padded)  # {"IXI": -0.556..., "XII": -0.354..., ...}
```


## Package layout

```
src/paulikit/
    __init__.py           Package metadata, public API summary.
    hamiltonian.py         Coupled-oscillator Hamiltonian construction
                            (independent NumPy reimplementation of the
                            tutorial notebook's SymPy version).
    pauli_utils.py          Minimal, dependency-free Pauli-matrix
                            helpers (label <-> matrix, reconstruction).
    algorithms/
        __init__.py
        fwht.py             The Fast Walsh-Hadamard Transform based
                            decomposition algorithm (see PLAN.md;
                            more algorithms planned here).
    testing/
        __init__.py
        fixtures.py         Known-good Hamiltonians and their
                            independently-verified expected Pauli
                            decompositions, for use by any algorithm's
                            tests.
    _native/                Optional compiled extension
                            (`pauli_label_native`, Cython/C++, wraps
                            `pauli_label.c`/`pauli_label_parallel.cpp`)
                            used by `algorithms/fwht.py` when available,
                            with a pure-Python fallback otherwise — see
                            "Native extension" above and PLAN.md Phase 3c.
    cli.py                  Command-line interface wiring the above
                            together into subcommands.
    meson.build             Per-directory Meson build rules (one per
                            subpackage above, plus a top-level
                            `meson.build` and `meson.options` at the
                            repository root of this package).
tests/
    test_fixtures.py        Self-consistency checks for the fixtures.
    test_fwht.py             Correctness tests for algorithms/fwht.py.
```

`hamiltonian.py` and `pauli_utils.py` sit at the package root (not
under `algorithms/`) because they are not algorithm-specific: every
current and planned decomposition algorithm needs the same Hamiltonian
construction and the same Pauli-matrix utilities.


## Running the tests

```bash
pytest
```

(from this directory; `pyproject.toml` sets `testpaths = ["tests"]`,
and the package must be installed - `pip install -e ".[test]"` - for
imports to resolve).


## Algorithms implemented

### Fast Walsh-Hadamard Transform (FWHT) — `paulikit.algorithms.fwht`

O(N² log N) for an N×N matrix (N = 2ⁿ), per
[Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d).
This is an **original implementation**: the algorithm's three steps
(XOR-index gather, Walsh-Hadamard Transform, phase-factor
multiplication) were independently re-derived from the symplectic
(X/Z) representation of Pauli operators and verified against a
from-scratch, definition-level brute-force decomposition before being
written in fast form — see `algorithms/fwht.py`'s module docstring for
the full derivation.

Verified two ways (see `tests/test_fwht.py`):
- Against a from-scratch brute-force reference on random Hermitian
  matrices (n = 1..4 qubits): exact match to floating-point precision.
- Against `testing.fixtures.ALL_FIXTURES` (real coupled-oscillator
  Hamiltonians at N=2, N=4): exact label-set and coefficient match.

Planned (see `PLAN.md`): Tensorized Pauli Decomposition (TPD), PHASE,
and C-ported variants of whichever algorithm profiling identifies as
worth porting — this is why `algorithms/` is a subpackage rather than
a single module.


## Reference baseline (not a dependency of the implementation)

PennyLane's `qml.pauli_decompose` was used during development purely
as a correctness and performance **reference point** — it is a
test/dev-only dependency (see `pyproject.toml`'s `test`/`dev` extras),
never imported by `paulikit.algorithms` itself. Both implementations
were run on the *exact same* `build_hamiltonian()` output at each N
(see `tests/test_benchmark_reference.py`, marked `slow` and excluded
from the default test run):

| N (oscillators) | qubits | Pauli terms | paulikit time | PennyLane time    | speedup |
|------------------|--------|-------------|----------------|--------------------|---------|
| 16               | 8      | 15360       | 0.0124s        | 5.6914s            | 459x    |
| 30               | 9      | 112384      | 0.0937s        | 45.9782s           | 491x    |
| 50               | 11     | 1261568     | 1.2371s        | 749.9998s          | 606x    |
| 100              | 13     | 20299776    | 24.6979s       | not attempted      | —       |

Both implementations agree exactly on term count at every N where
PennyLane finished (a correctness check, not just a performance one).
Unlike the original 2026-08-04 measurement, the N=50 PennyLane run
here completed rather than aborting on a timeout - both this test and
the reference implementation have since improved, and the machine load
at measurement time affects wall-clock numbers, so treat the exact
seconds as illustrative, not as a tight guarantee. N=100 still was not
attempted against PennyLane: a direct attempt (2026-08-26) ran for
over 26 minutes without finishing and was deliberately killed rather
than left running, both to avoid an indefinite wait and because the
machine's available memory was under real pressure by that point (see
`profiling/cache_locality/README.md`'s swap/resource-exhaustion notes
for what that looked like in practice). See `PLAN.md` Section 3.4 for
the full discussion, including why an earlier draft of this table
(using a synthetic proxy matrix rather than the real Hamiltonian)
understated PennyLane's actual cost on this problem.

`paulikit`'s own numbers above reflect the current, Phase 6-complete
implementation (native label kernel plus the sparse
`fwht_pauli_coefficients` output added in Phase 6 - see
`profiling/cache_locality/README.md`), not the Phase 1 pure-Python
baseline this table originally reported. That baseline densely
computed the full $2^n \times 2^n$ coefficient array regardless of
input sparsity, and generated Pauli-string labels with a per-term,
per-qubit Python loop. Both of those were identified as the actual
bottlenecks (via profiling, not guesswork) and fixed across Phases
3a-3c and 6:

| N (oscillators) | Phase 1 (baseline) | Phase 3b (sparse coefficients) | Phase 3c (+ native labels) | speedup vs. Phase 1 |
|------------------|---------------------|----------------------------------|-------------------------------|----------------------|
| 50               | 6.2213s             | 5.4957s                          | 2.1535s                       | 2.9x                |
| 100              | 126.3250s           | 107.2403s                        | 43.5629s                      | 2.9x                |

Phase 3b made `fwht_pauli_coefficients` skip the O(dim²) dense-array
construction for empty rows (2.0-3.1x on that function alone). Phase
3c wired in the native `pauli_label` kernel (see "Native extension"
above), closing most of the remaining gap. Term counts match exactly
across all versions at every N — a correctness re-confirmation, not
just a performance comparison. Full detail, including the design
exploration behind Phase 3b's sparsity fix, is in `PLAN.md`'s Section
5 (Phase 3b/3c write-ups) and `profiling/phase3b/README.md`.

This table does not yet include a Phase 6 column: Phase 6 (an
optional sparse-output mode for `fwht_pauli_coefficients`, see
`profiling/cache_locality/README.md`) removed a different cost - the
dense array's cache-locality/memory-footprint penalty and its
associated OOM risk at large N, not primarily wall-clock time on
these already-fast N=50/100 cases - so it's measured and reported
separately once that comparison sweep completes, rather than folded
into this table's per-phase story.

Two further memory-footprint fixes landed after Phase 6, still aimed
at the N=150 OOM rather than at wall-clock time on smaller N:

- `_walsh_hadamard_transform_rows` no longer force-copies its input; a
  new `overwrite_input` flag lets `fwht_pauli_coefficients` (which owns
  its gathered array exclusively) transform in place, and the
  per-stage butterfly loop no longer pre-copies both halves before
  combining them. Together these removed roughly 5.5 GiB of transient
  allocation at N=150.
- `fwht_pauli_coefficients` and `fwht_pauli_terms` gained an optional
  `chunk_size` parameter: active rows are processed in bounded blocks
  instead of one `(n_active, dim)` array all at once, bounding peak
  memory to roughly `chunk_size * dim` complex entries. This is the
  tiling technique from MIT 6.172 ("Performance Engineering of
  Software Systems") lecture 1, applied here to shrink working-set
  size rather than to improve cache reuse specifically, since each row
  transforms independently. `--chunk-size` is exposed on both the
  `decompose` and `benchmark` CLI subcommands. It is a manual knob for
  now, not auto-tuned - see the open item below.

Investigating N=150 with chunking surfaced a much larger, separate
bottleneck that neither fix touches: the coupled-oscillator
Hamiltonian is built and stored as a **dense** `numpy.ndarray` even
though it is extremely sparse in practice (at N=150, only 0.034% of
entries are nonzero), and `fwht_pauli_coefficients` upcasts that dense
array to `complex128` before doing anything else - roughly 4 GiB at
N=150, dwarfing every fix described above. `PLAN.md` Phase 8 (2026-08-27)
closes this: `build_hamiltonian(..., sparse=True)` and
`pad_to_power_of_two(..., sparse=True)` keep the Hamiltonian in
`scipy.sparse` form from construction through padding, and
`fwht_pauli_coefficients`/`fwht_pauli_terms` accept a `scipy.sparse`
operator directly - `scipy` is optional (see Installation above),
gated behind a clear `ImportError` rather than a silent dense
fallback. A real N=150 attempt with this in place confirms the
densification ceiling is cleared, but surfaced a **new, distinct**
memory ceiling one step later, inside `fwht_pauli_coefficients`'s own
output accumulator (unrelated to Hamiltonian sparsity). `PLAN.md`
Phase 9 (2026-08-27) closes that too: the chunked path now thresholds
each chunk against `atol` immediately and accumulates only surviving
`(x, z, coefficient)` triples in an amortized-growth array, instead of
one dense `(n_active, dim)` block regardless of `chunk_size` - plus an
opt-in `checkpoint_path` for crash/resume. Confirmed via real,
memory-capped N=150 runs (see `profiling/phase9/phase9_findings.md`):
the fix works exactly as designed, but the genuine result size at
N=150 (`atol=1e-10`) is ~134M terms (~4.3 GiB), which exceeds this
machine's available RAM once label-string generation and dict
construction are added on top - `fwht_pauli_terms`'s fully-materialized
`dict` contract has an inherent ceiling no memory size removes in
general. A mandatory streaming/generator output is scoped as `PLAN.md`
Phase 10 (not yet designed).

Exploiting Hamiltonian sparsity further (rather than the current
skip-empty-rows approach) and migrating to prebuilt wheels so the
native kernel becomes a hard requirement are the next optimization
targets — see `PLAN.md` Phase 4.


## Status

Phases 0-3c are complete: the original pure-Python FWHT implementation
and PennyLane benchmarking (Phase 1), profiling to find real hot spots
(Phase 2), a native `pauli_label` kernel with four bindings compared
(Phase 3a), a sparsity-aware `fwht_pauli_coefficients` (Phase 3b), and
migrating the build to meson-python with the native kernel wired into
the main `fwht_pauli_terms` pipeline (Phase 3c) — together a 2.9x
end-to-end speedup over the Phase 1 baseline at N=50/100 (see
"Reference baseline" above). Next: Phase 4 (final comparison/write-up)
and migrating to prebuilt wheels so the native extension becomes a
hard requirement rather than an optional fallback — see `PLAN.md` and
the parent repository's task list for current progress.


## License

GPL-3.0-or-later, matching the parent `openqcp-lab` repository. See
the repository root's `LICENSE` file.
