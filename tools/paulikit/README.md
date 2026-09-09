# paulikit

Performance-engineering tools for Pauli decomposition of arbitrary
complex matrices. Hermitian input is an optional fast path
(`assume_hermitian=True`, the default, which yields real coefficients
and *checks* the Hermiticity assumption rather than trusting it) — not
a restriction. General non-Hermitian matrices are fully supported and
are verified against both an independent projection oracle and
PennyLane (`verification/results/N20_nonhermitian_20260828.json`).
This package is not a tutorial — it exists to build
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
native (Cython/C++) `pauli_label` kernel, which materially reduces
label-generation cost — see [Native extension](#native-extension)
below. Figures live in `profiling/`, beside the data they came from,
rather than as a ratio here that would go stale.

**Recommended for development:** run `./configure` from this
directory. It creates/reuses a dedicated venv (default
`~/.venvs/paulikit`, deliberately outside the source tree — meson
rejects an absolute in-tree numpy include path if the venv lives
inside `tools/paulikit/`), prints an itemized capability/environment
diagnostic report (compiler, Cython, TBB, cache hierarchy, NumPy's
BLAS backend, etc. — see `PLAN.md`'s Phase 0.5 for the full list and
rationale), and generates a `Makefile` with the standard GNU set of
targets (`all`/`build`/`install`/`install-strip`/`installdirs`/
`check`/`test`/`installcheck`/`uninstall`/`docs`/`dist`/`TAGS`/
`clean`/`mostlyclean`/`distclean`/`maintainer-clean`/`report`) that
handle the `--no-build-isolation` editable-install sequencing below
correctly and automatically:

```bash
./configure                 # accepts --prefix/--docdir/--srcdir/VAR=value
                             # and the standard GNU dir-var options too, see --help
make                         # = make all = make build (editable install)
make check                   # run the test suite
```

If you'd rather not use `./configure`, the manual sequence it
automates is:

```bash
pip install numpy meson-python cython ninja
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required for editable installs: without it,
NumPy's include path gets baked in from a throwaway build-isolation
environment that goes stale on later rebuilds (this is NumPy's own
documented practice for meson-python editable installs, not a
paulikit-specific quirk) — and `numpy`/`meson-python`/`cython`/`ninja`
must already be installed in the target environment first, since
`--no-build-isolation` means pip won't fetch them into a throwaway
env for you the way it normally would. A regular, non-editable
`pip install .` does not need any of this.

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


## Reference implementations (not dependencies)

PennyLane's `qml.pauli_decompose` is used during development as a
**correctness** reference — it is a test/dev-only dependency (see
`pyproject.toml`'s `test`/`dev` extras) and is never imported by
`paulikit.algorithms` itself. Where both run to completion on the same
`build_hamiltonian()` output, they agree exactly on term count and on
coefficients within tolerance; that agreement is recorded in the
committed verification artifacts under `verification/results/`.

**No performance comparison is published here, deliberately.** A
speedup table in a README is a claim about two moving targets: it
decays as this library changes, as the reference library changes, and
as the machine it was measured on changes. Any such number would also
have to meet the protocol in
[`profiling/phase13/MEASUREMENT_METHODOLOGY.md`](profiling/phase13/MEASUREMENT_METHODOLOGY.md)
— replicated, interleaved, warmed, thermally recorded, with a real
hypothesis test — which a hand-maintained README table cannot
guarantee over time.

Timing figures that *are* published live in `profiling/`, each beside
the raw data it was computed from and the protocol under which it was
taken. Read them there, with their sample sizes and error bars, rather
than as a headline ratio here.

What this README does claim, and what the artifacts support:

- **Scale.** N=150 (15 qubits, 91,652,096 surviving terms) completes
  under a 2 GB memory cap, on a laptop.
- **Exactness.** Every one of those terms is verified individually —
  not sampled — against an independently derived projection oracle;
  see `verification/FINDINGS.md`.
- **Generality.** Hermitian input is a fast path, not a restriction;
  non-Hermitian matrices are supported and separately verified.

For why the algorithmic approach differs from a dense reference
implementation — sparsity-aware coefficients, a native label kernel,
streaming output — see `PLAN.md` Section 5 and `profiling/phase3b/README.md`.

## Status

Phases 0-3c are complete: the original pure-Python FWHT implementation
and PennyLane benchmarking (Phase 1), profiling to find real hot spots
(Phase 2), a native `pauli_label` kernel with four bindings compared
(Phase 3a), a sparsity-aware `fwht_pauli_coefficients` (Phase 3b), and
migrating the build to meson-python with the native kernel wired into
the main `fwht_pauli_terms` pipeline (Phase 3c) — together a
substantial end-to-end reduction over the Phase 1 baseline, quantified
in `profiling/` beside its raw data rather than as a ratio here.
Next: Phase 4 (final comparison/write-up)
and migrating to prebuilt wheels so the native extension becomes a
hard requirement rather than an optional fallback — see `PLAN.md` and
the parent repository's task list for current progress.


## License

GPL-3.0-or-later, matching the parent `openqcp-lab` repository. See
the repository root's `LICENSE` file.
