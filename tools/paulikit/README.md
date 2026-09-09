# paulikit

Exact Pauli decomposition of arbitrary complex matrices, at scales
where materialising the full coefficient set is the binding
constraint.

Any `2^n x 2^n` complex matrix can be written as a weighted sum over
the `4^n` `n`-qubit Pauli strings. That decomposition is what turns a
Hamiltonian into something a quantum algorithm can consume - it is the
input to linear-combination-of-unitaries (LCU) routines, to
Hamiltonian simulation, and to variational methods. Computing it is a
fast Walsh-Hadamard transform, which is well established and cheap in
theory.

The difficulty is not the transform. It is that the output has `4^n`
entries: at 15 qubits a dense decomposition is over a billion
coefficients, and implementations that build the result in memory
before returning it run out of memory long before they run out of
time.

`paulikit` addresses that directly:

- **Streaming output.** Peak resident memory is bounded by the chunk
  size, not by the term count. A 15-qubit decomposition yielding
  91,652,096 surviving terms completes under a 2 GB cap.
- **Exhaustive verification.** Every term is checked individually -
  not sampled - against an independently derived projection oracle.
- **Checkpoint and restart.** A binary chunk-framed checkpoint cheap
  enough to leave permanently enabled, so long decompositions survive
  interruption.
- **Cache-aware parallelism.** Chunk sizing is tuned against measured
  cache boundaries, and multi-core scaling is characterised under a
  documented measurement protocol rather than asserted.

Hermitian input is an optional fast path (`assume_hermitian=True`, the
default), which yields real coefficients and *checks* that assumption
rather than trusting it. It is not a restriction: general
non-Hermitian matrices are fully supported and separately verified.

Requires Python >= 3.10; the only runtime dependency is NumPy.


## Documentation

| | |
|---|---|
| [`docs/installation.md`](docs/installation.md) | Full build and install reference |
| [`docs/tutorial.md`](docs/tutorial.md) | Step-by-step walkthrough |
| [`docs/theory.md`](docs/theory.md) | Mathematical derivation |
| [`docs/background.md`](docs/background.md) | Physical motivation |
| [`docs/non_hermitian.md`](docs/non_hermitian.md) | Non-Hermitian operators |
| [`PLAN.md`](PLAN.md) | Research background and design rationale |


## Installation

```bash
pip install paulikit
```

From source, for development:

```bash
./configure && make          # creates a venv, generates a Makefile
make check                   # run the test suite
```

`./configure` prints a capability report (compiler, Cython, oneTBB,
cache hierarchy, NumPy's BLAS backend) and generates a Makefile with
the standard GNU targets. The build optionally compiles a native
Cython/C++ kernel; if the toolchain is unavailable it falls back to
pure Python automatically, with a warning the first time that path
runs.

See [`docs/installation.md`](docs/installation.md) for the full
reference, including editable-install sequencing and how to force the
native extension on or off.


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
    hamiltonian.py      Coupled-oscillator Hamiltonian construction
    pauli_utils.py      Pauli-matrix helpers (label <-> matrix)
    algorithms/fwht.py  The decomposition algorithm
    testing/fixtures.py Known-good operators and expected outputs
    _native/            Optional compiled kernel, pure-Python fallback
    cli.py              Command-line interface
tests/                  Test suite (pytest)
verification/           Exhaustive correctness runs and their artifacts
docs/                   Tutorial, theory, background, installation
```

See [`docs/package_layout.md`](docs/package_layout.md) for the
annotated tree.


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


## Correctness

`paulikit`'s output is verified three ways:

- **Exhaustive projection.** Every term of a decomposition is checked
  individually against an independently derived projection oracle -
  not sampled - up to 91,652,096 terms at 15 qubits. Artifacts and
  method: [`verification/`](verification/).
- **Cross-implementation.** Where PennyLane's `qml.pauli_decompose`
  can also run, both agree exactly on term count and on coefficients
  within tolerance. PennyLane is a test-only dependency and is never
  imported by `paulikit.algorithms`.
- **Regression suite.** 214 tests, including crash-recovery and
  checkpoint-format cases.

No performance comparison is published here. Benchmark tables in a
README go stale as either implementation changes, and any figure worth
citing has to meet the protocol in
[`profiling/phase13/MEASUREMENT_METHODOLOGY.md`](profiling/phase13/MEASUREMENT_METHODOLOGY.md) -
replicated, interleaved, thermally recorded, with a hypothesis test.
Measured figures live in [`profiling/`](profiling/), each beside the
raw data it came from.


## Status

Alpha. The API is usable and the correctness evidence is strong, but
the version is 0.x and signatures may still change.

Implemented and verified: the FWHT decomposition with a native
label kernel, sparsity-aware coefficients, streaming output with
bounded memory, chunked and parallel execution with cache-aware
auto-tuning, binary checkpoint/restart, and exhaustive verification to
91,652,096 terms.

Known gaps, tracked in [`PLAN.md`](PLAN.md):

- Prebuilt wheels are not yet published, so the native extension
  remains an optional accelerator rather than a hard requirement.
- CPU pinning and topology detection are Linux-only, with a documented
  fallback elsewhere; the non-Linux paths are not yet exercised in CI.
- A parallel-efficiency step at the 14-to-15 qubit boundary is
  measured but not explained
  ([`profiling/phase13/`](profiling/phase13/)).


## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).
