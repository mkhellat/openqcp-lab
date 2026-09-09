# Package layout

```
src/paulikit/
    __init__.py           Package metadata, public API summary.
    hamiltonian.py         Coupled-oscillator Hamiltonian construction
                            (NumPy implementation; the reference
                            symbolic version it was validated against
                            is kept in the project's research notes).
    pauli_utils.py          Minimal, dependency-free Pauli-matrix
                            helpers (label <-> matrix, reconstruction).
    algorithms/
        __init__.py
        fwht.py             The Fast Walsh-Hadamard Transform based
                            decomposition algorithm (
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
                            "Native extension" in the README.
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


