# Phase 3a binding-technique comparison

Per [PLAN.md](../PLAN.md) Phase 3a, the `pauli_label` C kernel
(`src/paulikit/_native/pauli_label.c`) is bound to Python four
separate ways, in this order: Cython, CFFI, ctypes, SWIG. Each
subdirectory here is an independent, standalone build against the
same C sources - none of this is wired into the main `paulikit`
package build yet. That decision happens once all four are built and
benchmarked (task #28 in the repo task list).

## Cython (`cython/`)

Build in place:

```bash
cd bindings/cython
python3 setup.py build_ext --inplace
```

Produces `pauli_label_cy.cpython-<tag>.so` (gitignored - build
locally). `pauli_label_cy.pyx` is the hand-written source (versioned);
`pauli_label_cy.c` is Cython-generated and gitignored, since it
reflows on every rebuild and isn't hand-edited.

**Correctness:** verified two ways before benchmarking -
1. Exhaustive match against the pure-Python `fwht_pauli_terms.pauli_label`
   reference for `n_qubits` 1-4 (340 cases, matching
   `tests/test_fwht.py`'s existing exhaustive-fixture range).
2. 50,000 random `(x, z)` pairs at production `n_qubits` (5, 8, 9, 11,
   13), zero mismatches, single-call path.
3. Batch entry point cross-checked against per-term Python calls at
   `n_qubits` in {8, 11, 13}, 5000 random terms each, exact list match.

**Benchmark** (label generation only, i.e. the `pauli_label` step of
`fwht_pauli_terms`, not the full function - isolates the actual
ported kernel from the surrounding filter/dict-building logic):

| N (oscillators) | qubits | terms | pure Python | Cython batch | speedup |
|---|---|---|---|---|---|
| 16 | 8 | 15,360 | 0.0338s | 0.0011s | 31.0x |
| 30 | 9 | 112,384 | 0.2611s | 0.0074s | 35.5x |
| 50 | 11 | 1,261,568 | 3.8762s | 0.1478s | 26.2x |

**End-to-end impact at N=100** (13 qubits, 20,299,776 terms):
replacing just the label-generation step with the Cython batch call
brings total `fwht_pauli_terms`-equivalent time from the all-Python
baseline's 126.3s (Section 3 benchmark table) down to ~40.2s
(38.6s `fwht_pauli_coefficients` dense-array computation + 1.6s
Cython label generation) - a **3.1x end-to-end speedup**. Confirms
the Phase 2 profiling prediction (labeling was ~60% of runtime at
N=50) and shows that once labeling is no longer the bottleneck, the
dense-array-vs-sparsity issue (Phase 3b) becomes the dominant cost at
large N, exactly as scoped in PLAN.md.

See `../profiling/README.md` for the Phase 2 profiling data this
comparison builds on.
