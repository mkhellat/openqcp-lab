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

## CFFI (`cffi/`)

Build (API mode - compiles a real extension via `set_source`, not
ABI mode's dlopen-a-prebuilt-.so approach, for a fairer comparison
against Cython's also-compiled extension):

```bash
cd bindings/cffi
python3 build_pauli_label_cffi.py
```

Produces `_pauli_label_cffi.cpython-<tag>.so` (gitignored). The
low-level generated extension (`_pauli_label_cffi`) is wrapped by
`pauli_label_cffi.py`, a small hand-written Python module exposing
the same `pauli_label`/`pauli_label_batch` signatures as the Cython
binding, for a like-for-like comparison.

**Correctness:** identical verification suite as Cython (exhaustive
`n_qubits` 1-4, 50,000 random cases at production sizes, batch
cross-check) - all pass, zero mismatches.

**Benchmark** (same methodology as Cython's table above):

| N (oscillators) | qubits | terms | pure Python | CFFI batch | speedup |
|---|---|---|---|---|---|
| 16 | 8 | 15,360 | 0.0378s | 0.0053s | 7.2x |
| 30 | 9 | 112,384 | 0.3051s | 0.0415s | 7.4x |
| 50 | 11 | 1,261,568 | 3.3028s | 0.4215s | 7.8x |

**End-to-end impact at N=100:** 40.578s `fwht_pauli_coefficients` +
7.228s CFFI label generation = ~47.8s (vs. 126.3s all-Python baseline)
- a **2.6x end-to-end speedup**, real but noticeably less than
Cython's 3.1x.

**Why CFFI is slower than Cython here:** the C call itself is
presumably comparable, but `pauli_label_cffi.py`'s batch wrapper has
to unpack the raw C buffer back into Python `str` objects one term at
a time via `ffi.buffer()` slicing + `bytes()` + `.decode()`, whereas
the Cython wrapper does the equivalent unpacking with `cdef`-typed
loop variables and direct buffer indexing - considerably less
per-element Python-object overhead. This is a real, reproducible
difference in wrapper-code cost, not a difference in the underlying C
kernel (which is bit-for-bit identical between the two bindings).

## ctypes (`ctypes/`)

Stdlib-only, no separate Python-extension build step - but ctypes
only `dlopen()`s an already-built shared library, so a plain `cc`
invocation is still needed first:

```bash
cd bindings/ctypes
./build_pauli_label_shared   # compiles libpauli_label.so
```

`pauli_label_ctypes.py` loads `libpauli_label.so` and exposes the
same `pauli_label`/`pauli_label_batch` signatures as the other
bindings.

**Correctness:** identical verification suite as Cython/CFFI -
exhaustive `n_qubits` 1-4, 50,000 random cases at production sizes,
batch cross-check. All pass, zero mismatches.

**Benchmark** (same methodology as above):

| N (oscillators) | qubits | terms | pure Python | ctypes batch | speedup |
|---|---|---|---|---|---|
| 16 | 8 | 15,360 | 0.0332s | 0.0048s | 7.0x |
| 30 | 9 | 112,384 | 0.2433s | 0.0267s | 9.1x |
| 50 | 11 | 1,261,568 | 3.3944s | 0.3051s | 11.1x |

**End-to-end impact at N=100:** 37.512s `fwht_pauli_coefficients` +
5.768s ctypes label generation = ~43.3s (vs. 126.3s all-Python
baseline) - a **2.9x end-to-end speedup**. In the same range as CFFI
(2.6x) and, at this scale, edges slightly ahead of it - both well
behind Cython's 3.1x/26-35x, for the same reason noted in the CFFI
section: the batch-unpacking loop back into Python `str` objects
(here, slicing `ctypes.create_string_buffer.raw` + `.decode()` per
term) carries more per-element overhead than Cython's typed-loop
direct indexing. The underlying C kernel and call pattern are
otherwise the same across all three bindings so far.

## SWIG (`swig/`)

Requires the `swig` package (`sudo pacman -S swig`; not a Python
package, so it isn't in `dev` extras). Build:

```bash
cd bindings/swig
python3 setup.py build_ext --inplace
```

Produces `_pauli_label_swig.cpython-<tag>.so` (gitignored) plus
SWIG-generated `pauli_label_wrap.c` and `pauli_label_swig.py`
(both gitignored - regenerated from `pauli_label.i` on every build,
never hand-edited). `pauli_label_swig_wrapper.py` is the hand-written
Python module exposing the same `pauli_label`/`pauli_label_batch`
signatures as the other three bindings.

**This binding needed real typemap work**, unlike the other three:
neither C function in `pauli_label.h` maps onto SWIG's default
argument handling, since both write into a caller-supplied buffer
via a raw `char *` parameter rather than returning a value. `pauli_label.i`
hand-writes: (1) an `argout` typemap so `pauli_label_str` allocates a
small stack buffer internally and returns a Python `str`, and (2)
`Py_buffer`-based `in` typemaps so `pauli_label_batch_raw` accepts
NumPy arrays and a writable Python buffer directly, without SWIG
owning any allocation - matching the C kernel's actual
no-allocation contract. This is a concrete instance of the
"three-language complexity cost" / "heaviest boilerplate" trade-off
PLAN.md's Section 4 tooling survey predicted for SWIG going in -
confirmed here, not assumed.

**Correctness:** identical verification suite as the other three
bindings - exhaustive `n_qubits` 1-4, 50,000 random cases at
production sizes, batch cross-check. All pass, zero mismatches.

**Benchmark** (same methodology as above):

| N (oscillators) | qubits | terms | pure Python | SWIG batch | speedup |
|---|---|---|---|---|---|
| 16 | 8 | 15,360 | 0.0337s | 0.0053s | 6.4x |
| 30 | 9 | 112,384 | 0.2429s | 0.0351s | 6.9x |
| 50 | 11 | 1,261,568 | 3.4679s | 0.5305s | 6.5x |

**End-to-end impact at N=100:** 35.734s `fwht_pauli_coefficients` +
6.600s SWIG label generation = ~42.3s (vs. 126.3s all-Python
baseline) - a **3.0x end-to-end speedup**, essentially tied with
ctypes (2.9x) and CFFI (2.6x), all behind Cython's 3.1x - despite
SWIG requiring substantially more binding-code effort (hand-written
typemaps) than any of the other three to get there.

## Summary across all four bindings

| binding | label-gen speedup (N=50) | end-to-end speedup (N=100) | binding effort |
|---|---|---|---|
| Cython | 26.2x | 3.1x | Low - typed Python-like syntax, no typemaps needed |
| CFFI | 7.8x | 2.6x | Low - declarative `cdef`-style header, small wrapper |
| ctypes | 11.1x | 2.9x | Low - stdlib only, manual `argtypes`/`restype` |
| SWIG | 6.5x | 3.0x | High - hand-written typemaps required for buffer args |

Cython is the clear winner on raw label-generation speed (its
typed-loop unpacking avoids the per-term Python-object overhead all
three C-binding techniques otherwise share) and ties for best
end-to-end result, for the lowest binding effort of the four. See
task #28 for the final decision on which binding `paulikit` ships
with, once oneTBB parallelization (task #29) is also factored in -
that may change the relative picture, since it targets throughput at
the C level rather than the Python-unpacking layer these numbers are
currently bottlenecked on for CFFI/ctypes/SWIG.

See `../profiling/README.md` for the Phase 2 profiling data this
comparison builds on.
