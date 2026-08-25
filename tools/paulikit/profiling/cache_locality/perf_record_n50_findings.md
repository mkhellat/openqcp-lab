# Cache locality investigation — perf record localization

Follow-on to `baseline_perf_stat.md`. Localizes the ~54-56% cache-miss
rate found there to specific code, using
`perf record -g -e cache-misses` + `perf report` on
`paulikit decompose --n-oscillators 50` (raw data not committed -
`perf.data` is large and machine-specific; the commands below
reproduce it).

## Method

```
perf record -g -e cache-misses -o perf_cachemiss_n50.data -- \
  paulikit decompose --n-oscillators 50
perf report -i perf_cachemiss_n50.data --stdio --sort=symbol -g none --percent-limit=1
```

## Finding: the dominant cost is NOT the native/TBB kernel

Flat (self-time) breakdown of cache-miss samples, top entries:

| symbol | self % | what it is |
|---|---|---|
| `_PyEval_EvalFrameDefault` | 11.33% | CPython bytecode interpreter loop |
| `CDOUBLE_subtract_X86_V3` | 9.97% | NumPy's generic complex128 subtract ufunc |
| `CDOUBLE_add_X86_V3` | 9.61% | NumPy's generic complex128 add ufunc |
| `raw_array_assign_array` | 4.76% | NumPy array-assignment internals |
| `_PyObject_Free` / `list_dealloc` / `dict_dealloc` | ~6% combined | CPython object teardown / refcounting |
| `_contig_to_contig` | 2.67% | NumPy dtype-cast/copy internals |

**None of paulikit's native Cython/C++/TBB kernel
(`pauli_label_native`, `pauli_label_parallel.cpp`) appears anywhere
near the top of this list.** The cache misses are overwhelmingly in
NumPy's generic ufunc machinery and CPython's own interpreter/object
lifecycle - i.e. in `fwht_pauli_coefficients`'s Python-level array
math (`algorithms/fwht.py`), not in the compiled extension. This
directly falsifies the framing (both the earlier Google AI Mode
transcript's and, implicitly, our own prior assumption going into this
investigation) that cache locality here is primarily a TBB/native-
kernel concern.

## Root cause identified: `fwht_pauli_coefficients` densifies

`src/paulikit/algorithms/fwht.py`, `fwht_pauli_coefficients`
(lines 202-218) and its caller `fwht_pauli_terms` (lines 297-319):

1. Line 202: `gathered_active = np.zeros((n_active, dim), dtype=complex)`
   - correctly sized to only the active (nonzero) rows. Good.
2. Line 216-217: `coefficients = np.zeros((dim, dim), dtype=complex)` then
   `coefficients[active_x] = active_coefficients` - **scatters the
   sparse result into a full dense `(dim, dim)` array**. At N=50,
   dim=2048, this is a 2048x2048 complex128 array = **64 MiB** - 8x
   larger than this machine's 8 MiB L3 cache, allocated and
   zero-filled on every call regardless of how sparse the actual
   result is.
3. `fwht_pauli_terms` line 299: `np.nonzero(np.abs(coefficients) > atol)`
   - scans the *entire* dense array to find the same nonzero rows
   `fwht_pauli_coefficients` already knew about via `active_x` one
   function call earlier. `np.abs()` alone allocates and touches a
   second full-size `(dim, dim)` float64 temporary (32 MiB more) just
   to build a boolean mask over data that's almost entirely zero.

This is a genuine structural bug, independent of TBB or the native
extension entirely: the function computes a sparse result internally,
then deliberately throws away that sparsity information by
materializing (and immediately re-scanning) a dense array far larger
than any cache level on this machine. This is very plausibly the
dominant source of the cache-miss rate measured in
`baseline_perf_stat.md` - though not yet proven quantitatively (see
"Not yet done" below).

## Why Phase 3b's "sparsity-aware" fix didn't catch this

Phase 3b (see `phase3b/README.md`, `PLAN.md`) made
`fwht_pauli_coefficients`'s *computation* sparsity-aware (skip
computing WHT for empty rows) but did not change its *return type* -
it still returns (and `fwht_pauli_terms` still consumes) a dense
`(dim, dim)` array. The compute-time fix was real and measured
(2.0-3.1x on that function, per `PLAN.md`/`README.md`), but the
memory-layout problem it left behind was never separately identified
or measured until this investigation. Worth being direct about: this
was a genuine gap in Phase 3b's own review, not something introduced
since - it's been sitting in the code the whole time.

## Not yet done

- Quantify how much of the 2.2s N=50 wall time is actually
  attributable to this densification vs. genuinely necessary work -
  `perf record` localizes *where* misses happen, not *how much time*
  they cost (out-of-order execution can hide some memory latency).
  Need a stall-cycle count (`perf stat -e cycle_activity.stalls_l2_miss`
  or similar) before and after any fix to make an honest before/after
  claim.
- A candidate fix (returning/consuming the sparse `(active_x,
  active_coefficients)` representation directly instead of
  densifying) is not yet designed, scoped, or implemented - this
  requires checking every caller of `fwht_pauli_coefficients`
  (public API - used by tests, benchmarks, and potentially external
  code) for backward-compatibility impact before deciding on an
  approach. That's the next step, to be scoped in `PLAN.md` before any
  code changes, per this project's established practice.
