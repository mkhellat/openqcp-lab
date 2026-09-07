"""Step 3: is `d5` (dict build) actually GIL-bound, or does part of it
release the GIL?

WHY THIS EXISTS. In summarizing the Phase-2 falsification result I
asserted that a threaded drain loop "won't help, dict inserts hold the
GIL" and stopped there. That was an assertion, not a measurement, and
`_build_real_terms` (fwht.py:771-783) is NOT purely dict inserts - it
is a mix:

    c_abs     = np.abs(coefficient_values)          # NumPy, releases GIL
    imag_abs  = np.abs(coefficient_values.imag)     # NumPy, releases GIL
    violation = imag_abs > np.maximum(atol, ...)    # NumPy, releases GIL
    violation.any()                                 # NumPy, releases GIL
    dict(zip(labels, coefficient_values.real.tolist()))
                       # .tolist() releases the GIL building the list;
                       # dict(zip(...)) does NOT - pure interpreter work

If the GIL-releasing half is a large fraction of d5, a threaded drain
loop could overlap that half across threads and is worth considering.
If it is a small fraction, the dismissal was right - but it will then
be right on the basis of a measurement instead of an assumption.

The same question applies to `d4`: the labeling scaling test showed
~62% of it is the Cython wrapper's per-term Python str loop
(GIL-held), with the remainder the C kernel. That C kernel does NOT
release the GIL either (`pauli_label_native.pyx` has no `with nogil:`
block - checked directly), so d4 is ~fully GIL-bound. Recorded here
for completeness rather than re-measured.

FALSIFIABLE: if GIL-releasing work is >40% of d5, my "threads won't
help" claim is WRONG and must be retracted. If it is <15%, the claim
stands on evidence.

Standalone microbenchmark - no pool, no contention. Measures the
composition of d5 at the real per-chunk term count, which is what the
threading question turns on. It deliberately does NOT try to predict a
wall-clock speedup; that would need the real contended run.
"""

import time

import numpy as np

from paulikit.algorithms.fwht import _build_real_terms, _pauli_label_batch

# Real per-chunk scale at N=150/chunk_size=2: T_x/C = 91,652,096/5595.
TERMS_PER_CHUNK = 91_652_096 // 5595
N_QUBITS = 14
REPS = 50
ATOL = 1e-10


def _make_inputs(n_terms):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**N_QUBITS, n_terms, dtype=np.uint32)
    z = rng.integers(0, 2**N_QUBITS, n_terms, dtype=np.uint32)
    labels = _pauli_label_batch(x, z, N_QUBITS)
    # Real coefficients are Hermitian-clean (negligible imag) - matching
    # that matters, since a violation would take the error branch.
    coeffs = (rng.standard_normal(n_terms) + 0j).astype(complex)
    return labels, coeffs


def _time(fn, reps=REPS):
    for _ in range(5):  # warm up
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    labels, coeffs = _make_inputs(TERMS_PER_CHUNK)
    print(f"terms/chunk = {TERMS_PER_CHUNK}  n_qubits = {N_QUBITS}  reps = {REPS}")
    print()

    full = _time(lambda: _build_real_terms(labels, coeffs, ATOL))

    # GIL-RELEASING half: the vectorized Hermiticity check + .tolist().
    def gil_releasing():
        c_abs = np.abs(coeffs)
        imag_abs = np.abs(coeffs.imag)
        violation = imag_abs > np.maximum(ATOL, 1e-6 * c_abs)
        violation.any()
        return coeffs.real.tolist()

    releasing = _time(gil_releasing)

    # GIL-HELD half: dict(zip(...)) over an ALREADY-materialized list,
    # so no NumPy work is counted here.
    values = coeffs.real.tolist()
    held = _time(lambda: dict(zip(labels, values)))

    print(f"{'component':<44} {'ms':>9} {'% of d5':>9}")
    print(f"{'full _build_real_terms (d5)':<44} {full*1e3:>9.3f} {100.0:>8.1f}%")
    print(f"{'  GIL-RELEASING (np checks + .tolist())':<44} "
          f"{releasing*1e3:>9.3f} {100*releasing/full:>8.1f}%")
    print(f"{'  GIL-HELD (dict(zip(...)))':<44} "
          f"{held*1e3:>9.3f} {100*held/full:>8.1f}%")
    print(f"{'  [sum of parts vs full]':<44} "
          f"{(releasing+held)*1e3:>9.3f} {100*(releasing+held)/full:>8.1f}%")

    pct = 100 * releasing / full
    print()
    print(f"GIL-releasing share of d5 = {pct:.1f}%")
    if pct > 40:
        print("--> 'threads won't help' is WRONG and must be retracted:")
        print("    a large fraction of d5 can overlap across threads.")
    elif pct < 15:
        print("--> 'threads won't help' STANDS, now on measurement:")
        print("    d5 is dominated by GIL-held dict insertion.")
    else:
        print("--> INCONCLUSIVE (15-40%): threading could recover some")
        print("    of d5 but not most of it - not a clean answer either way.")

    print()
    print("NOTE: sum-of-parts need not equal the full call exactly -")
    print("the parts re-allocate intermediates the fused call reuses.")
    print("The SHARE is the finding, not the absolute reconstruction.")


if __name__ == "__main__":
    main()
