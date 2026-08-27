"""Microbenchmark isolating dict_build's own sub-costs (PLAN.md Phase
11 scoping) - see phase11_dict_build_scoping_findings.md in this
directory for the results/analysis.

dict_build is the per-chunk Python loop in
fwht_pauli_terms/fwht_pauli_terms_iter that converts (label,
coefficient) pairs to a dict, found by full_pipeline_n150_findings.md
to be ~60% of total pipeline time at N=150. This isolates which
sub-part of that loop actually dominates - the per-term Hermiticity
check, or the dict construction itself - using a synthetic array
rather than a real N=150 run, so this stays fast and safe to iterate
on.

Usage: python dict_build_microbenchmark.py [n_terms]
(default n_terms: 1_000_000)
"""

import sys
import time

import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)
labels = [f"XYZ{i}" for i in range(n)]
coeffs = ((rng.random(n) + 1e-12) + 0j).astype(complex)
atol = 1e-10

# Variant A: today's real code path - per-term Hermiticity check
# (abs/max/compare, evaluated in the Python loop) plus a per-term
# dict insert.
t0 = time.perf_counter()
coeff_list = coeffs.tolist()
real_terms_a = {}
for label, c in zip(labels, coeff_list):
    if abs(c.imag) > max(atol, 1e-6 * abs(c)):
        raise ValueError("unexpected Hermiticity violation in synthetic data")
    real_terms_a[label] = float(c.real)
t_current = time.perf_counter() - t0

# Variant B: vectorized Hermiticity check (NumPy, once for the whole
# chunk) + dict(zip(...)) construction instead of an explicit loop.
t0 = time.perf_counter()
imag_abs = np.abs(coeffs.imag)
real_abs = np.abs(coeffs.real)
violation = imag_abs > np.maximum(atol, 1e-6 * real_abs)
if violation.any():
    raise ValueError("unexpected Hermiticity violation in synthetic data")
real_parts = coeffs.real.tolist()
real_terms_b = dict(zip(labels, real_parts))
t_vectorized = time.perf_counter() - t0

assert real_terms_a == real_terms_b, "vectorized variant must match today's result exactly"

print(f"n_terms={n:,}")
print(f"A (today: per-term check + loop insert):        {t_current:.4f}s")
print(f"B (vectorized check + dict(zip(...)) ctor):      {t_vectorized:.4f}s")
print(f"speedup: {t_current / t_vectorized:.2f}x")
