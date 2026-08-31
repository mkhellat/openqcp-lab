"""Break down _build_real_terms's own remaining sub-costs post-Phase-11
(the shared helper in src/paulikit/algorithms/fwht.py) to see where the
27.64s (39.1% of total pipeline time at N=150,
profiling/phase11/n150_post_implementation_findings.md) actually goes
now, before scoping any further optimization (dict_build itself, or as
a comparison point for the WHT-butterfly GPU question).

Usage: python dict_build_phase11b_microbenchmark.py [n_terms]
"""

import sys
import time

import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
rng = np.random.default_rng(0)
labels = [f"XYZ{i}" for i in range(n)]
atol = 1e-10
real_part = rng.random(n) + 1e-12
imag_part = (rng.random(n) - 0.5) * 1e-8 * real_part
coeffs = (real_part + 1j * imag_part).astype(complex)

# Sub-step 1: np.abs(coefficient_values) (full complex magnitude).
t0 = time.perf_counter()
c_abs = np.abs(coeffs)
t_c_abs = time.perf_counter() - t0

# Sub-step 2: np.abs(coefficient_values.imag).
t0 = time.perf_counter()
imag_abs = np.abs(coeffs.imag)
t_imag_abs = time.perf_counter() - t0

# Sub-step 3: the comparison itself (np.maximum + >).
t0 = time.perf_counter()
violation = imag_abs > np.maximum(atol, 1e-6 * c_abs)
_ = violation.any()
t_compare = time.perf_counter() - t0

# Sub-step 4: coefficient_values.real.tolist() (NumPy -> Python floats).
t0 = time.perf_counter()
real_list = coeffs.real.tolist()
t_real_tolist = time.perf_counter() - t0

# Sub-step 5: dict(zip(labels, real_list)) construction itself.
t0 = time.perf_counter()
result = dict(zip(labels, real_list))
t_dict_zip = time.perf_counter() - t0

total = t_c_abs + t_imag_abs + t_compare + t_real_tolist + t_dict_zip

print(f"n_terms={n:,}")
print(f"c_abs = np.abs(coeffs):            {t_c_abs:.4f}s ({100*t_c_abs/total:5.1f}%)")
print(f"imag_abs = np.abs(coeffs.imag):     {t_imag_abs:.4f}s ({100*t_imag_abs/total:5.1f}%)")
print(f"violation compare + .any():         {t_compare:.4f}s ({100*t_compare/total:5.1f}%)")
print(f"coeffs.real.tolist():               {t_real_tolist:.4f}s ({100*t_real_tolist/total:5.1f}%)")
print(f"dict(zip(labels, real_list)):        {t_dict_zip:.4f}s ({100*t_dict_zip/total:5.1f}%)")
print(f"total (sum of sub-steps):           {total:.4f}s")
assert len(result) == n
