"""Is dict construction itself the floor, or is there a cheaper
container that avoids per-entry hashing? Compares dict(zip(...))
against just zipping into a list of tuples (no hashing at all) and
against building without ever materializing Python str/float objects.
"""
import sys
import time
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
rng = np.random.default_rng(0)
labels = [f"XYZ{i}" for i in range(n)]
real_part = rng.random(n) + 1e-12
real_list = real_part.tolist()

# Variant A: dict(zip(...)) - today's real code (hashes n strings).
t0 = time.perf_counter()
d_a = dict(zip(labels, real_list))
t_a = time.perf_counter() - t0

# Variant D: list(zip(...)) - no hashing, just tuple packing (lower bound
# for "materialize label+value pairs without a hash table").
t0 = time.perf_counter()
pairs = list(zip(labels, real_list))
t_d = time.perf_counter() - t0

# Variant E: just the label list materialization cost alone (already
# paid before this function is even called, in _pauli_label_batch -
# included here only to show how much of A's time is "hashing n
# strings" vs. "the strings existing at all").
t0 = time.perf_counter()
_ = list(labels)
t_e = time.perf_counter() - t0

print(f"n_terms={n:,}")
print(f"A dict(zip(...)):            {t_a:.4f}s")
print(f"D list(zip(...)) [no hash]:  {t_d:.4f}s")
print(f"E list(labels) [copy only]:  {t_e:.4f}s")
print(f"hashing overhead (A - D):    {t_a - t_d:.4f}s ({100*(t_a-t_d)/t_a:.1f}% of A)")
