"""Is pauli_lcu's O(1) additional-memory claim true as stated?
Measure peak RSS BEFORE and DURING pauli_coefficients, so the delta
is the auxiliary space only."""
import sys, resource, numpy as np
n = int(sys.argv[1]); dim = 1 << n
def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
from pauli_lcu import pauli_coefficients
base = rss()
rng = np.random.default_rng(0)
a = rng.standard_normal((dim,dim)) + 1j*rng.standard_normal((dim,dim))
a = np.ascontiguousarray(a)
after_alloc = rss()
pauli_coefficients(a)
after_call = rss()
print(f"n={n:>2} dim={dim:>6}  matrix={dim*dim*16/2**20:8.1f} MiB")
print(f"   RSS before alloc : {base:9.1f} MiB")
print(f"   RSS after alloc  : {after_alloc:9.1f} MiB  (+{after_alloc-base:.1f})")
print(f"   RSS after call   : {after_call:9.1f} MiB  (+{after_call-after_alloc:.1f} = AUXILIARY)")
