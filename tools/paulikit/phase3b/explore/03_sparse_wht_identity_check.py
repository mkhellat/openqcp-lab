import numpy as np

# WHT of a single impulse at position q0 with value v, over length-dim vector:
# WHT(delta_{q0} * v)[z] = v * (-1)**popcount(q0 & z)
# This is exact for the unnormalized +-1 butterfly WHT used in fwht.py.
# So for a sparse row with nonzero entries {(q_i, v_i)}, the WHT output at z is:
#   sum_i v_i * (-1)**popcount(q_i & z)
# We can compute this for ALL z in O(k * dim) via broadcasting, without doing
# the O(dim log dim) butterfly at all.

rng = np.random.default_rng(0)
dim = 64
row = np.zeros(dim, dtype=complex)
nz_idx = rng.choice(dim, size=5, replace=False)
row[nz_idx] = rng.normal(size=5) + 1j*rng.normal(size=5)

# reference: standard butterfly WHT (mirroring fwht.py's algorithm)
def wht_dense(v):
    v = v.copy()
    span = 1
    while span < len(v):
        v = v.reshape(len(v)//(2*span), 2, span)
        left = v[:,0,:].copy(); right = v[:,1,:].copy()
        v[:,0,:] = left+right; v[:,1,:] = left-right
        v = v.reshape(-1)
        span *= 2
    return v

ref = wht_dense(row)

# sparse computation
def popcount(x):
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c

z_all = np.arange(dim)
result = np.zeros(dim, dtype=complex)
for qi, vi in zip(nz_idx, row[nz_idx]):
    signs = np.array([(-1)**popcount(int(qi) & int(z)) for z in z_all])
    result += vi * signs

print("max abs diff:", np.max(np.abs(result - ref)))
