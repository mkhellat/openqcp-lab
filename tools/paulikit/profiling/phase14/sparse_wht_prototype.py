"""A WHT of a k-sparse row is a sum of k sign patterns.

For a row with nonzeros v_j at positions q_j, the unnormalized WHT is
    W[z] = sum_j v_j * (-1)^popcount(q_j & z)
so cost is O(k*dim) instead of O(dim*log dim). At dim=16384, k=4 that
is 4*16384 vs 16384*14 -> 3.5x fewer term-ops in principle.

Verify correctness first, then measure.
"""
import numpy as np, time

LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def wht_dense(a):
    """current approach: full butterfly"""
    t = a.copy(); dim = a.shape[1]; rows = a.shape[0]
    scratch = np.empty((rows, dim//2), dtype=t.dtype)
    span = 1
    while span < dim:
        blocks = dim//(2*span)
        v = t.reshape(rows, blocks, 2, span)
        l = v[:,:,0,:]; r = v[:,:,1,:]
        h = scratch[:, :blocks*span].reshape(rows, blocks, span)
        np.subtract(l, r, out=h); np.add(l, r, out=l); r[...] = h
        t = v.reshape(rows, dim)
        span *= 2
    return t

def parity_mask(q, dim):
    """(-1)^popcount(q & z) for all z in [0,dim), as +-1 float"""
    z = np.arange(dim, dtype=np.uint32)
    v = (z & np.uint32(q)).astype(np.uint32)
    c = np.zeros(dim, dtype=np.uint8)
    for sh in (0,8,16,24):
        c += LUT[(v >> sh) & 0xFF]
    return 1.0 - 2.0*(c & 1)

def wht_sparse(rows_nz, dim):
    """rows_nz: list of (positions, values) per row"""
    out = np.zeros((len(rows_nz), dim), dtype=complex)
    for i,(qs, vs) in enumerate(rows_nz):
        acc = np.zeros(dim, dtype=complex)
        for q, v in zip(qs, vs):
            acc += v * parity_mask(int(q), dim)
        out[i] = acc
    return out

rng = np.random.default_rng(0)
for dim in (256, 4096, 16384):
    k = 4; nrows = 8
    rows_nz = []
    dense = np.zeros((nrows, dim), dtype=complex)
    for i in range(nrows):
        qs = rng.choice(dim, k, replace=False)
        vs = rng.standard_normal(k) + 1j*rng.standard_normal(k)
        rows_nz.append((qs, vs))
        dense[i, qs] = vs
    ref = wht_dense(dense)
    got = wht_sparse(rows_nz, dim)
    ok = np.allclose(ref, got)
    t0=time.perf_counter(); wht_dense(dense); td=time.perf_counter()-t0
    t0=time.perf_counter(); wht_sparse(rows_nz, dim); ts=time.perf_counter()-t0
    print(f"dim={dim:>6} k={k} rows={nrows} correct={ok}  "
          f"dense={td*1000:7.3f}ms  sparse={ts*1000:7.3f}ms  ratio={td/ts:5.2f}x")
