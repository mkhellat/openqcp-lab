"""Vectorized: all rows at once, no Python loop over rows.

Key structure: every active row has the SAME k positions? No - but the
positions come from q_nz, and the parity mask for position q is
(-1)^popcount(q & z). Build it for all needed q at once as a
(k_total, dim) sign matrix, then scatter-add v * mask into rows.
"""
import numpy as np, time
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def wht_dense(a):
    t=a.copy(); dim=a.shape[1]; rows=a.shape[0]
    sc=np.empty((rows,dim//2),dtype=t.dtype); span=1
    while span<dim:
        b=dim//(2*span); v=t.reshape(rows,b,2,span)
        l=v[:,:,0,:]; r=v[:,:,1,:]; h=sc[:,:b*span].reshape(rows,b,span)
        np.subtract(l,r,out=h); np.add(l,r,out=l); r[...]=h
        t=v.reshape(rows,dim); span*=2
    return t

def signs_for(qs, dim):
    """(-1)^popcount(q & z), shape (len(qs), dim), fully vectorized"""
    z = np.arange(dim, dtype=np.uint32)[None, :]
    v = (z & np.asarray(qs, dtype=np.uint32)[:, None])
    c = np.zeros(v.shape, dtype=np.uint8)
    for sh in (0, 8, 16, 24):
        c += LUT[(v >> sh) & 0xFF]
    return 1.0 - 2.0*(c & 1).astype(np.float64)

def wht_sparse_vec(row_idx, qs, vals, nrows, dim):
    """row_idx[i], qs[i], vals[i] = one nonzero. Vectorized over ALL
    nonzeros of ALL rows at once."""
    contrib = vals[:, None] * signs_for(qs, dim)     # (nnz, dim)
    out = np.zeros((nrows, dim), dtype=complex)
    np.add.at(out, row_idx, contrib) if False else None
    # np.add.at is slow; use bincount-style accumulation instead
    for r in np.unique(row_idx):
        out[r] = contrib[row_idx == r].sum(axis=0)
    return out

rng = np.random.default_rng(0)
for dim, nrows in ((4096, 64), (16384, 64), (16384, 256)):
    k=4
    row_idx=[]; qs=[]; vals=[]
    dense=np.zeros((nrows,dim),dtype=complex)
    for i in range(nrows):
        q=rng.choice(dim,k,replace=False); v=rng.standard_normal(k)+1j*rng.standard_normal(k)
        dense[i,q]=v
        row_idx += [i]*k; qs += list(q); vals += list(v)
    row_idx=np.array(row_idx); qs=np.array(qs); vals=np.array(vals)
    ref=wht_dense(dense)
    got=wht_sparse_vec(row_idx,qs,vals,nrows,dim)
    ok=np.allclose(ref,got)
    t0=time.perf_counter(); wht_dense(dense); td=time.perf_counter()-t0
    t0=time.perf_counter(); wht_sparse_vec(row_idx,qs,vals,nrows,dim); ts=time.perf_counter()-t0
    print(f"dim={dim:>6} rows={nrows:>4} correct={ok} dense={td*1000:8.3f}ms "
          f"sparse={ts*1000:8.3f}ms ratio={td/ts:5.2f}x")
