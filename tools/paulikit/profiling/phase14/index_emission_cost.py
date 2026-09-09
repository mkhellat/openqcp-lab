"""How much of our per-term cost is emitting explicit (x, z) indices -
work pauli_lcu does not do, because position encodes identity in its
dense output?

Decompose one chunk's work into stages and time each. Same shapes as
the real N=150 path (chunk_size=2, dim=16384).
"""
import numpy as np, time
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
PH  = np.array([1+0j,1j,-1+0j,-1j]); CPH = PH.conj()

dim, cs, atol = 16384, 2, 1e-9
rng = np.random.default_rng(0)
REPS = 60

def butterfly(a):
    t=a.copy(); rows=t.shape[0]
    sc=np.empty((rows,dim//2),dtype=t.dtype); span=1
    while span<dim:
        b=dim//(2*span); v=t.reshape(rows,b,2,span)
        l=v[:,:,0,:]; r=v[:,:,1,:]; h=sc[:,:b*span].reshape(rows,b,span)
        np.subtract(l,r,out=h); np.add(l,r,out=l); r[...]=h
        t=v.reshape(rows,dim); span*=2
    return t

def phase(cx, z_idx, nq):
    v=(cx & z_idx).astype(np.uint32)
    c=np.zeros(v.shape,dtype=np.uint8)
    for sh in range(0,nq,8): c += LUT[(v>>sh)&0xFF]
    return CPH[c&3]

def bench(name, fn):
    fn()
    ts=[]
    for _ in range(REPS):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    ts.sort(); return name, ts[REPS//2]

z_idx = np.arange(dim)[None,:]
cx    = np.array([[123],[456]], dtype=np.int64)
block = rng.standard_normal((cs,dim)) + 1j*rng.standard_normal((cs,dim))
# make it realistic: ~34% of entries survive atol (matches N=150)
block[np.abs(block) < 0.93] = 0

results = []
results.append(bench("1. butterfly (WHT)", lambda: butterfly(block)))
tb = butterfly(block)
results.append(bench("2. phase factor", lambda: phase(cx, z_idx, 14)))
ph = phase(cx, z_idx, 14)
results.append(bench("3. scale by phase", lambda: tb*(ph*(1.0/dim))))
coef = tb*(ph*(1.0/dim))
results.append(bench("4. threshold+nonzero (INDEX EMISSION)",
                     lambda: np.nonzero(np.abs(coef)>atol)))
ri, zi = np.nonzero(np.abs(coef)>atol)
ax = np.array([123,456])
results.append(bench("5. gather x/coeff (INDEX EMISSION)",
                     lambda: (ax[ri], coef[ri,zi])))
results.append(bench("6. narrow dtype (INDEX EMISSION)",
                     lambda: (ax[ri].astype(np.uint16), zi.astype(np.uint16))))

tot = sum(t for _,t in results)
print(f"per chunk (cs={cs}, dim={dim}), median of {REPS}:\n")
idx = 0.0
for nm, t in results:
    tag = ""
    if "INDEX EMISSION" in nm: idx += t; tag = "  <-- not paid by pauli_lcu"
    print(f"  {nm:<40} {t*1e6:9.1f} us  {100*t/tot:5.1f}%{tag}")
print(f"\n  {'TOTAL':<40} {tot*1e6:9.1f} us")
print(f"  {'of which index emission':<40} {idx*1e6:9.1f} us  {100*idx/tot:5.1f}%")
nterms = len(ri)
print(f"\n  terms surviving per chunk: {nterms} ({nterms/(cs*dim):.1%} density)")
print(f"  total       : {tot/nterms*1e9:7.1f} ns/term")
print(f"  index part  : {idx/nterms*1e9:7.1f} ns/term")
print(f"  transform   : {(tot-idx)/nterms*1e9:7.1f} ns/term   <- comparable to pauli_lcu's 51.06")
