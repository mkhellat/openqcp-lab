import numpy as np, time
rng = np.random.default_rng(0)
rows, dim = 3, 16384
c = rng.standard_normal((rows,dim)) + 1j*rng.standard_normal((rows,dim))
c[np.abs(c) < 1.0] = 0          # make ~some sub-threshold
atol = 1e-9

def cur(c, atol):
    return np.nonzero(np.abs(c) > atol)
def sq(c, atol):
    # |z|^2 > atol^2 avoids a sqrt per term; real**2+imag**2 is exact
    # in the same way abs() is, and the comparison is equivalent for
    # non-negative atol.
    r = c.real; i = c.imag
    return np.nonzero(r*r + i*i > atol*atol)

a = cur(c, atol); b = sq(c, atol)
assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "WRONG"
print("threshold correctness OK")
for nm, fn in (("abs()>atol", cur), ("re^2+im^2>atol^2", sq)):
    fn(c, atol); ts=[]
    for _ in range(9):
        t0=time.perf_counter(); fn(c, atol); ts.append(time.perf_counter()-t0)
    ts.sort(); print(f"{nm:>20}: {ts[4]*1000:7.3f} ms")

print()
# conj + divide vs precomputed conj-phase table * reciprocal
PH = np.array([1+0j, 1j, -1+0j, -1j])
CONJ_PH = PH.conj()
k = rng.integers(0,4,(rows,dim))
t = rng.standard_normal((rows,dim)) + 1j*rng.standard_normal((rows,dim))
def cur2(t,k,dim): return t * np.conj(PH[k]) / dim
def opt2(t,k,dim): return t * (CONJ_PH[k] * (1.0/dim))
assert np.allclose(cur2(t,k,dim), opt2(t,k,dim)), "WRONG"
print("phase/scale correctness OK")
for nm, fn in (("conj(PH[k])/dim", cur2), ("CONJ_PH[k]*(1/dim)", opt2)):
    fn(t,k,dim); ts=[]
    for _ in range(9):
        t0=time.perf_counter(); fn(t,k,dim); ts.append(time.perf_counter()-t0)
    ts.sort(); print(f"{nm:>20}: {ts[4]*1000:7.3f} ms")
