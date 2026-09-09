"""Isolate the butterfly's inner-loop cost. Same math, three ways."""
import numpy as np, time

def current(a):
    t = a.copy(); dim = a.shape[1]; span = 1
    while span < dim:
        t = t.reshape(t.shape[0], dim // (2*span), 2, span)
        l = t[:, :, 0, :]; r = t[:, :, 1, :]
        l, r = l + r, l - r
        t[:, :, 0, :] = l; t[:, :, 1, :] = r
        t = t.reshape(t.shape[0], dim)
        span *= 2
    return t

def scratch(a):
    """One preallocated scratch buffer, reused; no per-stage allocation."""
    t = a.copy(); dim = a.shape[1]; span = 1
    tmp = np.empty((a.shape[0], dim // 2), dtype=a.dtype)
    while span < dim:
        v = t.reshape(t.shape[0], dim // (2*span), 2, span)
        l = v[:, :, 0, :]; r = v[:, :, 1, :]
        s = tmp[:, :l.shape[1]*l.shape[2]].reshape(l.shape)
        np.subtract(l, r, out=s)      # r_new stashed
        np.add(l, r, out=l)           # l updated in place
        r[...] = s
        span *= 2
    return t

rows, dim = 3, 16384
rng = np.random.default_rng(0)
a = (rng.standard_normal((rows, dim)) + 1j*rng.standard_normal((rows, dim)))
ref = current(a)
assert np.allclose(scratch(a), ref), "scratch variant WRONG"
print("correctness: scratch matches current  OK")

for name, fn in (("current", current), ("scratch", scratch)):
    fn(a)                                   # warm
    ts = [ ]
    for _ in range(7):
        t0 = time.perf_counter(); fn(a); ts.append(time.perf_counter()-t0)
    ts.sort()
    print(f"{name:>8}: median {ts[3]*1000:7.3f} ms  min {ts[0]*1000:7.3f} ms")
