"""Split the per-chunk cost: task dispatch vs payload transfer."""
import numpy as np, time
from concurrent.futures import ProcessPoolExecutor

TERMS = 512*1024//32
def noop(i):        return i                      # no payload
def tiny(i):        return i, 0                   # no payload
def payload(i):
    r = np.random.default_rng(i)
    return (r.integers(0,1<<14,TERMS,dtype=np.intp),
            r.integers(0,1<<14,TERMS,dtype=np.intp),
            r.standard_normal(TERMS)+1j*r.standard_normal(TERMS))
def payload_gen_only(i):
    r = np.random.default_rng(i)
    x=(r.integers(0,1<<14,TERMS,dtype=np.intp),
       r.integers(0,1<<14,TERMS,dtype=np.intp),
       r.standard_normal(TERMS)+1j*r.standard_normal(TERMS))
    return len(x[0])                              # compute, return nothing big

if __name__ == "__main__":
    N=400
    with ProcessPoolExecutor(4) as ex:
        for nm, fn in (("empty task", noop), ("tiny return", tiny),
                       ("generate, return small", payload_gen_only),
                       ("generate, return 512KiB", payload)):
            list(ex.map(fn, range(4)))
            t0=time.perf_counter()
            for _ in ex.map(fn, range(N)): pass
            el=time.perf_counter()-t0
            print(f"{nm:>26}: {el:6.3f}s  {el/N*1000:6.3f} ms/task")
