"""Is shared memory materially cheaper than the pool's result pipe for
a 512 KiB numpy payload? Measure the round trip both ways."""
import numpy as np, time, pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

NCHUNK = 200
TERMS = 512*1024 // 32          # 32 B/term (intp+intp+complex128)

def make(seed):
    r = np.random.default_rng(seed)
    return (r.integers(0, 1<<14, TERMS, dtype=np.intp),
            r.integers(0, 1<<14, TERMS, dtype=np.intp),
            r.standard_normal(TERMS) + 1j*r.standard_normal(TERMS))

def w_pipe(i):
    return make(i)

_SHM = {}
def w_shm(args):
    i, name, off = args
    x, z, c = make(i)
    shm = _SHM.get(name) or shared_memory.SharedMemory(name=name)
    _SHM[name] = shm
    n = len(x)
    b = shm.buf
    xb = np.ndarray(n, dtype=np.intp, buffer=b, offset=off)
    zb = np.ndarray(n, dtype=np.intp, buffer=b, offset=off + n*8)
    cb = np.ndarray(n, dtype=complex, buffer=b, offset=off + n*16)
    xb[:] = x; zb[:] = z; cb[:] = c
    return i, n                      # tiny descriptor only

if __name__ == "__main__":
    per = TERMS*32
    print(f"{NCHUNK} chunks x {per/1024:.0f} KiB = {NCHUNK*per/2**20:.0f} MiB\n")

    with ProcessPoolExecutor(4) as ex:
        list(ex.map(w_pipe, range(4)))       # warm
        t0=time.perf_counter()
        tot=0
        for x,z,c in ex.map(w_pipe, range(NCHUNK)):
            tot += len(c)
        t_pipe=time.perf_counter()-t0
    print(f"via result pipe : {t_pipe:6.3f}s  ({NCHUNK*per/2**20/t_pipe:6.1f} MiB/s)")

    shm = shared_memory.SharedMemory(create=True, size=4*per + 4096)
    try:
        with ProcessPoolExecutor(4) as ex:
            list(ex.map(w_shm, [(i, shm.name, (i%4)*per) for i in range(4)]))
            t0=time.perf_counter()
            tot=0
            for i,n in ex.map(w_shm, [(i, shm.name, (i%4)*per) for i in range(NCHUNK)]):
                tot += n
            t_shm=time.perf_counter()-t0
        print(f"via shared mem  : {t_shm:6.3f}s  ({NCHUNK*per/2**20/t_shm:6.1f} MiB/s)")
        print(f"\nspeedup on transfer: {t_pipe/t_shm:.2f}x")
    finally:
        shm.close(); shm.unlink()
