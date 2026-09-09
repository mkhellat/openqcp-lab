"""Bit-trick candidates for the phase step. All verified against the
current implementation before timing."""
import numpy as np, time

LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

def popcount_current(v, n_bits):
    v = v.astype(np.uint32)
    c = np.zeros(v.shape, dtype=np.int64)
    for sh in range(0, n_bits, 8):
        c += LUT8[(v >> sh) & 0xFF]
    return c

def phase_current(x, z, n_bits):
    return 1j ** popcount_current(x & z, n_bits)

# --- candidate A: we only need popcount MOD 4, and only the low 2
# bits of it. But parity tricks give mod 2, not mod 4, so the full
# count is still needed. Keep the LUT, drop to uint8 accumulate.
def popcount_u8(v, n_bits):
    v = v.astype(np.uint32)
    c = np.zeros(v.shape, dtype=np.uint8)
    for sh in range(0, n_bits, 8):
        c += LUT8[(v >> sh) & 0xFF].astype(np.uint8)
    return c

# --- candidate B: replace 1j**k (a complex power) with a 4-entry
# gather. k is already in [0,4) after &3.
PHASE4 = np.array([1+0j, 1j, -1+0j, -1j], dtype=np.complex128)
def phase_lut(x, z, n_bits):
    k = popcount_current(x & z, n_bits) & 3
    return PHASE4[k]

# --- candidate C: skip materialising the phase array entirely and
# apply it to coefficients by sign/swap manipulation. Phase k acts on
# (re,im) as: 0->(re,im) 1->(-im,re) 2->(-re,-im) 3->(im,-re).
def apply_phase_inplace(coeff, k):
    out = coeff.copy()
    m1 = k == 1; m2 = k == 2; m3 = k == 3
    a = out[m1]; out[m1] = -a.imag + 1j*a.real
    out[m2] = -out[m2]
    a = out[m3]; out[m3] = a.imag - 1j*a.real
    return out

rng = np.random.default_rng(0)
n_bits = 14
N = 4_000_000
x = rng.integers(0, 1 << n_bits, N, dtype=np.int64)
z = rng.integers(0, 1 << n_bits, N, dtype=np.int64)
coeff = rng.standard_normal(N) + 1j*rng.standard_normal(N)

ref = phase_current(x, z, n_bits)
assert np.allclose(phase_lut(x, z, n_bits), ref), "LUT phase WRONG"
k = popcount_current(x & z, n_bits) & 3
assert np.allclose(apply_phase_inplace(coeff, k), coeff*ref), "inplace WRONG"
assert np.array_equal(popcount_u8(x & z, n_bits), popcount_current(x & z, n_bits))
print("correctness: all candidates match  OK\n")

def bench(name, fn, reps=7):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    ts.sort()
    print(f"{name:>34}: {ts[len(ts)//2]*1000:8.2f} ms")

bench("current  1j**popcount", lambda: phase_current(x, z, n_bits))
bench("LUT gather PHASE4[k&3]", lambda: phase_lut(x, z, n_bits))
bench("popcount only (current)", lambda: popcount_current(x & z, n_bits))
bench("popcount only (uint8 acc)", lambda: popcount_u8(x & z, n_bits))
bench("phase*coeff (current)", lambda: coeff * phase_current(x, z, n_bits))
bench("phase applied by sign/swap", lambda: apply_phase_inplace(coeff, k))
