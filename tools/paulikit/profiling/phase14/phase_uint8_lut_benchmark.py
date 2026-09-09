import numpy as np, time
LUT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
PHASE4 = np.array([1+0j, 1j, -1+0j, -1j], dtype=np.complex128)
LUT8_M3 = (LUT8 & 3)

def cur(x, z, n_bits):
    v = (x & z).astype(np.uint32)
    c = np.zeros(v.shape, dtype=np.int64)
    for sh in range(0, n_bits, 8):
        c += np.array([bin(i).count("1") for i in range(256)],
                      dtype=np.int64)[(v >> sh) & 0xFF]
    return 1j ** c

def best(x, z, n_bits):
    """uint8 popcount accumulate + &3 + 4-entry complex gather.
    uint8 wraps mod 256; since 256 % 4 == 0, (sum mod 256) & 3 ==
    sum & 3 exactly - the wrap is harmless for the phase."""
    v = (x & z).astype(np.uint32)
    c = np.zeros(v.shape, dtype=np.uint8)
    for sh in range(0, n_bits, 8):
        c += LUT8[(v >> sh) & 0xFF]
    return PHASE4[c & 3]

rng = np.random.default_rng(0); n_bits = 14; N = 4_000_000
x = rng.integers(0, 1 << n_bits, N, dtype=np.int64)
z = rng.integers(0, 1 << n_bits, N, dtype=np.int64)
assert np.allclose(best(x, z, n_bits), cur(x, z, n_bits)), "WRONG"
# stress the wrap argument: n_bits large enough to exceed 255 ones is
# impossible for uint32 (max 32), but assert the identity anyway
for t in (0, 1, 2, 3, 4, 31, 32, 255, 256, 257):
    assert (np.uint8(t % 256) & 3) == (t & 3)
print("correctness OK (incl. uint8 wrap identity)\n")
for nm, fn in (("current", cur), ("uint8+LUT gather", best)):
    fn(x, z, n_bits)
    ts = []
    for _ in range(7):
        t0 = time.perf_counter(); fn(x, z, n_bits); ts.append(time.perf_counter()-t0)
    ts.sort(); print(f"{nm:>20}: {ts[3]*1000:8.2f} ms")
