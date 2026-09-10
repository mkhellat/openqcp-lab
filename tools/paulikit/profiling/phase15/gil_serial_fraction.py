"""Phase 15 item 2: what fraction of per-chunk work HOLDS THE GIL?

Under threads, only GIL-free work runs concurrently. Both C kernels
release it; the sparse gather does not. Amdahl on that fraction caps
any threaded design, so measure it before choosing a technology.

Measured on a REAL N=150 chunk, median of many reps.
"""
import numpy as np, time
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import (_prepare_operator_for_fwht,
    _walsh_hadamard_transform_rows, _coefficients_from_transformed)

N=150; atol=1e-9
H=build_hamiltonian(N,_default_spring_constants(N),_default_masses(N),sparse=True)
Hp,nq=pad_to_power_of_two(H,sparse=True)
dim=Hp.shape[0]
op,is_sp,_d,n_qubits,p_nz,q_nz,x_nz,values_nz=_prepare_operator_for_fwht(Hp)
active_x,inverse=np.unique(x_nz,return_inverse=True)
order=np.argsort(inverse,kind="stable")
si,sq=inverse[order],q_nz[order]
sorted_values=values_nz[order]
cs=2; z_idx=np.arange(dim)[np.newaxis,:]; inv_dim=1.0/dim
lo=int(np.searchsorted(si,0)); hi=int(np.searchsorted(si,cs))

R=80
def bench(f):
    f(); ts=[]
    for _ in range(R):
        t0=time.perf_counter(); f(); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[R//2]

# --- GIL-HELD stages (Python/NumPy/scipy) ---
def alloc():   return np.zeros((cs,dim),dtype=complex)
def slice_v(): return sorted_values[lo:hi]
g=alloc()
def scatter(): g[si[lo:hi], sq[lo:hi]] = sorted_values[lo:hi]

t_alloc   = bench(alloc)
t_slice   = bench(slice_v)
t_scatter = bench(scatter)

# --- GIL-RELEASED stages (C kernels) ---
blk=alloc(); blk[si[lo:hi], sq[lo:hi]]=sorted_values[lo:hi]
t_wht = bench(lambda: _walsh_hadamard_transform_rows(blk.copy(), overwrite_input=True))
tw=_walsh_hadamard_transform_rows(blk.copy(), overwrite_input=True)
ax=active_x[0:cs]
t_coef= bench(lambda: _coefficients_from_transformed(tw, ax, z_idx, n_qubits, inv_dim, atol))

# the .copy() above is itself GIL-held; measure it to attribute honestly
t_copy = bench(lambda: blk.copy())

gil_held = t_alloc + t_slice + t_scatter
gil_free = t_wht + t_coef
tot = gil_held + gil_free
print(f"REAL N=150 chunk, chunk_size={cs}, dim={dim}, median of {R}\n")
print("  GIL-HELD (serialised under threads):")
for nm,t in (("np.zeros((cs,dim))",t_alloc),("values slice",t_slice),
             ("scatter assign",t_scatter)):
    print(f"    {nm:<26}{t*1e6:8.1f} us")
print(f"    {'subtotal':<26}{gil_held*1e6:8.1f} us   {100*gil_held/tot:5.1f}%")
print("\n  GIL-RELEASED (runs concurrently):")
for nm,t in (("butterfly [C]",t_wht),("fused coeffs [C]",t_coef)):
    print(f"    {nm:<26}{t*1e6:8.1f} us")
print(f"    {'subtotal':<26}{gil_free*1e6:8.1f} us   {100*gil_free/tot:5.1f}%")
print(f"\n  TOTAL                       {tot*1e6:8.1f} us")
print(f"  (block .copy() alone:       {t_copy*1e6:8.1f} us - GIL-held, in the C bench)")

f = gil_held/tot
print(f"\n  SERIAL FRACTION f = {f:.4f}")
print("  Amdahl ceiling S(p) = 1/(f + (1-f)/p):")
for p in (2,4,8,16,1e9):
    lab = "inf" if p>1e8 else str(int(p))
    print(f"    p={lab:>4}: {1/(f+(1-f)/p):6.2f}x")
