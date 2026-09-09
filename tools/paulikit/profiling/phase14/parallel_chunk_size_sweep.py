"""chunk_size sweep on the PARALLEL path, measuring CPU-seconds not
just wall. The autotuner picks for the sequential path; parallel has
a per-chunk dispatch+setup cost the sequential path does not."""
import sys, time, resource
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import parallel_decompose_arrays
N = int(sys.argv[1]); cs = int(sys.argv[2])
H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
Hp, _ = pad_to_power_of_two(H, sparse=True)
def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF); c=resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime
c0,t0=cpu(),time.perf_counter()
n=0; k=0
for x,z,co in parallel_decompose_arrays(Hp, chunk_size=cs):
    n+=len(co); k+=1
c1,t1=cpu(),time.perf_counter()
print(f"chunk_size={cs:>5} wall={t1-t0:7.3f}s cpu={c1-c0:7.3f}s "
      f"chunks={k:>5} cores={(c1-c0)/(t1-t0):.2f} terms={n}")
