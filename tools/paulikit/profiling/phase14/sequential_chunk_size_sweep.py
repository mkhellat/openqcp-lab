import sys, time
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import parallel_decompose_arrays
N = int(sys.argv[1]); cs = int(sys.argv[2])
H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
Hp, _ = pad_to_power_of_two(H, sparse=True)
t = time.perf_counter(); n = 0
for x, z, c in parallel_decompose_arrays(Hp, chunk_size=cs):
    n += len(c)
print(f"chunk_size={cs:>5} elapsed={time.perf_counter()-t:7.3f}s terms={n}")
