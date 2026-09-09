import json, sys, time, resource
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
impl, N = sys.argv[1], int(sys.argv[2])
sparse = impl == "paulikit"
H = build_hamiltonian(N, _default_spring_constants(N),
                      _default_masses(N), sparse=sparse)
Hp, nq = pad_to_power_of_two(H, sparse=sparse)
if impl == "pauli_lcu":
    from pauli_lcu import pauli_coefficients
    w = np.ascontiguousarray(Hp.astype(complex))
    t0 = time.perf_counter(); pauli_coefficients(w)
    el = time.perf_counter() - t0
    n = int(np.count_nonzero(np.abs(w) > 1e-9))
else:
    from paulikit.algorithms import autotune
    from paulikit.algorithms.fwht import parallel_decompose_arrays
    cs = autotune.recommended_chunk_size(Hp.shape[0])
    t0 = time.perf_counter(); n = 0
    for x, z, c in parallel_decompose_arrays(Hp, chunk_size=cs):
        n += len(c)
    el = time.perf_counter() - t0
print(json.dumps(dict(impl=impl, N=N, qubits=nq, dim=Hp.shape[0],
    elapsed=el, n_terms=n,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
