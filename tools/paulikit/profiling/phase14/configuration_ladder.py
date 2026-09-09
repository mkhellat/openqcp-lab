"""Isolate each difference between my pauli_lcu harness and today's
w4_c4 harness, one factor at a time, same N=150."""
import json, sys, time, resource
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms import autotune

mode = sys.argv[1]
N = 150
sc, ms = _default_spring_constants(N), _default_masses(N)

sparse_in = mode != "dense_seq"
unpadded = build_hamiltonian(N, sc, ms, sparse=sparse_in)
padded, nq = pad_to_power_of_two(unpadded, sparse=sparse_in)
cs = autotune.recommended_chunk_size(padded.shape[0])

t0 = time.perf_counter()
if mode == "parallel":
    from paulikit.algorithms.fwht import parallel_decompose_arrays
    n = 0
    for x, z, c in parallel_decompose_arrays(padded, chunk_size=cs):
        n += len(c)
else:
    from paulikit.algorithms.fwht import fwht_pauli_coefficients
    x, z, c = fwht_pauli_coefficients(padded, sparse=True,
                                      chunk_size=cs, atol=1e-9)
    n = len(c)
el = time.perf_counter() - t0
print(json.dumps(dict(mode=mode, dim=padded.shape[0], qubits=nq,
    chunk_size=cs, elapsed=round(el, 3), n_terms=n,
    peak_rss_mib=round(resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss / 1024))))
