"""One benchmark measurement, one process.

Conditions:
  pauli_lcu        - their released package, dense input (which it
                     requires), single core (which is what ships)
  paulikit_seq     - sequential chunked path
  paulikit_thread  - threaded drain, n_workers threads
  paulikit_process - process pool, for comparison

Chunks are CONSUMED AND DISCARDED, never accumulated: at 15-16 qubits
the result is 6-27 GiB and retaining it is not the use case. Peak RSS
is therefore the pipeline's, which is the number that matters.
"""
import json, resource, sys, time
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

impl, N = sys.argv[1], int(sys.argv[2])
nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
atol = 1e-9

sparse = impl != "pauli_lcu"
H = build_hamiltonian(N, _default_spring_constants(N),
                      _default_masses(N), sparse=sparse)
Hp, nq = pad_to_power_of_two(H, sparse=sparse)
dim = Hp.shape[0]

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

try:
    if impl == "pauli_lcu":
        from pauli_lcu import pauli_coefficients
        w = np.ascontiguousarray(Hp.astype(complex))   # must materialise
        t0 = time.perf_counter()
        pauli_coefficients(w)
        el = time.perf_counter() - t0
        n = int(np.count_nonzero(np.abs(w) > atol))
        del w
    else:
        from paulikit.algorithms.fwht import (
            fwht_pauli_coefficients, parallel_decompose_arrays)
        if impl == "paulikit_seq":
            t0 = time.perf_counter(); n = 0
            from paulikit.algorithms.fwht import _iter_chunked_coefficients
            # stream without accumulating: use the public chunked API
            # through parallel_decompose_arrays with 1 thread
            for x, z, c in parallel_decompose_arrays(
                    Hp, chunk_size=2, atol=atol, executor="thread",
                    n_workers=1):
                n += len(c)
            el = time.perf_counter() - t0
        else:
            ex = "thread" if impl == "paulikit_thread" else "process"
            t0 = time.perf_counter(); n = 0
            for x, z, c in parallel_decompose_arrays(
                    Hp, chunk_size=2, atol=atol, executor=ex, n_workers=nw):
                n += len(c)
            el = time.perf_counter() - t0
    print(json.dumps(dict(impl=impl, N=N, qubits=nq, dim=dim, n_workers=nw,
                          elapsed=el, n_terms=n, peak_rss_mib=rss(),
                          ok=True)))
except MemoryError as e:
    print(json.dumps(dict(impl=impl, N=N, qubits=nq, dim=dim, ok=False,
                          error="MemoryError", peak_rss_mib=rss())))
except Exception as e:
    print(json.dumps(dict(impl=impl, N=N, qubits=nq, dim=dim, ok=False,
                          error=f"{type(e).__name__}: {e}"[:200],
                          peak_rss_mib=rss())))
