"""One measurement, one process. variant=old restores the previous
one-at-a-time refill by monkeypatching the drain loop's behaviour via
a module flag; variant=new uses the shipped batched refill."""
import json, sys, time, resource
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms import fwht

variant, N, cs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
Hp, _ = pad_to_power_of_two(H, sparse=True)

if variant == "old":
    # Re-create the previous behaviour: refill ONE task per completed
    # future, from inside the per-future loop, after result() and
    # before the yield. Implemented by wrapping the generator's
    # internals is fragile, so instead run the equivalent loop here
    # using the same real worker functions.
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    import multiprocessing
    from paulikit.algorithms.fwht import (
        _prepare_operator_for_fwht, _parallel_worker_init,
        _parallel_worker_chunk, _physical_core_representative_cpus,
        _recommended_parallel_chunk_size, _per_worker_resident_bytes,
        _detect_available_worker_count, _index_dtype_for_dim)
    op, is_sp, dim, nq, p_nz, q_nz, x_nz = _prepare_operator_for_fwht(Hp)
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x); z_idx = np.arange(dim)[np.newaxis, :]
    pin = _physical_core_representative_cpus()
    nw = len(pin) if pin else _detect_available_worker_count()
    order = np.argsort(inverse, kind="stable")
    si, sp_, sq = inverse[order], p_nz[order], q_nz[order]
    starts = list(range(0, n_active, cs))
    pending = [(i, s, min(s+cs, n_active)) for i, s in enumerate(starts)]
    nxt = multiprocessing.Value("i", 0)
    t0 = time.perf_counter(); n = 0
    with ProcessPoolExecutor(max_workers=nw, initializer=_parallel_worker_init,
            initargs=(op, is_sp, si, sp_, sq, active_x, dim, nq, z_idx,
                      1e-9, pin, nxt)) as pool:
        it = iter(pending); inflight = set()
        def sub():
            item = next(it, None)
            if item is None: return False
            inflight.add(pool.submit(_parallel_worker_chunk, *item)); return True
        for _ in range(2*nw):
            if not sub(): break
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:                      # OLD: one at a time,
                ci, cx, cz, cc = fut.result()     # interleaved with work
                sub()
                n += len(cc)
    el = time.perf_counter() - t0
else:
    t0 = time.perf_counter(); n = 0
    for x, z, c in fwht.parallel_decompose_arrays(Hp, chunk_size=cs):
        n += len(c)
    el = time.perf_counter() - t0

print(json.dumps(dict(variant=variant, N=N, chunk_size=cs, elapsed=el,
    n_terms=n,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
