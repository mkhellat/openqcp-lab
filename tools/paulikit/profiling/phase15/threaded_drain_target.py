"""Phase 15 item 1: honest re-measurement of the threading premise.

The earlier probe reported 3.04x/2 and 5.42x/4 and was rejected as
untrustworthy: it reused 64 pre-built blocks, so they stayed
cache-warm and the "work" never touched fresh memory. Superlinear
speedup on compute-bound work is unphysical and was the tell.

This harness fixes that:
  * REAL chunks from a REAL N=150 operator, gathered fresh per chunk
    (the gather is GIL-held - that is the serial fraction, and it
    must be inside the timed region, not hoisted out).
  * Every chunk is distinct - the full 5595-chunk sweep, so nothing
    is reused and the memory traffic is what production pays.
  * One process per measurement, so peak RSS is that run's alone.
  * Sequential baseline uses the SAME per-chunk code path, so the
    comparison isolates threading, not implementation differences.
"""
import json, sys, threading, time, resource
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import (_prepare_operator_for_fwht,
    _walsh_hadamard_transform_rows, _coefficients_from_transformed)

n_threads = int(sys.argv[1]); N = int(sys.argv[2]); cs = int(sys.argv[3])
atol = 1e-9

H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
Hp, nq = pad_to_power_of_two(H, sparse=True)
dim = Hp.shape[0]
op, is_sp, _d, n_qubits, p_nz, q_nz, x_nz, values_nz = _prepare_operator_for_fwht(Hp)
active_x, inverse = np.unique(x_nz, return_inverse=True)
n_active = len(active_x)
order = np.argsort(inverse, kind="stable")
si, sq = inverse[order], q_nz[order]
sorted_values = values_nz[order]
z_idx = np.arange(dim)[np.newaxis, :]
inv_dim = 1.0 / dim
starts = list(range(0, n_active, cs))

def do_chunk(ci):
    """One chunk, end to end - gather (GIL-held) then both C kernels
    (GIL-released). Identical work in both conditions."""
    s = starts[ci]; e = min(s + cs, n_active)
    lo = int(np.searchsorted(si, s)); hi = int(np.searchsorted(si, e))
    g = np.zeros((e - s, dim), dtype=complex)
    g[si[lo:hi] - s, sq[lo:hi]] = sorted_values[lo:hi]
    t = _walsh_hadamard_transform_rows(g, overwrite_input=True)
    x, z, c = _coefficients_from_transformed(
        t, active_x[s:e], z_idx, n_qubits, inv_dim, atol)
    return len(c)

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime

n_chunks = len(starts)

# UNTIMED WARM-UP, inside this process, before the timed region.
#
# Without it the measurement is dominated by a cold-start CPU
# frequency ramp, not by the work: this machine runs the `powersave`
# governor with a 400 MHz floor and 4000 MHz ceiling, so a freshly
# spawned child begins on a downclocked core. Measured directly, the
# same 1-thread run came out bimodal at ~4.55s and ~2.25s - a clean
# factor of 2, with cores=1.00 in both modes - while the identical
# work measured in an already-warm process was consistently ~2.2s.
#
# Running a slice of the real work first pulls the core up to its
# operating frequency and first-touches the operator arrays, so the
# timed region measures compute rather than the ramp. This is the
# same confound recorded in profiling/phase13's cold-start work; it
# reappears whenever a measurement is put in a fresh process.
for _i in range(min(400, n_chunks)):
    do_chunk(_i)

c0, t0 = cpu(), time.perf_counter()
if n_threads == 1:
    total = sum(do_chunk(i) for i in range(n_chunks))
else:
    with ThreadPoolExecutor(n_threads) as ex:
        total = sum(ex.map(do_chunk, range(n_chunks)))
wall = time.perf_counter() - t0
cpu_s = cpu() - c0

print(json.dumps(dict(
    n_threads=n_threads, N=N, chunk_size=cs, n_chunks=n_chunks,
    wall=wall, cpu=cpu_s, cores=cpu_s / wall, n_terms=total,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
