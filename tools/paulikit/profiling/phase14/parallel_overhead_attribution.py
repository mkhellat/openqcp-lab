"""Where do the extra CPU-seconds go? Attribute by measuring the SAME
per-chunk work three ways: in-process, through a pool, and a pool doing
the work but returning nothing."""
import os, sys, time, resource, pickle
import numpy as np
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms import autotune
from paulikit.algorithms.fwht import (
    parallel_decompose_arrays, fwht_pauli_coefficients)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
H = build_hamiltonian(N, _default_spring_constants(N), _default_masses(N), sparse=True)
Hp, _ = pad_to_power_of_two(H, sparse=True)
cs = autotune.recommended_chunk_size(Hp.shape[0])

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime

# 1. sequential baseline
c0, t0 = cpu(), time.perf_counter()
x, z, co = fwht_pauli_coefficients(Hp, sparse=True, chunk_size=cs, atol=1e-9)
c1, t1 = cpu(), time.perf_counter()
seq_cpu, seq_wall, nterms = c1-c0, t1-t0, len(co)
print(f"sequential      wall={seq_wall:7.3f} cpu={seq_cpu:7.3f}")

# how big is the payload that crosses the pipe, in total?
nbytes = x.nbytes + z.nbytes + co.nbytes
print(f"  total result payload: {nbytes/2**20:8.1f} MiB across ~{(len(x)//max(cs,1))or 1} chunks")
del x, z, co

# 2. parallel, consuming every chunk (what callers actually do)
c0, t0 = cpu(), time.perf_counter()
n = 0
for a, b, c in parallel_decompose_arrays(Hp, chunk_size=cs):
    n += len(c)
c1, t1 = cpu(), time.perf_counter()
par_cpu, par_wall = c1-c0, t1-t0
print(f"parallel        wall={par_wall:7.3f} cpu={par_cpu:7.3f} terms={n}")

# 3. parallel, discarding each chunk immediately (same IPC, no
#    accumulation in the consumer)
c0, t0 = cpu(), time.perf_counter()
cnt = 0
for trip in parallel_decompose_arrays(Hp, chunk_size=cs):
    cnt += 1
c1, t1 = cpu(), time.perf_counter()
print(f"parallel/drop   wall={c1 and t1-t0:7.3f} cpu={c1-c0:7.3f} chunks={cnt}")

print()
print(f"overhead vs sequential: {par_cpu-seq_cpu:6.2f} cpu-s "
      f"({100*(par_cpu-seq_cpu)/seq_cpu:.0f}% of sequential)")
print(f"per chunk: {(par_cpu-seq_cpu)/max(cnt,1)*1000:.3f} ms")
print(f"payload/chunk: {nbytes/max(cnt,1)/1024:.1f} KiB")
