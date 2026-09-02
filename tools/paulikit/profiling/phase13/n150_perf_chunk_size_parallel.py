"""perf stat target: PARALLEL parallel_decompose (n_workers=8) at a
given chunk_size, N=150 - post-bounded-submission-fix (the earlier
n100_n150_parallel_decompose_findings.md perf numbers, if any existed,
would predate that fix). Same event set as
n150_perf_chunk_size_sequential.py and every other perf stat
measurement in this project - relies on perf's default counter
inheritance across forked/exec'd child processes (ProcessPoolExecutor
workers), NOT -a/system-wide mode, so counts should reflect only this
job's own worker tree, not the whole machine. Run under:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python n150_perf_chunk_size_parallel.py <chunk_size>
"""
import sys

from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
N_WORKERS = 8
chunk_size = int(sys.argv[1])

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
sparse = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(sparse, sparse=True)

total_terms = 0
for chunk_terms in parallel_decompose(padded, chunk_size=chunk_size, n_workers=N_WORKERS):
    total_terms += len(chunk_terms)
print(f"chunk_size={chunk_size} n_workers={N_WORKERS} total_terms={total_terms}")
