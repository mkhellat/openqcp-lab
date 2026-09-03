"""Long-running workload for a properly time-synchronized turbostat
capture - direct follow-up after the first attempt's turbostat window
did not overlap with the real ~29s workload (captured data showed
PkgWatt < 3.4W and PkgTmp < 58C throughout, far below the 15-23W/100C
measured during genuine sustained load earlier in this investigation -
a timing mismatch, not a real finding).

Loops the real parallel_decompose(n_workers=2, pinned) computation
repeatedly for a fixed wall-clock duration (default 90s) - long enough
that starting `sudo turbostat` a few seconds late or stopping it a few
seconds early still captures a wide window of genuine sustained
2-core load, unlike a single ~29s run where timing has to be nearly
exact.

Usage:
    OPENBLAS_NUM_THREADS=1 python turbostat_sync_workload.py [duration_s]
"""
import sys
import time

from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
N_WORKERS = 2
DURATION_S = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

print(f"Starting {DURATION_S:.0f}s of sustained pinned_2 load NOW - "
      f"start your turbostat capture immediately.", flush=True)

t_start = time.perf_counter()
n_runs = 0
total_terms = 0
while time.perf_counter() - t_start < DURATION_S:
    run_start = time.perf_counter()
    total = 0
    for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=N_WORKERS):
        total += len(chunk)
    total_terms += total
    n_runs += 1
    print(f"  run {n_runs} done at t={time.perf_counter()-t_start:.1f}s "
          f"(this run: {time.perf_counter()-run_start:.2f}s)", flush=True)

elapsed = time.perf_counter() - t_start
print(f"DONE. total_elapsed={elapsed:.2f}s n_runs={n_runs} total_terms={total_terms}")
