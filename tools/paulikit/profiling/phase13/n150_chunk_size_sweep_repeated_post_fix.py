"""Real N=150 chunk_size sweep with REPEATS (mean/stdev), n_workers
explicit - the clean, statistically-repeated redo of
n150_chunk_size_sweep_post_fix.py's single-run numbers, per direct
user instruction after execution-reliability problems (overlapping
background runs) made the earlier N=150 single-run sweep untrustworthy
and it was discarded entirely, not just supplemented.

Run FOREGROUND ONLY (no run_in_background, no ScheduleWakeup) per
direct user instruction - wait for full completion in one call.
CHUNK_SIZES/REPS deliberately kept smaller than the N=100 companion
script (3 sizes x 2 reps, not 4 x 3) to keep total wall-clock bounded
and reliable at N=150's larger per-run cost.
"""
import statistics
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter, parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
N_WORKERS = 8
CHUNK_SIZES = [2, 32, 128]
REPS = 2

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
dim = padded.shape[0]

print(f"N={N_OSCILLATORS} dim={dim} n_workers={N_WORKERS} reps={REPS}", flush=True)


def time_sequential(chunk_size, reps):
    times = []
    n = None
    for _ in range(reps):
        t0 = time.perf_counter()
        total = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
            total += len(chunk)
        times.append(time.perf_counter() - t0)
        n = total
    return times, n


def time_parallel(chunk_size, n_workers, reps):
    times = []
    n = None
    for _ in range(reps):
        t0 = time.perf_counter()
        total = 0
        for chunk in parallel_decompose(padded, chunk_size=chunk_size, n_workers=n_workers):
            total += len(chunk)
        times.append(time.perf_counter() - t0)
        n = total
    return times, n


results = []
for chunk_size in CHUNK_SIZES:
    seq_times, n_seq = time_sequential(chunk_size, REPS)
    par_times, n_par = time_parallel(chunk_size, N_WORKERS, REPS)
    assert n_seq == n_par, (chunk_size, n_seq, n_par)

    seq_mean = statistics.mean(seq_times)
    seq_stdev = statistics.stdev(seq_times) if len(seq_times) > 1 else 0.0
    par_mean = statistics.mean(par_times)
    par_stdev = statistics.stdev(par_times) if len(par_times) > 1 else 0.0
    speedup = seq_mean / par_mean

    results.append((chunk_size, seq_mean, seq_stdev, par_mean, par_stdev, speedup))
    print(f"chunk_size={chunk_size:>4}  n_workers={N_WORKERS}  "
          f"seq={seq_mean:.4f}s (+/-{seq_stdev:.4f})  "
          f"par={par_mean:.4f}s (+/-{par_stdev:.4f})  speedup={speedup:.3f}x  "
          f"seq_runs={[f'{t:.3f}' for t in seq_times]}  "
          f"par_runs={[f'{t:.3f}' for t in par_times]}", flush=True)

print(f"\nSummary (N={N_OSCILLATORS}, n_workers={N_WORKERS}, {REPS} reps each):")
print(f"{'chunk_size':>10}  {'n_workers':>9}  {'seq mean':>10}  {'par mean':>10}  {'speedup':>8}")
for chunk_size, seq_mean, seq_stdev, par_mean, par_stdev, speedup in results:
    print(f"{chunk_size:>10}  {N_WORKERS:>9}  {seq_mean:>8.4f}s  {par_mean:>8.4f}s  {speedup:>7.3f}x")
