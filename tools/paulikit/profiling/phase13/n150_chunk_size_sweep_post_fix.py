"""Real N=150 chunk_size sweep, re-run under the now-fixed bounded-
submission code (see n150_worker_count_sweep_findings.md's "Bug 2" -
the earlier chunk_size sweep in n100_n150_parallel_decompose_findings.md
predates that fix and ran under the unbounded-submission bug, so its
numbers are not trustworthy for this specific question).

Tests the per-task-IPC-overhead hypothesis directly: larger chunk_size
means fewer, bigger tasks, which should shrink IPC/pickling overhead's
relative share if that is truly the dominant limiter (per
n150_worker_count_sweep_findings.md's "Interpretation" section) - if
speedup stays flat even as chunk_size grows by orders of magnitude,
that would point elsewhere (real per-worker compute cost, or
something else not yet identified).

Records wall-clock (sequential fwht_pauli_terms_iter vs. parallel
n_workers=8) AND total process-tree RSS at each chunk_size - same
POSIX-only /proc/<pid>/status monitoring as
n150_worker_count_sweep.py, per the user's explicit request to always
include memory-footprint analysis alongside speedup measurement.
"""
import os
import subprocess
import threading
import time

from paulikit.algorithms.fwht import fwht_pauli_terms_iter, parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
N_WORKERS = 8
CHUNK_SIZES = [2, 8, 32, 128, 512]

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
dim = padded.shape[0]


def _rss_kib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _descendant_pids(root_pid: int) -> list[int]:
    try:
        out = subprocess.run(
            ["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    parent_of = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        parent_of[pid] = ppid

    descendants = []
    for pid in parent_of:
        p = pid
        seen = set()
        while p in parent_of and p not in seen:
            seen.add(p)
            if p == root_pid:
                descendants.append(pid)
                break
            p = parent_of[p]
    return descendants


class RssMonitor:
    def __init__(self, root_pid: int, interval: float = 0.1):
        self.root_pid = root_pid
        self.interval = interval
        self.peak_kib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> int:
        total = _rss_kib(self.root_pid)
        for pid in _descendant_pids(self.root_pid):
            total += _rss_kib(pid)
        return total

    def _run(self):
        while not self._stop.is_set():
            total = self._sample()
            self.peak_kib = max(self.peak_kib, total)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_kib = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


ROOT_PID = os.getpid()
print(f"N={N_OSCILLATORS} dim={dim} n_workers={N_WORKERS} root_pid={ROOT_PID}")

results = []
for chunk_size in CHUNK_SIZES:
    with RssMonitor(ROOT_PID) as mon:
        t0 = time.perf_counter()
        n_seq = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
            n_seq += len(chunk)
        seq_elapsed = time.perf_counter() - t0
    seq_peak_mib = mon.peak_kib / 1024

    with RssMonitor(ROOT_PID) as mon:
        t0 = time.perf_counter()
        n_par = 0
        for chunk in parallel_decompose(padded, chunk_size=chunk_size, n_workers=N_WORKERS):
            n_par += len(chunk)
        par_elapsed = time.perf_counter() - t0
    par_peak_mib = mon.peak_kib / 1024

    assert n_seq == n_par, (chunk_size, n_seq, n_par)
    speedup = seq_elapsed / par_elapsed
    results.append((chunk_size, seq_elapsed, seq_peak_mib, par_elapsed, par_peak_mib, speedup))
    print(f"chunk_size={chunk_size}: seq={seq_elapsed:.3f}s (rss={seq_peak_mib:.1f} MiB)  "
          f"par={par_elapsed:.3f}s (rss={par_peak_mib:.1f} MiB)  speedup={speedup:.3f}x")

print("\nSummary:")
for chunk_size, seq_e, seq_r, par_e, par_r, speedup in results:
    print(f"  chunk_size={chunk_size:>4}: seq={seq_e:7.3f}s/{seq_r:7.1f}MiB  "
          f"par={par_e:7.3f}s/{par_r:7.1f}MiB  speedup={speedup:.3f}x")
