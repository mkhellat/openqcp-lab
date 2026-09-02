"""Real N=150 n_workers sweep (1/2/4/8) - directly requested by the
user after observing memory spikes and asking for combined
memory-footprint + worker-count analysis in the same investigation
pass (PLAN.md Phase 13, see the requesting conversation).

Discriminates between two remaining hypotheses for why
parallel_decompose's speedup is far below linear
(n100_n150_parallel_decompose_findings.md): roughly-linear scaling
that degrades toward n_workers=8 would point at L3/memory-bandwidth
saturation; flat speedup even at low worker counts would point at
per-task dispatch/IPC overhead instead. Records wall-clock AND total
process-tree RSS (main process + all worker children) at each worker
count - POSIX-only (/proc/<pid>/status), no third-party dependency
(psutil not installed), matching this project's own established
memory-measurement discipline (autotune.py).
"""
import os
import subprocess
import threading
import time

from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

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
    """Polls total process-tree RSS (this process + all descendants)
    at a fixed interval on a background thread, records the peak."""

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
print(f"N={N_OSCILLATORS} dim={dim} root_pid={ROOT_PID}")

results = []
for n_workers in [1, 2, 4, 8]:
    with RssMonitor(ROOT_PID) as mon:
        t0 = time.perf_counter()
        total_terms = 0
        for chunk in parallel_decompose(padded, n_workers=n_workers):
            total_terms += len(chunk)
        elapsed = time.perf_counter() - t0
    peak_mib = mon.peak_kib / 1024
    results.append((n_workers, elapsed, peak_mib, total_terms))
    print(f"n_workers={n_workers}: elapsed={elapsed:.3f}s peak_rss={peak_mib:.1f} MiB "
          f"terms={total_terms}")

baseline_elapsed = results[0][1]
print("\nSummary (relative to n_workers=1):")
for n_workers, elapsed, peak_mib, terms in results:
    speedup = baseline_elapsed / elapsed
    print(f"  n_workers={n_workers}: speedup={speedup:.3f}x  peak_rss={peak_mib:.1f} MiB  "
          f"terms={terms}")
