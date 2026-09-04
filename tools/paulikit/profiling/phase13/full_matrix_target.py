"""Full performance/cache matrix target - calls the REAL shipped
parallel_decompose() API directly (per direct instruction: full
pipeline analysis, not a lower-level reimplementation), for one of 6
conditions the user specified:

  1. pinned,   n_workers=4, chunk_size=2
  2. unpinned, n_workers=4, chunk_size=2
  3. (pinned, default) n_workers=8, chunk_size=2
  4. pinned,   n_workers=2, chunk_size=2
  5. unpinned, n_workers=2, chunk_size=2
  6. n_workers=1 (sequential fwht_pauli_terms_iter - no pool at all)

"unpinned" is achieved by monkeypatching _physical_core_representative_cpus
to return None for the duration of the call - the exact same code path
already used when pinning is genuinely unavailable (e.g. non-Linux),
not a separate/different mechanism - so this measures a real,
already-supported behavior of the shipped code, not a synthetic
bypass.

Usage (foreground only, one job at a time):
    OPENBLAS_NUM_THREADS=1 perf stat --no-inherit \
      -e task-clock,cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses \
      python full_matrix_target.py <condition_name> [chunk_size]

condition_name: any key of CONDITIONS (condition_table.py), including
the w<n_workers>_c<n_cores> sweep configs. chunk_size is optional,
defaults to 2 (the single-process N=150 tuned value) - added to test
whether that value, tuned only for an uncontended process
(chunk_size_floor_scale_dependence_findings.md's own "does NOT show"
list flags multi-core contention as untested), still holds once
several workers share the machine's cache concurrently.
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import CONDITIONS as _CONDITIONS  # noqa: E402

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import fwht_pauli_terms_iter, parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

condition = sys.argv[1]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"
CHUNK_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 2


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
        p, seen = pid, set()
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
            self.peak_kib = max(self.peak_kib, self._sample())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_kib = self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

n_workers, cpu_list = _CONDITIONS[condition]
if condition != "seq_1":
    fwht._physical_core_representative_cpus = (
        (lambda cpus=cpu_list: cpus) if cpu_list is not None else (lambda: None)
    )

total_terms = 0
root_pid = os.getpid()

with RssMonitor(root_pid) as mon:
    t0 = time.perf_counter()
    if condition == "seq_1":
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=CHUNK_SIZE):
            total_terms += len(chunk)
    else:
        for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
            total_terms += len(chunk)
    elapsed = time.perf_counter() - t0

print(f"condition={condition} total_terms={total_terms} "
      f"elapsed={elapsed:.4f}s peak_rss_mib={mon.peak_kib/1024:.1f}")
