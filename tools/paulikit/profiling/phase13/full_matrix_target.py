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
      -e task-clock,mem_load_retired.l1_hit,mem_load_retired.l1_miss,\
mem_load_retired.l2_hit,mem_load_retired.l2_miss,L1-dcache-loads,\
L1-dcache-load-misses \
      python full_matrix_target.py <condition_name> l1l2

    OPENBLAS_NUM_THREADS=1 perf stat --no-inherit \
      -e task-clock,cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses \
      python full_matrix_target.py <condition_name> l3

condition_name in: pinned_4, unpinned_4, workers_8, pinned_2, unpinned_2, seq_1
"""
import os
import subprocess
import sys
import threading
import time

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import fwht_pauli_terms_iter, parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2

condition = sys.argv[1]

# Physical-core topology on this dev machine (checked directly, see
# scoping.md): core A=(0,4), B=(1,5), C=(2,6), D=(3,7). Each entry is
# (n_workers, explicit_cpu_list_or_None). None means unpinned
# (_physical_core_representative_cpus -> None, the same code path
# parallel_decompose already uses when pinning is genuinely
# unavailable). A list gives one CPU per worker, in claim order -
# workers 2+ on the SAME physical core as an earlier worker means that
# core is "doubled up" (both hyperthread siblings in use); a core
# never appearing means it is left completely idle.
_CONDITIONS: dict[str, tuple[int, list[int] | None]] = {
    "seq_1": (0, None),  # special-cased below (no pool at all)
    "pinned_2": (2, [0, 1]),
    "unpinned_2": (2, None),
    "pinned_4": (4, [0, 1, 2, 3]),
    "unpinned_4": (4, None),
    "workers_8": (8, [0, 1, 2, 3]),  # 8 logical CPUs, pinned default
    # 4-vs-2-physical-cores comparison (pinned4_4cores_vs_2cores_findings.md):
    "pinned_4_4cores": (4, [0, 1, 2, 3]),
    "pinned_4_2cores": (4, [0, 4, 1, 5]),
    # 2-vs-1-physical-core comparison:
    "pinned_2_2cores": (2, [0, 1]),
    "pinned_2_1core": (2, [0, 4]),
    # 3-vs-2-physical-cores comparison (2+1 packing on the 2-core side):
    "pinned_3_3cores": (3, [0, 1, 2]),
    "pinned_3_2cores": (3, [0, 4, 1]),
    # 5-vs-4-vs-3-physical-cores comparison (2+1+1+1, then 2+2+1 packing):
    "pinned_5_4cores": (5, [0, 4, 1, 2, 3]),
    "pinned_5_3cores": (5, [0, 4, 1, 5, 2]),
}

# Full enumeration for the publication-grade sweep
# (full_optimum_sweep_findings.md): every DISTINCT valid (n_workers,
# n_physical_cores_used) configuration on this machine (4 physical
# cores, 2 hyperthreads each - a core hosts at most 2 workers). Named
# w<n_workers>_c<n_cores> for unambiguous, systematic identification.
# Core assignment always fills cores in order A,B,C,D, doubling up
# (both hyperthread siblings) before moving to the next core, i.e. the
# "most packed" valid assignment for that (n_workers, n_cores) pair -
# matches the packing convention already used and verified in every
# earlier pinned_N_Mcores condition in this file.
_CORE_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]  # physical cores A,B,C,D


def _packed_cpu_list(n_workers: int, n_cores: int) -> list[int]:
    """One representative logical CPU per worker, spread as EVENLY as
    possible across exactly n_cores DISTINCT physical cores (using
    both hyperthread siblings of a core only when n_workers exceeds
    n_cores) - e.g. (4 workers, 3 cores) -> 2+1+1, not 2+2+(0 workers
    on a 3rd core). This is the only assignment that actually uses
    n_cores distinct cores for every (n_workers, n_cores) pair in
    range - a naive "pack cores 0..n_cores-1 to 2 each, ignore the
    rest" would silently collapse several distinct n_cores values onto
    the same CPU list (a real bug caught before running anything: an
    earlier version of this function did exactly that, generating only
    14 of the intended 17 configurations with silent duplicates)."""
    assert 1 <= n_cores <= 4
    assert n_cores <= n_workers <= 2 * n_cores
    base, extra = divmod(n_workers, n_cores)
    cpus = []
    for core_idx in range(n_cores):
        take = base + (1 if core_idx < extra else 0)
        cpus.extend(_CORE_PAIRS[core_idx][:take])
    assert len(cpus) == n_workers
    return cpus


_SWEEP_CONFIGS: dict[str, tuple[int, list[int]]] = {}
for _n_workers in range(1, 9):
    _min_cores = (_n_workers + 1) // 2  # ceil(n_workers / 2)
    _max_cores = min(_n_workers, 4)
    for _n_cores in range(_min_cores, _max_cores + 1):
        _SWEEP_CONFIGS[f"w{_n_workers}_c{_n_cores}"] = (
            _n_workers, _packed_cpu_list(_n_workers, _n_cores)
        )
_CONDITIONS.update(_SWEEP_CONFIGS)

assert condition in _CONDITIONS, f"unknown condition {condition!r}"


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
