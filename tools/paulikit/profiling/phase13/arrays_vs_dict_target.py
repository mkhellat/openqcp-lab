"""Task 7 target: does parallel_decompose_arrays actually deliver the
2.191x multi-core scaling measured by the SYNTHETIC arrays-only drain
loop in drain_gil_backpressure_results.jsonl, once it runs inside the
REAL shipped pipeline - real chunking, real auto-tuning, real CPU
pinning, and (in one of its two variants) real per-chunk checkpoint
I/O in the parent?

Three API variants, selected by sys.argv[1]:
  dict                  - parallel_decompose (existing, dict-yielding)
  arrays_no_checkpoint  - parallel_decompose_arrays, checkpoint_path=None
  arrays_with_checkpoint- parallel_decompose_arrays, checkpoint_path=<file>

Condition (n_workers/cpu pinning) selected by sys.argv[2], one of the
w<n>_c<c> keys in condition_table.CONDITIONS (same table used by
full_matrix_target.py and narrowed_index_dtype_sweep.py). chunk_size
is sys.argv[3], defaulting to 2 (the N=150 tuned value, per the task
brief).

Modeled directly on full_matrix_target.py: same RssMonitor, same
_physical_core_representative_cpus monkeypatch mechanism for pinning,
same stdout contract (space-separated key=value tokens) so the sweep
driver can parse it exactly like narrowed_index_dtype_sweep.py parses
full_matrix_target.py's output.

The checkpoint file is written to a fresh path under a per-run tmp
directory (given by sys.argv[4] when the variant is
arrays_with_checkpoint) so repeated reps never resume from a prior
rep's partial file - each rep must pay the FULL per-chunk write cost,
not a partially-skipped one.
"""
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import CONDITIONS as _CONDITIONS  # noqa: E402

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose, parallel_decompose_arrays
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150

variant = sys.argv[1]
assert variant in (
    "dict", "dict_with_checkpoint", "arrays_no_checkpoint", "arrays_with_checkpoint"
), (
    f"unknown variant {variant!r}"
)
condition = sys.argv[2]
assert condition in _CONDITIONS, f"unknown condition {condition!r}"
CHUNK_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
checkpoint_path = sys.argv[4] if len(sys.argv) > 4 else None
if variant in ("arrays_with_checkpoint", "dict_with_checkpoint"):
    assert checkpoint_path, f"{variant} requires a checkpoint path arg"


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
fwht._physical_core_representative_cpus = (
    (lambda cpus=cpu_list: cpus) if cpu_list is not None else (lambda: None)
)

total_terms = 0
root_pid = os.getpid()

with RssMonitor(root_pid) as mon:
    t0 = time.perf_counter()
    if variant == "dict":
        for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
            total_terms += len(chunk)
    elif variant == "arrays_no_checkpoint":
        for x, z, coeff in parallel_decompose_arrays(
            padded, chunk_size=CHUNK_SIZE, n_workers=n_workers
        ):
            total_terms += len(x)
    elif variant == "dict_with_checkpoint":
        # The control that decides whether the ~280s checkpoint cost
        # measured for arrays_with_checkpoint is INHERITED from the
        # shared, unmodified _append_parallel_checkpoint_chunk (which
        # writes one JSON line per surviving term, ~91.6M lines at
        # N=150) or INTRODUCED by the array path. Both variants call
        # the same checkpoint writer, so if this cell lands at a
        # similar magnitude the cost is pre-existing and unrelated to
        # this phase's change.
        for chunk in parallel_decompose(
            padded, chunk_size=CHUNK_SIZE, n_workers=n_workers,
            checkpoint_path=checkpoint_path,
        ):
            total_terms += len(chunk)
    else:  # arrays_with_checkpoint
        for x, z, coeff in parallel_decompose_arrays(
            padded, chunk_size=CHUNK_SIZE, n_workers=n_workers,
            checkpoint_path=checkpoint_path,
        ):
            total_terms += len(x)
    elapsed = time.perf_counter() - t0

print(f"variant={variant} condition={condition} total_terms={total_terms} "
      f"elapsed={elapsed:.4f}s peak_rss_mib={mon.peak_kib/1024:.1f}")
