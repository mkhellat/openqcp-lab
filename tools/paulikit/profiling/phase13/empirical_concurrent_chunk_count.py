"""Empirically, deterministically measures how many worker processes
are ACTUALLY executing on a CPU core at once during a real
parallel_decompose(n_workers=8) run at N=150 - not a theoretical bound,
not "at most n_workers" - a direct sample of /proc/<pid>/stat's process
state for every real worker PID, taken at fine intervals for the whole
run, from a separate monitoring process running concurrently.

Method: launch parallel_decompose(n_workers=8) in a subprocess (so this
script's own monitoring loop isn't competing with it for the GIL/a
core), discover the worker PIDs it spawns (children of that subprocess,
found via /proc/<ppid>/task/*/children or a full /proc scan), and every
~5ms read /proc/<pid>/stat for each worker PID to get its state (R =
running or runnable, S = sleeping, D = uninterruptible sleep) and its
processor field (which physical core it's currently on, per `man 5
proc`). Count of PIDs in state 'R' at a given sample IS the number
of workers the kernel considers runnable at that instant - not
identical to "executing on silicon right now" (only up to physical_core
count of R-state processes can truly be on a core simultaneously, the
rest are runnable but queued), but a direct, code-derived measurement,
not a stated ceiling.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python empirical_concurrent_chunk_count.py
"""
import os
import subprocess
import sys
import time

N_OSCILLATORS = 150
N_WORKERS = 8
SAMPLE_INTERVAL_S = 0.005

CHILD_SCRIPT = f"""
import os
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

print(f"CHILD_PID={{os.getpid()}}", flush=True)

spring_constants = _default_spring_constants({N_OSCILLATORS})
masses = _default_masses({N_OSCILLATORS})
unpadded = build_hamiltonian({N_OSCILLATORS}, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

total = 0
for chunk in parallel_decompose(padded, chunk_size=2, n_workers={N_WORKERS}):
    total += len(chunk)
print(f"CHILD_DONE total={{total}}", flush=True)
"""


def read_proc_stat_fields(pid: int):
    """Returns (state_char, processor_int) for a pid, or None if the
    process no longer exists (race with process exit is expected and
    handled by the caller, not treated as an error)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            raw = f.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm field can contain spaces/parens - split on the LAST ')'
    rparen = raw.rfind(b")")
    if rparen == -1:
        return None
    fields = raw[rparen + 2:].split()
    # per `man 5 proc`, fields after comm are 0-indexed from state (3rd
    # overall field); state is fields[0], processor (38th overall) is
    # fields[36] in this post-comm indexing (verified against `man 5
    # proc`'s field table: state=3, ..., processor=39, so post-comm
    # index = 39 - 3 = 36)
    try:
        state = fields[0].decode()
        processor = int(fields[36])
        return state, processor
    except (IndexError, ValueError):
        return None


def discover_children(parent_pid: int) -> set[int]:
    children = set()
    try:
        with open(f"/proc/{parent_pid}/task/{parent_pid}/children") as f:
            children.update(int(p) for p in f.read().split())
    except (FileNotFoundError, ProcessLookupError):
        pass
    return children


if __name__ == "__main__":
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
    )

    # Wait for the child to print its own PID, confirming it's alive
    # and past import time, before starting to sample.
    first_line = proc.stdout.readline()
    print(f"parent sees: {first_line.strip()}")
    child_pid = proc.pid

    worker_pids: set[int] = set()
    samples = []  # list of (timestamp, {pid: (state, processor)})
    r_state_counts = []  # count of PIDs in state 'R' per sample

    t_start = time.perf_counter()
    while proc.poll() is None:
        # Discover any new worker PIDs (ProcessPoolExecutor spawns
        # them once, near the start, but we keep checking in case of
        # a race at startup).
        worker_pids.update(discover_children(child_pid))

        snapshot = {}
        for pid in list(worker_pids):
            result = read_proc_stat_fields(pid)
            if result is not None:
                snapshot[pid] = result
        if snapshot:
            r_count = sum(1 for state, _ in snapshot.values() if state == "R")
            samples.append((time.perf_counter() - t_start, dict(snapshot)))
            r_state_counts.append(r_count)

        time.sleep(SAMPLE_INTERVAL_S)

    remaining_output = proc.stdout.read()
    print(remaining_output.strip())

    if not r_state_counts:
        print("ERROR: no samples collected - worker PIDs never discovered in time")
        sys.exit(1)

    print(f"\ntotal samples={len(r_state_counts)} "
          f"total worker PIDs ever seen={len(worker_pids)}")
    print(f"R-state count per sample: "
          f"min={min(r_state_counts)} max={max(r_state_counts)} "
          f"mean={sum(r_state_counts)/len(r_state_counts):.3f}")

    from collections import Counter
    dist = Counter(r_state_counts)
    print("Distribution of (# workers in state R) across all samples:")
    for count in sorted(dist):
        pct = 100 * dist[count] / len(r_state_counts)
        print(f"  {count} workers running: {dist[count]} samples ({pct:.1f}%)")

    # Also report which physical CPUs those R-state processes were
    # actually placed on, aggregated across all samples.
    cpu_usage = Counter()
    for _, snapshot in samples:
        for state, cpu in snapshot.values():
            if state == "R":
                cpu_usage[cpu] += 1
    print("\nWhich logical CPU R-state workers were found on (sample counts):")
    for cpu in sorted(cpu_usage):
        print(f"  cpu{cpu}: {cpu_usage[cpu]} R-state samples")
