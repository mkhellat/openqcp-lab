"""Combines worker-process CPU placement + system-wide process
placement + per-core frequency, sampled TOGETHER at every instant -
direct fix for a real gap the user caught: freq_scaling_check.py only
measured per-core frequency in isolation, with no way to tell WHICH
process was actually running on a given core when it showed elevated
activity. "The important question is why in hell were those other 6
cores also active?? How could you be sure that only 2 of the logical
cores were locked in?!! If you wanna rerun you must closely observe
all logical cores!!"

For every 0.2s sample: records (a) each real worker process's actual
logical-CPU placement (ps -o pid,psr, matching check_worker_placement.py's
method), (b) EVERY process on the system and which CPU it's on right
now (ps -e -o pid,psr,comm - not just descendants), and (c) all 8
cores' current frequency - all from the SAME instant, so cross-
referencing is possible after the fact (e.g. "cpu5 was at 3.8GHz at
sample 40 - what was running there?").

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python full_core_observation.py <condition>
condition in: pinned_2, unpinned_2, pinned_4, unpinned_4
"""
import os
import subprocess
import sys
import threading
import time
from collections import Counter

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2

condition = sys.argv[1]
assert condition in ("pinned_2", "unpinned_2", "pinned_4", "unpinned_4")
n_workers = 2 if "2" in condition else 4
pinned = condition.startswith("pinned")

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

_real_physical_core_representative_cpus = fwht._physical_core_representative_cpus
if pinned:
    fwht._physical_core_representative_cpus = _real_physical_core_representative_cpus
else:
    fwht._physical_core_representative_cpus = lambda: None

ROOT_PID = os.getpid()


def _read_freq_mhz(cpu: int) -> float | None:
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def direct_python_children(root_pid: int) -> set[int]:
    """Only DIRECT children of root_pid whose command is 'python' -
    i.e. the actual ProcessPoolExecutor worker processes, not
    transient `ps`/`subprocess.run` children this monitor itself
    spawns each sample (a real bug found by the user's own scrutiny:
    an earlier version used ALL descendants, which re-counted this
    script's own `ps` calls as "WORKER" on whatever core the scheduler
    happened to place that short-lived process - see
    full_core_observation_findings.md for the direct verification that
    caught this)."""
    out = subprocess.run(
        ["ps", "--ppid", str(root_pid), "-o", "pid=,comm="], capture_output=True, text=True
    ).stdout
    result = set()
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and "python" in parts[1]:
            result.add(int(parts[0]))
    return result


samples = []  # list of dicts: {cpu: [(pid, comm), ...], "freq": {cpu: mhz}}
stop = threading.Event()


def monitor():
    while not stop.is_set():
        worker_pids = direct_python_children(ROOT_PID)
        # Everything running RIGHT NOW on each CPU, system-wide.
        out = subprocess.run(
            ["ps", "-e", "-o", "pid=,psr=,comm="], capture_output=True, text=True
        ).stdout
        cpu_occupants = {cpu: [] for cpu in range(8)}
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) != 3:
                continue
            try:
                pid, cpu = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            comm = parts[2]
            if cpu in cpu_occupants:
                tag = "WORKER" if pid in worker_pids else ("SELF" if pid == ROOT_PID else comm)
                cpu_occupants[cpu].append((pid, tag))

        freqs = {cpu: _read_freq_mhz(cpu) for cpu in range(8)}
        samples.append({"occupants": cpu_occupants, "freq": freqs})
        time.sleep(0.2)


t = threading.Thread(target=monitor, daemon=True)
t.start()

t0 = time.perf_counter()
total = 0
for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
    total += len(chunk)
elapsed = time.perf_counter() - t0

stop.set()
t.join(timeout=2)

print(f"condition={condition} elapsed={elapsed:.4f}s terms={total} n_samples={len(samples)}")

# Per-core: how many samples had a WORKER present, and what else (by
# comm name) showed up there across the whole run.
print("\nPer-core occupancy summary (across the whole run):")
for cpu in range(8):
    worker_count = 0
    other_comms = Counter()
    freq_when_worker_present = []
    freq_when_worker_absent = []
    for s in samples:
        occ = s["occupants"][cpu]
        has_worker = any(tag == "WORKER" for _, tag in occ)
        if has_worker:
            worker_count += 1
        for pid, tag in occ:
            if tag not in ("WORKER", "SELF"):
                other_comms[tag] += 1
        freq = s["freq"][cpu]
        if freq is not None:
            (freq_when_worker_present if has_worker else freq_when_worker_absent).append(freq)

    def _fmt(vals):
        if not vals:
            return "n/a"
        return f"mean={sum(vals)/len(vals):.0f}MHz n={len(vals)}"

    print(f"  cpu{cpu}: worker present in {worker_count}/{len(samples)} samples  "
          f"freq|worker-present: {_fmt(freq_when_worker_present)}  "
          f"freq|worker-absent: {_fmt(freq_when_worker_absent)}")
    if other_comms:
        top = other_comms.most_common(5)
        print(f"          other processes seen here: {top}")

print("\nFull per-sample timeline (cpu: WORKER or top process there, freq MHz):")
for i, s in enumerate(samples):
    row = []
    for cpu in range(8):
        occ = s["occupants"][cpu]
        if any(tag == "WORKER" for _, tag in occ):
            label = "WORKER"
        elif occ:
            label = occ[0][1][:10]
        else:
            label = "-"
        freq = s["freq"][cpu]
        freq_str = f"{freq:.0f}" if freq is not None else "?"
        row.append(f"{label}:{freq_str}")
    print(f"  t={i*0.2:5.1f}s  " + "  ".join(f"cpu{c}={row[c]}" for c in range(8)))
