"""Launches parallel_decompose(n_workers=4) at N=150 and, from a
separate monitoring thread, samples which logical CPU (psr, via
`ps -o pid,psr`) each worker process is actually running on -
directly answers whether 4 workers land one-per-physical-core or get
placed however the OS scheduler likes (including both hyperthreads of
one physical core simultaneously).
"""
import os
import subprocess
import threading
import time

from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
N_WORKERS = 4
CHUNK_SIZE = 2

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

ROOT_PID = os.getpid()
samples = []
stop = threading.Event()


def descendant_pids(root_pid):
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
    parent_of = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            parent_of[int(parts[0])] = int(parts[1])
    result = []
    for pid in parent_of:
        p, seen = pid, set()
        while p in parent_of and p not in seen:
            seen.add(p)
            if p == root_pid:
                result.append(pid)
                break
            p = parent_of[p]
    return result


def monitor():
    while not stop.is_set():
        pids = descendant_pids(ROOT_PID)
        if pids:
            out = subprocess.run(
                ["ps", "-o", "pid=,psr="] + sum([["-p", str(p)] for p in pids], []),
                capture_output=True, text=True
            ).stdout
            placements = {}
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    placements[int(parts[0])] = int(parts[1])
            if placements:
                samples.append(dict(placements))
        time.sleep(0.15)


t = threading.Thread(target=monitor, daemon=True)
t.start()

total = 0
for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=N_WORKERS):
    total += len(chunk)

stop.set()
t.join(timeout=2)

print(f"terms={total}")
print(f"collected {len(samples)} placement samples")

# Aggregate: for each worker pid, which logical CPUs did it ever run on?
pid_to_cpus = {}
for sample in samples:
    for pid, cpu in sample.items():
        pid_to_cpus.setdefault(pid, set()).add(cpu)

physical_core_of = {0: "A", 4: "A", 1: "B", 5: "B", 2: "C", 6: "C", 3: "D", 7: "D"}

print("\nPer-worker observed logical-CPU placements (and physical core letter):")
for pid, cpus in sorted(pid_to_cpus.items()):
    cores = sorted({physical_core_of.get(c, "?") for c in cpus})
    print(f"  pid={pid}: logical CPUs seen={sorted(cpus)}  physical core(s)={cores}")

# Snapshot-level co-residency check: in any single sample, did two
# worker PIDs share the same physical core simultaneously?
print("\nCo-residency check (same physical core, same instant):")
collisions = 0
for sample in samples:
    core_occupants = {}
    for pid, cpu in sample.items():
        core = physical_core_of.get(cpu, "?")
        core_occupants.setdefault(core, []).append(pid)
    for core, pids in core_occupants.items():
        if len(pids) > 1:
            collisions += 1
print(f"  {collisions} / {len(samples)} samples had 2+ workers on the same physical core at once")
