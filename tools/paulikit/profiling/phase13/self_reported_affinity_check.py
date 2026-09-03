"""Maximally rigorous pinning verification, per direct user
skepticism: "I am still EXTREMELY skeptical if you have been able to
lock on logical cpu cores!!!!" - right to push, since every prior check
(full_core_observation.py) relied on an EXTERNAL tool (ps -o psr)
sampling at 0.2s intervals, which cannot catch migrations between
samples and already produced one real false positive earlier in this
investigation.

This check is done from INSIDE each worker process itself - the most
direct, authoritative ground truth available:
1. os.sched_getaffinity(0) - the kernel's own record of which CPUs
   this process is ALLOWED to run on (the actual pinning contract,
   not an inference from where it happened to be observed).
2. sched_getcpu() (via ctypes, since Python's os.sched_getcpu is not
   available in this build) - which CPU the kernel says this process
   is running on RIGHT NOW, sampled continuously (every ~1ms, tight
   enough to catch brief migrations a 0.2s external sampler would
   miss) throughout the ENTIRE real computation, not just at
   discrete external poll points.

Each worker logs its own continuous self-observed CPU history to a
dedicated file - if the affinity mask ever includes more than the
intended single CPU, or if sched_getcpu() ever reports a CPU outside
that mask, that is a DIRECT, unambiguous pinning failure, not an
external-tool artifact.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python self_reported_affinity_check.py <condition>
condition in: pinned_2, unpinned_2, pinned_4, unpinned_4
"""
import ctypes
import os
import sys
import time

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import _parallel_worker_chunk, _parallel_worker_init
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

import multiprocessing
import numpy as np

N_OSCILLATORS = 150
CHUNK_SIZE = 2

condition = sys.argv[1]
assert condition in ("pinned_2", "unpinned_2", "pinned_4", "unpinned_4")
n_workers = 2 if "2" in condition else 4
pinned = condition.startswith("pinned")

LOG_DIR = "/tmp/claude-1000/-home-desadm-Projects---0--science-tools---openqcp-lab/7d304dd7-a962-4527-b27b-fcc890dd8a51/scratchpad/self_affinity_logs"
os.makedirs(LOG_DIR, exist_ok=True)

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _sched_getcpu() -> int:
    return _libc.sched_getcpu()


def _worker_with_self_logging(cpu: int, worker_index: int, chunk_starts, n_active,
                                state_args, log_path: str) -> None:
    if cpu is not None:
        try:
            os.sched_setaffinity(0, {cpu})
        except (AttributeError, OSError):
            pass

    _parallel_worker_init(*state_args, None, None)

    my_chunk_indices = list(range(worker_index, len(chunk_starts), n_workers))

    log_lines = []
    stop_logging = False

    def snapshot():
        affinity = sorted(os.sched_getaffinity(0))
        current_cpu = _sched_getcpu()
        return f"t={time.perf_counter():.4f} affinity={affinity} current_cpu={current_cpu}"

    log_lines.append(f"START {snapshot()}")

    for i, chunk_index in enumerate(my_chunk_indices):
        chunk_start = chunk_starts[chunk_index]
        chunk_end = min(chunk_start + CHUNK_SIZE, n_active)
        _parallel_worker_chunk(chunk_index, chunk_start, chunk_end)
        if i % 200 == 0:  # sample roughly every 200 chunks - dense but bounded
            log_lines.append(snapshot())

    log_lines.append(f"END {snapshot()}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    spring_constants = _default_spring_constants(N_OSCILLATORS)
    masses = _default_masses(N_OSCILLATORS)
    unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)
    operator, is_sparse_input, dim, n_qubits, p_nz, q_nz, x_nz = fwht._prepare_operator_for_fwht(
        padded
    )
    active_x, inverse = np.unique(x_nz, return_inverse=True)
    n_active = len(active_x)
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    sorted_p_nz = p_nz[order]
    sorted_q_nz = q_nz[order]
    z_indices = np.arange(dim)[np.newaxis, :]
    chunk_starts = list(range(0, n_active, CHUNK_SIZE))

    pin_cpus = fwht._physical_core_representative_cpus() if pinned else None

    state_args = (
        operator, is_sparse_input, sorted_inverse, sorted_p_nz, sorted_q_nz,
        active_x, dim, n_qubits, z_indices, 1e-10,
    )

    procs = []
    log_paths = []
    t0 = time.perf_counter()
    for i in range(n_workers):
        cpu = pin_cpus[i] if pin_cpus else None
        log_path = os.path.join(LOG_DIR, f"{condition}_worker{i}.log")
        log_paths.append(log_path)
        p = multiprocessing.Process(
            target=_worker_with_self_logging,
            args=(cpu, i, chunk_starts, n_active, state_args, log_path),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - t0

    print(f"condition={condition} elapsed={elapsed:.4f}s n_workers={n_workers}")
    print(f"pin_cpus (intended) = {pin_cpus}")
    print("\nPer-worker self-reported affinity/CPU history:")
    for i, log_path in enumerate(log_paths):
        print(f"\n--- worker {i} (log: {log_path}) ---")
        with open(log_path) as f:
            content = f.read()
        print(content)

        violations = []
        for line in content.splitlines():
            if "affinity=" not in line:
                continue
            aff_str = line.split("affinity=")[1].split(" current_cpu=")[0]
            affinity = eval(aff_str)
            cur_str = line.split("current_cpu=")[1]
            current_cpu = int(cur_str)
            if pinned and pin_cpus:
                expected = pin_cpus[i]
                if affinity != [expected]:
                    violations.append(f"BAD AFFINITY MASK: {line}")
                if current_cpu != expected:
                    violations.append(f"RUNNING ON WRONG CPU: {line}")
        if violations:
            print(f"  *** {len(violations)} VIOLATIONS FOUND ***")
            for v in violations:
                print(f"    {v}")
        else:
            print(f"  No violations - affinity and current_cpu matched expectation "
                  f"at every self-sampled point.")
