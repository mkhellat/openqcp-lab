"""Verifies pinning correctness for the asymmetric core-packing
conditions (pinned_3_2cores, pinned_5_4cores, pinned_5_3cores, etc.) -
each real forked worker process writes its own self-reported
os.sched_getaffinity(0) to a dedicated file on its first chunk, exactly
matching the method already validated for pinned_4_2cores.

Usage:
    OPENBLAS_NUM_THREADS=1 python verify_asymmetric_pinning.py <condition>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_matrix_target import _CONDITIONS  # noqa: E402

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

condition = sys.argv[1]
n_workers, cpu_list = _CONDITIONS[condition]
fwht._physical_core_representative_cpus = lambda cpus=cpu_list: cpus

N_OSCILLATORS = 25  # small - only need a handful of chunks per worker
spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                        "..", "..", "tmp_aff_check")
LOG_DIR = "/tmp/claude-1000/-home-desadm-Projects---0--science-tools---openqcp-lab/7d304dd7-a962-4527-b27b-fcc890dd8a51/scratchpad/asym_aff_check"
os.makedirs(LOG_DIR, exist_ok=True)
for f in os.listdir(LOG_DIR):
    os.remove(os.path.join(LOG_DIR, f))

_real_chunk = fwht._parallel_worker_chunk


def wrapped(ci, cs, ce):
    aff = tuple(sorted(os.sched_getaffinity(0)))
    path = os.path.join(LOG_DIR, f"{os.getpid()}.txt")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(str(aff) + "\n")
    return _real_chunk(ci, cs, ce)


fwht._parallel_worker_chunk = wrapped

total = 0
for chunk in parallel_decompose(padded, chunk_size=2, n_workers=n_workers):
    total += len(chunk)

print(f"condition={condition} intended_cpu_list={cpu_list} n_workers={n_workers} terms={total}")
observed = []
for fname in sorted(os.listdir(LOG_DIR)):
    with open(os.path.join(LOG_DIR, fname)) as f:
        observed.append(f.read().strip())
print(f"observed affinities from {len(observed)} worker processes: {observed}")
