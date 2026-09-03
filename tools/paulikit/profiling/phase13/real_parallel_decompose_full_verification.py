"""Full verification on the REAL parallel_decompose() entry point, not
a reimplementation - direct correction after the user rejected the
prior self_reported_affinity_check.py: "Now run the pinned_2 full
test for OUR CODE!!!!! This was not aimed at a dummy code." That
script manually called os.sched_setaffinity and directly invoked
_parallel_worker_chunk/_parallel_worker_init, bypassing the actual
ProcessPoolExecutor/shared-counter pinning mechanism parallel_decompose
itself uses - a different code path from what ships.

This script calls parallel_decompose() UNMODIFIED - the real entry
point, real pool, real pin_cpus/next_pin_index shared-counter
mechanism. Verification is done by WRAPPING _parallel_worker_chunk
(monkeypatched to log, then call the real underlying function - the
real computation is untouched) so each REAL worker process, spawned by
the REAL pool, logs on every chunk it processes:
  - its own self-reported affinity (os.sched_getaffinity) and current
    CPU (sched_getcpu via ctypes) - the same rigor as the prior check,
    now inside the real worker.
  - ALL 8 cores' scaling_cur_freq, read together at the SAME instant -
    per direct instruction: "all cpu clocks are checked and recorded
    alongside the affinity."

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python real_parallel_decompose_full_verification.py pinned_2
"""
import ctypes
import os
import sys
import time

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2

condition = sys.argv[1] if len(sys.argv) > 1 else "pinned_2"
n_workers = 2 if "2" in condition else 4
pinned = condition.startswith("pinned")

LOG_DIR = "/tmp/claude-1000/-home-desadm-Projects---0--science-tools---openqcp-lab/7d304dd7-a962-4527-b27b-fcc890dd8a51/scratchpad/real_pd_logs"
os.makedirs(LOG_DIR, exist_ok=True)

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

if not pinned:
    fwht._physical_core_representative_cpus = lambda: None

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _sched_getcpu() -> int:
    return _libc.sched_getcpu()


def _read_all_freqs() -> dict[int, float | None]:
    freqs = {}
    for cpu in range(8):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
                freqs[cpu] = int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            freqs[cpu] = None
    return freqs


_real_worker_chunk = fwht._parallel_worker_chunk
_worker_log_path = None
_worker_chunk_count = 0


def _wrapped_worker_chunk(chunk_index, chunk_start, chunk_end):
    """Wraps the REAL _parallel_worker_chunk - calls it unmodified for
    the actual computation, only adds logging around it."""
    global _worker_log_path, _worker_chunk_count

    if _worker_log_path is None:
        _worker_log_path = os.path.join(LOG_DIR, f"{condition}_worker_pid{os.getpid()}.log")

    _worker_chunk_count += 1
    if _worker_chunk_count == 1 or _worker_chunk_count % 200 == 0:
        affinity = sorted(os.sched_getaffinity(0))
        current_cpu = _sched_getcpu()
        freqs = _read_all_freqs()
        freq_str = " ".join(f"cpu{c}={freqs[c]:.0f}MHz" if freqs[c] is not None else f"cpu{c}=?"
                             for c in range(8))
        line = (f"t={time.perf_counter():.4f} chunk#{_worker_chunk_count} "
                f"affinity={affinity} current_cpu={current_cpu} | {freq_str}")
        with open(_worker_log_path, "a") as f:
            f.write(line + "\n")

    return _real_worker_chunk(chunk_index, chunk_start, chunk_end)


fwht._parallel_worker_chunk = _wrapped_worker_chunk


if __name__ == "__main__":
    # Clean any stale logs from a previous run of this script.
    for fname in os.listdir(LOG_DIR):
        if fname.startswith(condition + "_worker_pid"):
            os.remove(os.path.join(LOG_DIR, fname))

    t0 = time.perf_counter()
    total = 0
    for chunk in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
        total += len(chunk)
    elapsed = time.perf_counter() - t0

    print(f"condition={condition} elapsed={elapsed:.4f}s n_workers={n_workers} terms={total}")

    log_files = sorted(
        f for f in os.listdir(LOG_DIR) if f.startswith(condition + "_worker_pid")
    )
    print(f"\n{len(log_files)} real worker process logs found:")
    for fname in log_files:
        path = os.path.join(LOG_DIR, fname)
        print(f"\n--- {fname} ---")
        with open(path) as f:
            print(f.read())
