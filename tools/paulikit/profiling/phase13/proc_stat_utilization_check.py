"""Checks REAL per-core CPU utilization (from /proc/stat's idle/total
jiffy counts, ground truth independent of HWP frequency reporting)
against scaling_cur_freq during a pinned_2 run - direct follow-up to
the user's sharp question: "It does not make sense the average of ALL
PHYSICAL CORES being on 3.x GHz during execution on pinned 2
workers!!! ... WHY ALL 8?!!"

This machine uses intel_pstate with HWP (Hardware P-States) -
hwp_dynamic_boost is exposed under /sys/devices/system/cpu/intel_pstate/,
meaning the HARDWARE, not the OS, autonomously sets per-core frequency,
and scaling_cur_freq is the kernel's APERF/MPERF-derived ESTIMATE of
effective frequency, not necessarily "how busy this core actually was."
This checks whether cores showing high scaling_cur_freq while running
only kernel idle-management threads (cpuhp/N, idle_inject/N,
ksoftirqd/N) are actually executing real work (per /proc/stat) or
genuinely idle despite the frequency reading.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python proc_stat_utilization_check.py pinned_2
"""
import sys
import threading
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

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

_real_physical_core_representative_cpus = fwht._physical_core_representative_cpus
if pinned:
    fwht._physical_core_representative_cpus = _real_physical_core_representative_cpus
else:
    fwht._physical_core_representative_cpus = lambda: None


def _read_proc_stat_percore():
    """Returns {cpu_index: (total_jiffies, idle_jiffies)}."""
    result = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line[3] == " ":
                continue
            parts = line.split()
            cpu_idx = int(parts[0][3:])
            fields = [int(x) for x in parts[1:]]
            # user nice system idle iowait irq softirq steal guest guest_nice
            idle = fields[3] + fields[4]  # idle + iowait
            total = sum(fields)
            result[cpu_idx] = (total, idle)
    return result


def _read_freq_mhz(cpu: int) -> float | None:
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


samples = []  # list of (proc_stat_snapshot, freqs)
stop = threading.Event()


def monitor():
    prev = _read_proc_stat_percore()
    while not stop.is_set():
        time.sleep(0.5)
        cur = _read_proc_stat_percore()
        freqs = {cpu: _read_freq_mhz(cpu) for cpu in range(8)}
        busy_pct = {}
        for cpu in range(8):
            t0, i0 = prev.get(cpu, (0, 0))
            t1, i1 = cur.get(cpu, (0, 0))
            dt, di = t1 - t0, i1 - i0
            busy_pct[cpu] = 100.0 * (1 - di / dt) if dt > 0 else None
        samples.append((busy_pct, freqs))
        prev = cur


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
print("\nPer-sample (0.5s intervals): cpu busy% (from /proc/stat) and freq (MHz):")
for i, (busy, freqs) in enumerate(samples):
    row = []
    for cpu in range(8):
        b = busy.get(cpu)
        f = freqs.get(cpu)
        b_str = f"{b:5.1f}%" if b is not None else "  n/a"
        f_str = f"{f:5.0f}" if f is not None else "  n/a"
        row.append(f"cpu{cpu}[{b_str},{f_str}MHz]")
    print(f"  t={i*0.5:5.1f}s  " + " ".join(row))

print("\nSummary - mean busy%/freq per core across whole run:")
for cpu in range(8):
    busy_vals = [b[cpu] for b, _ in samples if b.get(cpu) is not None]
    freq_vals = [f[cpu] for _, f in samples if f.get(cpu) is not None]
    mean_busy = sum(busy_vals) / len(busy_vals) if busy_vals else None
    mean_freq = sum(freq_vals) / len(freq_vals) if freq_vals else None
    print(f"  cpu{cpu}: mean_busy={mean_busy:.1f}%  mean_freq={mean_freq:.0f}MHz")
