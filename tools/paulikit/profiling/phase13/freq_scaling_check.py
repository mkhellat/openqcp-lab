"""Checks whether CPU frequency scaling (powersave governor, known
from cache_probe_extension_findings.md to swing 400MHz-3.3GHz during
measurement and corrupt an earlier wall-clock-based experiment) could
explain the pinned_2/pinned_4 wall-clock-regression and pinned_2's
elevated variance found in the Welch's t-test investigation - a
concrete, testable "is something off with measurement" hypothesis
raised directly by the user, rather than trusting the wall-clock
numbers as-is.

Samples all 8 logical CPUs' scaling_cur_freq every 0.2s throughout one
real parallel_decompose run per condition, reports per-core mean/min/
max/stdev frequency during that run.

Usage (foreground only):
    OPENBLAS_NUM_THREADS=1 python freq_scaling_check.py <condition>
condition in: pinned_2, unpinned_2, pinned_4, unpinned_4
"""
import statistics
import sys
import threading
import time

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


def _read_freq_mhz(cpu: int) -> float | None:
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _read_pkg_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


samples = {cpu: [] for cpu in range(8)}
temp_samples = []
stop = threading.Event()


def monitor():
    while not stop.is_set():
        for cpu in range(8):
            freq = _read_freq_mhz(cpu)
            if freq is not None:
                samples[cpu].append(freq)
        temp = _read_pkg_temp_c()
        if temp is not None:
            temp_samples.append(temp)
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

print(f"condition={condition} elapsed={elapsed:.4f}s terms={total}")
print(f"\nPer-core frequency during run (MHz):")
for cpu in range(8):
    vals = samples[cpu]
    if not vals:
        continue
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  cpu{cpu}: mean={mean:7.1f}  min={min(vals):7.1f}  max={max(vals):7.1f}  "
          f"stdev={stdev:6.1f}  n_samples={len(vals)}")

if temp_samples:
    t_mean = statistics.mean(temp_samples)
    t_stdev = statistics.stdev(temp_samples) if len(temp_samples) > 1 else 0.0
    print(f"\npackage temp (x86_pkg_temp, C): mean={t_mean:.1f}  min={min(temp_samples):.1f}  "
          f"max={max(temp_samples):.1f}  stdev={t_stdev:.1f}  n_samples={len(temp_samples)}")
    print(f"  timeline: {[f'{t:.0f}' for t in temp_samples]}")
