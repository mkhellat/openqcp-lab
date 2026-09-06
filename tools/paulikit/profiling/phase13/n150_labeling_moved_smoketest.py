"""Real N=150 wall-clock smoke test of the fix in
dag_extraction_with_labeling_moved.md (_pauli_label_batch relocated
from parallel_decompose's drain loop into _parallel_worker_chunk) -
runs the ACTUAL parallel_decompose generator end-to-end (drain loop,
checkpointing path untouched, generator semantics all exercised),
not a synthetic bypass of it.

Four conditions, named by the user's own w{n}_c{k} convention (n
workers, pinned across k distinct physical cores):

    w1     - 1 worker, 1 core (uncontended baseline)
    w2_c1  - 2 workers, BOTH pinned to the SAME 1 physical core
    w2_c2  - 2 workers, pinned to 2 distinct physical cores (no sharing)
    w8_c4  - 8 workers, pinned across this machine's 4 physical cores
             (2 workers per core - the historical contended case from
             perf_deep_dive_correlation.md's w8_c4 capture)

parallel_decompose's own pinning (_physical_core_representative_cpus)
always assigns one DISTINCT physical core per worker in round-robin -
n_workers=2 naturally gives w2_c2, n_workers=8 on this 4-core machine
naturally gives w8_c4, but there is no way to force 2 workers onto 1
shared core (w2_c1) through n_workers alone. This script monkeypatches
_physical_core_representative_cpus IN THIS SCRIPT ONLY (not the
library) to truncate the representative-CPU list to exactly k entries
before parallel_decompose reads it, which is enough to force w2_c1's
sharing without touching parallel_decompose's own pinning logic.

Thermal control (per thermal_controlled_welch_ttest.py's own finding:
this 15W-TDP laptop CPU genuinely throttles under sustained multi-core
load, intel_pstate's 'performance' governor does not prevent it) -
same technique reused directly, not reinvented: cool down to
<=COOLDOWN_TARGET_C before every single timed run (not just once at
the start), and log starting/mean/max package temp per run as a
covariate so any residual thermal confound is visible rather than
silently baked into the wall-clock numbers. Conditions are interleaved
per repetition (not run back-to-back by condition) to avoid a
monotonic drift confound between conditions.

Usage (foreground only, matches this investigation's own standing
foreground-only-measurement-jobs rule):
    OPENBLAS_NUM_THREADS=1 python n150_labeling_moved_smoketest.py
"""
import statistics
import threading
import time
from unittest import mock

from paulikit.algorithms import fwht
from paulikit.algorithms.fwht import parallel_decompose
from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

N_OSCILLATORS = 150
CHUNK_SIZE = 2
REPS = 3
COOLDOWN_TARGET_C = 55.0
COOLDOWN_TIMEOUT_S = 180

spring_constants = _default_spring_constants(N_OSCILLATORS)
masses = _default_masses(N_OSCILLATORS)
unpadded = build_hamiltonian(N_OSCILLATORS, spring_constants, masses, sparse=True)
padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

real_pin_cpus = fwht._physical_core_representative_cpus()
if real_pin_cpus is None or len(real_pin_cpus) < 4:
    raise SystemExit(
        f"expected >=4 physical cores for w8_c4, got {real_pin_cpus} - "
        "this smoke test assumes this machine's known 4-physical-core topology"
    )

CONDITIONS = [
    ("w1", 1, real_pin_cpus[:1]),
    ("w2_c1", 2, real_pin_cpus[:1]),
    ("w2_c2", 2, real_pin_cpus[:2]),
    ("w8_c4", 8, real_pin_cpus[:4]),
]


def _read_pkg_temp_c() -> float | None:
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def cooldown() -> float | None:
    start = time.perf_counter()
    while True:
        temp = _read_pkg_temp_c()
        if temp is not None and temp <= COOLDOWN_TARGET_C:
            return temp
        if time.perf_counter() - start > COOLDOWN_TIMEOUT_S:
            return temp  # give up, report whatever temp it settled at
        time.sleep(2)


def run_once(n_workers: int, pin_cpus_override: list[int]) -> dict:
    temp_before = _read_pkg_temp_c()
    temp_samples: list[float] = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            t = _read_pkg_temp_c()
            if t is not None:
                temp_samples.append(t)
            time.sleep(0.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    with mock.patch.object(
        fwht, "_physical_core_representative_cpus", return_value=pin_cpus_override
    ):
        t0 = time.perf_counter()
        total_terms = 0
        for chunk_terms in parallel_decompose(padded, chunk_size=CHUNK_SIZE, n_workers=n_workers):
            total_terms += len(chunk_terms)
        elapsed = time.perf_counter() - t0

    stop.set()
    thread.join(timeout=2)

    return {
        "elapsed": elapsed,
        "n_terms": total_terms,
        "temp_before": temp_before,
        "temp_mean": statistics.mean(temp_samples) if temp_samples else None,
        "temp_max": max(temp_samples) if temp_samples else None,
    }


if __name__ == "__main__":
    print(f"N={N_OSCILLATORS} dim={padded.shape[0]} chunk_size={CHUNK_SIZE} "
          f"physical_cores_detected={real_pin_cpus} reps={REPS}")
    print("(post-fix: _pauli_label_batch now runs in the worker, not the drain loop)")
    print(f"cooldown_target={COOLDOWN_TARGET_C}C cooldown_timeout={COOLDOWN_TIMEOUT_S}s")
    print()

    results: dict[str, list[dict]] = {name: [] for name, _, _ in CONDITIONS}

    for rep in range(REPS):
        for name, n_workers, pin_cpus in CONDITIONS:
            settled_temp = cooldown()
            r = run_once(n_workers, pin_cpus)
            results[name].append(r)
            print(f"rep={rep} {name:8s} (n_workers={n_workers}, pinned_to={pin_cpus}): "
                  f"terms={r['n_terms']} elapsed={r['elapsed']:.4f}s "
                  f"cooldown_settled={settled_temp} "
                  f"temp_before={r['temp_before']} "
                  f"temp_mean={r['temp_mean']} temp_max={r['temp_max']}", flush=True)

    print("\n=== Summary (mean over reps) ===")
    means = {}
    for name, _, _ in CONDITIONS:
        elapsed_vals = [r["elapsed"] for r in results[name]]
        mean_t = statistics.mean(elapsed_vals)
        means[name] = mean_t
        print(f"{name:8s}: mean={mean_t:.4f}s individual={[f'{v:.4f}' for v in elapsed_vals]}")

    print()
    baseline = means["w1"]
    for name, _, _ in CONDITIONS:
        print(f"  speedup vs w1: {name:8s} {baseline / means[name]:.3f}x")
