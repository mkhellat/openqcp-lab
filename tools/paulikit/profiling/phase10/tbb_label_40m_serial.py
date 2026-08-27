"""Serial-path half of the Phase 10 TBB-labeling re-measurement at
N=150-representative scale - see tbb_labeling_n150_findings.md in
this directory. Run standalone (not via pytest) under `perf stat`,
matching cache_locality/tbb_evaluation_findings.md's methodology:

    OPENBLAS_NUM_THREADS=1 perf stat -e task-clock,cycles,instructions,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,cycle_activity.stalls_total,\
cycle_activity.stalls_mem_any python tbb_label_40m_serial.py
"""

import numpy as np
from paulikit._native import pauli_label_native as native

rng = np.random.default_rng(0)
n_qubits = 14
dim = 2**n_qubits
n_terms = 40_000_000

x = rng.integers(0, dim, size=n_terms).astype(np.uint32)
z = rng.integers(0, dim, size=n_terms).astype(np.uint32)

labels = native.pauli_label_batch(x, z, n_qubits)
print(f"done, {len(labels)} labels")
