"""oneTBB-parallel half of the Phase 10 TBB-labeling re-measurement -
see tbb_label_40m_serial.py's docstring for the perf stat invocation
(same command, this script's filename in place of the serial one) and
tbb_labeling_n150_findings.md for the results/analysis.
"""

import numpy as np
from paulikit._native import pauli_label_native as native

rng = np.random.default_rng(0)
n_qubits = 14
dim = 2**n_qubits
n_terms = 40_000_000

x = rng.integers(0, dim, size=n_terms).astype(np.uint32)
z = rng.integers(0, dim, size=n_terms).astype(np.uint32)

labels = native.pauli_label_batch_parallel(x, z, n_qubits)
print(f"done, {len(labels)} labels")
