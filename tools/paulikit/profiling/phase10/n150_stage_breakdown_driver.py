"""Per-stage wall-clock breakdown of the real N=150 streaming
pipeline - see full_pipeline_n150_findings.md in this directory for
the results and analysis.

This driver imports a *timing-instrumented copy* of
paulikit/algorithms/fwht.py, not the installed package - the
instrumented copy is not committed (only this driver and the findings
doc are), per the findings doc's own "What this does NOT show" note.
To reproduce:

1. Copy src/paulikit/algorithms/fwht.py to a scratch location, e.g.
   fwht_instrumented.py.
2. Add near the top (after the existing imports):

       import time
       STAGE_TIMES = {"gather": 0.0, "wht": 0.0, "phase_threshold": 0.0,
                      "label": 0.0, "dict_build": 0.0}

3. In `_iter_chunked_coefficients`, wrap three sections with
   `_t0 = time.perf_counter()` / `STAGE_TIMES["<name>"] += time.perf_counter() - _t0`:
   - "gather": the gathered_chunk scatter (before the WHT call)
   - "wht": the `_walsh_hadamard_transform_rows(...)` call itself
   - "phase_threshold": the phase multiply through the
     `np.nonzero(np.abs(chunk_coefficients) > atol)` threshold
     (everything between the WHT call and the `yield`)
4. In `fwht_pauli_terms_iter`, wrap two more sections the same way:
   - "label": the `_pauli_label_batch(...)` call
   - "dict_build": the `_build_real_terms(...)` call (assume_hermitian
     branch) / the `complex_terms` dict comprehension (else branch) -
     both need `yield result` moved after the timer closes, since
     Phase 11 (2026-08-31) replaced the old inline dict-construction
     loop with a shared helper call (see
     ../phase11/n150_post_implementation_findings.md for a worked
     example of this exact instrumentation, post-Phase-11)
5. Update this script's `INSTRUMENTED_MODULE_PATH` below to point at
   your scratch copy, then run under a memory cap, e.g.:

       bash -c "ulimit -v 4000000; python n150_stage_breakdown_driver.py [--parallel-labels]"

   with a `free -m` polling loop guarding against real system memory
   exhaustion (same harness used throughout this project).
"""

import importlib.util
import sys
import time

INSTRUMENTED_MODULE_PATH = "/path/to/your/scratch/fwht_instrumented.py"

spec = importlib.util.spec_from_file_location("fwht_instrumented", INSTRUMENTED_MODULE_PATH)
fwht_instrumented = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fwht_instrumented)

from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two

parallel_labels = "--parallel-labels" in sys.argv

n = 150
spring_constants = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
masses = [1.0 + 0.05 * i for i in range(n)]

sparse = build_hamiltonian(n, spring_constants, masses, sparse=True)
padded_sparse, n_qubits = pad_to_power_of_two(sparse, sparse=True)

t0 = time.perf_counter()
total_terms = 0
n_chunks = 0
for chunk_terms in fwht_instrumented.fwht_pauli_terms_iter(
    padded_sparse, chunk_size=256, parallel_labels=parallel_labels
):
    n_chunks += 1
    total_terms += len(chunk_terms)
total_time = time.perf_counter() - t0

st = fwht_instrumented.STAGE_TIMES
accounted = sum(st.values())
other = total_time - accounted

print(f"parallel_labels={parallel_labels}")
print(f"chunks={n_chunks} total_terms={total_terms:,}")
print(f"total_time={total_time:.2f}s")
for stage, t in st.items():
    print(f"  {stage:16s} {t:7.2f}s ({100*t/total_time:5.1f}%)")
print(f"  {'unaccounted':16s} {other:7.2f}s ({100*other/total_time:5.1f}%)")
print("SUCCESS")
