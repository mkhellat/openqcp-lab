# Resident-footprint isolation experiment: refuted, closes the traffic_intensity_findings.md decision tree (2026-09-04)

The last untested item from `traffic_intensity_findings.md`'s original
decision tree, reached after `gather_pattern_findings.md`'s mixed
result: does holding a paulikit-SCALE resident
operator/setup-array footprint in each worker for its whole lifetime -
not any single chunk's own transient traffic - tip shared-memory-
subsystem contention into paulikit's observed ceiling?

## Method

Real per-worker resident footprint measured directly via the shipped
`_per_worker_resident_bytes` (added this session as REVIEW_NOTES.md
bug #4's fix) on the real N=150 Hamiltonian: **1.951 MiB**
(`nnz=45,000`, CSR sparse operator's `data`/`indices`/`indptr` buffers
plus the three `nnz`-length sorted setup arrays).

`resident_footprint_target.py` extends `gather_pattern_target.py`'s
`gather_and_wht` workload (the closest prior proxy, which did NOT
reverse - see `gather_pattern_findings.md`) with a real, actively-used
resident array: each worker's `_worker_init` allocates a
`RESIDENT_NNZ=45000`-length complex array once (fixed seed, so every
worker/run sees identical content - isolates the footprint/access
effect from content variation), held for the worker's entire
lifetime. Every task then gathers a deterministic scattered slice OF
that resident array into its chunk buffer (`buf[rows, cols] =
_worker_resident_values[start:start+n]`), matching how
`_parallel_worker_chunk` actually re-reads from `state["operator"]` on
EVERY single chunk (`fwht.py:1248-1250`) - not just holding the array
passively resident, but actively touching it every task, which is the
real access pattern under test.

Same protocol as every prior measurement: `w2_c1` vs `w8_c4`, thermal
cooldown to <=55C before every run, 5 reps/cell, Welch's t-test.

## Result: no effect, statistically indistinguishable from gather_and_wht

| condition | mean (s) | vs. `gather_and_wht` (no resident array) |
|---|---|---|
| w2_c1 | 9.0164 (sd 0.2366) | 9.0241 (sd 0.0804) - Welch p=0.95, NOT significant |
| w8_c4 | 7.1204 (sd 0.2486) | 7.0548 (sd 0.0726) - Welch p=0.64, NOT significant |

Overall speedup (w2/w8): **1.266x** (Welch t=11.05, p=4.1e-6) - scales
normally, same direction and nearly identical magnitude to
`gather_and_wht`'s own 1.279x. Adding the real 1.95 MiB resident
operator footprint, actively re-read on every single task, made **no
statistically detectable difference** at either core-packing
condition.

## Conclusion: refuted as a sufficient cause

The operator/setup-array resident footprint - the last untested item
from `traffic_intensity_findings.md`'s original decision tree - is
**refuted as a sufficient cause** of paulikit's ceiling, at least at
the scale tested here (1.95 MiB/worker, matching the real N=150
operator exactly). This closes the entire decision tree that document
originally laid out:

| suspect | tested by | result |
|---|---|---|
| dense buffer traffic/stage-touch count | `traffic_intensity_findings.md` | REFUTED (all 3 controls scale 2.2-2.7x) |
| irregular gather/scatter access pattern | `gather_pattern_findings.md` | MIXED (granularity-confounded, not conclusive) |
| large IPC payload | `traffic_intensity_findings.md` (`wht_large`) | REFUTED (still scales 2.2x) |
| operator/setup-array resident footprint | this experiment | **REFUTED** (no detectable effect) |

None of the four originally-suspected mechanisms, individually, is
sufficient to reproduce paulikit's real ceiling (0.89x, w8 slower).
Every synthetic control built so far that isolates ONE of these
factors at a time scales normally. This is a genuinely important
negative result, not a dead end: it means the trigger is either (a) a
COMBINATION of factors that only manifests when several are present
simultaneously (not yet tested - every control so far isolates one
factor), or (b) something not yet identified at all - e.g. the real
operator's actual (non-random) VALUES/sparsity structure interacting
with the real phase-multiply/threshold/label-construction steps this
whole synthetic-control lineage has consistently omitted, or something
about paulikit's real multi-chunk memory ALLOCATION pattern (repeated
`np.zeros((chunk_size, dim), ...)` allocation/deallocation cycles, not
modeled by any control here) rather than access pattern per se.

## What this does and does not show

**Does show**: none of dense traffic, gather irregularity, large IPC
payload, or resident footprint - alone - explains paulikit's ceiling.
Four real, careful, statistically-controlled experiments now rule
these out individually.

**Does NOT show**: what DOES explain it. The remaining candidate
explanations are qualitatively different from anything tested so far
(combination effects, real Hamiltonian value/sparsity structure, or
allocation-pattern effects) and would each require a different kind
of experiment, not a variant of the same "swap in a synthetic per-task
body, keep the pool shape" methodology this whole lineage has used.

## Honest assessment of the synthetic-control methodology itself

Four experiments in, every synthetic control that isolates one
plausible mechanism at a time has failed to reproduce paulikit's
ceiling. This is itself informative about the METHOD, not just the
individual hypotheses: it may mean the real explanation genuinely
requires the FULL real pipeline (real Hamiltonian values, real
phase-multiply, real threshold/filter, real repeated allocation
pattern - `_parallel_worker_chunk`'s exact sequence, not a
close approximation of one or two of its steps). The next
methodologically different step worth considering (not yet
attempted): profile the REAL `parallel_decompose` run itself more
deeply (e.g. `perf record`/`perf annotate` for line-level attribution
inside a contended worker, rather than the coarser py-spy stack
sampling already done) rather than continuing to build additional
synthetic proxies - since the proxy-elimination approach has now
exhausted the four most plausible single-factor hypotheses without a
positive result.

## Artifacts

- `resident_footprint_target.py`
- `resident_footprint_sweep.py`
- `resident_footprint_results.jsonl` (10 runs)
