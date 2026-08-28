# Correctness verification: findings

Date: 2026-08-28. Machine: Intel i7-8550U (same as all other
profiling in this project). See `README.md` for how to reproduce any
number below — every one traces to a JSON file under `results/`.

## 1. Why PennyLane can't verify paulikit at target scale

`qml.pauli_decompose`'s own documented theory section (source,
PennyLane 0.45.1): "This method internally uses a generalized
decomposition routine to convert the matrix to a weighted sum of
Pauli words ... in time O(n 4^n)." Its sparse-input support
("processed natively without converting to dense format") only
avoids materializing the *input* as dense — it does not reduce the
*output* cost, which is driven by the number of possible Pauli labels
(up to 4^n), independent of `nnz(H)`.

Confirmed empirically, not just from the docstring: a genuinely
sparse Hamiltonian (nnz=1250, 0.5% density) took PennyLane 34s at
N=25 (78,336 terms) and 51s at N=30 (112,384 terms). paulikit itself
computes 91.65 million terms at N=150 in ~100s. PennyLane cannot
reach N=150 in any practical time — a hard mathematical wall (the
*output* is exponentially large), not an implementation inefficiency.

Separately confirmed: `qml.pauli_decompose(H, check_hermitian=False)`
correctly decomposes non-Hermitian operators too (reconstructs to
1.57e-16, matching Hermitian-case precision) — the Hermitian check is
an opt-out safety gate, not a real capability limitation. Label
conventions match paulikit's exactly (leftmost char = qubit 0, same
as `paulikit.pauli_utils.pauli_string_to_matrix` and
`paulikit.algorithms.fwht.pauli_label`'s own docstrings).

## 2. The independent method: exhaustive per-term projection

For a Pauli label with symplectic bitmasks `(x_mask, z_mask)`
(convention: leftmost char = qubit 0, bit position `n-1-j` for qubit
`j` — matching `pauli_label`'s own docstring exactly, cross-checked
directly against the source, not just inferred from agreement):

```
c = Tr(H @ P_label^dagger) / dim
```

computed without ever materializing the full `(dim, dim)` matrix
`P_label`, using: `H[r, c]` contributes only where `r ^ c == x_mask`
(the X-part flips exactly the `x_mask` bits going from row to
column), with value `sign * phase` where
`sign = (-1)^popcount(r & z_mask)` and
`phase = (1j)^popcount(x_mask & z_mask)`. This is a direct
definitional projection, mathematically independent of paulikit's own
FWHT-based algorithm — not a re-derivation of the same shortcut.

## 3. Iteration history (kept for the record — the failures are the lesson)

| Version | Approach | Measured cost | Verdict |
|---|---|---|---|
| v1 | `sp.kron`-rebuild `P_label` per label | 2.75 ms/term | far too slow (70+ hr at N=150) |
| v2 | Per-label Python loop over H's nonzeros directly | 1.05 ms/term | still too slow |
| v3 | Python-dict bucket H's nonzeros by `r^c`, Python loop per label over its bucket | 14.26 μs/term at N=50 (looked great — 18s total) | **did not scale**: timed out (>100s) at N=80 once `nnz` grew 2.56x — the per-bucket Python loop still scales with bucket size, not O(1) |
| v4 | Fully vectorized NumPy: sort H's nonzeros by `r^c`, sort labels by `x_mask`, `np.searchsorted` to slice each label-group's matching range, grouped broadcast — **no Python-level loop over labels or nonzeros** | scales correctly, see results below | **final method**, implemented in `exhaustive_projection.py` |

User explicitly required real vectorization over accepting v3 with a
bigger timeout budget ("Vectorize properly with NumPy before locking
the design").

One real bug was caught and fixed during the extraction of v4 from
scratch-script to committed module: `np.unique(..., return_index=True,
return_inverse=True)` returns `(unique, index, inverse)` in that
fixed order regardless of kwarg order — an initial transcription swapped
`index`/`inverse` on unpacking, which silently produced wrong
coefficients (`max_abs_error` of 0.016 instead of ~1e-18) while still
running without error. Caught by rerunning the N=50 case through the
real module and comparing against the already-recorded scratch
number, per this project's standing "verify before trusting"
discipline — the mismatch was the tell.

## 4. Measured results

All values below are from `results/*.json` (see file names for exact
provenance). All errors are floating-point noise floor (~1e-17 to
1e-20), not approximation — this is an exact independent verification,
not a heuristic.

| N (oscillators) | qubits | dim | H.nnz | # terms | method | wall time | μs/term | max abs error | passed |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 8 | 256 | 800 | 24,448 | projection + PennyLane (dual oracle) | 0.05s / 8.8s | 2.0 | 1.39e-17 | yes, both |
| 20 (non-Hermitian) | 8 | 256 | 800 | 49,024 | projection + PennyLane (dual oracle) | 0.09s / 15.9s | 1.8 | 1.39e-17 | yes, both |
| 50 | 11 | 2048 | 5,000 | 1,261,568 | projection only | 5.5s | 4.3 | 3.47e-18 | yes |
| 80 | 12 | 4096 | 12,800 | 6,473,728 | projection only | 30.3s | 4.7 | 1.04e-17 | yes |
| 100 | 13 | 8192 | 20,000 | 20,299,776 | projection only | 124.8s | 6.2 | 2.60e-18 | yes |
| 150 | 14 | 16384 | 45,000 | 91,652,096 | projection only (streaming) | 369.7s | 4.0 | 8.67e-19 | yes |

N=150 required the streaming path (`--streaming --chunk-size 256`) —
see §6 below for why the non-streaming path cannot succeed here at
any memory size.

## 6. N=150's real memory ceiling (found the hard way, kept for the record)

Getting a real N=150 number took 4 attempts and 3 real OOMs before
finding the actual root cause - worth recording since it is a genuine
lesson about this project's own established API contract, not a bug
in the verification method:

1. **Attempt 1** (`fwht_pauli_terms(padded)` on a *dense* Hamiltonian,
   no `chunk_size`): OOM-killed. Root cause not yet understood at this
   point - blamed the projection method's memory use.
2. **Attempt 2**: added a memory-bounded broadcast cap inside
   `project_labels` (real, worthwhile hardening, kept) and retried the
   same dense/unchunked call. Still OOM-killed - the cap fixed a
   different, smaller problem than the actual one.
3. **Diagnosis**: instrumented stage-by-stage under `ulimit -v
   12000000` (a hard virtual-memory cap, so failures raise
   `MemoryError` cleanly instead of thrashing the whole machine's
   swap). Found the *actual* failure point: `fwht_pauli_terms`'s own
   internal `_pauli_label_batch` call, building the label-string dict.
   Cross-checked against this project's own prior profiling
   (`profiling/phase9/phase9_findings.md`,
   `profiling/phase10/phase10_streaming_findings.md`): **this is a
   known, already-documented ceiling** - the dict-returning
   `fwht_pauli_terms` API cannot complete at N=150 *regardless of
   available RAM* (previously measured failing even at 13.5 GiB),
   because it re-fuses every chunk's terms into one ~91.65M-entry dict
   before returning. Phase 10 had already built and validated the fix
   for exactly this: `fwht_pauli_terms_iter`, a generator yielding one
   chunk's terms at a time, never holding the combined result.
4. **Attempt 3, fixed**: rewrote the large-N path to consume
   `fwht_pauli_terms_iter` and verify each chunk immediately via the
   new `verify_terms_streaming` (accumulates only running summary
   statistics - `n_terms`, `max_abs_error`, `worst_label` - never the
   term dict itself). Also switched Hamiltonian construction to
   `build_hamiltonian(..., sparse=True)` +
   `pad_to_power_of_two(..., sparse=True)` end-to-end, avoiding ever
   materializing paulikit's own documented ~4 GiB dense-matrix cost at
   this N. Run under `ulimit -v 12000000` for safety. **Succeeded**:
   369.7s, 91,652,096 terms, max_abs_error=8.67e-19.

Lesson: `fwht_pauli_terms` (no streaming) is fine through at least
N=100 (20.3M terms, confirmed working); N=150 requires
`fwht_pauli_terms_iter` unconditionally, independent of how much RAM
is available - not a "nice to have for lower memory" option at this
scale, per the project's own Phase 10 findings.

## 5. Process notes

- Sampling was explicitly ruled out as a design option: every term
  paulikit outputs is individually checked, not a sample, at every N
  tested.
- A background research subagent investigating PennyLane's scaling
  behavior ran 40 minutes without a conclusive report; stopped, and
  the investigation was completed directly in a few minutes instead
  (read `qml.pauli_decompose`'s source, ran 2-3 targeted timing
  tests). See project memory `feedback_bound_investigation_agents`
  for the general lesson.
- The non-Hermitian dual-oracle check (§4, N=20 row) uses a
  deterministic antisymmetric-imaginary perturbation on the real
  Hamiltonian's own sparsity pattern (`run_verification.py`'s
  `make_non_hermitian`, seeded, default seed 42) — not an arbitrary
  random matrix — so it stays a physically meaningful test case (e.g.
  representative of a non-Hermitian effective Hamiltonian) rather than
  noise.
