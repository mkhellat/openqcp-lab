# Phase 3b — Sparsity-aware `fwht_pauli_coefficients`

Scoped in `PLAN.md` Section 5 (Phase 3b), started 2026-08-18. This
directory holds the working notes and exploratory scripts behind the
final implementation that lands in
`src/paulikit/algorithms/fwht.py`, so the design reasoning (including
approaches that were tried and rejected) stays reproducible and
inspectable rather than living only in commit messages.

## 1. Confirming the actual cost model (task #30)

`explore/01_sparsity_probe.py` measures, for the real
`build_hamiltonian()` output at matched N (16/30/50), what fraction of
the FWHT's `dim` "rows" (fixed `x` in the gather `gathered[x, q] =
operator[q ^ x, q]`) are entirely zero, and how many nonzero entries
an *active* row has on average:

| N (oscillators) | dim  | operator nnz | active rows / dim | avg nnz per active row |
|------------------|------|---------------|--------------------|--------------------------|
| 16               | 256  | 512           | 121/256 (47.3%)    | 4.23                    |
| 30               | 512  | 1800          | 440/512 (85.9%)    | 4.09                    |
| 50               | 2048 | 5000          | 1233/2048 (60.2%)  | 4.06                    |

This is the key number that shaped the whole design: the operator is
sparse (O(N) nonzeros), but the **rows are not** — a large and
N-dependent fraction (47-86% in this sample) of the `dim` WHT rows
have at least one nonzero entry. A "skip empty rows" optimization is
real but bounded by that fraction, not by the O(N)-vs-O(4^n) headline
sparsity ratio.

`explore/02_profile_dense_n50.py` (cProfile) confirms where the
*existing* dense `fwht_pauli_coefficients` actually spends time at
N=50 (2.26s total): the WHT butterfly (1.24s cumulative),
`_popcount_array`'s bit-serial Python loop (0.56s), and the O(dim²)
gather/glue in `fwht_pauli_coefficients` itself (0.44s tottime) — all
three are dense `dim x dim`-scale operations that don't use the
operator's sparsity at all today.

## 2. Design exploration (task #31)

`explore/03_sparse_wht_identity_check.py` first verifies, to machine
epsilon, the identity the "sparse-impulse WHT" approach depends on:
for a row with nonzero entries `{(q_i, v_i)}`, its Walsh-Hadamard
transform at every output index `z` is exactly
`sum_i v_i * (-1)**popcount(q_i & z)` — i.e. computable in `O(k * dim)`
for `k` nonzeros, instead of the standard `O(dim log dim)` butterfly.
With `k` averaging ~4 and `log2(dim)` reaching 13 at N=100, this
looked like the headline win going in.

It wasn't, once actually measured end-to-end. `explore/04` through
`explore/11` are the successive attempts, kept as-is (not cleaned up
into only the winner) because the negative results are as informative
as the positive one:

| script | approach | N=100 time | verdict |
|---|---|---|---|
| (baseline) | existing dense `fwht_pauli_coefficients` | 35.5s | current shipped behavior |
| `04_v1_per_row_loop.py` | sparse-impulse identity, Python loop over active rows | not run at N=100 (Python-loop overhead ~1200 iterations) | promising at N≤50 (4.6x), didn't scale further |
| `05_v2_vectorized_add_at.py` | fully vectorized via `np.add.at` scatter | 57.1s | **worse than dense** — `np.add.at` is an unbuffered scalar fallback, and the `(nnz, dim)` sign matrix is itself large |
| `06_v3_reduceat.py` | same but grouped via sorted `np.add.reduceat` instead of `add.at` | 57.1s (N=100 identical bottleneck) | fixed the scatter cost, but the `(nnz, dim)` sign-matrix build still dominates |
| `07_v4_fast_popcount.py` | v3 + 8-bit LUT popcount (replacing `_popcount_array`'s bit-serial loop) | 29.7s | real win from the LUT alone, still not competitive |
| `08_v5_active_rows_only.py` | v4 + only build the phase array for active rows, not all `dim` | 24.0s | further real win, still not the target |
| `09_v6_skip_empty_plus_cache.py` | **different strategy**: keep the dense butterfly, but skip all-zero rows entirely, use the LUT popcount, and cache the phase factor (it depends only on `n_qubits`, not the operator) across repeated calls | 22.4s (cold) / 17.6s (warm) | first result clearly and consistently beating dense |
| `10_v7_scatter_gather_winner.py` | v6, but also avoid the O(dim²) *gather* step — scatter nonzero entries directly into an `(n_active, dim)` array instead of fancy-indexing a full `(dim, dim)` gather first | **17.6s (no caching needed — same speed cold)** | **winner** — 35.5s → 17.6s, ~2.0x, exact to machine epsilon |
| `11_v8_sparse_impulse_combo.py` | combine v7's scatter-gather with the v1 sparse-impulse identity (skip the butterfly too) | 26.3s | the `(nnz, dim)` sign-matrix construction cost outweighs the butterfly savings at this nnz/dim ratio — reverts the v7 win |

**Root cause of why the "headline" sparse-impulse idea (v1/v8)
underperforms:** it trades an `O(dim log dim)` butterfly per active
row for an `O(k * dim)` explicit sign-matrix computation across *all*
nonzero entries. Because `active_rows / dim` stays as high as 47-86%
in this problem (not the ≪1% that would be needed for the impulse
trick to win), `n_active * dim` is already close to `dim²`, and the
per-entry sign matrix (`nnz * dim`, since `nnz ≈ 4 * n_active`) ends
up doing comparable or more total elementwise work than the highly
BLAS/vectorization-friendly butterfly, without the butterfly's
cache-friendly access pattern. The honest conclusion: **for this
specific Hamiltonian's sparsity profile, the win is in avoiding
wasted dense-array construction (gather, popcount, phase) around a
kept butterfly — not in replacing the butterfly's algorithm itself.**

## 3. Chosen design (implemented in `fwht.py`)

`fwht_pauli_coefficients` gets an internal fast path, functionally
identical to `explore/10_v7_scatter_gather_winner.py`:

1. Find the operator's nonzero entries directly (`np.nonzero`), not a
   dense `(dim, dim)` gather.
2. Compute `x = p ^ q` per nonzero entry and group by the distinct
   active `x` values (`np.unique`).
3. Scatter nonzero values directly into an `(n_active, dim)` array
   (not `(dim, dim)`) — this is the O(dim²) gather cost eliminated.
4. Run the existing `_walsh_hadamard_transform_rows` butterfly only on
   the active rows.
5. Compute the phase factor only for the active `x` values, using a
   fast 8-bit LUT popcount instead of the previous bit-serial Python
   loop (a general win, independent of sparsity).
6. Scatter the `(n_active, dim)` result back into the full
   `(dim, dim)` output array, preserving the existing function's
   public contract (dense array return) so `fwht_pauli_terms` and all
   existing tests/callers are unaffected.

This is an exact algorithm, not an approximation — output matches the
original dense implementation to machine epsilon (verified in
`explore/10` at N=16/30, and via the full correctness-gate re-run
against fixtures/PennyLane at N=2/4/16/30 — see task #33).

## 4. What this does *not* fix

`_popcount_array`'s bit-serial-loop replacement (the LUT) and the
phase-caching idea both help unconditionally, including for dense
non-sparse inputs — worth keeping in mind if `paulikit` is ever used
on a genuinely dense operator, since the row-skip optimization
degrades gracefully to zero benefit there (no active rows are ever
skipped) while the LUT/gather-avoidance changes still apply.

The magnitude of the win (~2x at N=100) is real but modest compared
to Phase 3a's C-porting results — worth stating plainly rather than
oversold, consistent with this project's "verify, don't guess"
discipline (see also the Phase 3a oneTBB honest-negative-finding
precedent in `bindings/README.md`).

## 5. Final benchmark (implemented version, matched N)

`fwht_pauli_coefficients` now in `src/paulikit/algorithms/fwht.py`,
measured directly (not the exploratory script) at the same matched N
as Phase 1/3a, using the same synthetic Hamiltonian generator as
`tests/test_benchmark_reference.py`:

| N (oscillators) | qubits | dim  | `fwht_pauli_coefficients` | `fwht_pauli_terms` (end-to-end) | Pauli terms |
|------------------|--------|------|-----------------------------|-----------------------------------|--------------|
| 16               | 8      | 256  | 0.0065s                     | 0.0544s                           | 15360        |
| 30               | 9      | 512  | 0.0435s                     | 0.3623s                           | 112384       |
| 50               | 11     | 2048 | 0.6361s                     | 5.4957s                           | 1261568      |
| 100              | 13     | 8192 | 17.5646s                    | 107.2403s                         | 20299776     |

Term counts match Section 3.4's PennyLane-cross-checked figures
exactly at every N, which is itself a correctness confirmation, not
just a performance measurement.

`fwht_pauli_coefficients` at N=100: **35.5s (old dense) -> 17.56s
(new), ~2.0x**, matching the exploration numbers above. End-to-end
`fwht_pauli_terms` at N=100: **126.3s (Phase 1 baseline) -> 107.2s,
~1.18x** — smaller than the coefficients-only speedup because
`fwht_pauli_terms` is still using the pure-Python `pauli_label` loop,
not yet wired to Phase 3a's Cython kernel (that integration - actually
using the C-ported `pauli_label` inside `fwht_pauli_terms` - is a
distinct, not-yet-done step; Phase 3a only built and benchmarked the
kernel/bindings standalone).
