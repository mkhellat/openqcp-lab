# Phase 9: chunked accumulator fix confirmed correct; real N=150 result size is the remaining ceiling

Recorded 2026-08-27, after implementing PLAN.md Phase 9 (the
space-complexity fix for `fwht_pauli_coefficients`'s chunked path -
see that phase's write-up for the design). Machine: 15.7 GiB RAM
total, ~11 GiB available at the time of these runs (`free -m`).

## What was fixed (recap)

Phase 8 cleared N=150's original ~4 GiB dense-Hamiltonian-densification
ceiling. That surfaced a *second*, distinct ceiling: even with
`chunk_size` set, `fwht_pauli_coefficients`'s old chunked path still
wrote every chunk's result into one `active_coefficients =
np.empty((n_active, dim), dtype=complex)` accumulator allocated up
front - at N=150, `45000 * 16384 * 16 bytes ≈ 11.8 GiB`, independent
of `chunk_size`'s value.

Phase 9 replaced that dense accumulator with:
- per-chunk thresholding (`atol`) *before* accumulation, so only
  surviving `(x, z, coefficient)` triples are ever kept past a
  chunk's own lifetime;
- an amortized-doubling growable array (`_GrowableArray`) for the
  three output arrays (`x_out`, `z_out`, `coeff_out`), so appending
  costs O(total survivors) amortized rather than O(chunks) full
  reallocations;
- an opt-in checkpoint/resume mechanism (`checkpoint_path`): each
  chunk's surviving triples are appended to a newline-delimited JSON
  file plus a small progress marker, so a crashed/interrupted run can
  resume from the next unfinished chunk instead of restarting from
  chunk 0.

All of the above is covered by `tests/test_chunked_accumulator.py`
(15 tests, including a checkpoint-interrupt-and-resume round trip and
a hand-truncated-progress-file crash simulation) - all passing before
these N=150 runs were attempted.

## N=150 runs, in order

Script: `n150_chunked_accumulator_test.py` (this directory). Harness:
a `bash -c "ulimit -v <cap>; python ..."` subshell with a `free -m`
polling loop (2s interval) that force-kills the process if real
available memory drops below a safety floor - the same discipline
used throughout this project's N=150 investigation (see
`../cache_locality/n150_oom_finding.md`).

| `ulimit -v` cap | Safety-floor | Result |
|---|---|---|
| 6 GB | 1500 MiB | `MemoryError` at `coeff_out.extend(...)` inside the accumulator's doubling step - needed 2.00 GiB for a `(133955584,)` complex128 array, i.e. the accumulator was already holding ~134M survivor terms and needed to grow past the cap. |
| 10 GB | 1000 MiB | Accumulator step **completed** - all three growable arrays (`x_out`, `z_out`, `coeff_out`) finished. Crashed one step later, inside `_pauli_label_batch`'s native call (`MemoryError` converting the 134M `(x, z)` pairs to label strings) - a **third**, still-separate O(n_terms) allocation on top of the triple. |
| 13.5 GB | 800 MiB | Killed by the safety loop itself (available system memory dropped to 630 MiB) - not the `ulimit`. This is decisive: the process was genuinely about to exhaust **real** machine memory, not an artificial test ceiling. |

## What this establishes

1. **The Phase 9 fix works exactly as designed.** The artificial
   `n_active * dim` (11.8 GiB) ceiling from the old chunked path is
   gone - confirmed by the 10 GB run completing the entire chunked
   accumulation step (all 134M survivor triples), something the
   pre-Phase-9 code could never have done at any memory size, since it
   allocated the full dense block regardless of `chunk_size`.

2. **The real, non-artificial number at N=150 (`atol=1e-10`) is ~134M
   terms.** That is genuine problem data, not a design defect - see
   PLAN.md Phase 9's original design-question framing (worked through
   with the user 2026-08-27): the question of whether 134M terms is
   itself a *useful* result was explicitly set aside in favor of
   treating this purely as a space-complexity/scalability engineering
   problem, on the basis that the block-rewrite technique is reusable
   for other large computational problems regardless of whether this
   particular result set is the "right" size physically.

3. **`fwht_pauli_terms`'s dict-return contract has more O(n_terms)
   copies than just the coefficient accumulator.** The 10 GB run
   surfaced a *third* full-size allocation
   (`_pauli_label_batch`) that Phase 9's chunked-accumulator fix does
   not address, since it happens after `fwht_pauli_coefficients`
   returns. A dict of 134M `label -> coefficient` entries would need
   to hold: the COO triple (~4.3 GiB), the label list (134M Python
   strings, each with real per-object overhead beyond the raw byte
   count), and the final dict (hash table overhead on top of that) -
   plausibly all coexisting during dict construction. This machine's
   ~11 GiB available does not have room for that peak.

4. **No memory-size fix survives arbitrary N.** Term count grows with
   N; a fixed RAM ceiling (whether 6 GB, 10 GB, or however much a
   given machine has) will always eventually be exceeded by a large
   enough N under the current fully-materialized-dict contract. This
   is the basis for the user's explicit "we MUST do the streaming"
   decision (2026-08-27) - not treated as optional/future work.

## Next: streaming output (mandatory follow-up, not yet implemented)

`fwht_pauli_terms` (or a new sibling function/mode) needs a
streaming/generator form that yields `(label, coefficient)` pairs (or
writes them incrementally) without ever materializing the full label
list or the full dict at once - removing the ceiling entirely rather
than raising it. The chunked/checkpointed COO accumulator from Phase 9
is the natural foundation: each chunk's surviving triples already
exist independently and briefly, before being folded into the
growable arrays; a streaming mode would instead convert each chunk's
triples to labels and yield them immediately, never holding more than
one chunk's worth of labels at a time. Design not yet started.
