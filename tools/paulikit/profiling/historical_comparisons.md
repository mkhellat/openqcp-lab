# Historical comparison figures (moved out of README)

These tables were removed from `README.md` on 2026-09-09 and are kept
here as a research record. They are **not** current claims and must
not be cited as such.

Two reasons for the move:

1. **A benchmark table in a README is a claim about two moving
   targets.** It decays as this library changes, as the reference
   library changes, and as the machine it was taken on changes.
   Nothing keeps it honest between edits.
2. **These numbers do not meet this project's own measurement
   protocol.** Every cell below is a single run (n=1) with no warm-up,
   no interleaving, no repetition, no recorded temperature, and no
   committed raw data. `phase13/MEASUREMENT_METHODOLOGY.md` section 1
   documents four rounds of this project's own figures being retracted
   for exactly that reason, and
   `phase13/core_scaling_replicated_findings.md` records n=1 figures
   here being wrong "some by 40 percentage points."

This file lives under `profiling/`, which stays with the research
monorepo and is not part of the standalone package repository.

---

## PennyLane comparison (2026-08-26, n=1 per cell)

| N | qubits | Pauli terms | paulikit | PennyLane | ratio |
|---|---|---|---|---|---|
| 16 | 8 | 15,360 | 0.0124 s | 5.6914 s | ~459x |
| 30 | 9 | 112,384 | 0.0937 s | 45.9782 s | ~491x |
| 50 | 11 | 1,261,568 | 1.2371 s | 749.9998 s | >=606x |
| 100 | 13 | 20,299,776 | 24.6979 s | not attempted | — |

**The N=50 cell is almost certainly a timeout, not a completion.**
749.9998 s is four nines short of 750 s - the signature of a
wall-clock read against a 750-second deadline. `PLAN.md` section 3.4
records the same cell from an earlier run as `>590s (aborted)`. No log
or result artifact was ever committed for the run claimed to have
finished, so the completion cannot be substantiated from the
repository. Treat 606x as a lower bound at best.

**This table and `PLAN.md` section 3.4 disagree by ~4x on N=16** -
0.0124 s / ~459x here against 0.0586 s / 115x there. The paulikit side
is explained by Phase 6 (native label kernel plus sparse
`fwht_pauli_coefficients`). The PennyLane side is **not**: the same
library, on the same input, on the same machine, measured 18% faster
between the two runs, and no change on our side accounts for it. Both
readings are n=1, so the gap sits inside this machine's known
run-to-run variation - which is the argument for re-measuring rather
than reconciling.

**A further caveat on what the comparison means.** PennyLane's
`qml.pauli_decompose` uses a Walsh-Hadamard transform of its own (its
docstring says so) at O(n * 4^n). So this is not an algorithm-versus-
algorithm comparison; both sides use the same transform, and the gap
is implementation and memory strategy at equal algorithmic footing.
That is a defensible thing to measure, but it is a different claim
from the one a bare speedup ratio implies.

## Phase progression (2026-08-18 era, n=1 per cell)

| N | Phase 1 (baseline) | Phase 3b (sparse coefficients) | Phase 3c (+ native labels) | ratio |
|---|---|---|---|---|
| 50 | 6.2213 s | 5.4957 s | 2.1535 s | 2.9x |
| 100 | 126.3250 s | 107.2403 s | 43.5629 s | 2.9x |

Phase 3b made `fwht_pauli_coefficients` skip the O(dim^2) dense-array
construction for empty rows. Phase 3c wired in the native
`pauli_label` kernel. Term counts matched exactly across all versions
at every N - a correctness re-confirmation independent of the timing.

**The methodological problem here is worse than n=1.** These three
columns were measured on *different builds on different days*, and
`MEASUREMENT_METHODOLOGY.md` section 5.2's closing line prohibits
exactly that: efficiency and speedup figures must come from a single
interleaved session, because comparing a baseline taken in one session
against a treatment taken in another reproduces the precise failure
the protocol exists to prevent.

Treat this as an engineering log of the order in which optimisations
landed, not as a measured speedup.

---

## What replaced these in the README

The README now makes only claims the committed artifacts support:

- **Scale** - N=150 (15 qubits, 91,652,096 terms) completes under a
  2 GB cap.
- **Exactness** - every term verified individually against an
  independent projection oracle (`verification/FINDINGS.md`).
- **Generality** - non-Hermitian input supported and separately
  verified.

Timing claims live in `profiling/`, each beside the raw data and the
protocol under which it was taken. If a performance comparison is
wanted for publication, it must be produced under
`phase13/MEASUREMENT_METHODOLOGY.md` - replicated, interleaved,
warmed, thermally recorded, with a hypothesis test - and against a
current reference implementation, which at the time of writing means
`pauli_lcu` (Riverlane) as well as PennyLane.
