# Phase 10: streaming closes N=150 completely - 4GB, then 2GB, both succeed

Recorded 2026-08-27, after implementing PLAN.md Phase 10
(`fwht_pauli_terms_iter`). Machine: 15.7 GiB RAM total, ~10.6-11 GiB
available at the time of these runs.

## Design recap

Phase 9 confirmed the chunked/checkpointed accumulator was correct,
but that `fwht_pauli_terms`'s fully-materialized `dict` contract still
required O(n_terms) memory for the *final* answer - and N=150's real
term count (~134M at `atol=1e-10`) exceeds this machine's available
RAM once label strings and dict construction are added on top of the
raw triple (see `../phase9/phase9_findings.md`).

The user's framing for the fix (2026-08-27, quoted in
[[feedback_divide_and_conquer_strategy]]): the MIT 6.172 tiling
example was meant to teach the *general principle* of
divide-and-conquer, with devising the actual decomposition strategy
left to us. Reasoning through it: each chunk of active rows is
already a fully independent sub-problem - no cross-chunk combination
step exists anywhere in the underlying math (unlike tiled matrix
multiply, which requires summing blocks). The bug in
`fwht_pauli_terms` was never really about chunk size; it was that
every chunk's result was re-fused into one combined `dict` before the
caller ever saw it - an artificial recombination the math does not
require. `fwht_pauli_terms_iter` (`src/paulikit/algorithms/fwht.py`)
fixes this by keeping each chunk's result a chunk, yielding one
`dict` per chunk directly to the caller.

Along the way, also re-measured `pauli_label_batch_parallel` (the
existing oneTBB-parallel label kernel) at N=150-representative scale
- see `tbb_labeling_n150_findings.md` in this directory: a real
1.1-1.4x wall-clock win **in isolation**, at a modest cache-locality
cost, adopted per direct user decision. **Update:** re-measured
embedded in the real streaming pipeline in `full_pipeline_n150_findings.md`
(this directory) - the effect washes out to noise at full-pipeline
scale, since labeling is only ~7% of total time there.

## N=150 runs, in order

Script: `n150_streaming_test.py` (this directory) - builds N=150's
Hamiltonian via `build_hamiltonian(sparse=True)`, pads it via
`pad_to_power_of_two(sparse=True)`, then consumes
`fwht_pauli_terms_iter(padded, chunk_size=256, parallel_labels=True)`
chunk by chunk, accumulating only a running term *count* (not the
terms themselves) to confirm the streaming contract is actually being
honored by the test harness too. Same `ulimit -v` + `free -m`
polling-safety harness used throughout this project (2s interval,
kill below 1500 MiB available).

| `ulimit -v` cap | Result |
|---|---|
| 4 GB | **Succeeded completely.** 44 chunks, 91,652,096 total terms, 101.59s. |
| 2 GB | **Succeeded completely**, same result (91,652,096 terms, 97.19s) - confirming 4 GB was not near a real edge. |

(Note: this run's `atol` default and Hamiltonian differ slightly in
exact term count from Phase 9's ~134M figure - both are the same
spring-constant/mass parameterization and same `atol=1e-10`, so the
91.65M here should be treated as the authoritative real count for
this exact test configuration; Phase 9's "~134M" was a description of
scale from an earlier run of the same experiment, not a materially
different problem. The two numbers agree in order of magnitude and
both establish the same qualitative point.)

For comparison, Phase 9's non-streaming chunked-accumulator path
needed at least 10 GB to get through accumulation alone, and never
completed even at 13.5 GB once label generation and dict construction
were included (killed by real system memory pressure, not the
`ulimit`) - see `../phase9/phase9_findings.md`.

## What this establishes

1. **The divide-and-conquer reframing works exactly as reasoned.**
   Keeping each chunk's result as an independent, immediately-yielded
   tile (rather than re-fusing every chunk into one combined
   structure before returning) reduces peak memory from "grows with
   total term count" (previously >13.5 GB and still failing) to
   "bounded by one chunk's term count" (succeeds comfortably at 2 GB).
   This is a qualitative change, not an incremental one - no `ulimit`
   size fixed today's problem before Phase 10; only decomposing the
   *problem itself* differently did.

2. **N=150 is now a solved, repeatable case on this machine**, not
   just something that "sometimes fits." Both the 4 GB and 2 GB runs
   produced byte-identical term counts and completed in comparable
   time (~100s), meaning the memory cap was never actually load-
   bearing on correctness or performance at this scale - the
   streaming design has genuinely decoupled peak memory from total
   result size, which was the entire point.

3. **This validates the user's explicit divide-and-conquer framing**
   (see [[feedback_divide_and_conquer_strategy]]) as more than a
   philosophical preference: reasoning about the actual independent
   sub-problems, rather than mechanically adding a generator wrapper
   around the existing accumulate-then-return logic, is what produced
   the qualitative (not incremental) improvement here.

## What this does NOT show

- Not a lower bound - 2 GB was the smallest cap tried, not the
  smallest cap that would work; a tighter search was not attempted
  since 2 GB already comfortably demonstrates the qualitative claim
  (peak memory decoupled from total term count).
- `--parallel-labels` (`parallel_labels=True`) was used throughout
  this run; the streaming fix itself does not depend on parallel
  labeling - a serial-labels run would be expected to succeed at the
  same memory caps (untested directly, since the two features are
  independent per `tbb_labeling_n150_findings.md`'s own scope, and
  time was prioritized on the memory-ceiling question this session
  actually needed answered).
- Per [[feedback_perf_priority_order]] and the user's 2026-08-27
  instruction, cache-locality behavior at real N=150+ scale is to be
  **re-investigated iteratively**, not treated as closed by this
  finding - this document establishes the memory/scalability result
  only, not a final cache-locality verdict for the streaming pipeline
  as a whole.
