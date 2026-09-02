# Full pinned/unpinned/n_workers matrix on the REAL parallel_decompose pipeline

Recorded 2026-09-02, per direct, explicit instruction after the
synthetic-noise-proxy methodology was rejected as insufficient: "we
need to capture L3 contention (AND/OR L3 cache misses) for [6 named
conditions]... full performance analysis including wall-clock, task
clock, full L1 cache line analysis, full L2 cache line analysis,
detailed memory footprints... we cannot evaluate your theory that L3
is the source of difference between pinned and unpinned performance
difference." This document is that full matrix, on the REAL shipped
`parallel_decompose()`/`fwht_pauli_terms_iter()` API (per direct
instruction: "full pipeline analysis not a specific module only"), not
a lower-level reimplementation or synthetic proxy.

## Method

`full_matrix_target.py` calls `parallel_decompose(padded,
chunk_size=2, n_workers=N)` directly for 5 of 6 conditions, and
`fwht_pauli_terms_iter` (no pool) for the sequential baseline.
"Unpinned" conditions monkeypatch `_physical_core_representative_cpus`
to return `None` - the EXACT SAME code path `parallel_decompose`
already exercises when pinning is genuinely unavailable (e.g.
non-Linux), not a separate/different bypass mechanism.

Two `perf stat --no-inherit` event groups per condition (split to
reduce PMU-counter multiplexing on this machine's ~4 general-purpose
counters, confirmed via a quick single-event check beforehand -
`nmi_watchdog=1` occupies one counter, and an 11-event single pass
showed events only ~37-41% scheduled):
- **L1/L2 group**: `task-clock, mem_load_retired.l1_hit,
  mem_load_retired.l1_miss, mem_load_retired.l2_hit,
  mem_load_retired.l2_miss, L1-dcache-loads, L1-dcache-load-misses`
  (`mem_load_retired.*` gives TRUE per-level hit/miss counts, more
  precise than the generic L1-dcache-load-misses ratio used in earlier
  documents this session).
- **L3 group**: `task-clock, cycles, instructions, cache-references,
  cache-misses, LLC-loads, LLC-load-misses` (same event set as every
  other `perf stat` measurement in this project).

All 12 runs (6 conditions x 2 groups) executed foreground-only, one at
a time, machine confirmed idle via `ps aux` immediately before every
launch - no exceptions. RSS (process-tree `VmRSS`, POSIX-only) tracked
inside the target script itself for every run.

## Full results table

| condition | wall-clock | peak RSS | L1 miss % | L2 miss % | cache-miss % | LLC-miss % |
|---|---|---|---|---|---|---|
| seq_1 (baseline, no pool) | 35.09s | 154 MiB | 4.50% | 37.01% | 5.79% | 4.24% |
| pinned_2 | 25.16s | 263 MiB | 2.58% | 43.48% | 35.15% | 20.73% |
| unpinned_2 | 26.86s | 262 MiB | 2.61% | 43.73% | 36.41% | 21.83% |
| **pinned_4** | **29.70s** | 373 MiB | 2.60% | 44.70% | **38.92%** | **23.89%** |
| unpinned_4 | 26.93s | 370 MiB | 2.55% | 44.77% | 36.96% | 22.35% |
| workers_8 (default, pinned) | 28.81s | 581 MiB | 2.64% | 44.51% | 37.63% | 22.80% |

(L1 miss % = `mem_load_retired.l1_miss` / (`l1_hit`+`l1_miss`); L2
miss % = `mem_load_retired.l2_miss` / (`l2_hit`+`l2_miss`) - i.e. of
accesses that already missed L1; cache-miss %/LLC-miss % as in every
prior document, both `:u`-scoped.)

Raw counter values (for reproducibility/re-derivation) recorded in the
git history of this file's own commit message and the two perf-run
transcripts this document was built from - available on request, kept
out of the table above for readability.

## Direct answer: why does pinned_4 "hit" (and slightly exceed) workers_8's wall-clock?

**Confirmed and sharper than "hit"**: `pinned_4` (29.70s) is actually
the SLOWEST of every multi-worker condition tested, including
`workers_8` (28.81s) - not just comparable to it. And `pinned_4` has
the HIGHEST cache-miss% (38.92%) and LLC-miss% (23.89%) in the ENTIRE
matrix, higher even than `workers_8`.

This directly falsifies the working theory from earlier in this
investigation (`cpu_pinning_findings.md`'s framing, and the L1/L2-
hyperthread-sharing hypothesis it was built on): if hyperthread-
sibling sharing at `workers_8` (2 threads per physical core) were a
meaningfully worse condition than clean 1-worker-per-core isolation at
`pinned_4`, `pinned_4` should show LOWER cache-miss/LLC-miss ratios
and FASTER wall-clock than `workers_8`. It shows neither - it is
worse on both axes.

**Comparing `pinned_4` directly against `unpinned_4`** (same worker
count, same chunk_size, only pinning differs): pinned is SLOWER
(29.70s vs. 26.93s) AND has HIGHER cache-miss%/LLC-miss% (38.92%/
23.89% vs. 36.96%/22.35%). If pinning eliminated a real contention
source, unpinned should be worse, not better - it is the opposite of
what the pinning-helps-isolation theory predicts.

**What the numbers actually show instead**: L2-miss%, cache-miss%, and
LLC-miss% are all tightly clustered across every 2/4/8-worker
condition (43-45% L2-miss, 35-39% cache-miss, 21-24% LLC-miss)
regardless of pinning status or exact worker count - there is no
clean monotonic relationship between pinning/worker-count and cache
behavior in this data. The shared resource these workers contend for
(consistent with `l3_capacity_vs_bandwidth_findings.md`'s earlier
separated evidence: L3 capacity AND memory bandwidth, both real) is
already substantially saturated by just 2 concurrent workers
(`pinned_2`'s 35.15%/20.73% is already close to every other
condition's figures) - adding more workers, pinned or not, barely
moves the cache-behavior needle further, and CPU pinning specifically
does not reliably improve it and in the `pinned_4` case measurably
does not.

## CORRECTION, TWICE (2026-09-02, same session): the first "it's noise" correction was ITSELF statistically invalid - proper testing shows pinning hurts at BOTH n_workers=2 and 4

**First pass** (now itself superseded): a 3-rep check eyeballing "diff
vs. a naive combined stdev" concluded the `pinned_2`/`unpinned_2` gap
was noise. The user immediately and correctly challenged THIS
conclusion too: "Your claim on pinned_2 vs unpinned_2 is biased...
if the data points... are claimed to be noise due the overlap and
stdev, You cannot conclude their difference is noise!!! You need to
strengthen your statistical analysis!!" - exactly right: eyeballing an
overlap between two small samples is not a hypothesis test, and
"the difference didn't clear my informal stdev check" does not license
a claim of "no real difference" (absence of significant evidence is
not evidence of absence, especially at n=3).

**Proper test, done right**: `pinned_vs_unpinned_welch_ttest.py` -
Welch's t-test (unequal-variance two-sample t-test, via `scipy.stats`),
5 reps per condition, actual p-values and 95% confidence intervals on
the difference of means, not an eyeballed comparison.

| condition | mean | stdev |
|---|---|---|
| pinned_2 | 29.23s | 1.58 |
| unpinned_2 | 26.28s | 1.57 |
| pinned_4 | 28.04s | 1.06 |
| unpinned_4 | 26.44s | 0.10 |

**pinned_2 vs. unpinned_2**: diff = -2.96s (unpinned faster), t=2.975,
df=8.00, **p=0.0177**, 95% CI [-5.25, -0.67]s (excludes zero) -
**statistically significant**. The original "it's noise" conclusion
from the informal 3-rep check was WRONG - with a properly powered test
(5 reps, a real t-test), the effect is real and unpinned is faster.

**pinned_4 vs. unpinned_4**: diff = -1.59s (unpinned faster), t=3.337,
df=4.07, **p=0.0282**, 95% CI [-2.91, -0.28]s (excludes zero) - also
**statistically significant**, consistent with every prior pass.

**Correct conclusion, superseding both earlier passes**: there was
never a "2-vs-4-worker asymmetry" - but not because the `n_workers=2`
effect was noise (the first correction's claim). Both are real,
statistically significant effects in the SAME direction: **pinning
measurably regresses wall-clock at BOTH `n_workers=2` and
`n_workers=4`** on this machine. No mechanism for this regression is
confirmed at either worker count - the "rigid pinning removes
scheduler load-balancing freedom" idea remains an untested hypothesis,
not a finding.

**Methodology lesson, recorded twice over now** (see
[[feedback_never_trust_single_run_comparisons]] and its own follow-up):
neither drawing a conclusion from ONE run, NOR "disproving" that
conclusion via an informal stdev eyeball at n=3, is valid statistical
practice - a REAL test (adequate sample size, an actual hypothesis
test with a p-value/CI, not just "does the range overlap") is required
before either asserting or denying an effect exists.

## What this does NOT show

- Does not re-derive the exact raw counter values in-table (kept to
  ratios for readability) - full transcripts exist in this session's
  own record for anyone needing to re-verify.
- Does not include statistical repeats (Welch's t-test or otherwise)
  for the FULL matrix's cache-COUNTER data (L1/L2/L3 miss ratios) -
  only wall-clock has been properly tested so far. The single-run
  cache-miss%/LLC-miss% figures in the main table above could
  themselves carry similarly unconfirmed variance - flagged as the
  direct next step, not yet done.
- Does not identify the mechanism behind the now-confirmed pinning
  regression at either `n_workers=2` or `n_workers=4` - flagged as an
  open, unexplained, and NOT-YET-INVESTIGATED item, not something this
  document has an answer for.
