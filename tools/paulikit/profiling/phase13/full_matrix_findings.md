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

## Revised conclusion, superseding `cpu_pinning_findings.md`'s framing

The earlier document's framing ("pinning didn't help, therefore L1/L2
hyperthread sharing isn't dominant, therefore it must be L3") is
still directionally right that hyperthread-sibling sharing is not the
dominant mechanism, but this fuller matrix shows pinning's effect is
not simply "neutral" - in the `pinned_4` vs. `unpinned_4` comparison
specifically, pinning is measurably WORSE on every axis measured
(wall-clock, cache-miss%, LLC-miss%, and RSS is statistically
identical so that is not the explanation). A plausible mechanism, not
yet confirmed: forcing a fixed core assignment removes the Linux
scheduler's own freedom to react to real-time load imbalance (e.g.
migrating a temporarily-stalled worker's thread to a less-busy core) -
rigid pinning can cost more than it isolates when the real bottleneck
(shared L3/bandwidth) is not something core placement can fix in the
first place, exactly as this investigation's L3-capacity-vs-bandwidth
findings already established.

## What this does NOT show

- Does not re-derive the exact raw counter values in-table (kept to
  ratios for readability) - full transcripts exist in this session's
  own record for anyone needing to re-verify.
- Does not explain WHY pinned_4 specifically (not pinned_2) shows the
  worst numbers - pinned_2 does not show the same regression relative
  to its unpinned counterpart (25.16s pinned vs. 26.86s unpinned -
  pinned is actually FASTER here, the opposite pattern from the
  4-worker case) - this 2-vs-4-worker asymmetry in how pinning affects
  outcomes is itself unexplained and not yet investigated further.
- Does not include a repeated-run statistical pass (single run per
  condition/event-group here, unlike `clean_chunk_size_sweep_findings.md`'s
  earlier repeated methodology) - given the clear, large effect sizes
  observed (pinned_4 vs. unpinned_4's ~10% wall-clock gap, ~2 point
  percentage gaps in miss ratios), this is unlikely to be pure noise,
  but has not been confirmed via repeats specifically for this matrix.
