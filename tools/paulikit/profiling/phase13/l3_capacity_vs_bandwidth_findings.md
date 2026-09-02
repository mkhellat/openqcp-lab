# L3-capacity vs. memory-bandwidth contention: separated, both real, L3-capacity dominant

Recorded 2026-09-02, direct follow-up to
`l3_contention_direct_evidence_findings.md`'s own "what this does NOT
establish" item: that document proved cross-core contention is real
but could not distinguish L3-capacity/eviction pressure from raw
memory-bandwidth/interconnect contention (both are consistent with a
tripled LLC-miss ratio). Per direct instruction, this is a fresh,
single experiment - all three conditions collected together, no reuse
of old numbers.

## Machine's real cache sizes (checked directly, not assumed)

`lscpu` + `/sys/devices/system/cpu/cpu0/cache/index*/size`: **per-core
L2 = 256 KiB** (4 instances, one per physical core), **shared L3 = 8
MiB** (1 instance, shared across all 4 physical cores). These sizes
directly drove the two noise-workload designs below.

## Design: three conditions, same measured worker, only the noise workload's size differs

`l3_capacity_vs_bandwidth_test.py`: one pinned worker (physical core
A) processes a real 1/4 share of N=150's actual workload (same
`_iter_chunked_coefficients` body as `l3_contention_isolation_test.py`),
under:

1. **`alone`**: no noise processes.
2. **`noise_l2_bound`**: 3 noise processes (physical cores B, C, D),
   each repeatedly touching a 64 KiB buffer - well under this core's
   own 256 KiB L2, so each noise process's own working set NEVER
   spills into L3 or DRAM. Any degradation here must come from
   something shared even by pure-L2-resident traffic - i.e. memory-
   bandwidth/interconnect contention, not L3 capacity (there is
   nothing for this noise to evict from L3 that it never touches).
3. **`noise_l3_exceeding`**: 3 noise processes, each touching a 64 MB
   buffer - ~8x the ENTIRE shared L3's capacity, guaranteeing real
   eviction pressure plus real DRAM round-trips of its own.

`perf stat --no-inherit` isolates the measured worker's own counters
from its noise-process children in every condition (confirmed:
`task-clock` closely tracks wall-clock in all three runs, while `user`
time correctly reflects the excluded noise processes' own CPU cost).

## Result: fresh, single-experiment data, all three conditions

| condition | wall-clock | vs. alone | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|---|
| alone | 4.62s | 1.00x | 0.41% | 0.40% |
| noise_l2_bound (64 KiB/core) | 5.48s | **1.19x** | 0.88% | 0.79% |
| noise_l3_exceeding (64 MB/core) | 14.78s | **3.20x** | 2.87% | 2.20% |

## Both mechanisms are real; L3-capacity/DRAM contention dominates

**L2-bound noise alone causes real, measurable degradation** (1.19x
wall-clock, cache-miss/LLC-miss ratios roughly doubled) despite the
noise processes' data never leaving their own private L2 - direct
evidence of genuine memory-bandwidth/interconnect contention (the
ring/mesh interconnect and memory-controller request queue are shared
by all cores' cache-fill/writeback traffic regardless of which cache
level ultimately serves a given access, so even L2-resident noise
generates real contention for that shared path).

**L3-exceeding noise causes substantially larger degradation** (3.20x
wall-clock total, and specifically 2.7x slower than the L2-bound
condition alone: 5.48s -> 14.78s) - a large ADDITIONAL effect
attributable to the L3-exceeding noise's own eviction pressure on the
measured worker's L3-resident data and its own real DRAM traffic,
which the L2-bound condition structurally cannot produce.

**Conclusion, now evidence-based rather than guessed**: both L3-
capacity contention and memory-bandwidth/interconnect contention are
real, separately confirmed mechanisms on this machine - but L3-
capacity/DRAM-traffic contention is the DOMINANT one, accounting for
roughly 2.7x of the total 3.2x degradation, versus bandwidth/
interconnect contention's smaller (but non-zero) ~1.19x share.

## What this still does NOT show

- Does not test intermediate noise-buffer sizes (e.g. sized to fit L3
  but exceed one core's L2 - a "shared-L3-but-not-evicting" condition)
  to further refine the boundary between the two mechanisms.
- Does not test this same three-way split at the REAL 4-simultaneous-
  worker `parallel_decompose` scale (still a controlled 1-worker-vs-
  noise proxy, not the actual parallel workload contending with
  itself).
- Does not distinguish "interconnect/ring bus" contention from
  "memory controller/DRAM channel" contention specifically within the
  bandwidth-contention finding - both are plausible sub-mechanisms for
  the L2-bound noise's effect, not separately isolated here.
