# Direct evidence for cross-core contention (L3/memory-bandwidth) - not inference by elimination

Recorded 2026-09-02, prompted by direct, correct pushback: the earlier
"pinning didn't help, therefore L3" claim
(`cpu_pinning_findings.md`) was a process-of-elimination inference,
not a directly measured comparison. "Do you have full evidence for
the L3 contention? Did you collect data on that for pinned and
unpinned executions? Let's collect evidence instead of assuming." This
document is that direct evidence.

## Design: isolate cross-core contention from L1/L2 hyperthread sharing entirely

`l3_contention_isolation_test.py`: one pinned worker (logical CPU 0 =
physical core A) processes exactly 1/4 of the real N=150 workload
(matching a real `parallel_decompose(n_workers=4)` worker's share -
the same `_iter_chunked_coefficients` per-chunk body, not a synthetic
proxy), measured under two conditions:

- **`alone`**: no other processes running.
- **`with_noise`**: 3 additional processes, each pinned to a DIFFERENT
  physical core (B, C, D - logical CPUs 1, 2, 3), each continuously
  reading/writing a ~64 MB buffer (large enough to spill past any
  per-core L1/L2, guaranteeing real cache/memory traffic, not a no-op
  spin loop) for the entire duration of the measured worker's run.

Critically, **the noise processes share ZERO L1/L2 with the measured
worker** - they are on entirely different physical cores. Any effect
on the measured worker's cache-miss ratio must come from something
shared across cores: L3 and/or memory bandwidth, not hyperthread
sibling sharing. `perf stat --no-inherit` isolates the measured
worker's own counters from its noise-process children (confirmed:
`task-clock` matched wall-clock in both runs, while `user` time in the
`with_noise` run - 66.1s - correctly reflects the (excluded) noise
processes' own CPU time, not counted in the `:u` figures reported).

## Result: direct, measured evidence of real cross-core contention

| condition | wall-clock | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|
| alone | 5.13s | 1.4% | 1.2% |
| **with 3-core noise (zero L1/L2 sharing)** | **16.23s** | **4.6%** | **3.6%** |

Wall-clock: **3.2x slower** with noise on other cores alone. Cache-miss
ratio: roughly **tripled** (1.4%->4.6%). LLC-miss ratio: **tripled**
(1.2%->3.6%). This is real, controlled, directly measured evidence -
not an inference from "pinning didn't help" - that cross-core
resource contention (consistent with L3 and/or memory bandwidth, the
two resources genuinely shared across all physical cores on this
single-socket machine) meaningfully degrades cache behavior and
wall-clock, even with zero hyperthread-sibling (L1/L2) sharing
involved at all.

## What this does and does NOT establish

**Does establish**: cross-core contention is real, substantial, and
does not require hyperthread sharing to occur - directly answering the
user's question with collected evidence, not assumption.

**Does NOT yet establish**: whether the contention is *specifically*
L3 capacity/eviction pressure, or memory-bandwidth saturation, or
both - `LLC-load-misses` measures L3 misses (i.e. requests that had to
go to DRAM), which tripling is consistent with either mechanism:
L3 eviction pressure from the noise processes' larger working set
would show up as more L3 misses, but so would raw memory-bandwidth
contention independently slowing down the DRAM-bound fraction of the
measured worker's own accesses without necessarily reflecting a true
L3-capacity effect. Distinguishing these two would need a differently
designed noise workload (e.g. one sized to fit entirely within its own
L2, generating memory-bus traffic without ever touching L3/DRAM, vs.
one deliberately larger than total L3 capacity) - not done here.

**Also not yet done**: comparing this same `alone`/`with_noise`
contrast at the REAL `parallel_decompose` scale (4 real workers doing
real divided work simultaneously, not 1 real worker + 3 synthetic
noise processes) - this test isolates the mechanism cleanly but is a
controlled proxy, not a replacement for confirming the real 4-worker
case shows a comparable magnitude of degradation.
