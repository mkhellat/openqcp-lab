# Cache locality investigation — baseline hardware counters

Started 2026-08-25. This is step 1 of a fundamental (not TBB-limited)
cache-locality investigation: establish ground truth with real hardware
performance counters before touching any code. See `PLAN.md` for the
phase this belongs to once scoped.

## Method

```
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  paulikit decompose --n-oscillators 50
```

N=50 chosen because it's the smallest size where Phase 3c's native+TBB
path already dominates the pure-Python baseline substantially (2.15s
per `README.md`'s benchmark table), so it exercises the compiled
kernel meaningfully without the multi-minute runtime of N=100.

Test machine: Intel Core i7-8550U (see `cpu_info.txt`) — 4 cores/8
threads, 128 KiB L1d, 1 MiB L2, 8 MiB shared L3, **single NUMA node**.
The single-NUMA-node caveat matters: this hardware cannot reveal
NUMA-topology cache effects even if they exist in the code, since
there's only one node to migrate across. Any conclusion here about
"no NUMA issue" is scoped to this machine, not proven in general.

## Results (3 runs, `paulikit decompose --n-oscillators 50`)

| metric | run 1 | run 2 | run 3 |
|---|---|---|---|
| wall time | 2.219s | 2.612s | 2.445s |
| cycles | 8.75B | 9.08B | 8.95B |
| instructions | 12.57B | 12.65B | 12.64B |
| cache-references | 228.8M | 235.7M | 233.7M |
| cache-misses | 123.5M | 130.4M | 130.1M |
| **cache-miss rate** | **54.0%** | **55.3%** | **55.7%** |
| L1-dcache-loads | 3.23B | 3.21B | 3.22B |
| L1-dcache-load-misses | 117.2M | 126.2M | 126.1M |
| L1 miss rate | 3.6% | 3.9% | 3.9% |
| LLC-loads | 31.2M | 31.3M | 30.4M |
| LLC-load-misses | 13.7M | 15.1M | 14.4M |
| **LLC miss rate** | **43.8%** | **48.2%** | **47.4%** |

(Sample rates on `:u` counters were 50-63%, not 100% — perf multiplexed
these events since more were requested than the CPU's PMU has counters
for. Absolute counts are scaled estimates, not exact. Rates (ratios)
are more trustworthy than the absolute counts for this reason — this
is a real methodological limitation of the current measurement, not
swept under the rug.)

## What this does and doesn't tell us yet

**Does tell us:** cache misses are a real, reproducible, non-trivial
phenomenon in this workload (consistent ~54-56% combined miss rate,
~44-48% LLC miss rate across 3 runs) — not noise, not a fabricated
concern. Worth investigating further.

**Doesn't yet tell us:** *where* in the code these misses originate
(dense/sparse coefficient array construction? label generation?
Python/Cython/C++ boundary crossings? the TBB parallel kernel
specifically?), or *how much of the 2.2s wall time* is actually
attributable to memory stalls versus compute — a 54% cache-miss rate
doesn't by itself mean 54% of runtime is lost to it; miss cost depends
on whether the CPU can hide the latency via out-of-order execution and
prefetching. That's IPC (instructions-per-cycle) territory:
12.57B instructions / 8.75B cycles ≈ 1.44 IPC — not catastrophically
stalled (a memory-bound kernel often sees IPC well under 1), but not
great either. This needs a proper stall-cycle breakdown
(`perf stat -e cycle_activity.stalls_l2_pending` or similar, or
`perf record`+`perf report` with `--sort=comm,dso,symbol` to localize
misses to specific functions) as the next step — not yet done.

## Honest scope note

This machine (an 8-thread ultrabook, no NUMA) is not representative
of a "production cluster server" the way the earlier discussion
imagined. Findings here establish real cache behavior on real
hardware, but conclusions about multi-socket NUMA effects, larger
core counts, or server-class cache hierarchies cannot be drawn from
this machine and shouldn't be claimed as if they can.
