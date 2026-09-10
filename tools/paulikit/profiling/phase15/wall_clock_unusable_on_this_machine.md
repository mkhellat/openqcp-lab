# Why wall clock cannot measure this workload on this machine

2026-09-10. Root cause of a bimodality that defeated three separate
harnesses and produced two wrong diagnoses before being isolated.

## The symptom

A single-threaded run of the full N=150 decomposition, in a fresh
process, alternates between **~2.2 s and ~4.5 s** — a clean factor of
two. Same code, same input, `cores` reported as ~1.00 in both modes.
Eight consecutive runs on a verified-idle machine:

```
2.291  4.991  2.479  2.451  2.506  5.076  4.851  2.411
```

The mode is not predictable, does not drift, and does not alternate
regularly. Roughly half the runs land in each.

## Two wrong diagnoses, recorded because they were plausible

**Wrong #1: cache-warm reused blocks.** Phase 14 blamed its
superlinear probe on reusing 64 pre-built blocks. A new harness used
all 5595 distinct chunks, gathered fresh inside the timed region —
and the superlinear result reappeared. So that diagnosis was wrong,
or at least not the whole story.

**Wrong #2: external CPU contention.** `ps` showed `rtk` at 83% CPU
and a transient `zsh` at 50%, which on a 4-core machine would produce
exactly a 2x. But **`ps` `%CPU` is a lifetime average** — the same
trap recorded in `feedback_measure_the_region_not_the_process`.
Measuring instantaneous usage from `/proc/<pid>/stat` deltas over a
3 s window showed `rtk` at **0%** and the machine 19.9% busy of 800%
available. The bimodality persisted with the machine verifiably idle.

Also ruled out, each by direct test:

- **The warm-up loop** — disabling the 400-chunk untimed warm-up
  changed nothing.
- **Core placement / hyperthread siblings** — the topology is 4
  physical cores with siblings (0,4 / 1,5 / 2,6 / 3,7), so landing on
  a busy sibling would give a clean 2x. Pinning with
  `taskset -c 0` did **not** remove the bimodality.
- **The derived tile** — `tile_for_cache(16384)` returned 1024 in
  every run, fast and slow alike.

## The actual cause: CPU frequency, proven by hardware counters

`perf stat -e instructions:u,cycles:u,cache-misses:u`:

| wall | instructions | cycles | cache misses |
|---|---|---|---|
| 2.201 s | 29.556 B | 9.496 B | 0.008 B |
| 2.203 s | 29.560 B | 9.531 B | 0.008 B |
| **4.563 s** | 29.559 B | 10.861 B | 0.009 B |
| 2.254 s | 29.555 B | 9.495 B | 0.008 B |
| 2.195 s | 29.557 B | 9.478 B | 0.007 B |
| **4.482 s** | 29.559 B | 10.681 B | 0.009 B |

**Instructions are identical to five significant figures.** Cache
misses are identical. Cycles rise only ~14%. But wall doubles.

Effective clock, cycles ÷ wall:

- fast mode: 9.49 B / 2.20 s ≈ **4.3 GHz**
- slow mode: 10.75 B / 4.5 s ≈ **2.4 GHz**

The core is running at roughly half speed. This machine uses
`intel_pstate` with the `powersave` governor (400 MHz floor,
4000 MHz ceiling) and turbo enabled (`no_turbo = 0`), and the
governor's choice of turbo versus base residency for a given run is
not controllable from userspace. It is the DVFS confound already
recorded in
`feedback_disable_dvfs_before_trusting_wallclock`, in a form that a
per-run frequency probe missed because the probe happened never to
land in the slow mode.

The extra ~14% cycles in the slow mode are consistent with running at
a lower frequency while memory latency stays fixed in nanoseconds —
the core stalls for more cycles waiting on the same DRAM.

## Consequence: measure cycles, not wall

For this workload on this machine, **wall clock has a ~2x
uncontrolled multiplicative error**, which is larger than any effect
worth measuring. Two runs of identical work differ by 2x for reasons
that have nothing to do with the code.

Instructions and cycles are frequency-invariant and were stable to
five significant figures across the same runs. So:

- **Speedup should be expressed as a ratio of total cycles** for a
  fixed amount of work. Perfect scaling consumes the same total
  cycles as the serial run; coordination and contention appear as
  extra cycles.
- **Instruction count is the correctness check on the harness** — if
  it moves between conditions, the conditions are not doing the same
  work.
- Wall may be reported as advisory, never as the basis of a ratio.

This is also the retrospective explanation for Phase 14's superlinear
probe (3.04x/2 threads, 5.42x/4) and for this phase's first sweep
(3.06x/2 threads, 153% efficiency): a serial baseline that landed in
the slow mode, divided into threaded runs that landed in the fast
one, manufactures an arbitrary speedup. The baseline's sd of 1.035
against 0.005–0.056 for the threaded rows was the visible tell.

## Standing rule for this machine

Any wall-clock ratio on a CPU-bound workload here must be treated as
unreliable unless the frequency state is pinned or the measurement is
in cycles. The earlier thermal protocol
(`phase13/MEASUREMENT_METHODOLOGY.md`) controls temperature, which is
necessary but demonstrably not sufficient — these runs were all below
the 55 °C cooldown gate.
