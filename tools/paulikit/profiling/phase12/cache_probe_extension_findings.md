# Building the empirical cache-latency probe extension: two real timing bugs found and fixed

Recorded 2026-09-01, while implementing PLAN.md Phase 12's chunk_size
auto-tuner. The design calls for measuring cache-latency boundaries
empirically at runtime (a new standalone Cython/C extension,
`src/paulikit/_native/cache_probe.{pyx,c,h}`) rather than trusting
declared cache-topology sources, after finding `lscpu`'s own "L2 1
MiB" figure was an aggregate-across-cores artifact (the real per-core
L2 on this machine, from `/sys/devices/system/cpu/cpu0/cache/index2/size`,
is 256 KiB - see PLAN.md Phase 12's design section). This document
records two distinct, real measurement-quality bugs found and fixed
while building the probe itself, both confirmed via direct
measurement, not assumption.

## Bug 1: `clock_gettime` timing is corrupted by CPU frequency scaling

The first working version of `cache_probe_run` used
`clock_gettime(CLOCK_MONOTONIC)` to time the pointer-chase loop.
Running it repeatedly showed the same 64 KiB buffer size reading
anywhere from **3ns to 42ns per access** across separate runs - a 14x
swing, on a supposedly deterministic pointer-chase over a fixed-size
buffer.

Traced to this machine's CPU frequency governor:

```
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
powersave
$ for i in 1 2 3 4 5; do cat /proc/cpuinfo | grep -m1 MHz; sleep 0.2; done
cpu MHz		: 3156.100
cpu MHz		: 3299.980
cpu MHz		: 3257.921
cpu MHz		: 1554.585
cpu MHz		: 400.002
```

An 8x frequency range (400 MHz - 3.3 GHz), observed swinging within
a one-second window under normal idle-ish load. `clock_gettime`
measures wall-clock time; if the core's actual clock speed changes
*during* the timed loop, the same number of cycles (the same amount of
real work) takes a different amount of wall-clock time depending on
which frequency the core happened to be running at - directly
corrupting any wall-clock-based per-access latency estimate.

**Fix**: switched to hardware cycle counters - `__rdtscp()` (x86_64,
via `<x86intrin.h>`), `mrs x, cntvct_el0` (aarch64), the `rdtime` CSR
read (riscv64) - the same three instructions `configure`'s own
`--probe-cache-latency` asm probe already uses, for exactly this
reason (see `configure`'s own comments). The invariant TSC (x86_64)
and its ARM/RISC-V equivalents tick at a fixed reference rate
regardless of the core's actual DVFS state, so cycle counts are
immune to this noise source. Confirmed by `configure`'s own probe
getting clean, sharp boundaries on this same noisy hardware under the
same load (`32KiB->64KiB(x1.70) 128KiB->256KiB(x1.60)
256KiB->512KiB(x1.92) 2048KiB->4096KiB(x5.25)`) - proof the underlying
hardware behavior was never actually noisy, only the wall-clock
measurement of it was.

Boundary detection only ever needs *ratios* between rows (matching
`configure`'s own `ratio > 1.3` threshold), so raw uncalibrated cycle
counts are sufficient - no cycles-to-seconds conversion is needed or
attempted.

## Bug 2: occasional scheduler preemption still corrupts individual readings

Switching to cycle counters fixed most of the noise, but did not fully
eliminate it: even with `__rdtscp`, one run out of several still
produced a corrupted reading (e.g. 8192 bytes reading 25.3 cycles/access
instead of the ~6 cycles/access every other run agreed on). This is a
different mechanism than DVFS: if the OS scheduler preempts the
process mid-timed-loop (to run something else, migrate it to another
core, etc.), the elapsed cycle count includes however long the process
was descheduled - cycle counters are immune to *frequency* noise, not
to being *paused*.

**Fix, two parts**, both in `cache_probe_run`:

1. **Core pinning** (Linux-only, via `sched_setaffinity`/`sched_getaffinity`,
   pinning to whichever CPU the probe is already running on - not a
   hardcoded CPU 0, since an external affinity restriction, e.g. a
   Slurm/PBS cgroup/cpuset on a shared HPC node, might not even
   include CPU 0; the original affinity mask is restored afterward).
   Reduces migration-driven noise specifically. A no-op stub on
   non-Linux platforms (macOS/BSD have weaker, different affinity
   APIs not chased here) - the second mitigation below is left to do
   the whole job there.
2. **Repeat-and-take-minimum**: each buffer size is measured `repeats`
   times (default 3), keeping the *minimum* cycle count across
   repeats as the estimate. This is the right statistic here (not a
   mean or median): a preemption can only *inflate* a reading (extra
   descheduled time adds to the measured delta), never deflate it
   below the buffer's true cache latency - so the minimum of a few
   repeats robustly rejects preemption outliers without needing to
   detect or classify them explicitly.

Confirmed via 4 repeated end-to-end runs after both fixes: all four
now show the same clean, monotonic step pattern matching this
machine's real per-core cache hierarchy (L1d 32 KiB, L2 256 KiB, L3
8 MiB shared, from `/sys/devices/system/cpu/cpu0/cache/`):

| buffer size | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| 8 KiB | 5.99 | 6.11 | 6.08 | 6.05 |
| 32 KiB | 5.88 | 6.13 | 6.15 | 6.09 |
| 64 KiB | 8.40 | 9.09 | 8.87 | 8.98 |
| 256 KiB | 9.60 | 9.93 | 8.94 | 11.42 |
| 512 KiB | 24.35 | 23.13 | 24.28 | 25.14 |
| 4 MiB | 23.44 | 23.58 | 32.59 | 22.82 |
| 8 MiB | 86.20 | 101.60 | 60.36 | 93.45 |
| 32 MiB | 187.21 | 205.71 | 183.55 | 241.45 |

(cycles/access; all 13 measured sizes omitted here for brevity, full
range 8 KiB-32 MiB doubling.) The step pattern - flat within L1
(~6 cycles), a rise entering L2 (~9-11), a bigger rise at the L2/L3
boundary (~23-30, staying roughly flat through the rest of L3's 8 MiB),
then a sharp final jump past L3 into DRAM latency territory
(60-240+) - is consistent across all four runs and matches the
`/sys`-reported per-core cache sizes closely.

## Why this matters beyond just "the probe works now"

Both bugs would have silently produced a working-looking but
*wrong* auto-tuning formula if not caught: bug 1 would have made the
chunk_size heuristic's cache-boundary detection unreliable on any
machine with a non-`performance` frequency governor (which is most
laptops/workstations and many cloud/HPC nodes with power-management
enabled) - not a rare edge case. Bug 2, left unfixed, would have made
the probe's *first invocation* per process (the one that gets cached
for the process's whole lifetime, per PLAN.md Phase 12's process-
lifetime-caching design) a roughly-1-in-4 chance of locking in a
corrupted boundary for the entire run - the kind of intermittent bug
that would be extremely hard to reproduce or diagnose later without
this document's direct measurement trail.

## What this does NOT show

- Only the x86_64 cycle-counter path (`__rdtscp`) has been exercised
  on real hardware here; the aarch64 (`cntvct_el0`) and riscv64
  (`rdtime`) paths are implemented (mirroring `configure`'s own
  per-arch asm probe exactly) but not yet run/verified on real or
  QEMU-emulated ARM64/RISC-V hardware - unlike `configure`'s own
  probe, which was QEMU-verified for those architectures (see
  PLAN.md's `configure` history). Follow-up verification, not done
  here.
- Core pinning's no-op fallback on non-Linux platforms (macOS/BSD) is
  untested - the repeat-minimum mitigation is expected to still help
  there, but this has not been measured on non-Linux hardware.
- Does not yet test behavior inside an HPC-style memory/CPU cgroup
  (a Slurm/PBS allocation restricting which CPUs are visible) -
  `sched_getcpu`-then-pin-to-that-CPU is designed to be safe inside
  such a restriction (see fix 2 above), but not directly verified
  under a real constrained cpuset.
