# Item 1: the threading premise is still not honestly measured

2026-09-10. **Status: BLOCKED on measurement environment. No speedup
number from this session should be quoted.**

## What was attempted

Phase 15 item 1 is a re-measurement of the threading probe that Phase
14 recorded as untrustworthy (3.04x on 2 threads, 5.42x on 4 -
superlinear, and therefore unphysical for compute-bound work). The
diagnosis then was that the probe reused 64 pre-built blocks, so they
stayed cache-warm and the work never touched fresh memory.

A new harness was built to remove that flaw: real chunks from a real
N=150 operator, gathered fresh inside the timed region (so the
GIL-held gather is included, as production pays it), all 5595 distinct
chunks with nothing reused, one process per measurement, and the
sequential baseline running the identical per-chunk code path.

Output correctness held throughout: 91,652,096 terms in every
condition.

## The result, and why it must not be quoted

| threads | wall mean | sd | apparent speedup | apparent efficiency |
|---|---|---|---|---|
| 1 | 3.978s | **1.035** | 1.00x | — |
| 2 | 1.299s | 0.038 | 3.06x | **153%** |
| 4 | 0.790s | 0.056 | 5.03x | 126% |
| 8 | 0.706s | 0.005 | 5.64x | 70% |

**153% efficiency is not physical.** The superlinear result from Phase
14 reappeared in a harness built specifically to eliminate its
supposed cause - which is itself the finding: the earlier diagnosis
(cache-warm reused blocks) was wrong, or at least incomplete.

Note the tell, and it is the one the standing rule names: **the
baseline's sd is 1.035 against 0.005-0.056 for the threaded rows.**
The instability is in the denominator. Any ratio computed from it is
meaningless.

## Root cause: contention from the measurement environment itself

Investigated directly rather than assumed.

**The baseline is bimodal, not drifting.** Six consecutive 1-thread
runs: 4.488, 4.492, 4.529, **2.317**, 4.454, 4.550 s - a clean factor
of two, with `cores` reported as 1.00 in every run. Same code, same
input, same process shape.

**It is not the CPU frequency ramp.** That was the first hypothesis,
since this machine runs the `powersave` governor with a 400 MHz floor
and 4000 MHz ceiling, and cold-start ramp is a known confound here
(phase13). An untimed in-process warm-up was added; **the bimodality
survived it**. Sampling `scaling_cur_freq` from inside the timed
region across six fresh processes gave 2.204-2.246 s at a median
3700-3796 MHz every time - no slow mode at all in that probe.

**It is external CPU contention.** `ps` during the runs showed first a
transient `zsh` at 50% CPU, then **`rtk` at 83% CPU** - the RTK proxy
that wraps shell commands in this environment. It runs concurrently
with the measurement and takes roughly a core, which on this 4-core
machine is exactly the intermittent factor of two. The clean probe
that showed no bimodality happened to run while nothing else was
active.

So the fast mode (~2.2s) is the true single-thread cost, and the slow
mode (~4.5s) is a contended run. An unknown mixture of the two lands
in every mean.

## Why this matters more than the number

The 8-thread row is the one that looks most credible - 70% efficiency,
sd 0.005 - and it is also the one whose baseline contamination is most
likely, since a contended baseline inflates every speedup above it.
The Amdahl prediction from the independently measured serial fraction
(f = 0.0336) is **1.93x at 2 threads and 3.63x at 4**; the harness
reported 3.06x and 5.03x. Measurement exceeding the theoretical
ceiling by that margin is a harness verdict, not a discovery.

Two conclusions stand regardless:

1. **The threading mechanism works.** Both C kernels genuinely release
   the GIL; `cores` rises from 1.00 to 6.46 across the sweep, and
   output stays bit-identical. Threads do run the compiled work
   concurrently, and peak RSS stays bounded (64 -> 78 MiB).
2. **The magnitude remains unmeasured**, for the second time. Phase
   14's number was rejected for the wrong reason; this one is
   rejected for a better-understood one.

## What a valid measurement requires

- **A quiet machine.** No RTK-proxied shell commands, no concurrent
  agent activity, nothing else scheduled during the sweep. This is
  the binding constraint and it cannot be satisfied while the
  measurement is driven from inside the same environment that is
  loading the CPU.
- Baseline stability as an explicit gate: reject the run unless the
  1-thread condition's spread is comparable to the threaded rows'
  (the 1.15x alarm in `phase13/MEASUREMENT_METHODOLOGY.md`).
- Sanity check against the Amdahl ceiling from f = 0.0336. Any
  measured speedup materially above it is a harness fault by
  construction.
- Consider pinning the measurement to specific cores so background
  load lands elsewhere, though that changes what is being measured
  and needs stating if used.

Until then: **the threaded design is justified by the serial fraction
(f = 0.0336, ceiling 3.63x at 4 threads) and by the dispatch-cost
evidence, not by any speedup this session produced.**
