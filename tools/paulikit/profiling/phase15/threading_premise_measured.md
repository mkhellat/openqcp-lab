# Item 1: the threading premise, measured

2026-09-10. **Result: a threaded drain scales. 1.87x on 2 threads,
3.44x on 4, both just under the Amdahl ceiling implied by the
independently measured serial fraction.**

Supersedes `threading_premise_measurement_blocked.md`, which recorded
the wall-clock attempt that could not produce a usable number. The
obstacle there was a ~2x uncontrolled frequency swing (see
`wall_clock_unusable_on_this_machine.md`); it affects the 1-core
baseline and the threaded runs alike, so the fix was to measure a
quantity it cannot distort, not to abandon the measurement.

## Method

`perf stat -e instructions:u,cycles:u` around each run. Cycles and
instructions are frequency-invariant; wall is recorded but advisory.

Real N=150 operator, all 5595 distinct chunks gathered fresh inside
the timed region (so the GIL-held gather is included as production
pays it), one process per measurement, identical per-chunk code in
every condition, 5 reps, interleaved, cooldown to 55 °C.

Speedup is derived from cycles: for fixed work spread over `t`
threads, if total cycles rise by a factor `r`, the achievable speedup
is `t / r`. Perfect scaling is `r = 1`.

## Results

| threads | cycles (B) | sd | instructions (B) | cycles vs t=1 | wall (s) |
|---|---|---|---|---|---|
| 1 | 9.488 | 0.072 | 29.557 | 1.00x | 2.202 |
| 2 | 10.141 | 0.206 | 29.968 | **1.07x** | 1.257 |
| 4 | 11.047 | 0.452 | 29.955 | **1.16x** | 0.746 |
| 8 | 16.274 | 0.228 | 29.940 | 1.72x | 0.729 |

Output was 91,652,096 terms in every condition.

| threads | implied speedup | wall speedup | Amdahl at f=0.0336 |
|---|---|---|---|
| 2 | **1.87x** | 1.75x | 1.93x |
| 4 | **3.44x** | 2.95x | 3.63x |
| 8 | 4.66x | 3.02x | 6.48x |

## Why this one is believable, where the previous two were not

Three independent consistency checks, all of which the earlier
superlinear results failed:

1. **Cycle-implied and wall speedups agree** (1.87 vs 1.75; 3.44 vs
   2.95). Two different metrics, same conclusion.
2. **Both sit just below the Amdahl ceiling** derived from the
   separately measured serial fraction f = 0.0336: 1.87 < 1.93 and
   3.44 < 3.63. The earlier sweep reported 3.06x and 5.03x against
   the same ceilings - physically impossible, and the tell that it
   was an artifact.
3. **Instruction counts are constant** at 29.55-29.97 B across every
   condition. The harness is doing identical work; had this moved,
   the conditions would not be comparable.

Note also the standard deviations: 0.072-0.452 B on cycles, i.e. a
few percent, against the 1.035 s baseline sd that poisoned the
wall-clock sweep. The instability was in the metric, not the system.

## Reading the numbers

**Threading is efficient at 2 and 4 threads.** Only 7% extra cycles
at 2 threads and 16% at 4 - that is the cost of coordination plus
shared-cache contention, and it is small. Efficiency is 94% and 86%
of the respective Amdahl ceilings.

**8 threads is past the useful point.** Cycles jump to 1.72x on a
machine with 4 physical cores and 4 hyperthread siblings - siblings
share execution units, so the second thread on a core adds cycles
without adding much throughput. Wall barely improves (0.746 ->
0.729 s). **4 threads is the operating point**, consistent with the
n_workers default already derived from physical-core count.

**The serial fraction behaves as predicted.** f = 0.0336 was measured
independently, before this sweep, from per-stage timings of a single
chunk. It predicted 1.93x and 3.63x; the sweep delivered 1.87x and
3.44x. A model built from one chunk's stage breakdown predicting a
full 5595-chunk parallel run to within 5% is strong evidence that
both measurements are sound.

## What this settles for Phase 15

Item 1 is answered: a threaded drain delivers real parallel speedup,
the mechanism works (both C kernels genuinely release the GIL), and
memory stays bounded (64 -> 78 MiB across the sweep).

Combined with item 2 (f = 0.0336, the gather is not a blocker) and
item 3 (ThreadPoolExecutor chosen on packaging grounds, since
dispatch is < 0.07% of a chunk either way), the design is justified
by measurement rather than by expectation.

Remaining work is implementation: wire a threaded drain into
`parallel_decompose_arrays` as an alternative to the process pool,
preserving the streaming contract, checkpointing, bounded memory and
chunk independence - none of which a change of drain touches.
