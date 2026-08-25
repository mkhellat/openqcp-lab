# The flat ~30% stall-floor mystery — solved: OpenBLAS thread-pool overhead

Follow-on to `steady_state_scaling_findings.md`, which found total-stall
cycles stuck at a flat ~29-32% across N=25/50/100 while cache-miss
ratio and mem-stall scaled cleanly with N. This was flagged as
unidentified and worth chasing before finalizing the dense-array fix
design - the user explicitly asked to resolve it first, suspecting it
could affect the fix design. It does not affect the fix design
directly, but it does affect how ALL prior `perf stat`/`perf record`
numbers in this directory should be read.

## Investigation

`perf record -g -e cycle_activity.stalls_total` at N=25 (chosen
because the dense array easily fits in L3 there, so it can't be the
cause of whatever this is) localized the dominant symbol immediately:

```
61.17%    59.50%  [.] blas_thread_server
```

**Nearly 60% of self-time in stall-cycle samples is
`blas_thread_server`** - an OpenBLAS worker-thread-pool function.
Grepping `hamiltonian.py` and `algorithms/fwht.py` for any explicit
BLAS-dispatching call (`np.dot`, `np.matmul`, `@`) found **none** -
paulikit's own FWHT code never issues a matrix-multiply. `numpy.show_config()`
confirms this environment's NumPy is linked against **scipy-openblas
0.3.33, `DYNAMIC_ARCH`, `MAX_THREADS=64`** - a threaded BLAS backend
that (per well-documented OpenBLAS behavior) spins up a worker-thread
pool that idles/spin-waits between calls, consuming CPU cycles on
whatever cores are available even when no BLAS work is actually
being dispatched.

## Confirmation via `OPENBLAS_NUM_THREADS=1`

Reproduce with `./run_openblas_comparison.sh [N_OSCILLATORS] [REPS]`
(defaults 25, 5 - matches the numbers below). Same fail-safe/Linux-only
conventions as the other scripts in this directory - see
`run_baseline_perf_stat.sh`'s header for the full reasoning.

Underlying commands, for reference:

```
perf stat -e cycles,cycle_activity.stalls_total,cycle_activity.stalls_mem_any -- \
  python steady_state_decompose.py --n-oscillators 25 --reps 5

OPENBLAS_NUM_THREADS=1 perf stat -e cycles,cycle_activity.stalls_total,cycle_activity.stalls_mem_any -- \
  python steady_state_decompose.py --n-oscillators 25 --reps 5
```

| | baseline | `OPENBLAS_NUM_THREADS=1` |
|---|---|---|
| wall time (5-rep mean) | 0.091s | 0.089s (unchanged, within noise) |
| total cycles | 5.61B | 2.16B (**2.6x fewer**) |
| total-stall cycles | 1.66B | 518M (**3.2x fewer**) |
| total-stall / cycles | 29.6% | 24.0% |

Wall-clock time is essentially unchanged, but total *cycles measured*
drops 2.6x and stall cycles drop 3.2x. This is the mechanistic
explanation: `perf stat`'s default `:u` counters sum across every
thread in the process. OpenBLAS's idle worker-thread pool (spun up
somewhere in NumPy's import/init path, running on otherwise-idle
cores of this 8-thread machine, never actually doing BLAS work
because paulikit's FWHT never calls into BLAS) was being counted
right alongside the main thread's real work - inflating both the
cycle count and the stall count throughout every `perf` measurement
in this investigation so far, without meaningfully affecting the
wall-clock number those measurements were nominally trying to explain.

## What this means for prior findings in this directory

**Does NOT invalidate**: the cache-miss *ratio* findings
(`baseline_perf_stat.md`, `n_scaling_findings.md`,
`steady_state_scaling_findings.md`) or the dense-array root-cause
localization (`perf_record_n50_findings.md`) - those used
`cache-references`/`cache-misses`, which are genuine memory-subsystem
events tied to actual data access, not confused by an idle thread
spinning in a register-level busy-loop (spin-waiting doesn't
typically generate cache-reference traffic the way real memory access
does - consistent with why the cache-miss numbers scaled cleanly with
N while total-stall didn't).

**DOES call into question**: `stall_cycles_n50_findings.md`'s
absolute stall-cycle percentages, and `compiler_flags_findings.md`'s
comparison of `-O2`/`-O3`/`-march=native` using
`cycle_activity.stalls_total` as one of the compared metrics - all of
those numbers include this OpenBLAS thread-pool noise mixed in. The
*directional* conclusions in those docs (compiler flags show no
meaningful difference; the mem-stall-specific numbers, not
total-stall, are the more trustworthy signal) likely still hold, since
the OpenBLAS overhead should affect all three build configurations
equally - but this hasn't been explicitly re-verified with
`OPENBLAS_NUM_THREADS=1` set, and should be before treating those
absolute percentages as final.

## Why total-stall stayed flat across N while mem-stall scaled

This resolves the puzzle directly: OpenBLAS thread-pool spin-wait
overhead is a roughly constant background cost, independent of N
(the worker pool spins at whatever ambient rate the runtime keeps it
alive, not proportional to the FWHT workload's size). As N grows and
the dense array increasingly exceeds cache, *mem-stall specifically*
grows on top of that constant floor - but total-stall is dominated
by the (N-independent) BLAS noise floor plus a (N-dependent, but
smaller in comparison at these N values) memory-stall contribution,
so the sum looks approximately flat.

## Update 2026-08-25: the trigger itself is now confirmed, not just inferred

The original version of this doc said the thread pool was "spun up
somewhere in NumPy's import/init path" without confirming exactly
where or when - flagged as a real gap when directly asked "did you
solve the mystery totally without any doubts?" (answer at the time:
no). Traced directly via `/proc/<pid>/task` thread counts (the
correct ground-truth measurement - Python's own `threading.active_count()`
is blind to native OS threads spawned by a C library, which is why an
earlier quick check with `threading` wrongly suggested no thread pool
existed):

```
threads before any import:        1
threads after `import numpy`:     8   <- jumps here, immediately
threads after `import scipy`:     8
threads after paulikit's own modules: 8   (unchanged)
```

**`import numpy` alone spawns the full 8-thread OpenBLAS pool**
(8 = this machine's core count), as a side effect of the shared
library loading - before any BLAS routine is ever called, and before
any of paulikit's own code runs. Those 8 idle threads then persist
for the rest of the process. This fully closes the mechanism: the
"noise floor" is genuinely N-independent and call-independent - it's
paid once, at import time, regardless of what paulikit's own code
does afterward. No remaining doubt about the trigger.

## Recommended fix for future measurements

Set `OPENBLAS_NUM_THREADS=1` (or `OPENBLAS_NUM_THREADS=0`/pin via
`threadpoolctl`) for all future `perf`-based measurements in this
directory, and re-baseline the affected docs' absolute numbers if a
clean, uncontaminated dataset is needed later. Not yet done: updating
`run_baseline_perf_stat.sh`/`run_steady_state_sweep.sh` to set this
automatically, and re-running the full N=25/50/100/150 sweep with it
set. This is a real methodology fix, tracked here, not yet applied
retroactively to already-committed numbers - doing so would require
re-running everything, which is a deliberate follow-up, not done in
this pass.

## Does this affect the dense-array fix design?

Not directly - the root-cause finding (dense-array densification in
`fwht_pauli_coefficients`) and its N-scaling confirmation both used
genuine memory-subsystem counters (`cache-references`/`cache-misses`),
which this OpenBLAS noise does not appear to have contaminated. The
fix design (sparse output instead of dense-array materialization) can
proceed on that basis. What changes is: (1) the expected *magnitude*
of the fix's benefit on `cycle_activity.stalls_total` specifically
should not be overstated, since a large chunk of that metric is
environment noise unrelated to paulikit's own code; (2) before/after
measurements of the eventual fix should use
`OPENBLAS_NUM_THREADS=1` to get a clean signal, or explicitly note
when they don't.
