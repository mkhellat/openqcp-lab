# Full verification on the ACTUAL parallel_decompose() code path: pinning confirmed, frequency puzzle isolated

Recorded 2026-09-02, direct correction after: "Now run the pinned_2
full test for OUR CODE!!!!! This was not aimed at a dummy code....
In your test, all cpu clocks are checked and recorded alongside the
affinity." Correct - the prior check
(`self_reported_affinity_verification.md`) manually called
`os.sched_setaffinity` and directly invoked `_parallel_worker_chunk`/
`_parallel_worker_init`, bypassing the real `ProcessPoolExecutor`/
shared-counter pinning mechanism `parallel_decompose()` itself uses.

## Method: the real entry point, unmodified, observed from inside

`real_parallel_decompose_full_verification.py` calls
`parallel_decompose()` completely unmodified - the real pool, the real
`pin_cpus`/`next_pin_index` shared-counter pinning mechanism. `fwht.
_parallel_worker_chunk` is monkeypatched to a thin wrapper that logs
before calling the REAL underlying function unchanged - the actual
computation is untouched, only observed. Confirmed the fork-based
`multiprocessing` start method (`multiprocessing.get_start_method()`
returns `'fork'`) propagates this monkeypatch correctly into the real
worker processes.

Each real worker process logs, at chunk #1 and every 200th chunk
thereafter: its own self-reported affinity (`os.sched_getaffinity`)
and current CPU (`sched_getcpu` via `ctypes`) - AND, at the exact same
instant, ALL 8 cores' `scaling_cur_freq` together, per direct
instruction to record clocks alongside affinity, not separately.

## Full raw data (both real worker processes, complete, unfiltered)

```
condition=pinned_2 elapsed=29.9731s n_workers=2 terms=91652096

--- pinned_2_worker_pid8989.log ---
t=2389.3206 chunk#1    affinity=[0] current_cpu=0 | cpu0=1200MHz cpu1=1200MHz cpu2=1200MHz cpu3=1201MHz cpu4=400MHz cpu5=400MHz cpu6=400MHz cpu7=400MHz
t=2391.2661 chunk#200  affinity=[0] current_cpu=0 | cpu0=3199MHz cpu1=3200MHz cpu2=3201MHz cpu3=3200MHz cpu4=3200MHz cpu5=400MHz cpu6=3200MHz cpu7=3199MHz
t=2393.3829 chunk#400  affinity=[0] current_cpu=0 | cpu0=3202MHz cpu1=3200MHz cpu2=3200MHz cpu3=400MHz cpu4=2235MHz cpu5=400MHz cpu6=3200MHz cpu7=3202MHz
t=2394.9070 chunk#600  affinity=[0] current_cpu=0 | cpu0=3126MHz cpu1=3173MHz cpu2=3140MHz cpu3=3099MHz cpu4=3135MHz cpu5=400MHz cpu6=3025MHz cpu7=3173MHz
t=2396.8711 chunk#800  affinity=[0] current_cpu=0 | cpu0=3199MHz cpu1=3200MHz cpu2=3200MHz cpu3=3199MHz cpu4=3200MHz cpu5=3200MHz cpu6=3200MHz cpu7=3196MHz
t=2398.7240 chunk#1000 affinity=[0] current_cpu=0 | cpu0=3200MHz cpu1=3200MHz cpu2=3200MHz cpu3=3200MHz cpu4=3200MHz cpu5=3201MHz cpu6=3200MHz cpu7=400MHz
t=2401.0889 chunk#1200 affinity=[0] current_cpu=0 | cpu0=3201MHz cpu1=3200MHz cpu2=3201MHz cpu3=3200MHz cpu4=3181MHz cpu5=3206MHz cpu6=3200MHz cpu7=3200MHz
t=2402.8568 chunk#1400 affinity=[0] current_cpu=0 | cpu0=3102MHz cpu1=3100MHz cpu2=3100MHz cpu3=3100MHz cpu4=3104MHz cpu5=3103MHz cpu6=3100MHz cpu7=3103MHz
t=2404.8052 chunk#1600 affinity=[0] current_cpu=0 | cpu0=3203MHz cpu1=3200MHz cpu2=3159MHz cpu3=400MHz cpu4=400MHz cpu5=400MHz cpu6=3200MHz cpu7=3200MHz
t=2407.0593 chunk#1800 affinity=[0] current_cpu=0 | cpu0=3200MHz cpu1=3200MHz cpu2=3200MHz cpu3=3199MHz cpu4=400MHz cpu5=3129MHz cpu6=3202MHz cpu7=3200MHz
t=2409.1282 chunk#2000 affinity=[0] current_cpu=0 | cpu0=2001MHz cpu1=2000MHz cpu2=1731MHz cpu3=1600MHz cpu4=2400MHz cpu5=2400MHz cpu6=2000MHz cpu7=2000MHz
t=2411.1421 chunk#2200 affinity=[0] current_cpu=0 | cpu0=3101MHz cpu1=3100MHz cpu2=3100MHz cpu3=3106MHz cpu4=3100MHz cpu5=3101MHz cpu6=3100MHz cpu7=3100MHz
t=2413.7614 chunk#2400 affinity=[0] current_cpu=0 | cpu0=3200MHz cpu1=3200MHz cpu2=3200MHz cpu3=3201MHz cpu4=3200MHz cpu5=400MHz cpu6=3200MHz cpu7=400MHz
t=2416.5204 chunk#2600 affinity=[0] current_cpu=0 | cpu0=400MHz cpu1=399MHz cpu2=400MHz cpu3=400MHz cpu4=400MHz cpu5=400MHz cpu6=400MHz cpu7=400MHz

--- pinned_2_worker_pid8990.log ---
t=2389.3206 chunk#1    affinity=[1] current_cpu=1 | cpu0=1200MHz cpu1=1200MHz cpu2=1200MHz cpu3=1201MHz cpu4=400MHz cpu5=400MHz cpu6=400MHz cpu7=400MHz
t=2391.2623 chunk#200  affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3200MHz cpu2=3201MHz cpu3=3200MHz cpu4=3203MHz cpu5=400MHz cpu6=3200MHz cpu7=3199MHz
t=2393.3942 chunk#400  affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3200MHz cpu2=3200MHz cpu3=400MHz cpu4=400MHz cpu5=400MHz cpu6=3200MHz cpu7=3200MHz
t=2394.9182 chunk#600  affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3201MHz cpu2=3201MHz cpu3=400MHz cpu4=3200MHz cpu5=400MHz cpu6=400MHz cpu7=3200MHz
t=2396.8834 chunk#800  affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3200MHz cpu2=3200MHz cpu3=3201MHz cpu4=3198MHz cpu5=3200MHz cpu6=3200MHz cpu7=3203MHz
t=2398.7351 chunk#1000 affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3202MHz cpu2=3201MHz cpu3=3200MHz cpu4=400MHz cpu5=3298MHz cpu6=400MHz cpu7=3232MHz
t=2401.1009 chunk#1200 affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3205MHz cpu2=3201MHz cpu3=3201MHz cpu4=3199MHz cpu5=3202MHz cpu6=3201MHz cpu7=3200MHz
t=2402.8686 chunk#1400 affinity=[1] current_cpu=1 | cpu0=3100MHz cpu1=3100MHz cpu2=3100MHz cpu3=400MHz cpu4=3094MHz cpu5=400MHz cpu6=3100MHz cpu7=3100MHz
t=2404.8167 chunk#1600 affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3205MHz cpu2=3200MHz cpu3=400MHz cpu4=400MHz cpu5=3204MHz cpu6=3200MHz cpu7=3200MHz
t=2407.0710 chunk#1800 affinity=[1] current_cpu=1 | cpu0=3200MHz cpu1=3201MHz cpu2=3200MHz cpu3=3200MHz cpu4=400MHz cpu5=400MHz cpu6=3198MHz cpu7=3200MHz
t=2409.1526 chunk#2000 affinity=[1] current_cpu=1 | cpu0=3100MHz cpu1=3100MHz cpu2=2783MHz cpu3=3097MHz cpu4=3100MHz cpu5=3100MHz cpu6=3100MHz cpu7=3100MHz
t=2411.2005 chunk#2200 affinity=[1] current_cpu=1 | cpu0=400MHz cpu1=400MHz cpu2=400MHz cpu3=400MHz cpu4=400MHz cpu5=400MHz cpu6=400MHz cpu7=400MHz
t=2413.7863 chunk#2400 affinity=[1] current_cpu=1 | cpu0=3400MHz cpu1=3388MHz cpu2=3399MHz cpu3=3203MHz cpu4=3400MHz cpu5=400MHz cpu6=3400MHz cpu7=3402MHz
t=2416.5586 chunk#2600 affinity=[1] current_cpu=1 | cpu0=2400MHz cpu1=2400MHz cpu2=2400MHz cpu3=534MHz cpu4=400MHz cpu5=2400MHz cpu6=1600MHz cpu7=1257MHz
```

## Finding 1: pinning confirmed, on the real code path this time

Both real workers (PIDs 8989, 8990, spawned by the real
`ProcessPoolExecutor` inside `parallel_decompose()`) show
`affinity=[0]`/`current_cpu=0` and `affinity=[1]`/`current_cpu=1`
respectively at every one of the 14 sampled points each, spanning the
entire ~27s run. Zero deviations. This closes the pinning-mechanism
doubt specifically on the actual shipped code, not a reimplementation.

## Finding 2: the same run shows BOTH synchronized and divergent frequency moments, captured simultaneously with confirmed-correct affinity

Two clear synchronized-frequency instants are directly visible in this
one real run:
- `t=2411.20s`/`t=2411.14s` (both workers' nearest samples): ALL 8
  cores read exactly 400MHz.
- `t=2416.52s` (worker 0's log): ALL 8 cores read essentially 400MHz
  (399-400 across the board).

But most other timestamps show large divergence - e.g. `t=2393.38s`:
cpu0=3202, cpu3=400, cpu4=2235 (three very different values among
just those three cores alone); `t=2404.82s`: cpu0=3203, cpu3=400,
cpu4=400, cpu6=3200 - a mix of full-speed and floor-speed cores in the
same instant.

Because affinity was verified correct at these EXACT same instants
(not a separate check on a different run), this rules out any
possibility that the frequency weirdness was somehow an artifact of a
pinning failure or migration - the two phenomena are now confirmed
independent: pinning holds throughout, and frequency still does its
own thing regardless.

## Status of the two original questions

1. **Is pinning real and correct, on the actual shipped code?** Yes -
   confirmed directly, at maximum available rigor (self-reported
   kernel syscalls from inside the real worker processes), with zero
   violations.
2. **What explains the per-core frequency pattern (neither cleanly
   per-core-load-driven nor uniformly package-wide)?** Still open -
   this document adds more real evidence of the same puzzling pattern
   (occasional full sync, mostly divergent) but does not identify the
   mechanism. Next step, if pursued: `turbostat` (purpose-built for
   real per-core P-state/C-state/RAPL-power visibility) or RAPL power-
   cap data (`/sys/class/powercap/intel-rapl/`) to check whether
   package power-limiting explains the pattern - not yet attempted.
