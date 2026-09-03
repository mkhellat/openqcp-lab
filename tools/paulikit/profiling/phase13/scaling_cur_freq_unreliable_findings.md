# scaling_cur_freq is NOT a per-core utilization proxy on this machine - HWP makes frequency largely package-wide

Recorded 2026-09-02, prompted by direct, sharp user pushback: "It does
not make sense the average of ALL PHYSICAL CORES being on 3.x GHz
during execution on pinned 2 workers!!! If you would have said 4 of
the logical cores have their clock mean at 3.x GHz, that might have
had some excuse, BUT WHY ALL 8?!!" - correct instinct. This document
confirms the mechanism directly rather than continuing to treat
`scaling_cur_freq` as trustworthy.

## The real question: is a mostly-idle core actually running at the frequency it reports?

Prior documents in this investigation (`freq_scaling_check.py`,
`full_core_observation.py`) reported `scaling_cur_freq` for all 8
cores during a `pinned_2` run and found cpu2-7 (running only kernel
idle-management threads: `cpuhp/N`, `idle_inject/N`, `ksoftirqd/N`,
`migration/N` - essentially no real work) at similar frequencies to
cpu0/cpu1 (the two actual, busy pinned workers). This was interpreted
as "package-wide throttling" without directly checking whether
`scaling_cur_freq` reflects real per-core execution at all.

## Root cause found: this machine runs Intel HWP (Hardware P-States)

`/sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost` exists on
this machine - confirming HWP is active. Under HWP, the CPU's own
firmware, not the OS, autonomously selects each core's operating
frequency in real time; `scaling_cur_freq` is the kernel's
APERF/MPERF-derived ESTIMATE of effective frequency over the last
sampling window, not necessarily a value that scales cleanly with
"how much real work this specific core just did."

## Direct ground-truth check: /proc/stat busy% vs. scaling_cur_freq, same instants

`proc_stat_utilization_check.py` samples REAL per-core CPU utilization
(from `/proc/stat`'s idle/total jiffy counters - the standard
ground-truth Linux utilization metric, independent of any frequency
self-reporting) every 0.5s alongside `scaling_cur_freq`, during a real
`pinned_2` run.

**Mean busy% and mean freq per core across the whole run:**

| cpu | mean busy% | mean freq (MHz) |
|---|---|---|
| cpu0 (worker) | 63.8% | 2516 |
| cpu1 (worker) | 62.8% | 2469 |
| cpu2 | 14.7% | 2455 |
| cpu3 | 13.1% | 2185 |
| cpu4 | 27.8% | 2362 |
| cpu5 | 32.3% | 2463 |
| cpu6 | 24.6% | 2123 |
| cpu7 | 15.9% | 2590 |

**The correlation the user correctly suspected wouldn't hold, doesn't
hold**: cpu0/cpu1 are ~4-5x busier than cpu2-7 (63-64% vs. 13-32%), but
their mean frequencies are all in the same 2100-2600 MHz band -
cpu7 (15.9% busy, among the least-busy cores) reports a HIGHER mean
frequency (2590 MHz) than cpu6 (24.6% busy, notably busier) at 2123
MHz. There is essentially no per-core relationship between actual
utilization and reported frequency in this data.

## Conclusion: scaling_cur_freq is NOT a reliable per-core utilization signal on this machine

Every claim earlier in this investigation that treated
`scaling_cur_freq` as evidence of "this core was doing real work" or
used it to reason about per-core contention is now known to be on
shaky ground - the mechanism, now confirmed via HWP's existence and
the direct busy%-vs-freq mismatch, is that frequency selection under
`intel_pstate`/HWP responds substantially to PACKAGE-level state
(thermal budget, the busiest core's demand, power delivery
constraints), not purely to each core's own individual load. Idle
cores "ride along" at a similar voltage/frequency plane to busy ones,
for reasons intrinsic to how modern Intel mobile SoCs share power
delivery and thermal budget across the die - not a bug in any of this
session's measurement scripts.

**The only trustworthy per-core signal collected in this investigation
is `/proc/stat`'s busy% (or the earlier `ps -o psr` process-placement
data, which is exact by construction)** - it behaves exactly as
expected: pinned workers show high, sustained busy%; unpinned/idle
cores show low busy% throughout. Frequency data should not be used
as a proxy for per-core computational activity on this machine going
forward without this caveat.

## What this does NOT show

- Does not re-derive the earlier DVFS/thermal-throttling conclusions
  (`dvfs_thermal_confound_findings.md`) - the package-temperature
  finding (100C, real thermal throttling) is independent of this
  frequency-reporting caveat and stands on its own (temperature is a
  direct physical measurement, not subject to the same HWP
  reporting-vs-reality gap).
- Does not identify exactly which HWP/firmware mechanism causes the
  package-wide frequency correlation - only that it exists and is
  large enough to defeat any attempt to read per-core frequency as a
  utilization signal.
- Does not re-check whether the earlier pinned/unpinned Welch's
  t-test wall-clock conclusions are affected by this - those used
  wall-clock time directly (a real, physical measurement), not
  frequency data, so they are not invalidated by this finding, but the
  MECHANISM discussion around them (e.g. "L1/L2 sharing" reasoning
  that implicitly assumed per-core frequency was meaningful) should be
  revisited with this caveat in mind.
