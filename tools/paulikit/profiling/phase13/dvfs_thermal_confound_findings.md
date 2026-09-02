# DVFS/thermal-throttling confound found in all wall-clock measurements so far

Recorded 2026-09-02, prompted directly by the user's own skepticism
after the pooled 15-rep analysis: "Are you sure we can trust these
numbers?! I feel some neat thing is off. How do you think we can
figure that out?" This document is that investigation's result.

## Investigation: is something physically confounding the measurements?

This machine's own earlier session work
(`cache_probe_extension_findings.md`) already documented that its
`powersave` CPU-frequency governor swings frequency 400MHz-3.3GHz
during measurement, corrupting an earlier `clock_gettime`-based
timing experiment until fixed with hardware cycle counters. The
`pinned`/`unpinned` Welch's t-test work in this investigation used
plain `time.perf_counter()` wall-clock timing throughout, with zero
protection against this - the exact failure mode already known to
exist on this hardware.

## Direct measurement, not assumption

`freq_scaling_check.py`: samples all 8 logical CPUs'
`scaling_cur_freq` every 0.2s throughout a real `parallel_decompose`
run. First check, under the original `powersave` governor:

- Every core swung from 400MHz to ~3700MHz DURING a single run (a
  ~9.25x range), with per-core frequency stdevs of 970-1275MHz.

## Attempted fix #1: switch governor to `performance` - did NOT work

`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
(run by the user, verified directly on all 8 cores afterward). Re-ran
the frequency check: frequency STILL swung 400MHz-4000MHz, stdevs
still 800-1150MHz - essentially unchanged.

**Root cause identified**: this CPU uses the `intel_pstate` driver
(`scaling_driver` = `intel_pstate`), not the generic kernel `cpufreq`
governor framework. Under `intel_pstate`, even `performance` mode only
changes the *target* P-state request to the hardware - Intel's own
firmware still dynamically throttles below that target under real
thermal/power constraints. Setting the governor was necessary but not
sufficient; it does not disable hardware-level throttling.

## Confirmed root cause: real thermal throttling

Added package-temperature logging (`x86_pkg_temp` thermal zone,
confirmed via `/sys/class/thermal/thermal_zone7/type`) alongside
frequency sampling. Result, one real `pinned_2` run under the
`performance` governor: package temperature climbed from 72C to
**100C** (the chip's thermal limit) over the ~23s run, with a full
temperature timeline showing sustained stretches at 95-100C.
Frequency dips visibly track the temperature curve - this is genuine
thermal throttling on a 15W-TDP mobile CPU (i7-8550U) under sustained
multi-core load, not a software/governor misconfiguration.

## Why this matters for every prior pinning comparison in this investigation

This directly implicates the instability seen across the 3 individual
5-rep Welch's t-test runs (`pooled_15rep_pinning_analysis.md`'s own
motivating observation) and the ~8x variance asymmetry between
`pinned_2` and `unpinned_2`: back-to-back benchmark runs with no
cooldown period mean each run's THERMAL STARTING STATE depends on how
hot the machine got during the previous run(s) - a real, uncontrolled
covariate that was never logged or controlled for in any measurement
before this document. This is a serious, real methodological gap, not
a minor caveat - every wall-clock number in this investigation prior
to this point should be treated as measured under uncontrolled
thermal history.

## Fix, per direct instruction

Two controls added together in `thermal_controlled_welch_ttest.py`:
1. **Cooldown between runs**: wait until package temp drops below a
   55C baseline (chosen from direct empirical observation - the chip
   settles to 52-58C within ~90s idle) before starting the next timed
   run, with a 180s timeout safeguard.
2. **Thermal state logged as a covariate for every run**: starting
   temp (before the run), mean temp during the run, and max temp
   during the run - so if timing still correlates with thermal state
   even under the cooldown control, that is now visible and checkable,
   not an invisible confound.

Results of the thermal-controlled re-test are recorded separately
(see the following findings document, once that run completes).
