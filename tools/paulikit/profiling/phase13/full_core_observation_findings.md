# Verified: pinned_2 correctly locks exactly 2 logical CPUs - an earlier apparent "leak" was a measurement-script bug

Recorded 2026-09-02, prompted by direct, correct user skepticism after
`freq_scaling_check.py`'s per-core frequency data showed similar
activity across ALL 8 cores during a `pinned_2` run, not just the 2
that should be pinned: "The important question is why in hell were
those other 6 cores also active?? How could you be sure that only 2 of
the logical cores were locked in?!! If you wanna rerun you must
closely observe all logical cores!!" - correct, `freq_scaling_check.py`
never actually tracked which PROCESS was on which core, only aggregate
per-core frequency, so it could not answer this question at all.

## First attempt found an alarming, but ultimately false, result

`full_core_observation.py` (first version) combined per-core frequency
with process placement (`ps -o pid,psr,comm`), using `descendant_pids`
(ALL descendants of the root process, recursively) to identify
"WORKER" processes. Result: `WORKER` appeared not just on cpu0/cpu1
but intermittently on cpu2 through cpu6 as well, throughout the run -
appearing to show the pinning mechanism leaking workers onto
unintended cores.

## Root cause: the monitoring script's OWN subprocess calls were mislabeled as workers

Directly verified via a minimal reproduction: launched a real
`parallel_decompose(n_workers=2)` run and dumped the process tree
(`ps --ppid <root>`) at t=3s. Found exactly 3 direct children of the
root process: 2 real Python worker processes (pinned to cpu0 and cpu1
respectively, confirmed via their `psr` field) - and a THIRD, a `ps`
process itself (the monitoring thread's own `subprocess.run(["ps",
...])` call, which is itself a transient child of the root process),
placed by the scheduler on cpu7 in that snapshot.

`full_core_observation.py`'s `descendant_pids()` counted ALL
descendants, including these transient `ps`/`subprocess.run` children
the monitor spawns every single sample - so the script was
intermittently mislabeling its own measurement tooling as a "WORKER"
appearing on whatever core the scheduler happened to place that
short-lived `ps` invocation. This was a real, self-inflicted false
positive in the measurement script, not evidence of an actual pinning
failure.

## Fix and re-verification

Replaced `descendant_pids` (all descendants) with
`direct_python_children` (only DIRECT children of the root process
whose command name is `python` - i.e. the actual pool workers, never
the monitor's own `ps` subprocess). Re-ran `pinned_2`:

- `cpu0`: worker present in 98/99 samples (mean freq while present:
  3387 MHz)
- `cpu1`: worker present in 98/99 samples (mean freq while present:
  3412 MHz)
- `cpu2` through `cpu7`: worker present in **0/99 samples** - zero
  worker activity on any of the 6 unpinned cores, for the entire run.

**Confirmed: `pinned_2` correctly locks exactly logical CPU 0 and
logical CPU 1 (physical cores A and B) - the pinning mechanism works
exactly as designed.** The apparent "leak" in the first attempt was
entirely a measurement-script artifact, not a real defect in
`_pin_current_process_to_cpu`/`_parallel_worker_init`.

## What cpu2-7 actually showed (real, not a bug)

Even with zero worker presence, cpu2-7 still showed real, elevated
frequency (3230-3354 MHz mean) throughout the run - what's actually
running there is normal per-CPU Linux kernel infrastructure threads
that exist on every core regardless of workload (`cpuhp/N`,
`idle_inject/N`, `migration/N`, `ksoftirqd/N`, per-CPU `kworker`
threads) plus a few incidental unrelated system processes
(`dbus-broker`, `systemd-userworker`, `wrapper-2.0`). Their elevated
frequency despite doing essentially no real work is consistent with
this machine's frequency scaling being at least partly PACKAGE-wide,
not purely per-core-load-driven - the whole chip's frequency/power
state responds to cpu0/cpu1's real work, not just those two cores'
own local load, matching this investigation's earlier DVFS/thermal
findings (`dvfs_thermal_confound_findings.md`).

## What this does NOT show

- Does not re-verify `pinned_4`, `unpinned_2`, or `unpinned_4` with
  this corrected script - only `pinned_2` was re-checked. `unpinned_2`
  in particular would be expected to show worker presence spread
  across more cores (by design - pinning is disabled), which should
  be verified directly rather than assumed given how wrong the
  informal assumption about `pinned_2` turned out to be.
- Does not explain why cpu2-7's frequency, while elevated, was
  slightly lower on average (~3230-3354 MHz) than cpu0/cpu1's
  worker-present frequency (~3387-3412 MHz) - consistent with, but not
  proof of, a package-wide-plus-per-core-load combination rather than
  pure package-wide uniformity.
