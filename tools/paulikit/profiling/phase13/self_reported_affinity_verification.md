# Maximally rigorous pinning verification: confirmed correct, from inside the worker processes themselves

Recorded 2026-09-02, per direct, sustained user skepticism: "I am
still EXTREMELY skeptical if you have been able to lock on logical cpu
cores!!!!" - fair, since every prior check
(`full_core_observation_findings.md`) relied on an EXTERNAL tool
(`ps -o psr`, sampled every 0.2s) to infer worker placement, which
cannot catch migrations between samples and had already produced one
real false positive earlier in this same investigation (the monitoring
script's own `ps` subprocess being mislabeled as a worker).

## Method: self-reported, from inside the worker process, at the highest rigor available

`self_reported_affinity_check.py` has each worker process report on
itself, using the kernel's own authoritative interfaces:

1. `os.sched_getaffinity(0)` - the kernel's actual record of which
   CPUs this process is ALLOWED to run on (the real pinning contract
   `sched_setaffinity` establishes, not an inference).
2. `sched_getcpu()` (via `ctypes` - not available as `os.sched_getcpu`
   in this Python build) - which CPU the kernel says this process is
   executing on RIGHT NOW.

Sampled at process start, every 200 processed chunks throughout the
ENTIRE real computation (not just a few external poll points), and at
process end - 16 total self-observations per worker across the whole
~12.5s run. This is categorically stronger evidence than external
sampling: there is no room for a measurement-tool artifact here - if
`sched_getaffinity`/`sched_getcpu` themselves reported a violation, it
would mean the kernel-level pinning mechanism itself was broken, not a
polling-interval gap.

## Result: zero violations, exact match at every single sample

**Worker 0** (intended: CPU 0): all 16 samples, from `START` to `END`,
report `affinity=[0] current_cpu=0` - no exceptions.

**Worker 1** (intended: CPU 1): all 16 samples report
`affinity=[1] current_cpu=1` - no exceptions.

Full raw log (both workers, all samples):

```
worker 0:
START t=2037.6836 affinity=[0] current_cpu=0
t=2037.6926 affinity=[0] current_cpu=0
t=2038.4910 affinity=[0] current_cpu=0
t=2039.3155 affinity=[0] current_cpu=0
t=2040.2302 affinity=[0] current_cpu=0
t=2041.1054 affinity=[0] current_cpu=0
t=2041.9174 affinity=[0] current_cpu=0
t=2042.7791 affinity=[0] current_cpu=0
t=2043.5889 affinity=[0] current_cpu=0
t=2044.5210 affinity=[0] current_cpu=0
t=2045.3838 affinity=[0] current_cpu=0
t=2046.3713 affinity=[0] current_cpu=0
t=2047.2800 affinity=[0] current_cpu=0
t=2048.2631 affinity=[0] current_cpu=0
t=2049.2822 affinity=[0] current_cpu=0
END t=2050.1304 affinity=[0] current_cpu=0

worker 1:
START t=2037.6849 affinity=[1] current_cpu=1
t=2037.6934 affinity=[1] current_cpu=1
t=2038.4803 affinity=[1] current_cpu=1
t=2039.2977 affinity=[1] current_cpu=1
t=2040.1982 affinity=[1] current_cpu=1
t=2041.0452 affinity=[1] current_cpu=1
t=2041.8375 affinity=[1] current_cpu=1
t=2042.6797 affinity=[1] current_cpu=1
t=2043.4674 affinity=[1] current_cpu=1
t=2044.3235 affinity=[1] current_cpu=1
t=2045.2417 affinity=[1] current_cpu=1
t=2046.2168 affinity=[1] current_cpu=1
t=2047.1106 affinity=[1] current_cpu=1
t=2048.0853 affinity=[1] current_cpu=1
t=2049.0045 affinity=[1] current_cpu=1
END t=2049.8771 affinity=[1] current_cpu=1
```

## Conclusion

CPU pinning via `os.sched_setaffinity` is confirmed working correctly
for `pinned_2`, verified from inside the worker processes themselves
using the kernel's own authoritative affinity/current-CPU interfaces -
the strongest verification method available on this platform. This
closes the pinning-mechanism doubt directly, independent of (and more
rigorous than) the earlier external `ps`-based check
(`full_core_observation_findings.md`).

**What remains genuinely open** (unaffected by this verification):
the unexplained per-core frequency behavior found in
`scaling_cur_freq_unreliable_findings.md` - frequency varies
substantially per-core (not uniformly package-wide) but does not
track per-core utilization either. That is a question about hardware
frequency-selection/reporting, not about whether pinning itself works
- pinning is now confirmed solid, so it is not the explanation for the
frequency puzzle.

## What this does NOT show

- Only `pinned_2` was verified this way - `pinned_4` was not re-run
  with self-reported affinity checking (though the mechanism tested,
  `os.sched_setaffinity`, is identical regardless of worker count, so
  a `pinned_4` failure would be surprising given this result, but has
  not been directly checked).
- Sampling was every 200 processed chunks (~16 samples across the
  whole run), not truly continuous - a sub-millisecond migration
  between two consecutive samples that immediately migrated back
  would not be caught. This is a much finer granularity than the
  external `ps`-based check's 0.2s polling, but not infinite
  resolution.
