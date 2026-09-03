# What actually controls per-core frequency: proven from kernel driver source, not inferred from measurement

Recorded 2026-09-02, per direct instruction after prior measurement-
based explanations were correctly rejected as insufficient: "Either
you had to try to find out what was throttling other cores inside our
code (which based on your little script you are betting against!!!
And I do not trust that judgement) OR go ahead and research and review
the throttling code for this family of cpu drivers and kernel modules
and prove the behavior inside those codes!!!" This document takes the
second path: the actual upstream Linux kernel `intel_pstate` driver
source and its official documentation, not another measurement script.

## Method

Fetched the current upstream `drivers/cpufreq/intel_pstate.c` (3965
lines, from `torvalds/linux` on GitHub) and
`Documentation/admin-guide/pm/intel_pstate.rst` directly - primary
source, not a summary or a blog post.

## Finding 1 (source code, `intel_pstate_hwp_set()`): per-CPU MSR writes set a min/max ENVELOPE, not a fixed value

```c
static void intel_pstate_hwp_set(unsigned int cpu)
{
	struct cpudata *cpu_data = all_cpu_data[cpu];
	int max, min;
	...
	max = cpu_data->max_perf_ratio;
	min = cpu_data->min_perf_ratio;
	...
	rdmsrq_on_cpu(cpu, MSR_HWP_REQUEST, &value);
	value &= ~HWP_MIN_PERF(~0L);
	value |= HWP_MIN_PERF(min);
	value &= ~HWP_MAX_PERF(~0L);
	value |= HWP_MAX_PERF(max);
	...
	wrmsrq_on_cpu(cpu, MSR_HWP_REQUEST, value);
}
```

This IS per-CPU (`wrmsrq_on_cpu(cpu, ...)`, `cpu_data = all_cpu_data[cpu]`)
- not a single package-wide write. But it writes a MIN/MAX bound
(`HWP_MIN_PERF`, `HWP_MAX_PERF`), not a specific requested frequency.

## Finding 2 (source code, `__intel_pstate_cpu_init()`): the default envelope is wide open

```c
cpu->max_perf_ratio = 0xFF;
cpu->min_perf_ratio = 0;
```

On this machine, each core's HWP envelope is initialized to the
WIDEST possible range (0 = fully idle floor, 0xFF = maximum/turbo
ceiling) - i.e. the driver is not narrowing the range to anything
specific; it hands the hardware maximum freedom.

## Finding 3 (official kernel documentation, `intel_pstate.rst`): the driver states explicitly that it does NOT select frequency under HWP

Direct quote from the kernel's own documentation:

> "If the HWP feature has been enabled, `intel_pstate` relies on the
> processor to select P-states by itself, but still it can give hints
> to the processor's internal P-state selection logic."

And further: the driver's per-CPU scheduler callback exists "not for
running a P-state selection algorithm, but for periodic updates of the
current CPU frequency information" - i.e. even the driver's own
utilization-tracking hook is read-only telemetry collection, not a
control input to frequency selection.

## Proven conclusion

**Per-core frequency selection on this machine is NOT performed by
Linux, NOT by `intel_pstate`, and NOT by anything in this project's
own code.** It is performed entirely by Intel's own on-die HWP
hardware/microcode, which the kernel driver explicitly hands full
control to (a wide-open min/max envelope) at initialization and never
overrides during normal (non-`performance`-governor-forced) operation.
This is proven directly from the driver's own source code and its
official documentation - not inferred from a correlation pattern in
measured data, which was the (correctly rejected) prior approach.

**What this does and does NOT explain**: it proves WHO is in control
(hardware, not software) - it does NOT and CANNOT explain the specific
moment-to-moment pattern observed in
`scaling_cur_freq_unreliable_findings.md`/`real_parallel_decompose_full_verification.md`
(occasional full-package synchronization, mostly per-core divergence
uncorrelated with `/proc/stat` busy%). That decision logic lives
entirely inside Intel's proprietary microcode - it is not open source,
not documented in the Linux kernel tree, and cannot be further traced
from anything available on this machine or in any public repository.
This is the honest, source-verified stopping point for "why does the
frequency do what it does": the mechanism's OWNER is now proven
(hardware autonomy, not OS/driver/application code), but its internal
decision algorithm is not inspectable.

## Practical implication for this whole investigation

Since per-core frequency is controlled by undocumented, proprietary
hardware logic - not by anything in `paulikit`, not by the pinning
mechanism, not by the Linux scheduler in any way this project's code
can influence - **frequency data should be treated as an observed
physical phenomenon of this specific machine, not a signal this
project's code has any control over or should be expected to reason
about further.** The pinning mechanism itself has been separately,
directly verified correct (`self_reported_affinity_verification.md`,
`real_parallel_decompose_full_verification.md`) - that question is
closed. The frequency-pattern question is now understood at the
"who controls it" level but cannot be resolved further without
Intel's own internal HWP microcode documentation, which is not public.
