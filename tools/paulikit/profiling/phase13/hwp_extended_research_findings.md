# Extended HWP research: a real candidate mechanism found (package-level HWP control), not confirmed as THE cause

Recorded 2026-09-02, delegated to a research subagent per direct
instruction after `hwp_kernel_source_investigation.md` proved WHO
controls per-core frequency (Intel's own HWP hardware, not Linux/our
code) but could not explain WHY the specific idle-vs-busy decorrelation
pattern happens. User asked directly: "do more extensive research...
to see if you can find anything on HWP for this processor family and
see why throttling is happening this way and if anyone has observed
similar thing? Are you sure there is nothing you could find on HWP for
this family?!!" This document reports that research's findings,
distinguishing confirmed fact from plausible-but-unconfirmed hypothesis
throughout, per explicit instruction not to manufacture an explanation
to fill gaps.

## The most load-bearing finding: package-level HWP control genuinely exists in silicon

**Primary source** (Intel SDM Vol. 3B, section 14.4.1, "HWP
Programming Interfaces"): HWP has TWO separate control registers, not
one:
- `IA32_HWP_REQUEST` (MSR 0x774) - per-logical-processor control hints
  (this is the register `intel_pstate_hwp_set()` writes via
  `wrmsrq_on_cpu()`, confirmed in `hwp_kernel_source_investigation.md`).
- `IA32_HWP_REQUEST_PKG` (MSR 0x772) - PACKAGE-WIDE control hints,
  applied to ALL logical processors on the package SIMULTANEOUSLY.
  Availability is signaled by `CPUID.06H:EAX[bit 11]`.

A per-core bit determines whether that specific core is "over-ruled
by" or "exempt from" the package-wide register. This is documented
directly by Intel's own diagnostic tool `x86_energy_perf_policy` (ships
with `linux-tools`, i.e. present on real systems, not theoretical):

> "A bit in per-CPU MSR_IA32_HWP_REQUEST indicates whether it is
> over-ruled-by or exempt-from MSR_IA32_HWP_REQUEST_PKG." The tool's
> own `--hwp-use-pkg` flag "specifies whether the per-cpu
> MSR_IA32_HWP_REQUEST should be over-ruled by MSR_IA32_HWP_REQUEST_PKG
> (1), or exempt from MSR_IA32_HWP_REQUEST_PKG (0)."

**Confidence: high that this mechanism exists in the silicon family.**
This is a real, named, documented hardware capability - not
speculation. It is architecturally consistent with (though not proof
of) the specific pattern measured: a core's frequency could be
influenced by another core's demand THROUGH this exact
package-override register, in a way that would appear as neither
purely-per-core nor purely-uniform-package-wide from the outside -
matching the "sometimes synchronized, mostly divergent" pattern found
in `scaling_cur_freq_unreliable_findings.md`.

**The honest gap**: neither Linux kernel source, kernel documentation,
nor any other source found states which mode (package-overruled vs.
exempt) Kaby Lake R actually defaults to, or whether `intel_pstate`
ever touches `IA32_HWP_REQUEST_PKG` at all for this chip family. This
remains unconfirmed - the mechanism is real and plausible, but NOT
proven to be the actual explanation for what was measured.

## Corroborating (but not identical) academic finding, different chip

Schöne et al., "Energy Efficiency Features of the Intel Skylake-SP
Processor and Their Impact on Performance," IEEE HPCS 2019
(arXiv:1905.12468) - peer-reviewed, secondary but credible. On a
server Xeon (Skylake-SP, NOT the same chip family as this machine's
Kaby Lake R mobile part), they independently found HWP frequency
selection does not track expected work intensity: "even memory-bound
workload run at the maximum allowed frequency, which is not what we
would expect" - to the point that they disabled HWP entirely for their
own measurements. This is NOT the same phenomenon (their busy-but-
stalled threads vs. our genuinely-idle sibling cores), but it is
independent, credible evidence that "HWP frequency does not reliably
track actual per-core work" is a known, real category of surprise in
the CPU architecture research community, not unique to this
investigation or this specific chip.

## What was searched and found nothing on point

- **LKML/kernel bugzilla**: the one superficially-matching historical
  thread (2016, "processor frequency very high even if in idle",
  kernel 4.6-rc1) was a different bug (utilization-callback ordering,
  fixed by commit `a4675fbc4a7a`), not applicable to current kernels
  and not about idle-vs-busy decorrelation specifically.
- **`turbostat` documentation**: precisely documents HOW `Bzy_MHz`/
  `Avg_MHz` are computed (APERF/MPERF/TSC deltas) but contains no
  explanatory prose about HWP's internal per-core-vs-package
  arbitration logic.
- **Hardware-enthusiast deep-technical press** (Chips and Cheese,
  AnandTech, or similar): searched specifically for architecture-level
  analysis of HWP's actual arbitration heuristics - found none. This
  specific algorithm does not appear to be covered at that depth
  anywhere in public hardware journalism.
- **FreeBSD's independent `hwpstate_intel(4)` driver** (a genuinely
  different OS's implementation, useful as independent corroboration):
  confirms the same package/per-core duality exists as a real,
  switchable driver-level control (`machdep.hwpstate_pkg_ctrl`) - not
  new information, but independent confirmation the mechanism is real
  hardware behavior, not a Linux-specific artifact.

## Conclusion, stated at the appropriate confidence level

**Confirmed fact**: Intel's own architecture manual documents a real
package-level HWP override mechanism (`IA32_HWP_REQUEST_PKG`) that
could architecturally produce exactly the kind of
sometimes-correlated, mostly-divergent per-core frequency pattern
measured on this machine.

**NOT confirmed**: that this specific mechanism is actually active or
responsible for the measured pattern on this specific chip
(i7-8550U / Kaby Lake R) under this specific kernel's `intel_pstate`
configuration. No source - Intel documentation, Linux kernel source/
docs, academic literature, or hardware press - connects the dots this
explicitly for this chip family. Per direct instruction not to
manufacture an explanation: this connection is NOT stated as
established fact anywhere found, and this document does not claim it
is - it names the most architecturally plausible confirmed-to-exist
mechanism, while being explicit that its actual role here is
unverified.

**Genuinely undocumented beyond this**: Intel's SDM itself declines to
specify HWP's autonomous decision algorithm internals ("the hardware's
view of workload scalability is implementation specific" - standard
SDM language for "this is proprietary microcode logic we don't
publish"). This is the real, honest stopping point for public-source
research on this question - not a failure to search hard enough, but
a genuine boundary of what Intel has published.

## What would be needed to go further (not pursued here)

- Directly reading `MSR_IA32_HWP_REQUEST_PKG` (0x772) and the per-core
  override bit in `MSR_IA32_HWP_REQUEST` (0x774) via `rdmsr` (from the
  `msr-tools` package) on this specific machine, to at least confirm
  whether the package-override mechanism is architecturally active
  here at all - would require root and reading raw MSRs, not yet
  attempted.
- `turbostat`'s own live output (not just its documentation) during a
  real run, cross-referenced against the `/proc/stat`+`scaling_cur_freq`
  data already collected - `turbostat` reports C-state residency and
  more precise frequency data than `scaling_cur_freq` alone, and might
  show whether idle cores are actually in a deep C-state (which would
  itself explain apparently-elevated "current" frequency readings as
  an artifact of exiting/entering C-states around the sampling
  instant, a different and simpler explanation than package-level HWP
  coupling) - not yet checked.
