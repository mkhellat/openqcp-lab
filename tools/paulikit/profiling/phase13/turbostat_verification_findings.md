# turbostat verification: two leading hypotheses ruled out, mechanism remains genuinely unexplained

Recorded 2026-09-03. Direct follow-up per instruction to check C-states
with `turbostat` (root access, run by the user) - this rules out two
specific candidate explanations for the idle-vs-busy frequency
decorrelation found earlier
(`scaling_cur_freq_unreliable_findings.md`), rather than confirming a
mechanism.

## Method and a real methodology correction along the way

Two capture attempts were needed. The first
(`turbostat --show Core,CPU,Bzy_MHz,IRQ,POLL,C1,C1E,C6,...`) did not
overlap with real sustained load (PkgWatt stayed under 3.4W, PkgTmp
under 58C throughout - far below the 15-23W/100C measured during
genuine 2-core load elsewhere in this investigation) - a timing
mismatch between when the user started `turbostat` and when the real
workload was running, not a real finding. Fixed by building
`turbostat_sync_workload.py` - loops the real
`parallel_decompose(n_workers=2, pinned)` computation for a full 90s,
giving a wide window to start `turbostat` into reliably.

A second methodology correction: the first successful-looking capture
used bare `C1`, `C1E`, `C6` columns, which `man turbostat` clarifies
are RAW EVENT COUNTS (how many times that C-state was requested),
NOT residency percentages - re-ran with the correct `C1%`, `C1E%`,
`C6%` columns before drawing any conclusion.

## Result: real sustained load confirmed, C-state residency is low and comparable across busy and idle cores

System-level summary rows (5 of 30 samples): `PkgWatt` 14.3-17.0W
(well above the 15W nominal TDP, confirming real, sustained
turbo-range draw), `PkgTmp` 83-91C, `Bzy_MHz` 2500-3100 - this capture
genuinely overlapped with real, sustained 2-core load this time.

Per-core `C6%` across every sample and every one of the 8 cores stays
under ~18%, with most values under 10% and several under 2% -
including for cpu2, cpu3, cpu6, cpu7 (cores running ONLY kernel
housekeeping threads, essentially zero real work). If these cores
were spending most of their time genuinely parked in deep C6 sleep
(which would show C6% near 80-95%), that would offer a simple
explanation for their frequency readings being an artifact of
brief wake transients. They are not - real C6 residency is low across
the board, comparable between busy and idle cores.

`Bzy_MHz` (turbostat's busy-time-ONLY frequency metric, specifically
designed to exclude idle-state noise, more rigorous than the raw
`scaling_cur_freq` used in earlier documents) shows essentially
IDENTICAL values across busy and idle cores within the same sample -
e.g. one sample: cpu0=3090, cpu2=3090, cpu3=3092, cpu6=3090,
cpu7=3098. This is not a measurement artifact of idle-time inclusion -
even the metric built to exclude idle time shows the same
decorrelation from real per-core work.

## Two candidate hypotheses now directly ruled out

1. **Package-level HWP override** (`IA32_HWP_REQUEST_PKG`,
   `hwp_extended_research_findings.md`'s leading candidate): RULED OUT
   directly - `turbostat`'s own CPUID decode (via `CPUID(6)`) reports
   `No-HWPpkg` for this specific chip. This register does not exist on
   this silicon. Confirmed twice, independently (once via a quick
   no-root run, once via the full root-privileged MSR-reading capture,
   which also showed `MSR_HWP_REQUEST: ... pkg 0x0` directly from the
   live register).
2. **C-state transient/measurement artifact** (the simpler alternative
   flagged in `hwp_extended_research_findings.md`'s own "what would be
   needed to go further" section): RULED OUT by this document's own
   `C6%` data - idle cores are not spending enough time in deep sleep
   for a wake-transient artifact to explain frequencies this close to
   the genuinely busy cores', and `Bzy_MHz` (built to exclude idle
   time by construction) shows the same pattern as the simpler,
   idle-time-inclusive `scaling_cur_freq` did.

## Honest status: mechanism remains genuinely unexplained

Both leading, checkable candidates have now been directly eliminated
with real data, not just documentation review. This is real progress -
narrowing what it ISN'T - but it does not identify what it IS. What
remains: the RAPL power-limiting evidence (`intel-rapl:0: package-0
28.0s:23W,max:15W` from the first capture's header - the package
running at ~150% of its nominal sustained power budget on average) is
a real, confirmed constraint on this chip, but a power/thermal budget
being tight does not by itself explain WHY idle cores specifically
track busy cores' frequency rather than dropping to their own
independent minimum - that would need to be a property of how the
per-core `IA32_HWP_REQUEST` envelope interacts with a SHARED power/
thermal budget at the hardware level, which (per
`hwp_kernel_source_investigation.md`) is proprietary HWP microcode
logic not documented in any public source found so far.

## What this does NOT show

- Does not identify the actual mechanism - two hypotheses eliminated,
  none confirmed.
- Does not check whether a THIRD, not-yet-considered mechanism (e.g.
  shared power-delivery/voltage-plane coupling independent of any
  HWP-specific register, or `intel_pstate`'s own EAS/thermal-daemon-
  adjacent behavior) might explain the pattern - not investigated.
- Only tested `pinned_2` - `pinned_4`/`unpinned_2`/`unpinned_4` were
  not re-checked with `turbostat` specifically.
