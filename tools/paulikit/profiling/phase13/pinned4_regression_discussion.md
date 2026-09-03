# Discussion: why is pinned_4 slower than unpinned_4? Cache misses and throttling both ruled out

Recorded 2026-09-03. Direct discussion following the "let's accept the
throttling mechanism is unexplained and move on" decision - revisiting
what is and isn't actually established about the ONE finding that
survived every round of statistical scrutiny in this investigation:
`pinned_4` is significantly slower than `unpinned_4`
(`thermal_controlled_results.md`: diff=-0.88s, p=0.0088, Cohen's
d=-2.26).

## Question 1 - SUPERSEDED: cache-miss claim did not survive repetition, and the "L3 shared" reasoning was flawed anyway

**Two separate corrections, both from direct user pushback.** First,
the reasoning offered below (originally: "L1/L2 dedication doesn't
help against L3 contention") was itself flawed - the user caught it
directly: L3 pressure is a SHARED, CONSTANT factor between `pinned_4`
and `unpinned_4` (both have all 4 physical cores contending for the
same L3 either way) - it cannot explain a DIFFERENCE between the two
conditions, only something that actually differs between them
(pinned vs. not) can. Second, and more importantly: the underlying
claim this reasoning was trying to explain turned out not to be real
at all. `pinned4_cache_miss_welch_ttest_findings.md` - 5 properly
repeated `perf stat` runs per condition, real Welch's t-test - found
NEITHER cache-miss ratio (p=0.157) NOR LLC-miss ratio (p=0.086) is
statistically significant between `pinned_4` and `unpinned_4`. If
anything, `pinned_4`'s point estimates are numerically slightly LOWER
(better) than `unpinned_4`'s - the opposite direction from the
original single-run claim. This is the THIRD finding in this
investigation to not survive proper repetition, after the
`pinned_2`/`unpinned_2` wall-clock question's own multi-round reversal.

**Corrected conclusion**: cache/LLC-miss ratios are NOT the
explanation for the wall-clock gap - they are statistically
indistinguishable between conditions, while wall-clock is not. This
rules out a cache-level explanation entirely, sharpening rather than
answering the real question (see below).

## Question 2: is throttling severity the explanation for the wall-clock gap? Checked directly - no.

The thermal-controlled test logged temperature as a covariate
specifically to check this. Comparing `pinned_4` vs `unpinned_4`'s
actual per-run temperatures:

| condition | temp_before (all runs) | temp_mean (5 runs) |
|---|---|---|
| pinned_4 | 55.0C (all 5) | 92.4, 93.4, 93.4, 95.0, 94.2 (avg ~93.7C) |
| unpinned_4 | 54-55.0C (all 5) | 91.6, 91.6, 93.7, 94.4, 94.5 (avg ~93.2C) |

Starting temperature is identical by construction (the cooldown
protocol normalizes it to 55C for both). Mean in-run temperature is
nearly identical between the two conditions (~93.7C vs ~93.2C, a
~0.5C difference, well within the run-to-run spread each condition
shows on its own). **If throttling severity were driving the
wall-clock gap, `pinned_4` should show meaningfully higher
temperatures than `unpinned_4` to explain running slower - it does
not.** Both conditions throttle to essentially the same degree.

**Direct answer: no, this was not implicitly assumed, and checking it
directly shows throttling severity is NOT a good candidate explanation
for the pinned_4-vs-unpinned_4 wall-clock gap.** Something else is
causing the difference - the temperature data available does not
support "pinned_4 runs hotter, therefore throttles harder, therefore
runs slower" as the mechanism.

## What remains genuinely open

- The mechanism behind the confirmed wall-clock regression is still
  unexplained. Ruled out so far, each with real data: cache/LLC-miss
  ratios (`pinned4_cache_miss_welch_ttest_findings.md`), throttling
  severity (this document, temperature-covariate check), and the two
  HWP-related hypotheses for the separate idle-vs-busy frequency
  puzzle (`turbostat_verification_findings.md`). The earlier "rigid
  pinning removes scheduler load-balancing freedom" idea remains an
  untested hypothesis, not confirmed or denied by anything collected
  here - and is now the most direct remaining candidate, since the
  only thing that actually differs between `pinned_4` and
  `unpinned_4` is whether the OS scheduler retains freedom to migrate
  each process. Scheduler-level metrics (migration counts, run-queue
  latency, context-switch overhead) have not been measured at all in
  this investigation.
- Per the session's own decision to stop pursuing the HWP root-cause
  question further (option 3: accept as characterized-but-unexplained
  hardware behavior), this document does not attempt further
  mechanism investigation - it only checks and reports on the two
  specific candidate explanations the user asked about directly.
