# Thermal-controlled re-test: n_workers=4 pinning regression holds, n_workers=2 does not

Recorded 2026-09-02. Direct follow-up to `dvfs_thermal_confound_findings.md`'s
discovery that all prior wall-clock measurements were taken under
uncontrolled thermal state (package temp reaching 100C mid-run, no
cooldown between back-to-back runs). Per direct instruction ("log
thermal state as a covariate, add cooldown between runs"),
`thermal_controlled_welch_ttest.py` waits until package temp drops
below 55C before each timed run and logs starting/mean/max temperature
for every run.

## Important limitation of the "control," stated honestly

The cooldown successfully normalizes STARTING temperature - every one
of the 20 runs below began at 51-55C. But every single run, in EVERY
condition, still heats up to 91-95C mean / 99-100C max DURING the run
itself - the chip hits its thermal ceiling internally on every run
regardless of starting point. So this is a control on starting
condition, not a throttling-free measurement - real in-run throttling
still happens in every condition tested. This is a genuine, remaining
limitation of running this comparison on a 15W-TDP laptop CPU under
sustained multi-core load, not fully resolved by the cooldown alone.

## Raw data (5 reps each, cooldown before every run)

**`pinned_2`**: elapsed = [25.5659, 23.4198, 23.4426, 22.0875, 22.2080]s;
temp_before = [51.0, 55.0, 55.0, 55.0, 55.0]C;
temp_mean = [87.9, 91.1, 93.1, 94.2, 94.8]C

**`unpinned_2`**: elapsed = [22.6317, 22.4731, 21.5605, 21.7064, 21.8425]s;
temp_before = [55.0, 55.0, 55.0, 55.0, 55.0]C;
temp_mean = [89.5, 92.1, 92.5, 92.7, 93.5]C

**`pinned_4`**: elapsed = [24.3901, 24.0863, 23.7664, 23.6127, 23.7804]s;
temp_before = [55.0, 55.0, 55.0, 55.0, 55.0]C;
temp_mean = [92.4, 93.4, 93.4, 95.0, 94.2]C

**`unpinned_4`**: elapsed = [23.4518, 23.4199, 22.3794, 22.8287, 23.1783]s;
temp_before = [54.0, 55.0, 55.0, 55.0, 55.0]C;
temp_mean = [91.6, 91.6, 93.7, 94.4, 94.5]C

(Max temp for every single run in every condition: 99-100C - omitted
from the per-run listing above since it is constant across the whole
dataset.)

## Results

| comparison | mean pinned | mean unpinned | diff | 95% CI | p | Cohen's d | significant? |
|---|---|---|---|---|---|---|---|
| n_workers=2 | 23.34s (sd=1.40) | 22.04s (sd=0.48) | -1.30s | [-3.01, +0.41] | 0.107 | -1.25 | **No** |
| n_workers=4 | 23.93s (sd=0.31) | 23.05s (sd=0.45) | -0.88s | [-1.45, -0.30] | **0.0088** | -2.26 | **Yes** |

## Interpretation

The `n_workers=4` pinning regression (unpinned faster than pinned) is
now the most trustworthy version of this finding across three
methodological iterations of this investigation (uncontrolled ->
pooled 15-rep -> thermal-controlled) - it reproduces with a properly
normalized starting thermal condition and remains statistically
significant (p=0.0088). No mechanism for it is confirmed; it remains
a real, open puzzle.

The `n_workers=2` comparison is NOT significant under thermal control
(p=0.107), reversing the pooled-15-rep result. Since max in-run
temperature is essentially identical across pinned_2/unpinned_2 (both
reach 99-100C), this is less likely to be explained by a leftover
starting-temperature difference and more likely reflects either a
genuinely smaller/harder-to-detect effect at n_workers=2, or that the
earlier uncontrolled pooled result was itself partly an artifact of
uncontrolled thermal history after all - both are plausible given the
data in hand, and this document does not have enough evidence to
choose between them.

## Absolute wall-clock note

All conditions in this thermal-controlled run are notably FASTER in
absolute terms (21.6-25.6s) than the earlier uncontrolled runs
(22.5-31.0s across the three prior 5-rep sets) - consistent with
per-run thermal state being a real driver of absolute wall-clock time
generally (a chip that starts cooler completes more total work before
hitting its throttle ceiling), independent of the pinned-vs-unpinned
question specifically.

## What this does NOT show

- Does not eliminate in-run throttling (every run still reaches
  99-100C) - only starting-state variance was controlled.
- Does not identify why n_workers=4 shows a robust effect while
  n_workers=2 does not - both were subject to the same cooldown
  protocol and reach comparable max temperatures.
- Does not test whether a longer cooldown (fully idle, not just
  below 55C) or a shorter/smaller workload (avoiding the 99-100C
  ceiling entirely) would change either result - not yet tried.
- Does not identify a mechanism for the confirmed n_workers=4
  regression - still genuinely open.
