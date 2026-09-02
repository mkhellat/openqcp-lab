# Pooled 15-rep analysis: pinning regression confirmed, but a DVFS/thermal confound was then discovered (see thermal-controlled re-test)

Recorded 2026-09-02. Direct follow-up after 3 independent 5-rep Welch's
t-test runs showed the `pinned_4`/`unpinned_4` effect reproducing
cleanly but the `pinned_2`/`unpinned_2` effect swinging between
significant (p=0.018, p=0.019) and non-significant (p=0.44) across the
three runs. Per direct instruction ("Pool all 15 reps into one test...
to make sure we have proper stats"), all three runs' raw data was
pooled into one n=15-per-condition Welch's t-test.

**IMPORTANT - read before trusting this document's conclusion**: this
analysis was done BEFORE a serious DVFS/thermal-throttling confound in
the measurement method was discovered (package temperature reaching
100C mid-run, frequency swinging 400MHz-4000MHz even under the
`performance` governor - see the thermal-controlled re-test that
follows this document). The numbers below are real and the pooling
methodology is sound, but the underlying wall-clock measurements may
themselves be confounded by uncontrolled thermal state carried over
between back-to-back runs. This document is kept as the intermediate
step it was, not retracted, but its conclusion should be read as
provisional pending the thermal-controlled re-test's own result.

## Raw data: all 15 reps per condition (3 independent 5-rep runs pooled)

**`pinned_2`** (n=15): 31.0346, 29.0202, 26.7858, 29.2874, 30.0434,
28.4597, 26.9018, 27.8769, 27.1043, 26.4341, 25.3943, 25.9890, 27.2510,
25.7940, 26.4881

**`unpinned_2`** (n=15): 26.4270, 25.5481, 24.8193, 25.6951, 28.8890,
25.7875, 25.7686, 26.4263, 26.0189, 26.2672, 25.9027, 25.8784, 25.9652,
26.1500, 25.6140

**`pinned_4`** (n=15): 29.5095, 28.8330, 27.1887, 27.3204, 27.3382,
27.4671, 27.3254, 27.3153, 27.4533, 27.4403, 27.1024, 27.2382, 27.1692,
27.1296, 27.3358

**`unpinned_4`** (n=15): 26.4864, 26.3334, 26.3521, 26.5503, 26.5010,
26.6112, 26.5234, 26.3534, 26.5198, 26.6371, 25.8623, 26.3728, 26.2734,
26.4305, 26.5186

## Pooled Welch's t-test results

| comparison | mean (pinned) | mean (unpinned) | diff | 95% CI | p-value | Cohen's d |
|---|---|---|---|---|---|---|
| n_workers=2 | 27.59s (sd=1.65) | 26.08s (sd=0.87) | -1.51s | [-2.52, -0.51] | 0.0050 | -1.15 (large) |
| n_workers=4 | 27.54s (sd=0.68) | 26.42s (sd=0.19) | -1.12s | [-1.51, -0.74] | 0.000014 | -2.25 (very large) |

With n=15 per condition, both effects are statistically significant
with large effect sizes. `pinned_2`'s non-significant third individual
run was a smaller-sample artifact regressing toward zero, not evidence
the pooled effect isn't real - pooling correctly recovers the
signal.

## A distinct, separately-noted observation: pinned_2's variance is ~8x higher

`pinned_2`'s stdev (1.65s) is roughly 8x larger than `unpinned_2`'s
(0.87s) in the pooled data - a real, separate finding from the
mean-shift result, prompting the direct question that led to the DVFS/
thermal investigation: "why is pinned_2's variance so much higher...
is this statistically meaningful... do you need to check higher-order
moments?" This variance asymmetry is exactly what motivated checking
whether an uncontrolled physical variable (frequency scaling, thermal
throttling) was contaminating the measurements - which it was (see the
follow-up investigation).

## Status: superseded pending thermal-controlled re-measurement

This document's numbers stand as a real, correctly-computed pooled
statistical result on the data as collected - but that data was
collected without thermal-state control or logging. See
`freq_scaling_check.py`'s findings (package temp reaching 100C
mid-run) and `thermal_controlled_welch_ttest.py` for the corrected
methodology (cooldown between runs, temperature logged as a covariate)
and its own, separately-recorded result.
