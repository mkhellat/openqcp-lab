# Physical-core scaling, replicated

Recorded 2026-09-09 by `core_scaling_replicated.py`. **This supersedes
`core_efficiency_findings.md` entirely** - every efficiency number in
that document was computed from n=1 per cell and every one of them was
wrong, some by 40 percentage points. Raw data:
`core_scaling_replicated_results.jsonl` (48 runs).

Protocol: one worker per PHYSICAL core, 6 reps per cell, conditions
INTERLEAVED (w1, w4, w1, w4, ...) rather than blocked, cooldown to 55C
before every timed run, first rep discarded as warm-up. No
checkpointing. Every run started at 54-55C; none started above target.

## Result

| N | qubits | dim | cs | w1_c1 | w4_c4 | speedup | efficiency | adversarial | Welch p |
|---|---|---|---|---|---|---|---|---|---|
| 150 | 14 | 16384 | 2 | 23.35s +-0.97 | 8.07s +-0.11 | 2.894x | **72.4%** | 67.7% | 3.0e-06 |
| 160 | 14 | 16384 | 2 | 28.11s +-0.75 | 9.31s +-0.36 | 3.020x | **75.5%** | 69.8% | 7.3e-09 |
| 180 | 15 | 32768 | 1 | 72.87s +-3.63 | 27.94s +-1.02 | 2.608x | **65.2%** | 56.4% | 3.1e-06 |
| 200 | 15 | 32768 | 1 | 87.22s +-3.11 | 33.36s +-0.76 | 2.614x | **65.4%** | 60.5% | 9.1e-07 |

All spreads within a cell are <=1.13x. "Adversarial" is min(w1) /
max(w4) - the most hostile reading of the same data, and the number to
quote to a skeptic.

**Efficiency tracks QUBIT COUNT, not N.** Within 14 qubits it does not
decline (N=160 is slightly *higher* than N=150); within 15 qubits it is
flat to 0.2 points. Across the boundary it steps down ~9 points, from
~74% to ~65%.

**Four physical cores deliver 2.6-3.0x throughout.** There is no
degradation with more cores at any size measured, and no superlinearity
at any size either.

## The measurement fault that produced three wrong answers

Every earlier version of this table was built on n=1 cells. The
`w4_c4` runs were always stable (spread 1.03-1.09x), so nothing looked
wrong - but the `w1_c1` BASELINE, which is the numerator of every
efficiency figure, is badly unstable on its first run of a session.

N=150's `w1_c1` sequence, in order:

```
rep0: 29.79s   <- first run of the session
rep1: 24.11s
rep2: 24.06s
rep3: 24.00s
rep4: 22.38s
rep5: 22.22s
```

Discarding rep0 takes the spread from 1.34x to 1.09x and the sd from
2.77 to 0.97. Every inflated baseline quoted in this investigation -
30.37s, 37.49s, 115.85s, 131.59s - was a first-run-of-session
measurement. There is also a mild continuing drift after warm-up
(24.1 -> 22.2s), which is why conditions are now INTERLEAVED: in a
blocked design that drift lands entirely on whichever condition ran
second and manufactures an effect.

Three successive "corrections" were each made from another n=1
measurement, so each replaced one artifact with another:

| N | reported over time | replicated |
|---|---|---|
| 150 | 113.2% -> 88% -> 73.4% | **72.4%** |
| 160 | 102.3% | **75.5%** |
| 180 | 73% -> 66.3% -> 87.3% | **65.2%** |
| 200 | 85.9% | **65.4%** |

The 87.3% figure for N=180 was used to RETRACT the 66.3% figure. That
retraction was itself wrong: 66.3% was approximately right, and the
87.3% that replaced it came from a single 115.85s baseline sitting 16
standard deviations above the replicated mean of 72.87s.

**A separate claim also retracted here:** that 55C was "unreachable on
this machine" and had to be raised to 65C. It is reachable - the
machine idles in the mid-40s. Cooldowns only hit their ceiling inside a
dense measurement session, where the next run starts before the
previous run's heat has dissipated. That is sustained load, not a
hardware floor, and the "thermal artifact" story built on it was wrong.
The 55C protocol is restored, and every run in this table met it.

## What still holds from the earlier work

- **Packed hyperthread siblings underperform spread physical cores.**
  `w2_c1` (two threads on ONE core) is not two cores, and phase 13's
  original ladder conflated them. That comparison used consistent
  baselines and survives.
- **`w4_c4` beats `w8_c4`** - four workers on four cores beat eight
  workers on the same four, using ~200 MiB less.
- **Do not raise `chunk_size` above the auto-tuned value.** The trend
  across that sweep was measured consistently even though its absolute
  values came from the bad regime.

## Open

**What causes the ~9-point step at the qubit boundary.** `dim` doubles
there, and it sets the gather/scatter row width independently of
`chunk_size`. Two other explanations are already falsified on their own
evidence (per-worker buffer size; chunk count / IPC volume - see
`core_efficiency_findings.md`, whose falsifications remain valid even
though its efficiency numbers do not). The step is now much smaller
than previously believed, so it may not be worth chasing.
