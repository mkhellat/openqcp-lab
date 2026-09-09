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

## The qubit-boundary step is `dim`, not chunk count

Resolved 2026-09-09. At the boundary three variables move together, so
"dim doubles" was never a single hypothesis. Two were already
controlled by the data above, and the third was tested directly.

| cell | dim | buffer/worker | chunks | terms/chunk | efficiency |
|---|---|---|---|---|---|
| N=150 cs=2 | 16384 | 512 KiB | 5,595 | 16,381 | **72.4%** |
| N=180 cs=1 | 32768 | 512 KiB | 16,120 | 16,382 | **65.2%** |
| N=180 cs=2 | 32768 | 1024 KiB | 8,060 | 32,764 | **62.2%** |

- **Per-worker buffer: refuted.** The auto-tuner halves `chunk_size` at
  15 qubits precisely to hold it at 512 KiB, so the first two rows
  share the putative cause and differ in effect.
- **Terms per chunk: never differed across the boundary.** 16,381 at
  N=150 against 16,382 at N=180, so it cannot explain the step. It does
  double in the third row (32,765), which is the price of using
  `chunk_size` as the lever - see the caveat below.
- **Chunk count: refuted by the third row.** Forcing `chunk_size=2` at
  N=180 halves the chunk count (16,120 -> 8,060), closing the gap with
  N=150 from 2.9x to 1.44x. Efficiency did not recover - it fell
  slightly, to 62.2% (Welch p=5.8e-08, spreads 1.03-1.06x). If chunk
  count drove the step, this cell should have moved toward 72%.

**`dim` is what remains, and the mechanism is coherent.** Each chunk
gathers and scatters across a row of `dim` complex128 entries: 256 KiB
at 14 qubits, 512 KiB at 15. That footprint scales with `dim`
regardless of `chunk_size`, so the tuner cannot compensate for it.
Against a 256 KiB per-core L2, a 14-qubit row fits and a 15-qubit row
does not.

Caveat: `chunk_size` is the only lever that moves chunk count, and
moving it also doubles the per-worker buffer (to 1024 KiB, 50% of L3
in aggregate at four workers) and doubles terms per chunk. So the
third row is not a clean single-variable change, and a perfectly
isolated test of chunk count is not available through this knob.

What makes the conclusion hold anyway is the DIRECTION of the failure.
If chunk count drove the step, halving it should have moved efficiency
toward 72%. It moved the other way, to 62.2%. Every confound
introduced alongside it would have to be not merely present but
strong enough to mask a recovery AND overshoot it - and the two
confounds (bigger buffer, more terms per chunk) both push in the same
direction the result already went. Chunk count is refuted as the
explanation for the boundary step; whether it has a small effect of
its own is not settled here.
