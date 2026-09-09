# The single-worker baseline has a warm-up band, and it corrupted every efficiency figure

Recorded 2026-09-09 by `w1_core_rotation.py`. Raw data:
`w1_core_rotation_results.jsonl`.

## The problem

`w1_c1` at N=150 measured ~22-25s in some contexts and ~30-31s in
others. Since that number is the NUMERATOR of every parallel-efficiency
figure, the ambiguity moves efficiency between 72% and 93% - so no
efficiency number could be quoted until it was resolved.

Two explanations were falsified first:

- **"fast only when a `w4_c4` ran just before it"** - four consecutive
  `w1_c1` runs with no `w4_c4` anywhere still gave 24.83-25.33s.
- **"a long idle gap makes it slow"** - a deliberate 120s idle cost
  only ~2.5s (25.77s vs 23.48s), not the ~6s the bands differ by.

## The test: rotate the pinned core

Every earlier `w1_c1` measurement pinned to cpu0, so "which core" was a
constant and could not be seen. That matters because cpu0 is not
interchangeable on Linux - measured on this machine it takes **12.2M
interrupts against 3.8-5.5M for every other CPU**, a 2.7x skew. A run
pinned there competes with interrupt handling in a way a run on cpu3
does not.

So: 3 reps on each of cpus 0-3 (one hyperthread of each physical
core), rotating, recording wall time, CPU time, and context switches.

## Result: it is the FIRST RUN, not the core

| cpu | mean | sd | min | max | invol ctx sw |
|---|---|---|---|---|---|
| 0 | 26.46s | 3.77 | 24.25 | **30.82** | 1517 |
| 1 | 25.02s | 1.69 | 23.56 | 26.87 | 806 |
| 2 | 24.52s | 0.46 | 24.20 | 25.05 | 565 |
| 3 | 24.60s | 0.80 | 24.08 | 25.52 | 3659 |

The slow band belongs to the first runs of the session and decays
across them, independent of core:

```
cpu0 rep0: 30.82s   <- first run of the whole session
cpu1 rep0: 26.87s
cpu2 rep0: 25.05s
cpu3 rep0: 24.08s
cpu0 rep1: 24.25s   <- SAME core, 6.6s faster than its rep0
```

Excluding rep0, every core agrees: **24.10, 24.25, 24.28, 24.86s**,
overall 23.56-25.52s at sd 0.55 and spread 1.08x.

**Core asymmetry is real but negligible.** cpu0 does carry more
interrupt load (1172 involuntary switches vs cpu2's 519 excluding
rep0), but costs ~0.05s for it. And the interrupt story is refuted
outright by cpu3: it has by far the MOST involuntary context switches
(5300, 10x cpu2) yet is within 0.6s of the fastest. If preemption drove
the timing, cpu3 would be slowest. It is not.

## Consequence

**~24s is the true steady-state `w1_c1` time at N=150.** Every ~30s
figure ever recorded for it - 29.79s, 30.37s, 30.99s, and by extension
the inflated 37.49s and 42.11s from hot sessions - was a first-run
measurement.

This is why the earlier sweeps were systematically wrong: they ran ONE
baseline per invocation, so *every* baseline they recorded was a first
run, while the parallel condition that followed was already warm. The
artifact loaded entirely onto the numerator.

Steady-state efficiency at N=150 is **24.37s / 8.07s = 3.02x = 75.5%**.

**Protocol going forward:** discard the first run of a session, or
issue an untimed warm-up run before measuring. `core_scaling_
replicated.py` already reports both with and without the first rep;
the without-first-rep column is the one to read.

## Open

What actually warms up. Each measurement is a separate subprocess, so
nothing carries over inside the process - the candidates are OS-level:
page cache for the venv's compiled extension modules, CPU frequency
ramp from idle, or first-touch page allocation. Untested, and not
needed for the practical fix.
