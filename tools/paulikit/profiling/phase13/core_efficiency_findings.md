# Physical-core scaling, and one falsified explanation for it

Recorded 2026-09-08. Measured by `worker_scaling_no_checkpoint.py` and
`core_efficiency_vs_working_set.py`, one worker per PHYSICAL core
throughout (no hyperthread siblings), thermally controlled to 55C
before every run, n=1 per cell.

## The headline: scaling was never the problem, the ladder was

Earlier phase-13 conditions doubled hyperthread siblings onto each
core (`w2_c1` = cpus 0,4 - one core; `w8_c4` = all eight logical CPUs
on four cores). Measured against a one-worker-per-core ladder instead:

| workers | packed (siblings) | spread (distinct cores) |
|---|---|---|
| 1 | 30.368s - 1.00x | 30.368s - 1.00x |
| 2 | `w2_c1` 19.296s - 1.57x | **`w2_c2` 13.823s - 2.20x** |
| 4 | `w4_c2` 10.746s - 2.83x | **`w4_c4` 8.641s - 3.51x** |
| 8 | `w8_c4` 9.069s - 3.35x | - (only 4 physical cores exist) |

N=150, `chunk_size=2`. **4 physical cores give 3.51x, or 88%
efficiency**; 2 cores give 2.20x, mildly superlinear (two private L2s
serving half the working set each). The second hyperthread is worse
than useless here: 4 workers on 4 cores (8.641s) beats 8 workers on
the same 4 cores (9.069s).

`w4_c4` is the best configuration at both sizes - it beats `w8_c4`
while using half the workers and ~200 MiB less memory.

At N=180 the same ladder gives 1.882x (94%) and 2.933x (73%).

## FALSIFIED: the working set does not explain the N=150 -> N=180 drop

4-core efficiency falls 88% -> 73% between the two sizes. The proposed
explanation was cache: N=180 doubles `dim` to 32768, so cores contend
for memory instead of being served from private L2.

**That explanation is refuted by the data that motivated it.** The
auto-tuner halves `chunk_size` to 1 at N=180 precisely to hold the
per-worker buffer at `chunk_size * dim * 16` = 512 KiB - the *same*
buffer as N=150 at `chunk_size=2`. Identical putative cause, different
effects (88% vs 73%), so the buffer size is not the cause. The
contrapositive settles it without further measurement.

The match is closer than just the buffer. At those two settings:

| | N=150, cs=2 | N=180, cs=1 |
|---|---|---|
| buffer/worker | 512 KiB | 512 KiB |
| terms per chunk | 16,381 | 16,382 |
| IPC bytes per chunk | ~320 KiB | ~320 KiB |
| **chunks** | **5,595** | **16,120** |
| **total IPC** | **1.83 GB** | **5.28 GB** |

Per-chunk shape is essentially identical; what differs is 2.9x more
chunks and 2.9x more total IPC. That points at per-chunk drain-loop
and IPC cost, not at cache residency. **Untested** - stated here as
the surviving candidate, not as a conclusion.

## What DID survive: oversized chunks wreck multi-core efficiency

A separate, narrower question - within a fixed problem, does
`chunk_size` affect core scaling? Measured at N=180:

| chunk | buf/worker | 4x buf | w1_c1 | w4_c4 | speedup | efficiency |
|---|---|---|---|---|---|---|
| 1 | 512 KiB | 2 MiB | 75.75s | 28.58s | 2.651x | **66.3%** |
| 2 | 1 MiB | 4 MiB | 74.50s | 29.11s | 2.559x | **64.0%** |
| 4 | 2 MiB | 8 MiB | 82.92s | 42.27s | 1.962x | **49.0%** |
| 8 | 4 MiB | 16 MiB | 81.48s | 50.87s | 1.602x | **40.0%** |

Single-core time barely moves (75-83s) while 4-core time nearly
doubles (28.6s -> 50.9s). That is contention, not extra work: one core
does not care how big the buffer is, four cores care a great deal.

**The cliff is at private L2 (1 MiB/core), not shared L3 (8 MiB).** An
L3 explanation would predict degradation only once the 4-worker
aggregate exceeds 8 MiB, i.e. at `chunk_size=8`. In fact efficiency is
already halved at `chunk_size=4`, and the decline begins between
`chunk_size` 2 and 4 - exactly where the per-worker buffer crosses
1 MiB. This is independent confirmation that Phase 12's auto-tuner is
right to size chunks against L2.

Practical consequence: **do not raise `chunk_size` above what the
auto-tuner picks.** At N=180 it picks 1, which is also the fastest and
most efficient setting measured.

## Caveats

n=1 per cell. Run-to-run variance on this machine has been observed at
~15% for a single condition, so third digits are not meaningful and
the *ordering* of adjacent cells (e.g. 66.3% vs 64.0%) is not
established. The large gaps - packed vs spread, and `chunk_size` 1-2
vs 4-8 - are far outside that variance and are solid.
