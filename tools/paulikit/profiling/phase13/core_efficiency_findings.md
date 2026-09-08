# Physical-core scaling: 86-113% efficiency, and two falsified theories

Recorded 2026-09-08. Measured by `worker_scaling_no_checkpoint.py` and
`core_efficiency_vs_working_set.py`, one worker per PHYSICAL core
throughout (no hyperthread siblings), n=1 per cell.

**This document supersedes an earlier version of itself.** Its
headline figure - 73% 4-core efficiency at N=180, later refined to
66.3% - was a MEASUREMENT ARTIFACT and is retracted. See "The
retraction" below.

## The headline: scaling was never the problem, the ladder was

Earlier phase-13 conditions doubled hyperthread siblings onto each
core (`w2_c1` = cpus 0,4 - one physical core; `w8_c4` = all eight
logical CPUs on four cores). Against a one-worker-per-core ladder:

| workers | packed (siblings) | spread (distinct cores) |
|---|---|---|
| 1 | 30.368s - 1.00x | 30.368s - 1.00x |
| 2 | `w2_c1` 19.296s - 1.57x | **`w2_c2` 13.823s - 2.20x** |
| 4 | `w4_c2` 10.746s - 2.83x | **`w4_c4` 8.641s - 3.51x** |
| 8 | `w8_c4` 9.069s - 3.35x | - (only 4 physical cores exist) |

N=150, `chunk_size=2`. The second hyperthread is worse than useless
here: 4 workers on 4 cores (8.641s) beats 8 workers on the same 4
cores (9.069s). **`w4_c4` is the best configuration at every size
measured**, using half the workers and ~200 MiB less than `w8_c4`.

## Efficiency across the qubit boundary: a STEP, not a slide

All at auto-tuned `chunk_size`, thermally controlled at a *reachable*
65C target:

| N | qubits | dim | auto cs | w1_c1 | w4_c4 | speedup | efficiency |
|---|---|---|---|---|---|---|---|
| 150 | **14** | 16384 | 2 | 37.49s | 8.28s | 4.525x | **113.1%** |
| 160 | **14** | 16384 | 2 | 44.28s | 10.82s | 4.092x | **102.3%** |
| 180 | **15** | 32768 | 1 | 115.85s | 33.16s | 3.494x | **87.3%** |
| 200 | **15** | 32768 | 1 | 131.59s | 38.29s | 3.436x | **85.9%** |

Within a qubit count efficiency is flat (113/102 at 14; 87/86 at 15).
Across the boundary it steps down. There is **no monotonic decay with
N**: N=200 is 24% larger than N=180 and scales the same. The 14-qubit
cells are superlinear - four private L2s serving a quarter of the
working set each.

Practical answer to "should we expect worse and worse efficiency, and
eventually degradation, as N grows": **no, not on this evidence.**
86-87% on 4 real cores at 15 qubits is good scaling, and it did not
degrade from 180 to 200.

## The retraction

An earlier version of this document reported N=180 at 73%, later
measured at 66.3%, and built two hypotheses on that number. **The
number was wrong.** Re-measured under a working cooldown, N=180 is
**87.3%**.

The cause was the thermal control itself. `COOLDOWN_TARGET_C` was
55C - a temperature this machine never reaches; it idles at 69-75C
after sustained multi-core work. Every cooldown therefore ran its full
240s timeout and started the run anyway at 69-85C. Runs were **not**
thermally matched despite the setting claiming they were, and a hot,
throttled single-core baseline finishes fast relative to a hot 4-core
run, deflating the apparent speedup.

The tell was a `w1_c1` baseline of 75.75s in the old regime against
115.85s in the new one - a 53% discrepancy on identical work, which
should have been questioned before any theory was built on it. Two
independent confirmations that the rerun is the trustworthy one: the
two 15-qubit cells now agree tightly (87.3% vs 85.9%) where they
differed by 20 points before, and cooldowns now complete in 2-52s
instead of always hitting the 240s ceiling.

Target is now 65C, which is reachable, so runs are genuinely matched.

## FALSIFIED: per-worker buffer size

Proposed: efficiency falls because the working set per worker grows.
Refuted by its own motivating data - the auto-tuner halves
`chunk_size` at 15 qubits precisely to hold the per-worker buffer at
512 KiB, so N=150/cs=2 and N=180/cs=1 share the putative cause but not
the effect. One cause cannot produce two effects; the contrapositive
settles it without further measurement.

## FALSIFIED: chunk count and inter-process volume

Proposed: 15-qubit problems have ~2.9x more chunks, so per-chunk drain
and IPC cost dominates. Refuted by doubling the chunk count at fixed
N:

| N | chunk | chunks | efficiency |
|---|---|---|---|
| 150 | 1 | 11,189 | 104.8% |
| 150 | 2 | 5,595 | 113.1% |

Doubling chunks at N=150 leaves efficiency superlinear. If chunk count
drove a collapse to 66%, this cell should have collapsed. It did not.

## What DID survive: oversized chunks wreck multi-core efficiency

A narrower question - within a fixed problem, does `chunk_size` affect
core scaling? At N=180 (measured in the old thermal regime, so the
absolute values are suspect; the TREND across rows, all measured the
same way, is the finding):

| chunk | buf/worker | 4x buf | w1_c1 | w4_c4 | speedup | efficiency |
|---|---|---|---|---|---|---|
| 1 | 512 KiB | 2 MiB | 75.75s | 28.58s | 2.651x | 66.3% |
| 2 | 1 MiB | 4 MiB | 74.50s | 29.11s | 2.559x | 64.0% |
| 4 | 2 MiB | 8 MiB | 82.92s | 42.27s | 1.962x | 49.0% |
| 8 | 4 MiB | 16 MiB | 81.48s | 50.87s | 1.602x | 40.0% |

Single-core time barely moves (75-83s) while 4-core time nearly
doubles. That is contention, not extra work. The cliff sits at private
L2 (1 MiB/core), not shared L3: an L3 story predicts trouble only at
`chunk_size=8`, but efficiency is already halved at 4.

**Do not raise `chunk_size` above what the auto-tuner picks.**

## Open: chunk_size should probably depend on the core count too

Raised by the user, and the data supports it. `chunk_size` is
currently chosen from `dim` and a memory budget, targeting **one
core's private L2**. But the cache a chunk actually competes for
depends on how many cores are running:

- 1 core: the chunk has a full 1 MiB L2 and all 8 MiB of L3.
- 4 cores: each still has its own 1 MiB L2, but the four now share one
  8 MiB L3 - 2 MiB each in aggregate terms.

So the same `chunk_size` is a different proposition at `w1_c1` than at
`w4_c4`, and the tuner cannot see which is coming. The table above is
what that looks like when it goes wrong: `chunk_size=4` puts the
aggregate at exactly L3 and efficiency halves.

Note what this does NOT explain: at auto `chunk_size` every N measured
above sits at 512 KiB/core and 2 MiB aggregate - identical cache
pressure at 14 and 15 qubits - yet efficiency still steps from ~102%
to ~87%. So a core-count-aware `chunk_size` is worth pursuing on its
own merits, but the qubit-boundary step needs a different explanation.

## Still open: what causes the step at the qubit boundary

`dim` is the surviving candidate - it doubles at the boundary, and it
sets the gather/scatter row width independently of `chunk_size`.
Untested, and stated here as a candidate rather than a conclusion.

## Caveats

n=1 per cell. Run-to-run variance on this machine has been observed at
15-23% for a single condition, so third digits are not meaningful and
the ordering of adjacent cells (113.1% vs 102.3%) is not established.
The large, repeated separations - packed vs spread, 14-qubit vs
15-qubit, `chunk_size` 1-2 vs 4-8 - are well outside that.
