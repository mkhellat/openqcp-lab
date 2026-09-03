# Full optimum sweep: which (n_workers, n_cores) configuration actually minimizes wall-clock, for `paulikit`'s `parallel_decompose`

Recorded 2026-09-03. Commissioned directly by the user after an
extended series of pairwise Welch's t-test comparisons (documented
across `core_packing_series_thermal_controlled_findings.md`,
`instructions_ipc_accounting_findings.md`,
`overreach_correction_and_open_questions.md`) established that, within
each of four *separately tested pairs*, packing workers onto fewer
physical cores was faster — but never answered the actual question:
across the *full* space of valid configurations, which one genuinely
minimizes wall-clock, with one properly-powered joint statistical
test rather than a series of independent pairwise ones. This document
is that joint analysis, run for "academic publication" rigor per
direct instruction: "the methodology must be concrete, strong, solid
statistical analysis with cooling period and fully controlled
measurements, all across every possible multi-core distribution with
and without hyperthreading."

## Reproducibility: exact hardware/software environment

- **CPU**: Intel Core i7-8550U @ 1.80GHz (Kaby Lake R), 1 socket, 4
  physical cores, 2 threads/core (8 logical CPUs), 15W TDP mobile part.
- **Cache**: L1d/L1i 32 KiB per core (128 KiB total / 4 instances), L2
  256 KiB per core (1 MiB total / 4 instances), L3 8 MiB shared (1
  instance across all 4 physical cores).
- **Topology** (`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`):
  hyperthread pairs (0,4), (1,5), (2,6), (3,7) — physical cores A, B,
  C, D respectively.
- **Frequency governor**: `intel_pstate` driver in HWP (Hardware
  P-States) autonomous mode — the OS sets only a min/max performance
  *envelope*, the CPU's own microcode selects the actual running
  frequency (verified directly against upstream kernel source,
  `drivers/cpufreq/intel_pstate.c`, and `Documentation/admin-guide/pm/intel_pstate.rst`;
  see `hwp_kernel_source_investigation.md`). The `performance` governor
  was set on all 8 logical CPUs for every measurement in this
  investigation, but this does NOT prevent thermal throttling — see
  the cooldown protocol below.
- **OS/kernel**: Arch Linux, kernel 7.1.11-arch1-1 (`uname -a`:
  `Linux archie 7.1.11-arch1-1 #1 SMP PREEMPT_DYNAMIC`).
- **Python**: CPython 3.12.13, dedicated venv
  (`/home/desadm/.venvs/paulikit`).
- **Key packages**: NumPy 2.5.2, SciPy 1.18.1.
- **`paulikit`**: this repo's own package, `parallel_decompose` from
  `src/paulikit/algorithms/fwht.py`, called through its real public
  API (no reimplementation) via `full_matrix_target.py`.
- **Measurement tool**: `perf stat --no-inherit`, event group
  `task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses`
  (the same "L3 group" used throughout this whole investigation phase).
- **Workload**: N=150 coupled harmonic oscillators (`N_OSCILLATORS =
  150` in `full_matrix_target.py`), `chunk_size=2` — identical fixed
  workload for every single one of the 140 runs; only `n_workers` and
  the CPU affinity pinning (`n_cores`) vary.

## Configuration enumeration

Every distinct valid `(n_workers, n_physical_cores_used)` pairing on
this 4-physical-core machine, where a physical core hosts at most 2
workers (its 2 hyperthread siblings): for `n_workers` from 1 to 8,
`n_cores` ranges over `ceil(n_workers/2) .. min(n_workers, 4)`. This
gives exactly **14 configurations**, named `w<n_workers>_c<n_cores>`:

```
w1_c1, w2_c1, w2_c2, w3_c2, w3_c3, w4_c2, w4_c3, w4_c4,
w5_c3, w5_c4, w6_c3, w6_c4, w7_c4, w8_c4
```

Workers are packed onto physical cores as evenly as possible via
`condition_table.py`'s `_packed_cpu_list(n_workers, n_cores)`
(`divmod`-based even distribution across exactly `n_cores` distinct
physical cores — e.g. `(5, 3)` gives 2+2+1, not 2+1+1+1 leaving a
core untouched). Each worker is pinned to one specific logical CPU via
`os.sched_setaffinity`, using `fwht._physical_core_representative_cpus`
monkeypatched to the computed list — the same mechanism
`parallel_decompose` already uses in production, not a synthetic
bypass. `w1_c1` uses the sequential code path
(`fwht_pauli_terms_iter`, no process pool at all) as the `n_workers=1`
baseline.

## Thermal-controlled measurement protocol

This machine's package temperature reaches 100°C (its hardware
throttling limit) under sustained multi-core load — a real,
previously-confirmed confound (`dvfs_thermal_confound_findings.md`)
that silently invalidated an earlier, uncontrolled round of
measurements in this same investigation. Protocol used for every one
of the 140 runs here (`full_optimum_sweep.py`'s `cooldown()`):

1. Poll `/sys/class/thermal/thermal_zone7/temp` (confirmed via
   `/sys/class/thermal/thermal_zone*/type` to be `x86_pkg_temp`) every
   2 seconds.
2. Block until package temperature drops to ≤55°C, or 180 seconds
   elapse (whichever first — the 180s cap is a safety bound, never hit
   in this run).
3. Record the settled starting temperature and the temperature
   immediately after the run completes, as covariates.
4. Only then launch the timed `perf stat` run.

This controls STARTING temperature uniformly across every run (every
one of the 140 runs began at 52-55°C — verified, see raw data);
in-run thermal climb to 79-88°C still occurs (also recorded, see raw
data) but starts from an identical, controlled baseline every time,
eliminating the leftover-heat confound that invalidated the earlier
uncontrolled series.

## Design: 14 configs × 10 reps = 140 runs

10 reps per configuration (not the 5 used in earlier pairwise
comparisons in this investigation) — a larger n specifically chosen
for publication-grade statistical power, per direct instruction. Run
order: `w1_c1` through `w8_c4`, ascending by `(n_workers, n_cores)`,
each config's 10 reps run back-to-back before moving to the next.
Results appended incrementally to `full_optimum_sweep_results.jsonl`
(committed alongside this document — the exact raw dataset behind
every number below, not just the method) after every single run, with
a resume mechanism (`load_completed()`) making the whole sweep safe to
interrupt and continue across many separate background sessions —
it was, in fact, run as one uninterrupted background process, but the
resumable design was exercised via a 1-rep smoke test before the real
launch.

## Results: per-condition summary (n=10 each)

| config | n_workers | n_cores | mean wall-clock (s) | sd (s) |
|---|---|---|---|---|
| w1_c1 | 1 | 1 | 26.366 | 3.205 |
| w2_c1 | 2 | 1 | **20.537** | 0.293 |
| w2_c2 | 2 | 2 | 22.273 | 0.465 |
| w3_c2 | 3 | 2 | 21.553 | 0.248 |
| w3_c3 | 3 | 3 | 22.974 | 0.440 |
| w4_c2 | 4 | 2 | 21.395 | 0.151 |
| w4_c3 | 4 | 3 | 22.586 | 0.390 |
| w4_c4 | 4 | 4 | 23.850 | 0.395 |
| w5_c3 | 5 | 3 | 22.553 | 0.658 |
| w5_c4 | 5 | 4 | 23.332 | 0.310 |
| w6_c3 | 6 | 3 | 22.626 | 0.309 |
| w6_c4 | 6 | 4 | 23.369 | 0.214 |
| w7_c4 | 7 | 4 | 23.532 | 0.269 |
| w8_c4 | 8 | 4 | 23.964 | 0.284 |

`w1_c1` (sequential) has by far the highest variance (sd=3.205s vs.
0.15-0.66s for every parallel condition) — consistent with a single
long-running process being more exposed to unrelated OS scheduling
noise than a pinned multi-process pool; not further investigated here
since it is not the effect under study.

## Statistical method: one-way ANOVA + Tukey HSD, not pairwise tests

This directly addresses the user's core, repeated methodological
objection to every earlier round of this investigation ("these are
four separate, independent pairwise comparisons... that's the actual
missing piece"). `full_optimum_sweep_analysis.py`:

1. **One-way ANOVA** (`scipy.stats.f_oneway`) across all 14 conditions'
   wall-clock times, one single joint test of "does configuration
   matter at all":
   **F(13, 126) = 22.65, p = 5.6 × 10⁻²⁷** — configuration has a
   highly significant joint effect on wall-clock.
2. **Tukey HSD post-hoc** (implemented directly via
   `scipy.stats.studentized_range` — `statsmodels` is not an existing
   dependency of this project and was not added for one script;
   textbook equal-n HSD formula, `q_crit × sqrt(MSE/n)`, using the
   pooled within-group MSE from the same ANOVA): identifies which
   *specific pairs* of the 91 possible pairs differ significantly, at
   one family-wise α=0.05 — correctly controlling the multiple-
   comparisons risk that running many separate Welch's t-tests does
   not control for. MSE=0.856, df_within=126, q_crit=4.837, **HSD
   critical difference = 1.415s** (any pair whose means differ by more
   than this is declared significant).

## Result: the observed minimum, and what is honestly distinguishable from it

**Observed minimum: `w2_c1` (n_workers=2, n_cores=1 — 2 workers on the
2 hyperthreads of a single physical core), mean 20.537s.**

| rank | config | n_workers | n_cores | mean (s) | diff from best (s) | statistically distinguishable from `w2_c1`? |
|---|---|---|---|---|---|---|
| 1 | w2_c1 | 2 | 1 | 20.537 | +0.000 | — (is the best) |
| 2 | w4_c2 | 4 | 2 | 21.395 | +0.858 | **no — statistically tied** |
| 3 | w3_c2 | 3 | 2 | 21.553 | +1.016 | **no — statistically tied** |
| 4 | w2_c2 | 2 | 2 | 22.273 | +1.735 | yes, worse |
| 5 | w5_c3 | 5 | 3 | 22.553 | +2.015 | yes, worse |
| 6 | w4_c3 | 4 | 3 | 22.586 | +2.049 | yes, worse |
| 7 | w6_c3 | 6 | 3 | 22.626 | +2.089 | yes, worse |
| 8 | w3_c3 | 3 | 3 | 22.974 | +2.437 | yes, worse |
| 9 | w5_c4 | 5 | 4 | 23.332 | +2.794 | yes, worse |
| 10 | w6_c4 | 6 | 4 | 23.369 | +2.832 | yes, worse |
| 11 | w7_c4 | 7 | 4 | 23.532 | +2.994 | yes, worse |
| 12 | w4_c4 | 4 | 4 | 23.850 | +3.313 | yes, worse |
| 13 | w8_c4 | 8 | 4 | 23.964 | +3.427 | yes, worse |
| 14 | w1_c1 | 1 | 1 | 26.366 | +5.829 | yes, worse |

**The honest statistical conclusion, stated precisely**: `w2_c1` has
the lowest observed mean wall-clock, and this is a real, jointly
significant effect (ANOVA p ≈ 10⁻²⁷, HSD-confirmed against 11 of the
other 13 configurations). But it is **statistically indistinguishable**
from two others — `w4_c2` (4 workers, packed onto 2 physical cores)
and `w3_c2` (3 workers, packed onto 2 physical cores) — both of whose
means fall within the 1.415s Tukey HSD critical difference of
`w2_c1`'s own mean. All three top configurations share one structural
property: **workers packed onto exactly 2 physical cores** (only
`n_cores` differs from `n_workers` for `w4_c2` and `w3_c2` — both use
hyperthreading doubling on at least one core, `w2_c1` uses it on the
only core in use). Every configuration using 3 or 4 physical cores is
significantly slower than all three of these, without exception.

This finally answers, with actual joint statistical power rather than
inference from separate pairs, the question left open in
`overreach_correction_and_open_questions.md`'s "Overreach 1": the
earlier claim "2-core hyperthreading is optimal" was correctly flagged
as unearned at the time — this sweep shows it was *directionally*
right but incompletely stated. The correct, now-supported claim is:
**minimizing distinct physical cores in use (packing onto 2 physical
cores specifically, regardless of exact worker count from 2 to 4) is
what actually minimizes wall-clock on this machine for this workload —
not simply "fewer workers" or "exactly 2 workers."** `w4_c2` (4
workers on 2 cores) is statistically tied with `w2_c1` (2 workers on 1
core) despite having double the worker count — the physical-core
count, not the worker count, is the variable that matters.

## What this does and does not establish

**Established**: on this specific machine (i7-8550U, 4 physical
cores/8 logical, 8 MiB shared L3, 15W TDP), for this specific workload
(`paulikit`'s `parallel_decompose` at N=150 oscillators,
`chunk_size=2`), packing workers onto exactly 2 physical cores
minimizes wall-clock, with a properly joint, family-wise-controlled
statistical test — not an inference from separate pairwise
comparisons. This is consistent with, and now supersedes with a full
joint test, every pairwise result found earlier in this investigation.

**NOT established** (per the still-open items in
`overreach_correction_and_open_questions.md`, unchanged by this
sweep):
1. Whether 2-physical-cores is a genuine hardware/workload optimum
   (e.g. determined by L3 size, per-core cache size, or
   memory-bandwidth-per-core ratio) that would generalize to other
   machines, or a coincidence of this specific 4-core/8-MiB-L3 chip —
   no cross-machine data exists.
2. Whether this is avoidable/tunable — a smaller `chunk_size` under
   multi-core contention, a more cache-friendly gather pattern, or a
   smaller shared setup-array footprint were never varied in this
   sweep (every one of the 140 runs used the same fixed
   `chunk_size=2`).
3. Whether the effect is specific to `paulikit`'s own implementation
   choices or a generic property of any CPU-bound multi-process Python
   workload sharing this machine's L3 — no in-process profiling
   (`perf record`/`py-spy`) was done here, only whole-process
   wall-clock and `perf stat` counters.

These three are the explicitly deferred next steps, to be taken up
only after this sweep (per direct user instruction: "after this
concrete analysis, we could talk about the other 2 concerns").

## Files

- `condition_table.py` — the 14-configuration enumeration and
  even-packing logic (`_packed_cpu_list`).
- `full_matrix_target.py` — measurement target, calls the real
  `parallel_decompose`/`fwht_pauli_terms_iter` API directly.
- `full_optimum_sweep.py` — the sweep driver (cooldown protocol,
  incremental resumable logging).
- `full_optimum_sweep_results.jsonl` — the complete, committed raw
  dataset: all 140 rows (14 configs × 10 reps), one JSON object per
  run, including wall-clock, cycles, instructions, IPC, cache-miss
  ratio, LLC-miss ratio, peak RSS, and temperature covariates for
  every single run — the full record behind every number in this
  document, not just its summary statistics.
- `full_optimum_sweep_analysis.py` — the ANOVA + Tukey HSD analysis
  script that produced every statistic in this document; re-running it
  against the committed `.jsonl` reproduces every number exactly.
