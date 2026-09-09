# Measurement methodology for parallel-efficiency claims

Status: proposed protocol, 2026-09-09. Supersedes the ad-hoc practice
used earlier in Phase 13, under which every reported efficiency figure
was subsequently retracted.

This document specifies how wall-clock and efficiency numbers for
`paulikit` are to be obtained, what is controlled, what is only
observed, and what the platform cannot control. It is written to be
reproducible by a third party on different hardware.

---

## 1. Why a protocol was necessary

Across three sessions, 4-core efficiency at N=150 was reported as
113.2%, then 88%, then 73.4%, and at N=180 as 73%, then 66.3%, then
87.3%. Each figure was computed from a single run per cell, and each
"correction" was itself computed from another single run. Two of the
corrections were later found to be retractions of correct results.

The failure had a single cause. The parallel measurements were always
stable - `w4_c4` at N=150 varied 1.03x across repeats - but the
single-worker baseline `T_1`, which is the *numerator* of every
efficiency figure, varied by up to 1.61x depending on when in a
session it ran. Because sweeps executed one baseline per invocation,
every baseline they recorded was a first-of-session run while the
parallel condition that followed was not.

The protocol below exists to make that class of error impossible.

---

## 2. The confound, identified

### 2.1 The effect

The first execution after a machine has been quiet is systematically
slow. Measured at N=150, `w1_c1`, in run order within one session:

| run | wall time |
|---|---|
| 0 | 29.79 s |
| 1 | 24.11 s |
| 2 | 24.06 s |
| 3 | 24.00 s |
| 4 | 22.38 s |
| 5 | 22.22 s |

Dropping run 0 takes the spread from 1.34x to 1.09x and the standard
deviation from 2.77 to 0.97. Every anomalously slow baseline recorded
in this investigation - 30.37 s, 30.99 s, 37.49 s, 42.11 s, 115.85 s,
131.59 s - was a first-of-session measurement.

The effect is **not specific to the single-worker configuration**.
Placed first on a quiet machine, `w4_c4` is inflated 11.8% and
`w1_c1` 32.9%, while the same conditions run later in the same session
are inflated 1.3% and 9.5%. It is positional, not configurational.

### 2.2 Causes eliminated

Each candidate was tested by removing it and observing whether the
effect persisted.

| candidate | test | outcome |
|---|---|---|
| Preceding condition | four consecutive `w1_c1` runs, no `w4_c4` between | still fast (24.8-25.3 s) - **refuted** |
| Idle gap | deliberate 120 s idle mid-session | cost 2.5 s, not the ~6 s effect - **refuted** |
| Core placement | pinned CPU rotated across all four physical cores | effect follows the first run, not the core; cores agree within 24.10-24.86 s excluding it - **refuted** |
| Interrupt load on cpu0 | cpu0 carries 12.2 M interrupts vs 3.8-5.5 M elsewhere | costs ~0.05 s; cpu3 has 10x cpu2's involuntary context switches yet runs within 0.6 s of the fastest - **refuted** |
| Memory fragmentation | would raise cycles for constant instructions | cycles flat at +0.3% - **refuted** |
| Page cache (cold shared objects) | would raise instructions and cycles | both flat at +0.3% - **refuted** |
| Kernel descheduling | would lower `task-clock / wall` | flat at 1.182-1.191 across cold and warm - **refuted** |

### 2.3 Cause identified: CPU frequency ramp

`perf stat` on a cold-then-warm sequence isolates it. Retired
instructions are constant to 1.004x across runs, so the work is
identical; cycles are constant to 1.033x; but wall time varies 1.156x.
The same cycles take more wall-seconds, which is a clock-rate effect.

Effective clock, computed as `cycles / task-clock`, rises
monotonically as the session proceeds:

| run | start temp | effective clock | wall |
|---|---|---|---|
| 0 (cold) | 53 C | 2.61 GHz | 24.75 s |
| 1 | 66 C | 2.68 GHz | 24.62 s |
| 2 | 79 C | 2.72 GHz | 23.47 s |
| 3 | 76 C | 2.82 GHz | 23.55 s |

The magnitudes close: the cold run's clock is 4.6% below the warm mean
and its wall time is 4.5% above. The slower clock fully accounts for
the longer run, leaving no residual.

Note the direction. The clock rises *as the CPU heats*, which is the
opposite of thermal throttling. The governor is `powersave` on
`intel_pstate`; an idle machine sits near 800 MHz against a 4.0 GHz
maximum, and sustained load is what raises it. **A cold machine is a
slow machine.** This also explains why a short (6 s) synthetic busy
loop failed to reproduce the warm state in an earlier test: the ramp
requires sustained load over minutes, which a real decomposition
provides and a brief loop does not.

---

## 3. What the platform cannot control

An attempt was made to clamp die temperature before each run, so that
thermal state could be held rather than merely observed. **It does not
work on this machine**, for four independently measured reasons:

1. **Sensor noise.** Per-core readings jump by 7 C between samples
   0.1 s apart on a completely idle machine. A single instantaneous
   reading cannot gate a control loop.
2. **Inter-core spread.** Cores differ by 3-4 C typically and by up to
   12 C transiently. Any tolerance tight enough to be meaningful is
   narrower than the hardware's own variation, so the target condition
   is frequently unsatisfiable.
3. **Non-linear thermal response.** A full-throttle synthetic load
   takes the package from 50 C to 94 C in approximately one second; a
   35% duty cycle never reaches 60 C in 120 s. There is no stable
   operating point between runaway and ineffective, and a
   feedback-controlled duty cycle still overshot to 74 C.
4. **Uncontrolled excursions.** Cores reached 70-80 C while the
   controlling loop held its duty cycle at 2%, indicating thermal
   input from outside the experiment's control.

This is a limitation of the measurement platform - a laptop with
aggressive turbo, small thermal mass, and firmware-controlled fan -
**not of the technique**. On a workstation with a locked P-state,
disabled turbo, fixed-speed fans, and better-behaved sensors,
temperature clamping would likely be tractable and would be the
preferable control. Any replication on such hardware should attempt it
and report whether the cold-start effect persists once clock is
pinned; that is the cleanest available test of the mechanism in
section 2.3.

Because temperature cannot be *controlled* here, it is **recorded** -
per-core, at the start of every run - and reported as an observed
covariate rather than a held constant.

---

## 4. The protocol

### 4.1 Warm-up

Before any timed measurement, execute a **fixed, standard, untimed
warm-up load**. This is experimental setup, not post-hoc data
exclusion - the same class of action as allowing an instrument to
reach operating temperature.

The warm-up must be **independent of the workload under measurement**.
Warming with the measured workload itself is circular: it makes the
warm-up a function of the problem size and the condition being tested,
so two studies are no longer warmed identically and the protocol is
not portable to anyone measuring something else.

The standard used here is `synthetic_ipc_control.py` at `w4_c4` with
`busywork_n=1200` - a paulikit-free multi-core load already validated
in this phase, sharing only the coarse shape of the real workload
(many small CPU-bound tasks over `ProcessPoolExecutor`). It runs
approximately 43 s.

**Duration matters and must be checked.** The same script at its
default `busywork_n=150` runs in 4.0 s, which is *not* sufficient: a
6 s synthetic burst was measured not to reproduce the warm state,
because the P-state ramp needs sustained load rather than a brief one.
The harness warns if the warm-up completes in under 20 s.

Discarding the first *recorded* run is a weaker fallback, acceptable
only where a standard warm-up is impractical. It is an exclusion
applied after seeing the data, and it depends on the replicator's
session structure matching ours. **Results in section 6 were obtained
under that weaker fallback**, before this standard was adopted; they
are not invalidated by it, but a replication should use the warm-up
above.

### 4.2 Interleaving

Conditions are executed in alternation - `T_1, T_p, T_1, T_p, ...` -
never in blocks. Residual drift is measurable even after warm-up
(24.11 s falling to 22.22 s across four repeats). Under a blocked
design that drift is assigned wholly to whichever condition ran during
it, manufacturing an effect; under alternation both conditions absorb
it equally.

This is not hypothetical. An early cause-analysis run with four
treatment arms in fixed order produced a textbook monotone
dose-response - 30.28, 27.04, 25.32, 24.87 s - that vanished entirely
when the arm order was rotated. Times tracked *position*, not
treatment.

### 4.3 Replication

Minimum **five** timed repetitions per cell after warm-up. Report
mean, standard deviation, range, and the max/min spread.

A cell whose spread exceeds **1.15x** is flagged as unstable and its
mean is not quoted; the run is repeated. This threshold is set from
observed behaviour: clean cells here run 1.03-1.13x.

### 4.4 Cooldown between runs

A **fixed-duration** pause between runs, reported with the results.

A cooldown *to a temperature threshold* was considered and rejected.
Its endpoint is reproducible but its duration is not - it varies with
ambient temperature, fan state, and prior load, taking seconds on a
cold day and minutes on a warm one. Since section 3 establishes that
thermal state cannot be guaranteed on this platform anyway, fixing the
duration controls the variable that is actually controllable, and the
resulting temperatures are recorded as data.

Where a replicating platform *can* hold temperature, a
threshold-based cooldown is preferable and should be used instead.

### 4.5 Correctness gate

Every run reports its total term count. All runs in a comparison must
agree exactly, and the value must match the analytically known count
where one exists (91,652,096 at N=150). A timing result from a run
that computed the wrong answer is discarded before any timing is
interpreted.

---

## 5. Reported quantities

Four quantities are recorded per run. Each answers a different
question, and reporting only one of them was how earlier errors went
undetected.

| quantity | source | role |
|---|---|---|
| **wall time** | `time.perf_counter` around the decomposition | the reported result |
| **retired instructions** | `perf stat -e instructions:u` | validity gate |
| **IPC** | `instructions / cycles` | mechanism, and run-validity check |
| **CPU time / wall** | `getrusage(RUSAGE_CHILDREN)` | occupancy check |

### 5.1 Efficiency is defined on wall time

Parallel efficiency is `E_p = T_1 / (p * T_p)`, with `T_1` and `T_p`
both obtained under this protocol. No counter-derived quantity can
replace it:

- **IPC cannot.** Two workers each at IPC 2.0 report the same IPC as
  one worker at IPC 2.0 while doing twice the work per unit time. IPC
  measures pipeline efficiency per core, not throughput across cores.
- **CPU time / wall cannot.** It is mean cores busy - approximately
  4.0 for any four-worker run regardless of whether those cores
  accomplish anything. It measures occupancy, not productivity.

### 5.2 What the counters are for

**Retired instructions gate the comparison.** If two runs execute
materially different instruction counts they are not measuring the
same work and their times are not comparable. Measured spread here is
1.004x, which validates the assumption rather than assuming it.

**IPC explains the shortfall and validates individual runs.** Earlier
phase-13 work established that spreading workers across more physical
cores leaves retired instructions statistically unchanged (differences
under 0.2%) while IPC falls significantly (p < 0.0001 at three of four
worker counts). The efficiency loss is *stalling*, not extra work.

IPC is also **immune to the cold-start confound**: on a cold run it
deviates by -0.0% (1.484 against a warm mean of 1.485) while wall time
on the same runs is inflated 11.5%. That makes it a strong
pre-specifiable validity criterion - a run whose IPC departs from its
condition's established value was disturbed and should be excluded, on
grounds that do not depend on its timing.

**CPU time / wall confirms the configuration ran as intended.**
Measured: 1.17 for one worker, 2.25 for two, 4.34 for four. A
`w4_c4` run reporting 2.1 did not use four cores, whatever its
configuration said.

---

## 6. Results under this protocol

N = 150, 160, 180, 200; one worker per physical core; auto-tuned
`chunk_size`; 6 repetitions per cell, interleaved, first repetition
discarded; no checkpointing. All runs started at 54-55 C; all term
counts correct. Raw data: `core_scaling_replicated_results.jsonl`
(48 runs).

| N | qubits | `T_1` (s) | `T_4` (s) | speedup | efficiency | adversarial | Welch p |
|---|---|---|---|---|---|---|---|
| 150 | 14 | 23.35 ± 0.97 | 8.07 ± 0.11 | 2.894x | **72.4%** | 67.7% | 3.0e-06 |
| 160 | 14 | 28.11 ± 0.75 | 9.31 ± 0.36 | 3.020x | **75.5%** | 69.8% | 7.3e-09 |
| 180 | 15 | 72.87 ± 3.63 | 27.94 ± 1.02 | 2.608x | **65.2%** | 56.4% | 3.1e-06 |
| 200 | 15 | 87.22 ± 3.11 | 33.36 ± 0.76 | 2.614x | **65.4%** | 60.5% | 9.1e-07 |

"Adversarial" is `min(T_1) / max(T_4) / 4` - the least favourable
reading the same data permits, and the figure to quote when a
conservative bound is wanted.

All cell spreads are within 1.13x. Efficiency is flat within a qubit
count and steps down by approximately 9 points across the 14-to-15
qubit boundary, where `dim` doubles from 16,384 to 32,768.

### 6.1 A caveat on hyperthread conditions

Conditions that place two workers on the two hyperthreads of a single
physical core (`w2_c1`, `w4_c2`, `w8_c4`) are **not** measurements of
2, 4, or 8 cores. Earlier phase-13 work used such a ladder and its
apparent sublinear scaling was an artifact of that conflation. Any
claim about core scaling must place one worker per physical core, and
must state the CPU topology it assumes.

---

## 7. Threats to validity

**Sample size.** Five to six repetitions per cell resolves the
separations reported here, which are large. It does not resolve
adjacent cells - 72.4% against 75.5% is within the observed
run-to-run variation and no ordering between them is claimed.

**Single platform.** All measurements are from one 4-core/8-thread
i7-8550U laptop. The frequency-ramp mechanism is specific to
`intel_pstate` under the `powersave` governor; platforms with a fixed
P-state may show no cold-start effect at all, in which case the
warm-up step is harmless but unnecessary.

**Temperature observed, not controlled.** See section 3. Reported
temperatures are covariates. A replication that *can* clamp
temperature should do so and report whether the results change.

**Mechanism established at one magnitude.** The frequency ramp is
identified from a run whose cold penalty was 4.5%, with effect sizes
closing to within 0.1 points. Cold penalties of 11.5% and 28-33% were
observed in other sessions but were not instrumented with performance
counters. The mechanism is consistent across them but its
completeness at larger magnitudes is inferred, not measured.

**Efficiency percentages are session-scoped.** `T_1` and `T_p` must
come from the same interleaved session. Comparing a baseline from one
session against a parallel time from another reproduces exactly the
error this protocol was written to prevent.
