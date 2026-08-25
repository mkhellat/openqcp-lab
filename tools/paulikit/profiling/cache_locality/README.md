# Cache locality investigation

A "fundamental, not TBB-limited" investigation into paulikit's cache
behavior, started 2026-08-25. Every measurement here is backed by a
checked-in, runnable, fail-safe script (not just prose commands) so a
reader can reproduce every number independently and extend the
investigation on their own. This README is the map: read it first,
then follow the "Findings, in order" section to the individual docs.

**Everything here targets `paulikit.algorithms.fwht.fwht_pauli_terms`**
and its helper `fwht_pauli_coefficients` (`src/paulikit/algorithms/fwht.py`),
run via `paulikit decompose --n-oscillators N` on the CLI's default
synthetic Hamiltonian (see `cli.py`'s `_default_spring_constants`/
`_default_masses`).

## Quick start (reproduce everything yourself)

All scripts are Linux-only (they depend on the `perf_events` subsystem
and `/proc`) and bash, not POSIX `sh` - see each script's own header
for why. Prerequisites:

```bash
# From tools/paulikit/, with the native extension built:
pip install -e . --no-build-isolation

# Unprivileged perf hardware-counter access (once per boot, or add to
# /etc/sysctl.d/ to persist):
sudo sysctl kernel.perf_event_paranoid=1
```

Then, from this directory:

```bash
./run_baseline_perf_stat.sh          # cache-miss ratio, N=50, 3 runs
./run_perf_record_localize.sh        # localizes misses to code symbols, N=50
./run_steady_state_sweep.sh          # cache-miss ratio + stalls across N=25/50/100/150
./run_openblas_comparison.sh         # isolates OpenBLAS thread-pool noise, N=25
```

Every script validates its own preconditions (perf available,
`perf_event_paranoid` low enough, paulikit importable) and fails
loudly with an actionable message rather than producing a partial or
misleading result. Output files are timestamped and never overwrite
each other; none of the timestamped `*.txt`/`*.data` outputs they
produce are committed to the repo (they're regenerable and
machine-specific) - only the reference findings docs and the scripts
themselves are.

**A note on your own hardware**: every finding here was measured on
one specific machine (see `cpu_info.txt` - an 8-thread Intel
Core i7-8550U, 8 MiB L3, single NUMA node). Absolute numbers (cache
sizes, miss ratios, timings) will differ on your machine. The
*structural* findings (dense-array-vs-cache-size scaling, OpenBLAS
thread noise) should reproduce qualitatively anywhere with a similar
NumPy/OpenBLAS setup, but re-run the scripts on your own hardware
before trusting any specific number.

## Findings, in order

Read these in this order - each one either builds on or corrects the
one before it. Don't read only the last one; the corrections
(especially steps 6-7) are as important as the original findings.

1. **[`baseline_perf_stat.md`](baseline_perf_stat.md)** - ground
   truth: real hardware cache-miss counters exist and are reproducible
   (~54-57% miss ratio at N=50), not noise. Script:
   `run_baseline_perf_stat.sh`.

2. **[`perf_record_n50_findings.md`](perf_record_n50_findings.md)** -
   localizes the misses to code: dominated by NumPy's own ufunc
   machinery and CPython object churn, **not** paulikit's native
   Cython/C++/TBB kernel. Root cause identified:
   `fwht_pauli_coefficients` materializes a full dense `(dim, dim)`
   array (64 MiB at N=50, 8x this machine's L3) that's mostly zero,
   then `fwht_pauli_terms` re-scans that entire array with
   `np.nonzero(np.abs(...))`. Script: `run_perf_record_localize.sh`.

3. **[`stall_cycles_n50_findings.md`](stall_cycles_n50_findings.md)** -
   corrects the framing: cache-miss *ratio* (54-57%) is not the same
   as wall-time *cost*. Direct stall-cycle measurement: ~30% of
   cycles stalled on any resource, ~19% specifically on memory loads.
   **Caveat (see finding 7 below): this doc's absolute numbers
   include OpenBLAS noise, not yet re-measured cleanly.**

4. **[`n_scaling_findings.md`](n_scaling_findings.md)** - tests the
   dense-array hypothesis across N=25/50/100 (one-shot CLI timing).
   Cache-miss ratio scales cleanly with array-vs-cache size (17.2% ->
   ~55% -> 59.1%), supporting the hypothesis - but total-stall stays
   flat across a 128x range in array size, an unexplained puzzle at
   the time. **Superseded by finding 6 for the N-scaling numbers
   specifically** (this one's measurement method was later found to
   have its own confound - see finding 5).

5. **[`n150_oom_finding.md`](n150_oom_finding.md)** - N=150
   OOM-killed this machine (15 GiB RAM) on the **unmodified,
   untouched** existing code, after ~5 minutes, without completing
   even one decompose call. Upgrades the dense-array issue from a
   performance concern to a real robustness risk: it can crash a
   user's process outright, not just run slower than necessary.

6. **[`steady_state_scaling_findings.md`](steady_state_scaling_findings.md)** -
   corrects finding 4's methodology: one-shot CLI timing conflates
   process-startup overhead (import machinery, first-touch page
   faults, GC) with algorithm cost, proportionally worse at small N.
   Redone with an in-process, warmed-up driver
   (`steady_state_decompose.py`). Corrected N=25 cache-miss ratio is
   *higher* than the original (22.7-24.6% vs. 17.2%) - the startup
   noise was masking the algorithm's own signal, not just adding to
   it. Cache-miss ratio and mem-stall now scale monotonically and
   cleanly with N; total-stall remains flat, confirming the puzzle
   was real, not a startup-noise artifact. Scripts:
   `steady_state_decompose.py`, `run_steady_state_sweep.sh`.

7. **[`stall_floor_mystery_solved.md`](stall_floor_mystery_solved.md)** -
   solves the flat-total-stall puzzle from findings 4 and 6: ~60% of
   sampled stall-cycle self-time is `blas_thread_server` - NumPy's
   linked OpenBLAS backend spinning an idle worker-thread pool that
   paulikit's own FWHT code never actually calls into. Confirmed via
   `OPENBLAS_NUM_THREADS=1`: cuts measured cycles 2.6-2.7x and stall
   cycles 3.2-3.3x with wall-clock time essentially unchanged - proof
   this is cross-thread perf-counter noise, not real algorithmic
   work. **This means finding 3's and
   [`compiler_flags_findings.md`](compiler_flags_findings.md)'s
   absolute `cycle_activity.stalls_total` percentages are upper
   bounds contaminated by this noise, not clean signal** - flagged
   honestly rather than silently left standing; re-running with
   `OPENBLAS_NUM_THREADS=1` set is a recommended follow-up, not yet
   done. Does **not** invalidate the dense-array root cause or its
   N-scaling confirmation (those used genuine memory-subsystem
   counters, unaffected by this noise). Script:
   `run_openblas_comparison.sh`.

8. **[`compiler_flags_findings.md`](compiler_flags_findings.md)** -
   tests whether `-O2`/`-O3`/`-march=native` affect the measured
   cache behavior. Confirms the build already uses `-O3` (an
   undocumented Meson `release`-buildtype default, not a deliberate
   choice) but finds no measurable difference between `-O2`, `-O3`,
   and `-O3`+`-march=native` at N=50 - with a mechanistic explanation
   (the hot cache-miss path lives in NumPy's pre-built binary, which
   paulikit's own compiler flags can't touch). Also flags
   `-march=native` as a real portability hazard for Phase 5's
   prebuilt wheels, independent of this null performance result. Not
   yet wrapped in a checked-in script (mutates the local build
   in-place; reconfiguring build state safely under the same
   fail-safe conventions as the read-only scripts is a candidate
   follow-up).

9. **[`stall_floor_mystery_solved.md`](stall_floor_mystery_solved.md)**
   (updated) - the OpenBLAS finding's trigger, not just its mechanism,
   is now directly confirmed: `import numpy` alone spawns the full
   8-thread pool (traced via `/proc/<pid>/task`, before any BLAS call
   and before any paulikit code runs). Prompted by being asked
   directly whether the mystery was "solved totally without any
   doubts" - it wasn't yet at that point, and now is.

10. **[`tbb_not_actually_used_finding.md`](tbb_not_actually_used_finding.md)** -
    corrects an assumption this whole investigation (and the earlier
    Google AI Mode transcript that motivated it) had been carrying:
    TBB is not actually invoked in the production decompose path
    today. `fwht_pauli_terms` calls the *serial* label-generation
    kernel, not the TBB-parallelized one - confirmed both by reading
    `pauli_label_native.pyx`'s own source (which documents Phase 3a
    already found parallelizing this specific loop barely helped) and
    by directly sampling OS thread counts before/during/after a
    native-kernel call (stays flat at 8 throughout, no TBB workers
    appear). Retroactively explains finding 8's null compiler-flags
    result even more completely: `-march=native` etc. apply to code
    that compiles but never runs in the hot path.

## Current honest state (as of the last finding above)

- **Confirmed root cause**: `fwht_pauli_coefficients` densifies a
  sparse result into a full `(dim, dim)` array, then
  `fwht_pauli_terms` re-scans the whole thing. Cache-miss ratio and
  memory-stall cycles both scale cleanly with how much that array
  exceeds cache size. This is real, robustness-relevant (see the
  N=150 OOM), and not explained away by compiler flags or TBB
  behavior.
- **Not yet designed or implemented**: an actual fix. The open,
  explicitly-flagged tension to resolve during design: Phase 3b's
  sparse-*computation* optimization must not be accidentally traded
  away for a "cache-friendlier" dense layout, or vice versa - any fix
  needs to be measured, not assumed, against both the cache-locality
  metrics here and Phase 3b's original sparsity-computation gains.
- **Known confound to control for in future measurements**: OpenBLAS
  thread-pool noise. Set `OPENBLAS_NUM_THREADS=1` for any new
  `perf`-based measurement in this directory unless specifically
  investigating BLAS behavior itself.
- **Known machine limitation**: this investigation's dev machine
  OOMs at N=150 with the current code. A fix should make N=150 (and
  beyond) memory-safe by making memory usage proportional to
  `n_active x dim` rather than `dim^2` - a prediction to verify once
  a fix exists, not yet measured.

## Extending this investigation

If you're picking this up fresh: run the quick-start scripts above on
your own hardware first, compare against the numbers in each doc, and
note where your results diverge (different CPU generation, cache
sizes, core count, or NumPy/OpenBLAS build will all shift the
specifics even if the structural findings hold). `steady_state_decompose.py`
is the right base to extend for any new in-process, warmed-up
measurement - avoid one-shot CLI timing for anything below roughly
N=50, where finding 6 showed process-startup noise can dominate or
distort the signal.
