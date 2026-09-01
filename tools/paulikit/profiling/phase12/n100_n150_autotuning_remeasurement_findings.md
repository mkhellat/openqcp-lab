# Real N=100/N=150 re-measurement of Phase 12's auto-tuning: real wins, two real bugs

Recorded 2026-09-01, at direct user request to verify Phase 12's
auto-tuning (`autotune.py`, `fwht.auto_decompose()`) end-to-end, not
mocked - the implementation session only unit-tested the formulas via
monkeypatching (`tests/test_autotune.py`), with no real N=100/N=150
wall-clock measurement. This document is that measurement, plus two
real bugs found along the way. Machine: same as every other N=150
finding in this project (15.7 GiB RAM, 8 cores, 8 MiB shared L3, per-
core L2 256 KiB).

## Headline result: the auto-tuned `chunk_size` is a real, substantial win

| N | dim | fixed (chunk_size=256) | auto-tuned (chunk_size=8) | speedup |
|---|---|---|---|---|
| 100 | 8192 | 16.17s (mean of 5) | 6.96s (mean of 5) | **2.32x** |
| 150 | 16384 | 69.32s | 33.90s | **2.04x** |

Both runs produced identical term counts (20,299,776 at N=100;
91,652,096 at N=150) confirming correctness is unaffected - this is a
pure speedup, not a different (and possibly wrong) computation.
`autotune.recommended_chunk_size(dim)` returned **8** at both N=100
and N=150 on this machine - the floor constant
(`_min_chunk_size_floor()`), not a cache-derived value, because the
empirically-measured L2 boundary (262144 bytes, matching this
machine's real per-core L2 exactly - see
`cache_probe_extension_findings.md`) divided by each row's byte size
(`dim * 16`) rounds down to 2 at dim=8192 and to 1 at dim=16384, both
below the floor of 8.

This means the win being measured here is **not yet actually testing
the cache-boundary-targeting part of the formula** - at both N tested,
`recommended_chunk_size` degenerates to "use the floor," which
happens to coincide with `chunk_size_cache_locality_findings.md`'s
own August sweep finding that very small `chunk_size` values
consistently beat 256 (e.g. N=100: `chunk_size=4` gave 12.633s vs.
`chunk_size=256`'s 20.508-22.369s in that original sweep - the same
qualitative direction, now confirmed again independently and via a
different code path). The floor is doing real, useful work here, but
this run doesn't yet exercise the cache-targeting branch at a large
enough `dim` to matter - see "What this does NOT show."

## Correctness sanity check (N=25/50, not mocked)

`auto_decompose_correctness_check.py` ran `auto_decompose()` end-to-
end (no mocking) at N=25 and N=50, comparing against `fwht_pauli_terms`
as ground truth:

```
N=25 dim=512 estimated_dense_bytes=4,194,304 budget=11,091,021,824
  expected_path=dense actual_path=dense OK
  terms=78336 max_coefficient_error=0.000e+00 elapsed=0.0656s
N=50 dim=2048 estimated_dense_bytes=67,108,864 budget=11,091,021,824
  expected_path=dense actual_path=dense OK
  terms=1261568 max_coefficient_error=0.000e+00 elapsed=1.0251s
ALL CORRECTNESS CHECKS PASSED
```

Bit-identical results (`max_coefficient_error=0.0`) at both N,
confirming `auto_decompose()`'s dense-path branch produces exactly the
same output as calling `fwht_pauli_terms` directly - it is a pure
routing decision, not a different computation.

## Bug 1: `auto_decompose()`'s dense-path memory estimate is a real underestimate — genuinely dangerous on a memory-constrained host

`auto_decompose()` estimates the dense path's peak footprint as
`dim**2 * 16` bytes (one `(dim, dim)` complex128 array) and compares
against half the available memory budget. At N=150 (dim=16384) this
estimate is 4.00 GiB. **The real dense path failed to complete under
both a `ulimit -v 8000000` (~7.6 GiB) cap and a `ulimit -v 12000000`
(~11.4 GiB) cap** - nearly 3x the naive estimate, and still not
enough:

```
--- ulimit -v 8000000, via auto_decompose() end-to-end ---
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 699.
MiB for an array with shape (11189, 16384) and data type uint32
  (raised inside fwht_pauli_coefficients's _popcount_array call)

--- ulimit -v 12000000, isolated fwht_pauli_terms(padded) call ---
MemoryError
  (raised inside _pauli_label_batch -> _native.pauli_label_batch)
```

Both failures were **caught cleanly as Python exceptions** (`ulimit
-v` triggers a catchable `MemoryError`/`ArrayMemoryError`, not a
kernel OOM-kill) - no crash, no corrupted state, no hung process. But
the second run's real system memory did climb to a genuinely
uncomfortable point before Python's own allocator gave up: `free -h`
polling during that run showed available memory drop from ~10 GiB to
as low as **177 MiB free / 1.1 GiB available**, with swap usage
climbing to 2.5 GiB, before the process's own `MemoryError` let it
exit and memory recovered fully (back to 12 GiB available within
seconds of exit). No system-wide OOM-kill occurred in this instance,
but this was close enough to real memory pressure that a host with
less headroom than this one's 15.7 GiB, or a host already running
other memory-hungry processes, could plausibly hit a real OOM-kill
rather than a clean Python exception - especially since the failure
point (which specific allocation fails first) is not deterministic
across the pipeline's several large intermediate arrays.

**Why this matters**: `_DENSE_MEMORY_SAFETY_FRACTION = 0.5` was
chosen (per `PLAN.md`'s own "known gaps" note) as "a reasonable-
looking placeholder, not derived from measurement" - this measurement
shows the placeholder is not just imprecise but **wrong in the unsafe
direction**. `auto_decompose()`'s whole reason for existing is to
protect a caller (especially on a memory-constrained or shared HPC
node) from an accidental dense-path OOM; a formula that underestimates
real peak usage by ~3x undermines exactly that guarantee. This is not
a performance nitpick - it is a correctness/safety gap in the feature
as shipped.

**Root cause, from the two tracebacks**: the dense path allocates
several large intermediate arrays beyond the final `(dim, dim)`
coefficient array the estimate accounts for - at minimum an
intermediate `uint32` array shaped `(n_active, dim)` inside
`_popcount_array` (699 MiB observed at `n_active=11189`, not the full
`dim=16384` rows, meaning this specific array is smaller than the
final coefficient array yet still contributed to exhausting the
budget alongside everything else already resident) and further
allocations inside `_pauli_label_batch`/the native label kernel. The
naive `dim**2 * 16` estimate accounts for only one of several
concurrently-live arrays in the real pipeline - it was never a
worst-case bound of the *whole* dense code path, only of its final
output array.

**Not fixed in this document** - reporting, not silently patching,
per the task's own instruction to stop and report bugs rather than
work around them quietly. A real fix needs either (a) a substantially
larger safety margin (e.g. `_DENSE_MEMORY_SAFETY_FRACTION` well below
0.5, informed by a proper peak-RSS measurement across several
representative N, not just N=150), or (b) accounting for the other
known-large intermediate arrays explicitly in the estimate, or (c)
both. This is flagged as the highest-priority follow-up from this
whole re-measurement.

## Bug 2: the cache probe is not idempotent when called repeatedly in the same process

Discovered while investigating an unexpected discrepancy in
`recommended_chunk_size(8192)` between two ad-hoc interactive checks
(one returned 8, another returned 32, from what looked like identical
inputs). Traced to calling the underlying probe twice in the same
process:

```
$ python3 -c "from paulikit._native import cache_probe as c; \
  print(c.probe_cache_boundaries()[0]); print(c.probe_cache_boundaries()[0])"
(8192, 5.85)
(8192, 14.17)      # 2.4x higher on the immediately-following call
```

Repeated over 5 trials, this reproduces probabilistically (1 of 5
trials showed a 2.4x jump on the small-buffer end; the rest were
within ~10% noise) - not a hard determinism bug, but a real,
measurable instability. When it does occur, the elevated small-buffer
reading blurs `_detect_l2_boundary_bytes_via_probe()`'s ratio-based
boundary detection (the L1-to-L2 jump, normally a clean ~1.4x, drops
below the 1.3x threshold and gets missed, pushing the detected "L2
boundary" out to a much larger, wrong buffer size - 4 MiB was observed
in one such case, 16x too large).

**Real-world impact is currently limited but not zero**:
`recommended_chunk_size()` calls the probe **at most once per
process** (its own module-level cache short-circuits on every
subsequent call - see `autotune.py`'s own docstring) and 10/10 fresh-
process calls in this session's testing returned the correct,
consistent value (8) - so this bug does not currently corrupt real
`auto_decompose()`/`fwht_pauli_terms_iter` usage through the public
API as shipped. It would matter if: (a) a future caller invokes
`paulikit._native.cache_probe.probe_cache_boundaries()` directly,
bypassing `autotune`'s cache, or (b) `autotune`'s own cache is ever
reset and re-primed within a single long-lived process (not something
the current API does, but plausible for a future long-running-service
use case), or (c) the *first* call itself happens to be an unlucky
noisy one - this document's evidence is that first calls were
reliable across 10 trials, but 10 trials is not proof of zero
probability, only a bound on it.

**Not fixed in this document**, same reporting-not-patching stance as
Bug 1. The existing repeat-and-minimum mitigation
(`cache_probe_extension_findings.md`) operates *within* one
`probe_cache_boundaries()` call's repeated timed passes at a given
buffer size; it does not protect against a *between-calls* systematic
shift of the kind observed here (which looks like a different noise
regime than the preemption-outlier noise that mitigation targets -
possibly cache/TLB/branch-predictor state left over from the first
call's own large `mmap` allocations affecting the second call's small-
buffer measurements, though this was not conclusively diagnosed).

## Was N=150 tested through the streaming path or the dense path?

Both, deliberately, to answer this exactly rather than assume it -
this was flagged as an open question before running anything. On this
machine (10-11 GiB typically available), **`auto_decompose()` picks
the *dense* path at N=150** (`estimated_dense_bytes=4.00 GiB` sits
under `budget * 0.5 ≈ 5.7 GiB`), which is what surfaced Bug 1 above -
had `auto_decompose()` picked streaming here (as it would on a
machine with less available memory, or once Bug 1's estimate is
corrected upward), Bug 1 would not have been visible in this
particular run. The streaming-path chunk_size comparison (the 2.04x
headline number) was measured by calling `fwht_pauli_terms_iter`
directly with each chunk_size, independent of which path
`auto_decompose()` itself would choose - this is the fair,
apples-to-apples comparison the task asked for.

## Method

- `auto_decompose_correctness_check.py` (this directory) - N=25/50,
  full correctness check against `fwht_pauli_terms`.
- `n100_autotuned_chunk_size_comparison.py` (this directory) - N=100,
  5-repeat wall-clock comparison of `fwht_pauli_terms_iter` at
  chunk_size=256 vs. `autotune.recommended_chunk_size(dim)`, plus
  `auto_decompose()` itself, same-process, same methodology as
  `streaming_vs_dense_comparison.py`/`chunk_size_cache_locality_findings.md`.
- `n150_autotuned_chunk_size_comparison.py` (this directory) - N=150,
  single-run wall-clock comparison (single run, not 5-repeat, given
  each N=150 pass costs 30-70s - matches this project's existing
  N=150 findings docs' own convention of single, `perf`-stat-backed
  runs rather than repeated wall-clock sampling at this scale), plus
  `auto_decompose()` end-to-end (this is what surfaced Bug 1). Run
  under `ulimit -v 8000000` initially.
- `n150_auto_decompose_dense_peak_check.py` (this directory) -
  follow-up isolating just the dense-path call under a larger
  `ulimit -v 12000000` cap, to find whether Bug 1 was an artifact of
  an artificially-tight cap or a genuine underestimate (confirmed the
  latter - failed again, at a different allocation point).
- All large runs monitored with `free -h` polling every 10-15s
  throughout (this project's standard safety harness) - see Bug 1's
  own section for the memory-pressure numbers observed.
- Full test suite (`pytest -q`, 99 tests) re-run at the end - all
  pass, no regressions from anything in this document (both bugs
  found are pre-existing behavior of already-shipped code being
  exercised for the first time at real scale, not something newly
  broken by this measurement session).

## What this does NOT show

- **Does not exercise the cache-boundary-targeting branch of
  `recommended_chunk_size` at meaningful scale** - both N=100 and
  N=150 hit the floor (8), not a cache-derived value, because `dim`
  is large enough that `L2_bytes // (dim * 16)` rounds to 1-2. A
  smaller N (where `dim` is small enough for the cache-derived value
  to exceed the floor) would be needed to actually test that branch's
  real-world benefit, not just the floor's. Not attempted here -
  scope was N=100/150 specifically per the task.
- **Does not re-derive the small-chunk-size floor itself** (still the
  placeholder constant of 8, flagged as an open gap in `PLAN.md`
  already) - this measurement's own results (chunk_size=8 beating 256
  by 2x+ at both N tested) are consistent with the floor being *at
  least* not obviously wrong at this scale, but do not test whether an
  even smaller value (4, 2, 1) would do better or would instead hit
  the per-chunk fixed-overhead wall the floor exists to avoid (per the
  original N=25 `chunk_size=1` finding of 57% slower than dense).
- **Does not fix either bug found** - both are reported, not patched,
  per the task's explicit instruction. Bug 1 in particular should be
  treated as blocking before recommending `auto_decompose()` for
  memory-constrained/HPC use without a manual `chunk_size`/streaming
  override, which is exactly the scenario Phase 12's design was
  supposed to protect.
- **Does not test the cgroup-aware memory-budget path for real** (no
  live cgroup-capped run was achieved on this host, per the earlier
  `configure` fix's own "What this does NOT show" - same gap, not
  re-attempted here).
- **Single-run (not repeated) measurement at N=150** for both the
  chunk_size comparison and the dense-path peak-memory check, unlike
  N=100's 5-repeat protocol - run-to-run variance at N=150 was not
  characterized here (matches this project's existing N=150 findings
  docs' own convention, but is worth noting as a limitation of this
  specific document).
- **The cache-probe non-idempotency (Bug 2) was not root-caused
  precisely** - flagged as "possibly" TLB/branch-predictor/cache
  residue from the first call, not confirmed via `perf stat` or
  further isolation. A dedicated follow-up would be needed to nail
  down the mechanism if it's ever worth fixing.
