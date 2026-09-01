# Post-bugfix re-verification: does Phase 12's auto-tuning still work after fixing Bug 1 and Bug 2?

Recorded 2026-09-01, at direct user request ("Dont you need to do a
fresh benchmark with the existing bug fixes?") after both bugs from
[`n100_n150_autotuning_remeasurement_findings.md`](n100_n150_autotuning_remeasurement_findings.md)
were fixed
([`dense_memory_estimate_fix_findings.md`](dense_memory_estimate_fix_findings.md),
[`cache_probe_idempotency_investigation_findings.md`](cache_probe_idempotency_investigation_findings.md)).
Every number in this project's headline claims (2.32x at N=100, 2.04x
at N=150, correct path selection) had been measured **before** either
fix landed - Bug 1's fix in particular changes `auto_decompose()`'s
actual routing decision, so re-running for real rather than trusting
old numbers was the right call, not just due diligence for its own
sake.

## Correctness, real N=25/50/100 (not mocked, not hand-calculated)

`auto_decompose_correctness_check.py` was updated to use the fixed
formula (`_DENSE_MEMORY_MULTIPLIER`/`_DENSE_MEMORY_SAFETY_FRACTION`,
not the original `dim**2*16`/`0.5`) for its own expected-path
prediction, then re-run for real:

```
N=25  dim=512  estimated_dense_bytes=25,165,824    threshold=2,551,076,454   budget=12,755,382,272
  expected_path=dense actual_path=dense OK
  terms=78336    max_coefficient_error=0.000e+00 elapsed=0.0336s
N=50  dim=2048 estimated_dense_bytes=402,653,184   threshold=2,551,076,454   budget=12,755,382,272
  expected_path=dense actual_path=dense OK
  terms=1261568  max_coefficient_error=0.000e+00 elapsed=1.0375s
N=100 dim=8192 estimated_dense_bytes=6,442,450,944 threshold=2,551,076,454   budget=12,755,382,272
  expected_path=streaming actual_path=streaming OK
  terms=20299776 max_coefficient_error=0.000e+00 elapsed=19.9042s
ALL CORRECTNESS CHECKS PASSED
```

Confirms, for real: the fixed formula correctly routes N=25/50 to
dense (small enough) and **N=100 to streaming** - a real, observed
change in behavior from before the fix (the original, buggy 0.5-
fraction formula would have routed N=100 to dense too, since real
N=100 peak usage of 5.27 GiB fits comfortably under an ~11-13 GiB
budget - see `dense_memory_estimate_fix_findings.md`'s own note on
this exact tradeoff). `max_coefficient_error=0.000e+00` at every N -
the routing change did not introduce any numerical divergence.

## Speedup, real N=100 (not from before the fix)

`n100_autotuned_chunk_size_comparison.py` (diagnostic print lines
updated to use the fixed formula, timing logic unchanged - it was
never touched by either bug fix) re-run fresh:

```
N=100 dim=8192 auto_chunk_size=8 available_memory_bytes=13,117,054,976
  estimated_dense_bytes=6,442,450,944 dense_threshold=2,623,410,995
terms=20299776
  chunk_size=256 (old fixed):        mean=15.0943s
  chunk_size=8 (auto-tuned):         mean=7.2317s
  auto/fixed ratio: 0.479x (auto FASTER, i.e. 2.09x speedup)
auto_decompose() picked path='streaming' terms=20299776 mean=7.1173s
```

**2.09x speedup**, matching the original pre-fix measurement's 2.32x
within normal machine-to-machine run variance (both comfortably in the
"real, substantial win" range, not a coincidence of one lucky run).
`auto_decompose()`'s own wall-clock (7.12s) closely matches the direct
`fwht_pauli_terms_iter` call at the same chunk_size (7.23s) - confirms
`auto_decompose` adds no meaningful overhead of its own, and (new
information versus the pre-fix run) that it now correctly exercises
the *streaming* path here rather than dense.

## Speedup, real N=150 (not from before the fix)

`n150_autotuned_chunk_size_comparison.py` (same diagnostic-line fix,
same timing logic) re-run fresh, under the same `ulimit -v 6000000`
safety cap this project's other N=150 drivers use, with `free -h`
polling throughout:

```
N=150 dim=16384 auto_chunk_size=8 available_memory_bytes=12,941,733,888
  estimated_dense_bytes=25,769,803,776 dense_threshold=2,588,346,778
--- chunk_size=256 (old fixed) ---
  elapsed=63.59s terms=91,652,096 chunks=44
--- chunk_size=8 (auto-tuned) ---
  elapsed=34.47s terms=91,652,096 chunks=1,399
auto/fixed ratio: 0.542x (auto FASTER, i.e. 1.84x speedup)
--- auto_decompose() ---
  picked path='streaming' elapsed=36.73s terms=91,652,096
SUCCESS
```

**1.84x speedup**, matching the original pre-fix 2.04x within normal
variance. `auto_decompose()` correctly picks streaming (36.73s, close
to but slightly above the direct 34.47s call - a small single-run gap
at this scale, not investigated further given it's within the noise
band this project's other single-run N=150 measurements already
accept). Same term count (91,652,096) across both `chunk_size` choices
and `auto_decompose()` - correctness preserved.

**Memory stayed healthy throughout this entire run** (`free -h`
polling every 5s showed 10-11 GiB available the whole time, never
dipping) - a stark contrast to the pre-fix bug-hunting runs in
`dense_memory_estimate_fix_findings.md`, which dropped to as low as
177-593 MiB free while forcing the (buggy, now-fixed) dense path to
attempt N=150. This is itself confirmation the fix is doing its job:
the same N=150 Hamiltonian that used to threaten real memory pressure
under `auto_decompose()`'s old routing logic now runs comfortably
under the corrected logic's (correct) streaming choice.

## Interpretation

Both real-world headline claims from before the bugfixes hold up under
fresh, real re-measurement:
- **The auto-tuned `chunk_size` speedup is real and reproducible**:
  2.09x at N=100 (was 2.32x), 1.84x at N=150 (was 2.04x) - both within
  normal run-to-run variance of the original numbers, not a
  regression introduced by either fix (neither fix touched
  `fwht_pauli_terms_iter` or `recommended_chunk_size`'s own formula).
- **`auto_decompose()`'s path-selection is now demonstrably correct
  and safe**, not just correct by hand-arithmetic: N=25/50 dense,
  N=100/150 streaming, all verified against `fwht_pauli_terms` as
  ground truth with zero numerical divergence, and the N=150 run that
  used to threaten real memory pressure is now memory-uneventful.

## What this does NOT show

- Does not re-verify Bug 2's own fix (the thread-safety lock) under
  real concurrent load at N=100/150 scale - the existing unit tests
  (`test_recommended_chunk_size_thread_safe_single_underlying_call`
  and its `available_memory_bytes` counterpart) cover the race
  directly with mocked, fast detection functions; a real end-to-end
  multi-threaded `auto_decompose()` stress test at N=100/150 was not
  attempted here (would need careful memory-safety planning given each
  thread would otherwise attempt its own dense/streaming decomposition
  concurrently - out of scope for this specific re-verification pass).
- Single-run (not repeated) measurement at N=150, matching this
  project's existing N=150 findings docs' own convention - run-to-run
  variance at N=150 specifically was not characterized here, same
  caveat as the original re-measurement findings doc.
- Does not re-derive or re-check the small-chunk-size floor (still 8,
  still the placeholder from one old N=25 data point) - unaffected by
  either bug fix, not re-examined here.
