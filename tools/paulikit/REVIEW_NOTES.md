# paulikit review notes (2026-09-04)

Two parts: (A) Phase 13 parallelism study status from project memories /
RESUME-HERE checkpoints; (B) code-level issues found in a fresh pass.

**Status of (B), 2026-09-04**: High #1, High #2, Medium #3, and
Medium #4 all fixed, tested, committed, pushed (4 atomic commits).
Low #5 and Low #6 deliberately left unfixed — see their own entries
below for why.

Source for (A): Claude project memory
`~/.claude/projects/.../memory/project_paulikit_phase13.md`
(+ matching `profiling/phase13/*findings.md`).

---

## A. Phase 13 parallelism — where we left off

### Established (evidence-backed)

- `parallel_decompose` shipped; real speedup ~1.27–1.56× (not linear).
- Bugs already fixed in-tree: per-worker memory budget wiring; unbounded
  pool submission (`max_in_flight`).
- Thermal confound real on this machine (package → 100°C); cooldown
  protocol required before trusting wall-clock.
- Full sweep 14 configs × 10 reps (ANOVA + Tukey HSD): observed min
  **`w2_c1`** (2 workers / 1 physical core); statistically tied with
  `w4_c2` / `w3_c2`. Every 3–4-core config significantly slower.
  **Physical-core count (~2), not worker count, is what matters here.**
- Extra cycles when spreading cores = lower IPC (same instruction
  count), not extra work.
- Chunk_size retuning under contention does **not** close the gap.
- Code-specificity: synthetic ProcessPoolExecutor control **scales**
  on more cores; paulikit does **not** → ceiling is paulikit’s
  memory-traffic pattern (or memory-bound workloads), not generic
  multiprocessing.
- Bandwidth hypothesis confirmed via corrected child-inheriting
  `perf stat`: paulikit cache-miss ~21–60× synthetic. Refined
  mechanism: miss *ratio* flat w2→w8; effective parallelism stuck
  ~2× while workers block in IPC — serialization on shared memory
  traffic intensity.
- Methodology caveat: older `--no-inherit` perf numbers only measured
  the launcher; not retroactively re-run.

### RESUME-HERE #4 next steps (not done)

1. **Cache-blocked / D&C WHT** — name only (MIT 6.172 lec8 analogy);
   no design, prototype, or footprint math yet.
2. Optionally strengthen **greedy-scheduler bound violation** claim
   with a proper multi-P / T_inf study.

### Traffic-intensity suspect (2026-09-04) — tested

Dense WHT / 14-touch / even large-IPC controls **scale** on w8_c4
(~2.2–2.7× vs w2_c1; eff. P ~6–7.6). Paulikit does not (0.89×).
So **dense buffer traffic alone is not sufficient**. Next isolate:
irregular gather/scatter (+ operator resident set). See
`profiling/phase13/traffic_intensity_findings.md`.

### Not established

- Whether 2-core optimum generalizes beyond i7-8550U.
- Whether cache-blocked WHT would restore multi-core scaling.
- Whether gather/irregular access is sufficient (not yet tested).

---

## B. Fresh code review (checkpoint / autotune)

### High

1. **`recommended_chunk_size(dim)` caches without `dim`** (`autotune.py`)
   - Confirmed: after `dim=512` → 32, cached `dim=16384` still 32;
     fresh call → 2. Cache L2 bytes; compute per `dim`.
   - **FIXED** 2026-09-04: only the dim-independent L2-boundary probe
     result is cached now (`_cached_l2_bytes`); `recommended_chunk_size`
     recomputes the actual chunk_size fresh every call. Regression test
     added (`test_recommended_chunk_size_recomputes_per_dim_despite_l2_cache`).

2. **Crash mid-append can break resume** (`fwht.py`)
   - Truncated JSONL line → `json.loads` fails on resume.
   - Atomic write or skip/truncate bad trailing line on load.
   - **FIXED** 2026-09-04: new shared `_read_checkpoint_triples` helper
     tolerates a `json.JSONDecodeError` on the file's LAST line only (an
     earlier line failing to parse is still a real, raised corruption).
     Regression test added
     (`test_streaming_checkpoint_resume_survives_truncated_inflight_line`).

### Medium

3. Duplicate triples on over-record resume (COO path; dict last-wins OK).
   - **FIXED** 2026-09-04: `_read_checkpoint_triples` now dedupes by
     `(x, z)`, keeping the last occurrence - closes the gap for any
     consumer that reads the raw checkpoint arrays directly rather than
     through a dict-combine path. Regression tests added
     (`test_read_checkpoint_triples_dedupes_by_x_z_keeping_last`,
     `test_streaming_checkpoint_resume_dedupes_over_recorded_chunk`).
4. Parallel memory budget ignores per-worker **operator** copies.
   - **FIXED** 2026-09-04: new `_per_worker_resident_bytes` estimates
     each worker's fixed resident footprint (operator copy + sorted
     setup arrays, `O(nnz)` for sparse input) and
     `_recommended_parallel_chunk_size` now subtracts it from the
     per-worker budget before bounding chunk_size. Regression tests
     added in `test_parallel_decompose.py`.

### Low

5. No `fsync` before progress advance.
   - **Deliberately NOT fixed**: `fsync` on every chunk's checkpoint
     append would add a real per-chunk disk-sync cost on a path whose
     whole design goal is to stay "negligible next to each chunk's
     O(chunk_size * dim * log dim) transform cost" (see
     `_append_checkpoint_chunk`'s own context) - at N=150/chunk_size=2,
     chunks run ~4.7ms each (Phase 13 measurement), a scale where an
     `fsync` call's few-ms cost is not negligible. The existing
     triples-then-progress-marker write order plus the Bug 2 fix above
     already make the actual failure mode (process crash mid-write)
     safely resumable - only a harder failure (power loss) is left
     unprotected, judged not worth the throughput cost per the
     performance-over-reliability priority for this workload.
6. `available_memory_bytes` process-lifetime cache can go stale.
   - **Deliberately NOT fixed**: this cache sits on the auto-tuning hot
     path (`auto_decompose`/`parallel_decompose`'s memory-budget
     decisions), and per-process caching there is a deliberate existing
     design choice, not an oversight (see the function's own
     docstring). Removing or time-bounding it reintroduces real
     per-call `/proc/meminfo` I/O for a benefit that only matters if
     available memory changes meaningfully within one process's
     lifetime - not this package's typical single-run usage pattern.
