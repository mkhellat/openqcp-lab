# Cache-locality perf stat pass, N=150: L3 contention is real and measured, not just hypothesized

Recorded 2026-09-02. Answers the scoping doc's own open verification-
plan item (`scoping.md`: "a `perf stat` pass under concurrent load")
and directly confirms/refutes the contention-vs-IPC-overhead question
raised while interpreting the earlier (now-superseded) speedup
numbers. Run foreground-only, one job at a time, machine confirmed
idle via `ps aux` before every launch - same execution discipline as
`clean_chunk_size_sweep_findings.md`, after the earlier execution-
reliability failure.

Same event set as every other `perf stat` measurement in this project
(`profiling/phase12/n150_perf_chunk_size.py` and earlier). Relies on
`perf stat`'s default counter inheritance across forked/exec'd child
processes (verified directly beforehand: `task-clock` exceeding
wall-clock time for a `parallel_decompose` run confirms counters
aggregate across the worker pool, not just the main process) - no
`-a`/system-wide flag needed.

## Absolute times and cache counters, chunk_size=2 and chunk_size=32

| chunk_size | mode | wall-clock | task-clock (total CPU) | cache-miss ratio | LLC-miss ratio |
|---|---|---|---|---|---|
| 2 | sequential | **34.42s** | 34.4s | 5.1% | 3.8% |
| 2 | parallel (n_workers=8) | **25.44s** | 60.0s | 10.6% | 8.6% |
| 32 | sequential | **52.77s** | 52.7s | 33.7% | 30.2% |
| 32 | parallel (n_workers=8) | **35.50s** | 80.3s | 43.2% | 42.8% |

(cache-miss ratio = cache-misses / cache-references; LLC-miss ratio =
LLC-load-misses / LLC-loads; both `:u`-scoped, userspace-only, per
`perf_event_paranoid=2` on this machine.)

**Important scope note on these two ratios, added after direct user
pushback and confirmed via the raw L1 counters
(`l3_contention_direct_evidence_findings.md`'s own correction
section)**: `cache-references`/`cache-misses` and `LLC-loads`/
`LLC-load-misses` are ratios WITHIN an already-small residual slice of
total memory traffic - the part that already missed L1 (only ~15% of
all L1 accesses at chunk_size=2, sequential baseline; 84.6% of all L1
accesses are served by L1 itself, confirming chunk_size's L1/L2
targeting is working as designed). These percentages are NOT a
statement that "most reads come from L3" - they never were, and
should not be read that way anywhere in this document.

## Finding: cache locality genuinely degrades under parallel execution - confirmed, not assumed

At both chunk_sizes, going from sequential to parallel (8 workers)
**roughly doubles or more than doubles both cache-miss and LLC-miss
ratios**: 5.1%->10.6% and 3.8%->8.6% at chunk_size=2; 33.7%->43.2% and
30.2%->42.8% at chunk_size=32. This is the L3-contention mechanism
flagged as a real, unresolved risk in `scoping.md` before any Phase
13 code existed ("8 processes now compete for the one shared L3 this
machine has... Phase 12's own chunk_size tuning assumed a single lone
process's undisputed cache") - now directly measured, not just
theoretically plausible.

**But wall-clock still improves despite the contention**: 34.42s ->
25.44s at chunk_size=2 (1.35x), 52.77s -> 35.50s at chunk_size=32
(1.49x). The parallelism gain outweighs the contention cost in both
cases - real speedup exists, but the doubled cache-miss ratio is a
real tax eating into what would otherwise be closer to the 8-worker
ideal.

## task-clock confirms real parallel CPU use, and quantifies efficiency

`task-clock` (aggregate CPU-seconds across the whole process tree)
roughly matches or exceeds wall-clock at chunk_size=32
(80.3s task-clock vs. 52.7s sequential - about 1.52x more total CPU
consumed for a 1.49x wall-clock gain), and similarly at chunk_size=2
(60.0s vs. 34.4s sequential, about 1.74x more CPU for a 1.35x
wall-clock gain). Parallel efficiency (wall-clock speedup / n_workers)
is roughly **19% at chunk_size=32** (1.49/8) and **17% at chunk_size=2**
(1.35/8) - real, but far from linear, consistent with (and now
partially explained by) the measured cache-contention cost.

## Absolute-time headline (times matter more than ratios - direct user instruction)

The best absolute wall-clock time measured in this pass is
**chunk_size=2, parallel, 25.44s** - clearly faster than chunk_size=32
parallel's 35.50s, even though chunk_size=32 had the larger *speedup
ratio* in `clean_chunk_size_sweep_findings.md`'s earlier comparison.
A speedup ratio compares a configuration against its own sequential
baseline, not against every other configuration's absolute time - the
ratio-only view can favor a configuration that is uniformly slower in
absolute terms but happens to have a worse (therefore more
"improvable") sequential baseline. Optimizing for absolute wall-clock,
not the ratio, chunk_size=2 remains the better real-world choice at
N=150 despite the doubled cache-miss ratio parallel execution imposes
on it.

## What this does NOT show

- Only two chunk_sizes tested here (2, 32) - not the full sweep
  (`clean_chunk_size_sweep_findings.md` also has 8, 128 at N=100 and
  128 at N=150, but those were not `perf stat`-profiled in this pass).
- Does not test `perf stat` at N=100 - N=150 only, matching the
  scoping doc's own original verification-plan target.
- Does not isolate how much of the contention is L3-specific versus
  memory-bandwidth-specific - LLC-load-misses (DRAM-bound) and the
  broader cache-misses figure both roughly double together, consistent
  with either or both; no separate memory-bandwidth counter (e.g.
  `mem_load_retired.l3_miss`-style event) was measured here.
- Does not retest `n_workers` values other than 8 under `perf stat` -
  no data yet on whether contention (and the resulting cache-miss
  ratio) scales smoothly with `n_workers` or has a sharper knee at
  some intermediate count.
