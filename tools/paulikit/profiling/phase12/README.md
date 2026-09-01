# Phase 12: `chunk_size` as a cache-locality lever; auto-tuning scoping

Scoped 2026-08-27, design finalized and implemented 2026-09-01,
real-world re-measured the same day (2 real bugs found, both fixed
the same day - see below). See `../../PLAN.md` Phase 12 for the
design/implementation narrative and the two distinct auto-decisions in
scope (auto-picking `chunk_size`, and auto-picking streaming vs.
dense); this directory holds the supporting measurement.

This phase was prompted by the user directly questioning whether
`chunk_size=256` — used as an example default throughout Phase 9/10's
own findings docs — was actually well-chosen, rather than accepted as
tuned.

## Finding

[`chunk_size_cache_locality_findings.md`](chunk_size_cache_locality_findings.md) -
a controlled `chunk_size` sweep at N=25/50/100
(`streaming_vs_dense_comparison.py`'s general shape), then `perf stat`
at N=100 across `chunk_size` in {4, 32, 256}
(`chunk_perf_n100_cs4.py`/`chunk_perf_n100_cs32.py`/`chunk_perf_n100_cs256.py`,
raw output in the matching `.txt` files). Result: `chunk_size=256` is
measurably suboptimal at **every** N tested — never the fastest option
in any comparison. The mechanism is confirmed, not just correlated:
cache-miss ratio scales cleanly with `chunk_size`'s working-set size
(`chunk_size * dim * 16 bytes`) relative to this machine's cache
hierarchy — 7.3% at `chunk_size=4` (fits in L2), 21.3% at
`chunk_size=32` (fits in L3), 44.6% at `chunk_size=256` (4x over L3).

`chunk_size` was designed (Phase 6/9) purely as a memory-footprint
bound; this finding shows it is independently, and often more
impactfully at N≤100 scale, a **cache-locality lever** — the two
considerations point the same general direction (smaller is generally
better) but for different reasons, and neither alone justified the
example default of 256.

## Building the cache-latency probe: two real timing bugs

[`cache_probe_extension_findings.md`](cache_probe_extension_findings.md) -
the finalized design (`../../PLAN.md` Phase 12) uses an *empirical*
pointer-chase probe rather than declared cache-topology sources
(`lscpu`'s "L2 1 MiB" was found to be a cores-aggregate artifact — see
PLAN.md's design section). Building that probe
(`src/paulikit/_native/cache_probe.{pyx,c,h}`) surfaced two real,
non-obvious measurement bugs, both found and fixed via direct
measurement: (1) `clock_gettime`-based timing was corrupted by this
machine's `powersave` CPU-frequency governor swinging 400 MHz-3.3 GHz
mid-measurement (fixed via hardware cycle counters — RDTSCP/CNTVCT_EL0/
`rdtime`, matching `configure`'s own asm probe's instruction choice);
(2) even with cycle counters, occasional scheduler preemption still
corrupted individual readings (fixed via CPU pinning +
repeat-and-take-minimum). Confirmed clean, monotonic, cache-hierarchy-
matching results across 4 repeated runs after both fixes.

## Real N=100/N=150 re-measurement: a real 2x+ win, and two real bugs

[`n100_n150_autotuning_remeasurement_findings.md`](n100_n150_autotuning_remeasurement_findings.md) -
after `auto_decompose()`/`autotune.py` were implemented and only
unit-tested via mocking, this re-runs them for real against the actual
N=100/N=150 Hamiltonians. **The auto-tuned `chunk_size` is a genuine,
substantial win**: 2.32x faster at N=100 (16.17s -> 6.96s), 2.04x
faster at N=150 (69.32s -> 33.90s), both against the old fixed
`chunk_size=256`, with identical term counts confirming correctness is
unaffected. On this machine `recommended_chunk_size` returns the
floor value (8) at both N, not yet exercising the cache-boundary-
targeting branch of the formula at meaningful scale - see the findings
doc's own "What this does NOT show."

**Two real bugs found:**
1. **FIXED same day** - see
   [`dense_memory_estimate_fix_findings.md`](dense_memory_estimate_fix_findings.md).
   `auto_decompose()`'s dense-path memory estimate underestimated real
   peak usage by 3x+ (confirmed even worse on re-check: still failing
   under an 18 GiB cap, 4.5x the naive estimate), in the unsafe
   direction. Root-caused by hand-tracing every concurrently-live array
   in the dense path, confirmed via a real `resource.getrusage`
   peak-RSS sweep at N=50/75/100 (5.3-6.5x measured ratio). Fixed via
   a `_DENSE_MEMORY_MULTIPLIER = 6.0` constant plus independently
   tightening `_DENSE_MEMORY_SAFETY_FRACTION` from `0.5` to `0.2` -
   deliberately conservative (now sometimes streams where dense would
   have fit, an intentional tradeoff for a safety-critical decision,
   not an oversight). 2 new regression tests pin both constants and
   the N=150 real-numbers outcome.
2. **Investigated and resolved same day** - see
   [`cache_probe_idempotency_investigation_findings.md`](cache_probe_idempotency_investigation_findings.md).
   Root-caused precisely: a large-buffer pass leaves cache/TLB state a
   later small-buffer warm-up doesn't fully clear. A bigger warm-up
   floor helped but never fully eliminated it - reverted once
   confirmed this exact scenario cannot occur in shipped code, since
   `recommended_chunk_size` calls the probe at most once per process.
   Re-checked (per direct instruction) whether that per-process cache
   really holds under real parallelism: safe for multi-process HPC
   jobs and `os.fork()` by construction, but a genuine gap was found
   and fixed - the cache wasn't thread-safe against concurrent
   *threads* in one process. Fixed with a lock, verified via mutation
   testing that the new regression tests actually catch the race.

## Takeaway if you only read one thing

`chunk_size=256` (used throughout this project's own Phase 9/10 docs)
is not a tuned value — auto-tuning it delivers a real, measured 2x+
speedup at both N=100 and N=150 against that old default, with
correctness confirmed via identical term counts. The implementation as
first shipped 2026-09-01 had a real safety gap (the streaming-vs-dense
memory-budget estimate underestimating the dense path's true peak
footprint by 3x+), **fixed the same day** - see
`dense_memory_estimate_fix_findings.md`. The cache-probe
non-idempotency was investigated, found to only be a theoretical issue
for the shipped code path, and a real (different) thread-safety gap
was found and fixed instead - see
`cache_probe_idempotency_investigation_findings.md`. Both bugs from
the re-measurement are now closed, verified via mutation testing where
applicable and 103 tests total passing - `auto_decompose()` is safe to
recommend for memory-constrained/HPC use as shipped.
