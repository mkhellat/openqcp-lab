# Phase 12: `chunk_size` as a cache-locality lever; auto-tuning scoping

Scoped 2026-08-27, design finalized and implemented 2026-09-01,
real-world re-measured the same day (2 real bugs found, not yet
fixed - see below). See `../../PLAN.md` Phase 12 for the
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

**Two real bugs found, not yet fixed:**
1. **`auto_decompose()`'s dense-path memory estimate underestimates
   real peak usage by ~3x, in the unsafe direction.** At N=150
   (dim=16384), the estimate is 4.00 GiB; the real dense path failed
   under both a ~7.6 GiB and an ~11.4 GiB memory cap (clean Python
   exceptions, not a crash, but real system memory dropped to as low
   as 177 MiB free during one run before recovering). This undermines
   the exact guarantee `auto_decompose()` exists to provide,
   especially on memory-constrained/shared HPC nodes - flagged as the
   highest-priority follow-up.
2. **The cache probe is not idempotent when called repeatedly in the
   same process** - a second `probe_cache_boundaries()` call can
   (probabilistically, ~1 in 5 trials observed) return a wildly
   different, wrong L2 boundary due to elevated small-buffer noise.
   Real-world impact is currently limited by `recommended_chunk_size`'s
   own per-process caching (calls the probe at most once), confirmed
   reliable across 10/10 fresh-process trials - but the underlying
   instability is real and not yet understood precisely.

## Takeaway if you only read one thing

`chunk_size=256` (used throughout this project's own Phase 9/10 docs)
is not a tuned value — auto-tuning it delivers a real, measured 2x+
speedup at both N=100 and N=150 against that old default, with
correctness confirmed via identical term counts. But the
implementation as shipped 2026-09-01 has a real, unfixed safety gap:
the streaming-vs-dense memory-budget estimate underestimates the dense
path's true peak footprint by roughly 3x, which could plausibly cause
a real OOM on a less memory-generous host than this one - this should
be treated as a blocker before recommending `auto_decompose()` (as
opposed to `fwht_pauli_terms_iter` with an explicit `chunk_size`) for
memory-constrained or HPC use.
