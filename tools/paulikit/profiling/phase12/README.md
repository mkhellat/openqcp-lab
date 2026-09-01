# Phase 12: `chunk_size` as a cache-locality lever; auto-tuning scoping

Scoped 2026-08-27, design finalized 2026-09-01, implementation in
progress. See `../../PLAN.md` Phase 12 for the design/implementation
narrative and the two distinct auto-decisions in scope (auto-picking
`chunk_size`, and auto-picking streaming vs. dense); this directory
holds the supporting measurement.

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

## Takeaway if you only read one thing

`chunk_size=256` (used throughout this project's own Phase 9/10 docs)
is not a tuned value — it is measurably worse than smaller alternatives
at every N tested, for a confirmed cache-locality reason on top of its
original memory-footprint rationale. This motivates PLAN.md Phase 12:
a principled, auto-computed default (targeting a working-set size that
fits a cache level, informed by an *empirical* pointer-chase probe
rather than a declared-topology source found to be ambiguous), with
manual override always available per the user's explicit,
non-negotiable requirement, and explicit HPC-node correctness
(cgroup-aware memory budgeting, per-node cache re-probing) as a design
target, not an afterthought. Design finalized 2026-09-01; the cache
probe extension itself is built and verified (see above) — the
`chunk_size`/streaming-vs-dense auto-tuning formulas that consume its
output are still being implemented, see `../../PLAN.md` Phase 12 for
current status.
