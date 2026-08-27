# Phase 12: `chunk_size` as a cache-locality lever; auto-tuning scoping

Scoped 2026-08-27, not yet designed in detail. See `../../PLAN.md`
Phase 12 for the design/implementation narrative and the two
distinct auto-decisions in scope (auto-picking `chunk_size`, and
auto-picking streaming vs. dense); this directory holds the supporting
measurement.

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

## Takeaway if you only read one thing

`chunk_size=256` (used throughout this project's own Phase 9/10 docs)
is not a tuned value — it is measurably worse than smaller alternatives
at every N tested, for a confirmed cache-locality reason on top of its
original memory-footprint rationale. This motivates PLAN.md Phase 12:
a principled, auto-computed default (targeting a working-set size that
fits a cache level, informed by the machine's actual cache sizes), with
manual override always available per the user's explicit,
non-negotiable requirement. Not yet designed in detail as of this
writing — see `../../PLAN.md` Phase 12 for the open questions (target
cache level, tradeoff against per-chunk fixed overhead, portable
cache-size detection).
