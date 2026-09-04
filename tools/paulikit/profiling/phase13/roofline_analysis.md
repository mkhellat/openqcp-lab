# Roofline analysis: bounding cache-blocked WHT's ceiling before building it (2026-09-04)

`dag_gst_master_analysis.md` motivated a cache-blocked/recursive WHT
(section 3c/4b) via Master's theorem, but explicitly could not predict
*how much* it could help - Master describes work/span asymptotics, not
memory-traffic magnitude. This closes that gap with the Roofline model
(Williams, Waterman, Patterson 2009), the standard tool in
performance-engineering practice for exactly this question: is a
kernel compute-bound or memory-bandwidth-bound, and by what factor?

This targets a single point in the Roofline framework: DAG B, the
per-chunk WHT butterfly (`_walsh_hadamard_transform_rows`), run
single-threaded within one worker (matches how it actually executes
today - see `dag_gst_master_analysis.md` section 1, "not exposed to
`ProcessPoolExecutor`"). It does **not** model the aggregate multi-
worker bandwidth contention (DAG C) - that is a separate, already
partially-explored question (see the blocked `uncore_imc` attempt in
`bandwidth_hypothesis_sweep.py`).

## 1. Measured single-core DRAM bandwidth (this machine)

`roofline_bandwidth_probe.py`: STREAM-triad-style `c = a + 2*b` over
256 MiB float64 arrays (~32x the 8 MiB shared L3, forcing genuine DRAM
traffic, not cache hits), single-threaded (`OPENBLAS_NUM_THREADS=1`),
thermal-cooldown-controlled (same protocol as every other measurement
in this investigation), 8 reps:

```text
mean bandwidth: 7.18 GB/s (stdev 0.23)
```

Sane against this CPU's spec: i7-8550U is dual-channel DDR4-2400
(~38.4 GB/s theoretical aggregate peak); a single core sustaining
~7.18 GB/s (~19% of theoretical peak) is consistent with the known
single-core memory-level-parallelism limits of this microarchitecture
- one core cannot saturate the full memory-controller bandwidth alone,
which is expected and not a measurement error.

## 2. Peak compute ceiling (this machine)

```text
peak_single_core_GFLOPS = base_clock_GHz * SIMD_width_doubles * FMA_units * 2
                        = 1.8 * 4 * 2 * 2
                        = 28.80 GFLOPS
```

AVX2 (256-bit = 4 doubles), 2 FMA units/core, base clock (not max
turbo 4.0 GHz - AVX2 workloads on this generation cannot sustain max
turbo, a well-documented "AVX offset" throttling effect; base clock is
the conservative, defensible choice for a sustained-throughput
ceiling).

**Ridge point** (AI where the bandwidth ceiling meets the compute
ceiling): `28.80 / 7.18 = 4.01 FLOPs/byte`. Below this AI, a kernel is
memory-bound; above it, compute-bound.

## 3. paulikit WHT kernel's arithmetic intensity

Per chunk (`chunk_size=2`, `dim=16384`, N=150's real workload):

- **Work**: 14 stages (`log2(16384)`), each stage does
  `chunk_size*dim/2 = 16384` independent complex butterfly pairs, each
  pair = 1 complex add + 1 complex sub = 4 real FLOPs (2 real +
  2 imaginary components, add and subtract).
  `total_flops = 16384 * 4 * 14 = 917,504` FLOPs per chunk.
- **Memory traffic, worst case** (no cache reuse across stages - the
  CURRENT code's actual behavior, since each stage's
  `transformed.reshape(...)` + slice-assign touches the full
  `(chunk_size, dim)` array = 512 KiB, which does **not** fit the
  256 KiB per-core L2, so each of the 14 stages plausibly re-reads +
  re-writes the whole array from/to a cache level beyond L2):
  `total_bytes = 2 * 512 KiB * 14 = 14,336 KiB` (1 read + 1 write per
  stage, all 14 stages).

```text
AI_worst_case = 917,504 / (14,336*1024) = 0.0625 FLOPs/byte
```

**64x below the ridge point (4.01)** - this kernel is *severely*
memory-bound, not just "somewhat."

**Cross-check against real measured performance** (sanity check the
model, not just trust the arithmetic): Roofline predicts an achievable
ceiling of `7.18 * 0.0625 = 0.449 GFLOPS` at this AI. Real measured
performance (`dag_gst_master_analysis.md`'s own `W_chunk ≈ 4.71ms`,
`w1_c1`, single lone worker): `917,504 FLOPs / 0.00471s ≈ 0.195
GFLOPS` - **below** the Roofline ceiling (0.43x of it), exactly the
expected relationship (Roofline gives an upper bound; real code always
sits at or below it, with the gap explained by non-ideal access
patterns, Python/NumPy dispatch overhead, and the AI calculation
excluding the gather/scatter/phase-multiply/threshold steps
surrounding the pure butterfly). The model is not contradicted by
measurement - it correctly predicts memory-bound behavior in the right
ballpark.

## 4. The actual bound on cache-blocking's benefit

**Best case**: perfect cache-blocking (the array stays L1/L2-resident
for the ENTIRE 14-stage butterfly - one compulsory DRAM read-in at the
start, one compulsory write-out at the end, zero re-reads/re-writes in
between):

```text
best_case_bytes = 2 * 512 KiB = 1,024 KiB   (was 14,336 KiB)
AI_best_case = 917,504 / (1024*1024) = 0.8750 FLOPs/byte
```

**AI improves by exactly 14x** (= `log2(dim)`, the stage count - not a
coincidence: eliminating per-stage DRAM round-trips scales directly
with how many stages there are).

```text
Roofline-predicted achievable GFLOPS, worst case (current): 0.449
Roofline-predicted achievable GFLOPS, best case (perfect blocking): 6.28
Max theoretical wall-clock speedup from cache-blocking: 14.0x
```

**But**: `AI_best_case = 0.875` is *still* 4.6x below the ridge point
(4.01). **Even a theoretically perfect cache-blocked WHT never becomes
compute-bound on this hardware - it stays memory-bound, just less
severely so.** This sets a hard, falsifiable ceiling: no cache-
blocking redesign, however good, can push this single-chunk kernel's
wall-clock speedup past ~14x, and realistic implementations (which
cannot achieve perfectly zero re-traffic - partial stage overlap,
imperfect blocking boundaries, Python/NumPy call overhead) will land
meaningfully below that.

## 5. What this does and does not tell us

**Does tell us**: cache-blocking is worth prototyping - a real,
bounded 14x ceiling on the *single-chunk, single-core* kernel is a
substantial available win, not a marginal one. It also tells us
exactly what to measure against once a prototype exists (AI should
move toward, not just below, 0.875 FLOPs/byte if the blocking is
working as intended - a concrete, falsifiable target, not just "faster
wall-clock").

**Does NOT tell us**: whether cache-blocking fixes the OBSERVED
multi-core collapse (DAG C, the ~7x-worse-than-GST, IPC-blocked
w8_c4 behavior). This Roofline analysis is single-core; DAG C is a
multi-worker resource-contention phenomenon layered on top. Per
`dag_gst_master_analysis.md` section 3e (added this session), IPC-
blocking's TRIGGER is still unidentified - the leading suspect is the
irregular gather/scatter pattern, not dense per-stage WHT traffic
(already refuted as sufficient by `traffic_intensity_findings.md`).
Cache-blocking the butterfly might reduce paulikit's per-worker memory
traffic INTENSITY enough to also relieve some of DAG C's contention
(a plausible compounding effect, consistent with the reasoning in
section 3c/4b), but that is a separate, untested claim - this analysis
only bounds the single-chunk, single-core piece.

## 6. Numbers quick-reference

```text
Measured single-core DRAM bandwidth      = 7.18 GB/s (stdev 0.23, n=8)
Peak single-core compute (AVX2 FMA)      = 28.80 GFLOPS
Ridge point                              = 4.01 FLOPs/byte
paulikit WHT AI, worst case (today)      = 0.0625 FLOPs/byte  (64x below ridge)
paulikit WHT AI, best case (perfect CB)  = 0.8750 FLOPs/byte  (4.6x below ridge, still memory-bound)
Roofline ceiling, worst case             = 0.449 GFLOPS
Roofline ceiling, best case              = 6.28 GFLOPS
Real measured (w1_c1)                    = 0.195 GFLOPS  (0.43x of worst-case ceiling - model not contradicted)
Max theoretical speedup from cache-blocking = 14.0x  (== log2(dim), single-chunk/single-core only)
```

## Artifacts

- `roofline_bandwidth_probe.py`
