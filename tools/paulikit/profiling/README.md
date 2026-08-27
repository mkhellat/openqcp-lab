# Performance engineering — index

This directory holds every profiling/measurement artifact behind
`paulikit`'s performance work, in the order the work was actually done.
It is the single entry point: read this file first, then follow the
links into the subdirectories for full detail. See
[`../PLAN.md`](../PLAN.md) for the design/implementation narrative
behind each phase — this README is the *measurement* trail, PLAN.md is
the *decision* trail.

**Layout:**
- Loose files directly in this directory (`profile_target.py`,
  `cprofile_n50.prof`, `pyspy_n50.svg`, …) are Phase 2's original
  cProfile/line_profiler/py-spy baseline artifacts.
- [`phase3b/`](phase3b/README.md) — sparsity-aware
  `fwht_pauli_coefficients` design exploration (Phase 3b).
- [`cache_locality/`](cache_locality/README.md) — the cache-locality
  investigation, Phase 6 through Phase 10's correction (13 findings).
- [`phase9/`](phase9/phase9_findings.md) — chunked-accumulator
  space-complexity fix, N=150 measurement (Phase 9).
- [`phase10/`](phase10/README.md) — streaming output, TBB
  re-measurement, N-scaling table, Phase 11 scoping (Phase 10).
- [`../bindings/README.md`](../bindings/README.md) — the C/Cython/CFFI/
  ctypes/SWIG binding comparison (Phase 3a; lives under `bindings/`,
  not `profiling/`, but is part of this same chronological trail).

## Chronological summary

### Phase 2 — cProfile / line_profiler / py-spy baseline (2026-08-16)

The starting point for all native-porting and algorithmic work below.
`profile_target.py` builds the matched-N real coupled-oscillator
Hamiltonian used by `tests/test_benchmark_reference.py`, then calls
`fwht_pauli_terms` on it. Profiled at N=50 (2048×2048, 11 qubits,
~6.2s plain run time) — the largest matched-benchmark size that still
runs in single-digit seconds, so it could be profiled repeatedly.

**Finding: `pauli_label` dominates, not the FWHT math.** At N=50,
`fwht_pauli_terms` takes 11.7s total. By cumulative time: `pauli_label`
(1,261,568 calls) is **59.6%** (6.980s), `fwht_pauli_coefficients` (the
actual FWHT — gather + WHT + phase) is 15.5% (1.812s), the WHT
butterfly itself is 8.9%, `_popcount_array` is 4.0%. By self time the
gap is starker: `pauli_label` alone costs 4.902s — more than 2x the
*entire* `fwht_pauli_coefficients` core algorithm's self time (0.274s).

`line_profiler` found one line inside `fwht_pauli_terms` accounting for
88.1% of the function's own measured time
(`real_terms[pauli_label(x, z, n_qubits)] = float(c.real)`), and inside
`pauli_label` itself, the per-qubit Python loop (bit-shifting, masking,
a dict lookup per qubit per term, run 13,877,248 times at N=50) as
~92% of the function's own cost. `py-spy` (100 Hz sampling, ~0
instrumentation overhead) independently confirmed the same hot lines:
four specific source lines inside `pauli_label` account for **~33% of
every sample taken across the entire program run**.

**Conclusion:** the FWHT algorithm itself (XOR gather, Walsh-Hadamard
butterfly, phase-factor multiplication) is fast and already vectorized
in NumPy — not a target for optimization. The bottleneck is
`pauli_label`'s per-term, per-qubit, pure-Python string-building loop,
called once per nonzero coefficient. This directly motivated Phase 3a
below: a native port of the WHT butterfly would speed up only ~15% of
runtime, while label formatting was 60-90% depending on term count.

Full detail, including the line-by-line and py-spy breakdowns:
[`README.md`'s original content is preserved below](#phase-2-artifacts-and-full-detail)
— see the "Artifacts in this directory" list at the end of this file.

### Phase 3a — `pauli_label` C port + binding comparison (2026-08-16, complete)

Ported the confirmed Phase 2 bottleneck to C
(`src/paulikit/_native/pauli_label.c`) and bound it to Python four
separate ways to compare technique, not just speed: Cython, CFFI,
ctypes, SWIG. Full detail:
[`../bindings/README.md`](../bindings/README.md).

**Results:** Cython won on both raw speed (26.2x label-gen speedup at
N=50, vs. 6.5-11.1x for the other three) and binding effort (lowest of
the four; SWIG needed hand-written typemaps for buffer arguments — the
highest-effort binding, as predicted going in) — retained as the
binding `paulikit` ships with. End-to-end impact at N=100: swapping in
the Cython batch label call alone (standalone comparison, not yet
wired into the shipped pipeline) cut `fwht_pauli_terms`-equivalent
time from the 126.3s all-Python baseline to ~40.2s (**3.1x**), leaving
`fwht_pauli_coefficients`'s dense-array computation (38.6s) as the new
dominant cost — exactly Phase 3b's scope.

oneTBB parallelization was added on top (`pauli_label_batch_parallel`,
`tbb::parallel_for` over independent terms). Standalone C++ benchmark:
3.9-4.1x on 8 cores (below the paper's ~7x/8-core reference point,
plausibly memory-bandwidth- rather than compute-bound). **Through the
actual Python boundary, parallelization barely helped (1.1-1.25x)** —
isolated to the Python list-of-`str` construction cost that already
dominated wall-clock time, confirming (one level further down the
stack) the same per-term Python-object-construction cost Phase 2
found.

### Phase 3b — sparsity-aware `fwht_pauli_coefficients` (2026-08-16 to 2026-08-18, complete)

`fwht_pauli_coefficients` always computed the *full* dense
`2**n × 2**n` coefficient array regardless of input sparsity. Full
detail, including 8 attempted variants (several regressions) before
landing on the winner: [`phase3b/README.md`](phase3b/README.md).

**Key measurement first:** the operator itself is sparse (O(N)
nonzeros), but the WHT *rows* are not uniformly sparse — 47-86% of
rows have at least one nonzero entry depending on N. This ruled out
the initially expected "sparse-impulse WHT" fix (an `O(k·dim)`
identity replacing the `O(dim log dim)` butterfly per row), since
active-row count stays too close to `dim` for that trade to win.

**Implemented fix:** skip the O(dim²) full gather entirely; scatter
nonzero entries directly into an `(n_active, dim)` array; run the
existing dense WHT butterfly only on active rows; compute the phase
factor only for active rows; replace `_popcount_array`'s bit-serial
loop with an 8-bit LUT popcount. Exact algorithm, not an
approximation — verified against the full test suite and PennyLane
cross-checks.

**Results:**

| N (oscillators) | `fwht_pauli_coefficients` (old dense) | (new) | speedup | `fwht_pauli_terms` end-to-end (old) | (new) | speedup |
|---|---|---|---|---|---|---|
| 50  | 1.971s  | 0.636s  | 3.1x  | 6.2213s   | 5.4957s   | 1.13x |
| 100 | 35.47s  | 17.56s  | 2.0x  | 126.3250s | 107.2403s | 1.18x |

**Honest scope note:** the end-to-end speedup (1.13-1.18x) was much
smaller than the coefficients-only speedup (2.0-3.1x) because
`fwht_pauli_terms` still used the pure-Python `pauli_label` loop, not
Phase 3a's Cython kernel — that integration was Phase 3c.

### Phase 3c — wire the native kernel in; adopt meson-python (2026-08-18, complete)

Two changes done together, since the packaging decision blocked the
integration decision. Full detail: `../PLAN.md` Phase 3c.

1. **Build-system migration to meson-python** (the same backend
   NumPy/SciPy use). The Cython `pauli_label` kernel from Phase 3a is
   now packaged inside the library as
   `paulikit._native.pauli_label_native`, gated behind a `native`
   Meson feature option (`auto`/`enabled`/`disabled`). Still optional
   with a pure-Python fallback (with a one-time `UserWarning` if the
   fallback fires) — a deliberate, temporary compromise pending
   prebuilt-wheel CI (Phase 5).
2. **Wiring:** `fwht_pauli_terms` now calls the native batch kernel
   when available, falling back to pure Python otherwise.

**Results** (matched N, same Hamiltonian generator as Phase 1/3a/3b):

| N (oscillators) | Phase 1 baseline | Phase 3b (sparse coeffs, Python labels) | Phase 3c (native labels) | speedup vs. Phase 1 | speedup vs. Phase 3b |
|---|---|---|---|---|---|
| 50  | 6.2213s   | 5.4957s   | 2.1535s | 2.9x | 2.6x |
| 100 | 126.3250s | 107.2403s | 43.5629s | 2.9x | 2.5x |

Term counts match exactly at every N across all three implementations
— a correctness confirmation, not just a performance measurement. This
2.9x-over-Phase-1 figure is the headline number the main
[`../README.md`](../README.md) "Reference baseline" table reports.

### Phase 6 through Phase 10's correction — cache-locality investigation (2026-08-25 to 2026-08-27)

The single largest investigation in this directory: 13 findings
tracing a real, measured cache-locality/memory-footprint bug from
first discovery through to full resolution, spanning PLAN.md Phases 6,
8, 9, and 10. Summarized as a group here — **read
[`cache_locality/README.md`](cache_locality/README.md) directly for
the full numbered trail**, not reproduced finding-by-finding below.

**What it found (Phase 6 root cause, 2026-08-25):**
`fwht_pauli_coefficients` computed its result sparsely (Phase 3b's
optimization, genuinely preserved) but then materialized a full dense
`(dim, dim)` complex128 array anyway, which `fwht_pauli_terms`
immediately re-scanned. Cache-miss ratio and memory-stall cycles both
scaled cleanly with how far this array exceeded cache size (~23% at
N=25, up to ~58.5% at N=100). At N=150 this **OOM-killed the
investigation's dev machine** (15 GiB RAM) without even completing one
decompose call — a real crash risk, not just a performance nicety.

**Corrections along the way, made visible rather than silently
folded in:**
- An early N-scaling measurement method conflated process-startup
  overhead with algorithm cost; redone with an in-process, warmed-up
  driver, which *raised* the measured N=25 cache-miss ratio (17.2% →
  22.7-24.6%) rather than lowering it.
- ~60% of measured stall-cycle self-time turned out to be NumPy's
  OpenBLAS thread pool spinning idle — cross-thread perf-counter
  noise, not real algorithmic work (`OPENBLAS_NUM_THREADS=1` cut
  measured cycles 2.6-2.7x with wall-clock time unchanged). This
  flagged several earlier absolute stall-percentage numbers as
  contaminated upper bounds, without invalidating the underlying
  dense-array root cause (measured via genuine memory-subsystem
  counters, unaffected by this noise).
- TBB was assumed relevant (per the original motivating transcript)
  but was found not even wired into the production path at all
  (finding 10), then directly measured and found to have no
  cache-locality effect even if it were (finding 11) — see the Phase
  10 section below for a *third* re-measurement at N=150 scale that
  reversed the practical recommendation once more.
- The dense-vs-sparse A/B test the fix needed (finding 12) found
  cache-miss ratio and stall percentage were statistically
  indistinguishable between dense and sparse through N=100, and
  wall-clock speedup was small (0-4.5%) — Phase 6's real, measured
  benefit through N=100 is **memory footprint and crash-avoidance**,
  not cache locality or wall-clock time, a materially more precise
  claim than "sparse fixes cache locality."

**The fix, implemented in stages across Phases 6, 8, 9, 10:**
- **Phase 6** (2026-08-25/26): `fwht_pauli_coefficients(operator,
  sparse=False)` — an optional, reversible sparse output mode (user's
  explicit preference over a one-way breaking change), plus two
  follow-on memory fixes found while chasing N=150: removing an
  unneeded array copy in the WHT butterfly (~5.5 GiB less transient
  allocation at N=150) and an optional `chunk_size` row-tiling
  parameter (MIT 6.172-style loop tiling, applied for memory footprint
  rather than cache reuse).
- **Phase 8** (scoped 2026-08-26, implemented 2026-08-27): the
  Hamiltonian itself was dense — at N=150, `11475×11475` with only
  45,000 nonzero entries (0.034% density) but 1.05 GiB dense, padded
  and upcast to a ~4 GiB allocation *before* any decomposition
  algorithm ran. `build_hamiltonian(..., sparse=True)` and
  `pad_to_power_of_two(..., sparse=True)` added (`scipy.sparse`, gated
  behind an optional `[sparse]` install extra). Verified: 15 new
  tests, sparse/dense parity bit-for-bit. A real N=150 attempt under a
  6 GB `ulimit -v` cap cleared this ceiling — and immediately hit a
  new one, scoped as Phase 9.
- **Phase 9** (scoped and implemented 2026-08-27): `chunk_size` only
  bounded *transient* per-chunk memory; the chunked loop still wrote
  every chunk into one dense `(n_active, dim)` accumulator allocated
  up front — 11.8 GiB at N=150, independent of `chunk_size`. Replaced
  with atol-thresholding *inside* the chunk loop plus an
  amortized-doubling growable COO accumulator, with opt-in
  checkpoint/resume. Confirmed via real capped runs (see
  [`phase9/phase9_findings.md`](phase9/phase9_findings.md)): a 10 GB
  cap now completes accumulation of the genuine ~134M survivor terms
  at N=150 — but materializing that as label strings + a
  `dict[str, float]` afterward is a **third**, still-separate O(n_terms)
  allocation this fix didn't address; a 13.5 GB attempt was killed by
  a safety monitor after real available memory dropped to 630 MiB.
  Conclusion, confirmed with the user: no fixed-RAM fix survives
  arbitrary N — only streaming removes the ceiling. Scoped as Phase 10
  as mandatory, not optional, follow-up.
- **Phase 10** (scoped and implemented 2026-08-27): see its own
  section below.

Full "current honest state" summary of this whole thread is in
[`cache_locality/README.md`](cache_locality/README.md#current-honest-state-as-of-the-last-finding-above).

### Phase 10 — streaming/generator output (2026-08-27, implemented and verified)

The direct resolution of Phase 9's conclusion. Full detail:
[`phase10/README.md`](phase10/README.md).

**Design reframing before implementation:** rather than bolting a
generic generator onto the existing accumulate-then-return shape, the
actual defect was identified as `fwht_pauli_terms` re-fusing every
independent chunk's result into one combined `dict` before the caller
ever saw it — an artificial recombination step the underlying math
never required (unlike, e.g., tiled matrix multiply, which genuinely
needs to sum blocks). The correct divide-and-conquer fix keeps each
chunk a chunk all the way out to the caller.

**Implementation:** a new sibling function, `fwht_pauli_terms_iter`
(not a `stream: bool` flag), requires `chunk_size`, and yields one
`dict` per chunk as an in-process generator; the existing
`checkpoint_path` mechanism remains a secondary, opt-in resumability
path. 16 new tests, full suite 71/71 passing.

**Real N=150 result:** 91,652,096 terms streamed to completion under
both a 4 GB and a 2 GB `ulimit -v` cap (~100s each, identical term
counts) — compared to Phase 9's finding that the non-streaming
accumulator needed at least 10 GB just to finish accumulation and
never completed even at 13.5 GB once labels and dict construction were
included. **N=150 is now a solved, repeatable case.**

**Sub-findings within Phase 10 (see
[`phase10/README.md`](phase10/README.md) for the full numbered list):**
1. [`tbb_labeling_n150_findings.md`](phase10/tbb_labeling_n150_findings.md)
   — the oneTBB label kernel re-measured in isolation at N=150-scale (a
   synthetic 40M-pair benchmark): a real 1.1-1.4x wall-clock win, at a
   modest cache-locality cost.
2. [`phase10_streaming_findings.md`](phase10/phase10_streaming_findings.md)
   — the core streaming result above.
3. **[`full_pipeline_n150_findings.md`](phase10/full_pipeline_n150_findings.md)
   — corrects finding 1.** Re-measured embedded in the real streaming
   pipeline (not isolated): `--parallel-labels` delivers **no
   measurable full-pipeline benefit** at N=150 (97.97s vs. 96.96s,
   cache-miss/stall percentages flat to within a point), because a
   per-stage breakdown found **dict construction — not labeling, not
   the WHT butterfly — dominates at ~60% of total pipeline time** (vs.
   ~21% WHT, ~7% labeling). This directly parallels Phase 2's original
   N=50 finding, with the specific bottleneck having moved from
   label-string formatting to dict construction now that labels are
   TBB/Cython-fast.
4. [`n_scaling_streaming_findings.md`](phase10/n_scaling_streaming_findings.md)
   — the first-ever successful N=25/50/100/150 streaming timing table:
   0.069s → 1.329s → 22.064s → 101.310s, **linear** scaling with term
   count (~4.5x growth in both from N=100 to N=150), consistent with
   finding 3's dict-construction-dominated diagnosis.
5. [`phase11_dict_build_scoping_findings.md`](phase10/phase11_dict_build_scoping_findings.md)
   — scopes Phase 11 (below): the per-term Hermiticity check, not dict
   construction itself, is the dominant sub-cost within dict-building;
   vectorizing it plus `dict(zip(...))` construction gives a
   **2.7-3.2x** speedup on a synthetic 1M/10M-term benchmark.

### Phase 11 — `dict_build` optimization (scoped 2026-08-27, not yet implemented)

Not yet started as of this writing. `dict_build` — the per-chunk
Python loop converting `(label, coefficient)` pairs into a `dict` —
was found (Phase 10, finding 3 above) to be ~60% of total pipeline
time at N=150. Scoping work (see
[`phase10/phase11_dict_build_scoping_findings.md`](phase10/phase11_dict_build_scoping_findings.md))
broke this down further via a standalone microbenchmark: the per-term
Hermiticity check (`abs(c.imag) > max(atol, 1e-6 * abs(c))`, evaluated
one Python object at a time) is the single largest sub-cost — a fully
vectorizable NumPy operation currently paid for one term at a time.
Removing it (replaced with one vectorized check before the loop)
produced a 3.76x speedup at 1M terms and 2.58x at 10M terms in
isolation. A secondary win: `dict(zip(...))`'s C-level constructor
beats an explicit per-item insert loop by a further ~30-40%. Not yet
implemented against the real pipeline — see `../PLAN.md` Phase 11 for
the remaining open design questions (preserving per-term error-message
specificity, whether to apply to the non-streaming path too).

## Current state: what's solved vs. open

- **Solved and shipped:** the N=50/N=100 label-generation and
  dense-array bottlenecks Phase 2 found (Phases 3a-3c, 2.9x end-to-end
  at N=100 over the Phase 1 baseline); the N=150 OOM the cache-locality
  investigation found (Phases 6, 8, 9, 10 together — N=150 now
  completes in ~100s under a 2 GB memory cap via streaming).
- **Measured but explicitly not a lever:** TBB/oneTBB parallelization
  of label generation. Confirmed unhelpful three separate times, for
  three different reasons (not wired into the production path;
  wouldn't help the dense-array bottleneck if it were; doesn't move
  the needle once dict construction dominates at N=150 scale). Still
  available as an opt-in `--parallel-labels` flag, since it is not
  harmful, just not currently the highest-leverage target.
- **Open, scoped, not yet implemented:** Phase 11 (`dict_build`
  vectorization) — the current highest-leverage remaining optimization
  target, with a scoped fix and a measured 2.7-3.2x isolated-benchmark
  upside not yet applied to the real pipeline. Also open: Phase 5
  (prebuilt wheels, to make the native extension a hard requirement)
  and Phase 7's remaining items (TBB false-sharing/partitioner tuning
  — explicitly conditional on TBB re-entering the hot path, which it
  has not; statistical rigor for `perf`-based measurements; PyPI
  publishing strategy). See `../PLAN.md` for full status on all of
  these.
- **Known measurement confound to control for in any new work in this
  directory:** OpenBLAS thread-pool noise. Set `OPENBLAS_NUM_THREADS=1`
  for any new `perf`-based measurement unless specifically
  investigating BLAS behavior itself (see
  `cache_locality/stall_floor_mystery_solved.md`).

---

## Phase 2 artifacts and full detail

The rest of this file is Phase 2's original profiling write-up,
preserved in full since it's the actual evidence Phases 3a/3b/3c were
scoped against.

Follows [`../PLAN.md`](../PLAN.md) Section 5, Phase 2: `cProfile` +
`snakeviz` for an initial pass, `line_profiler` for per-line detail on
whatever cProfile flags as hot, `py-spy` as a no-instrumentation
sampling cross-check. All three tools agree on the same hot spot.

### Setup

`profile_target.py` builds the same matched-N real coupled-oscillator
Hamiltonian used by `tests/test_benchmark_reference.py` (so results
are directly comparable to the recorded PennyLane benchmark), then
calls `fwht_pauli_terms` on it. N=50 (2048x2048, 11 qubits, ~6.2s
plain run time) was chosen as the profiling target: the largest
matched-benchmark size that still runs in single-digit seconds, so it
can be profiled repeatedly. N=100 (~8192x8192) was timed once for
reference but is too slow to iterate on directly.

Reproduce:

```bash
# cProfile
python3 -m cProfile -o profiling/cprofile_n50.prof profiling/profile_target.py
snakeviz profiling/cprofile_n50.prof   # interactive flame graph, browser GUI

# line_profiler (requires @profile-decorated copy or kernprof -l)
kernprof -l -v <script importing fwht.pauli_label / fwht.fwht_pauli_terms>

# py-spy
py-spy record -o profiling/pyspy_n50.svg -- python3 profiling/profile_target.py
```

### Finding: `pauli_label` dominates, not the FWHT math

At N=50, `fwht_pauli_terms` takes 11.7s total. Breakdown by
**cumulative** time (`cprofile_n50.prof`, sorted by `cumulative`):

| function | cumtime | % of total |
|---|---|---|
| `fwht_pauli_terms` (whole call) | 11.708s | 100% |
| `pauli_label` (1,261,568 calls) | 6.980s | **59.6%** |
| `fwht_pauli_coefficients` (the actual FWHT: gather + WHT + phase) | 1.812s | 15.5% |
| `_walsh_hadamard_transform_rows` (the WHT butterfly itself) | 1.042s | 8.9% |
| `_popcount_array` | 0.463s | 4.0% |

By **self** time (`tottime`, excludes callees) the gap is even more
stark: `pauli_label` alone costs 4.902s of self time — more than
2x the *entire* `fwht_pauli_coefficients` core algorithm's self time
(0.274s self, i.e. almost all of its 1.812s cumulative is the WHT and
gather steps it calls, not label formatting).

**The algorithmic core (FWHT itself) is not the bottleneck.** The
bottleneck is the pure-Python bookkeeping step that turns numeric
`(x, z)` bitmask coefficients into `"IXYZ"`-style string dict keys,
run once per nonzero term (1.26M times at N=50).

### Line-level detail (`line_profiler`, `kernprof -l -v`)

Inside `fwht_pauli_terms`, one line accounts for 88.1% of the
function's own measured time:

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
...
1261568   32265959.6     25.6     88.1      real_terms[pauli_label(x, z, n_qubits)] = float(c.real)
```

Inside `pauli_label` itself (called 1,261,568 times, `n_qubits=11`,
so 13,877,248 inner-loop iterations):

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
...
   1261568     650627.3      0.5      2.8      letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
   1261568     349118.4      0.3      1.5      chars = []
  15138816    4184285.3      0.3     18.2      for qubit in range(n_qubits):
  13877248    3843422.5      0.3     16.7          bit = n_qubits - 1 - qubit
  13877248    4012536.9      0.3     17.5          xj = (x_mask >> bit) & 1
  13877248    3992917.2      0.3     17.4          zj = (z_mask >> bit) & 1
  13877248    5104560.2      0.4     22.2          chars.append(letters[(xj, zj)])
   1261568     820816.5      0.7      3.6      return "".join(chars)
```

Two contributing causes, both fixable without touching the FWHT
algorithm:
1. **`letters` dict is rebuilt on every call** (2.8% of `pauli_label`'s
   own time) instead of being a module-level constant.
2. **The per-qubit Python loop itself** (lines 208-212, ~92% combined)
   is pure per-bit interpreted-Python work repeated 11-13 times per
   term, 1.26M times over. This is the real cost: bit-shifting,
   masking, and a dict lookup per qubit per term, none of which is
   vectorizable in the current one-call-per-term design.

### Cross-check (`py-spy`, sampling, ~0 instrumentation overhead)

`py-spy record` (100 Hz sampling) on the same N=50 target agrees with
the instrumented profilers without their overhead. Self-time samples
landing inside `pauli_label`, broken down by exact source line
(`pyspy_n50.svg` flame graph, leaf-frame percentages):

| line | content | % of all samples |
|---|---|---|
| fwht.py:212 | `chars.append(letters[(xj, zj)])` | 16.75% |
| fwht.py:210 | `xj = (x_mask >> bit) & 1` | 7.74% |
| fwht.py:211 | `zj = (z_mask >> bit) & 1` | 6.48% |
| fwht.py:213 | `return "".join(chars)` | 1.90% |

These four lines alone account for **~33% of every sample taken
across the entire program run** (not just within `pauli_label`) —
independent confirmation, via a fundamentally different measurement
method, of the same hot spot line_profiler found.

### Conclusion

The FWHT algorithm itself (`fwht_pauli_coefficients`: the XOR gather,
the Walsh-Hadamard butterfly, the phase-factor multiplication) is
fast and already vectorized in NumPy — it is not the target for any
future optimization work. The actual bottleneck is
`pauli_label`'s per-term, per-qubit, pure-Python string-building loop
in the `fwht_pauli_terms` convenience wrapper, called once per
nonzero coefficient.

This matters for any future Phase 3 (C porting) decision per
PLAN.md: a native port of the WHT butterfly would speed up only
~15% of current runtime, and *none* of the label-formatting cost
(~60-90% of runtime depending on term count) — porting `pauli_label`
(or vectorizing/batching label generation in NumPy, or making label
generation lazy so it only runs for terms the caller actually
inspects) is the higher-leverage target if wall-clock time on the
convenience `fwht_pauli_terms` API is the concern.
`fwht_pauli_coefficients` (the numeric-array API, no label strings)
does not pay this cost at all.

### Artifacts in this directory

- `profile_target.py` — shared setup, builds the matched-N Hamiltonian.
- `cprofile_n50.prof` — raw cProfile output, open with `snakeviz` for
  an interactive flame graph.
- `cprofile_n50_tottime.txt` — text dump, sorted by self time.
- `pyspy_n50.svg` — py-spy sampling flame graph (open in a browser).
- `pyspy_n50.speedscope.json` — same py-spy run, speedscope format
  (upload to https://www.speedscope.app/ for an interactive view).
