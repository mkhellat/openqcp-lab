# Pauli Decomposition Performance Engineering — Plan

Status: Phases 0-3c complete. Phase 4 (final write-up), Phase 5
(prebuilt wheels + hard-require the native extension), Phase 6
(sparse output for `fwht_pauli_coefficients` - fixes a real cache-
locality/robustness bug found via profiling, see
`profiling/cache_locality/README.md`), and Phase 7 (remaining items
from the 2026-08-25 Gemini-transcript review: TBB false sharing/
partitioner tuning - conditional on TBB re-entering the hot path,
statistical rigor for perf measurements, PyPI publishing strategy)
are all scoped but not started. Phase 5 scoped 2026-08-19; Phase 6
and Phase 7 scoped 2026-08-25.
Last updated: 2026-08-25.


## 1. Problem statement

`tutorials/coupled_harmonic_oscillators/N_coupled_harmonic_oscillators_1_D.ipynb`
implements a quantum algorithm for Hamiltonian simulation of N coupled
classical oscillators (based on
[Exponential Quantum Speedup in Simulating Coupled Classical Oscillators](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041)).
Before the Hamiltonian can be Trotterized and simulated, it must be
decomposed into a sum of Pauli-string terms.

The notebook's current `decompose_to_pauli_terms` function is a dense,
symbolic (SymPy), brute-force decomposition: for an n-qubit padded
Hamiltonian it loops over all $4^n$ Pauli strings, builds each as a
symbolic `TensorProduct`, and computes `(matrix * tensor_product).trace()`
symbolically. This does not scale:

- For N=30 oscillators, the padded Hamiltonian needs roughly n≈9
  qubits, i.e. ~262,000 symbolic trace evaluations on 512×512 symbolic
  matrices.
- The untracked, in-progress `N_coupled_harmonic_oscillators_1_D_N_30.ipynb`
  notebook only actually reaches N=4 in its executed cells (not N=30,
  despite the filename), and has two `## to be revised with new Coded
  and tested functions` placeholder cells sitting exactly at the Pauli
  decomposition and PauliTerms-conversion steps — direct evidence this
  is where the existing approach broke down.
- Separately, `suzuki_trotter()` (used downstream to Trotterize the
  decomposed Hamiltonian) has its own Classiq-API-version incompatibility
  (`list[PauliTerm]` vs. `SparsePauliOp`), tracked under task #5 in the
  main repo task list, orthogonal to this performance work.

This project's goal: build an original, fast, correct Pauli
decomposition implementation that scales to N=30 (target) and ideally
further (stretch: N=100+), through disciplined performance engineering
— not by simply calling an existing library.


## 2. Literature research (2026-08-04)

Web research this session surfaced the current state of the art for
Pauli decomposition performance:

- **Naive dense/symbolic** (current repo approach): effectively
  O(4ⁿ) terms × per-term symbolic overhead. The dominant cost is not
  the asymptotic complexity alone but the enormous constant factor of
  symbolic (SymPy) tensor products and traces per term.
- **PennyLane's `qml.pauli_decompose`** (channel-state duality + Bell
  basis Walsh-Hadamard transform): O(n·4ⁿ), natively supports
  `scipy.sparse` input without densifying. Already a pinned dependency
  of this repo (`pennylane==0.45.1`).
- **Fast Walsh-Hadamard Transform (FWHT) method**
  ([Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d),
  2025): O(N² log N) for an N×N matrix (N=2ⁿ) — i.e. O(n·4ⁿ) but with
  a dramatically smaller constant than symbolic approaches, since it
  operates numerically via three steps:
  1. XOR-index permutation of matrix elements (in-place, O(1) extra space)
  2. Fast Walsh-Hadamard Transform applied per row (the structured
     Hadamard matrix has only 2 nonzero entries per column)
  3. Phase-factor multiplication (bitwise operations)

  Reported ~7x speedup on 8 cores (parallelizable across rows/terms).
  This machine has 8 cores (`nproc` = 8), making this a directly
  relevant target.
- **Tensorized Pauli Decomposition (TPD)** ([arXiv:2310.13421](https://arxiv.org/abs/2310.13421)):
  favorable scaling for structured/special-case Hamiltonians (e.g.
  TFIM); worth benchmarking against FWHT once a baseline exists, since
  the coupled-oscillator Hamiltonian has its own specific (near
  block-sparse) structure that TPD might exploit better.
- **PHASE** (hierarchical, geometry-aware; combines TPD + FWHT via
  recursive mesh partitioning): further asymptotic improvement for
  geometry-structured problems; a later stretch goal, not part of the
  initial implementation.

Naive baseline note: some sources cite up to O(2⁵ⁿ) for the least
careful brute-force implementations — worth keeping in mind as a
worst-case anchor when comparing.


## 3. Baseline reference measurements (2026-08-04)

**Not the deliverable** — PennyLane's existing `qml.pauli_decompose`
was benchmarked purely to (a) get a real performance target to beat or
match, and (b) validate correctness of whatever original implementation
is built here, independent of it.

### 3.1 Dense random Hermitian matrix (worst case, no sparsity)

| N (oscillators) | qubits | matrix dim | time     | Pauli terms |
|------------------|--------|------------|----------|-------------|
| 2                | 3      | 8          | 0.0072s  | 36          |
| 4                | 4      | 16         | 0.0250s  | 136         |
| 8                | 6      | 64         | 0.5820s  | 2080        |
| 12               | 7      | 128        | 3.0204s  | 8256        |
| 16               | 8      | 256        | 14.3645s | 32896       |
| 20               | 8      | 256        | 14.8779s | 32896       |
| 30               | 9      | 512        | 64.7801s | 131328      |

### 3.2 Sparse matrix matching the real Hamiltonian's sparsity pattern

The coupled-oscillator Hamiltonian's off-diagonal block B has only
O(N) nonzero entries in an O(N²)-dimensional padded matrix — genuinely
sparse, unlike the dense random case above. Using a
`scipy.sparse.csr_matrix` encoding with the same sparsity structure:

| N (oscillators) | qubits | matrix dim | time    | Pauli terms |
|------------------|--------|------------|---------|-------------|
| 16               | 8      | 256        | 0.3723s | 1024        |
| 30               | 9      | 512        | 1.2879s | 2304        |
| 50               | 11     | 2048       | 9.3741s | 11264       |
| 100              | 13     | 8192       | 94.8331s| 53248       |

This is the realistic performance regime for this problem — the
repo's actual Hamiltonian is sparse, not dense.

### 3.3 Correctness cross-check

Reconstructed the repo's own N=2 Hamiltonian (per `prepare_hmatrix`'s
documented pattern in `N_coupled_harmonic_oscillators_1_D.ipynb`, with
k00=1.0, k01=2.0, k11=3.0, m0=1.0, m1=2.0) and decomposed it with
`qml.pauli_decompose`:

- 12 nonzero Pauli terms (matches the notebook's own documented claim
  "out of the 64 Pauli terms ... only 12 have nonzero coefficients"
  for N=2).
- Reconstructing H from the decomposition matched the original matrix
  to machine epsilon (max error ≈ 2.22e-16).

### 3.4 Matched comparison: paulikit vs. PennyLane on the *real* Hamiltonian

Section 3.2's sparse-matrix comparison used a synthetic tridiagonal-
band matrix as a stand-in for "the real Hamiltonian's sparsity
pattern," not the actual `build_hamiltonian()` output. That stand-in
happened to decompose to far fewer Pauli terms at each N than the
real coupled-oscillator Hamiltonian does, which understated
PennyLane's real runtime on this problem and made the eventual
comparison look more favorable to `paulikit` than the fair, matched
comparison actually shows. Section 3.2's numbers are retained above
for the historical record, but should not be read as apples-to-apples
against `paulikit`'s own benchmark numbers below.

Re-run with both implementations decomposing the exact same
`build_hamiltonian()` output at each N (see
`tests/test_benchmark_reference.py`, marked `slow` and excluded from
the default test run since PennyLane takes minutes — and, at N=50,
more than 10 minutes — at larger N):

| N (oscillators) | qubits | Pauli terms | paulikit time | PennyLane time    | speedup |
|------------------|--------|-------------|----------------|--------------------|---------|
| 16               | 8      | 15360       | 0.0586s        | 6.7369s            | 115x    |
| 30               | 9      | 112384      | 0.4019s        | 48.3211s           | 120x    |
| 50               | 11     | 1261568     | 6.2213s        | >590s (aborted)    | >95x    |
| 100              | 13     | 20299776    | 126.3250s      | not attempted      | —       |

At N=16 and N=30, both implementations were run to completion and
agree exactly on term count (a correctness check, not just a
performance one). At N=50, the PennyLane comparison run was killed by
a 590-second timeout without finishing — a real data point in its own
right (PennyLane's `pauli_decompose` takes over ten minutes on this
matrix), not a gap to read as "unknown." Attempting N=100 with
PennyLane was not pursued given N=50 already exceeded ten minutes;
the time cost was judged not worth it once the trend was this clear.

`paulikit`'s own time at N=100 (126.3s) is worth noting honestly: it
is markedly worse than its N=16/N=30 speedup ratio would suggest,
because the current implementation computes the *full* dense
$2^n \times 2^n$ coefficient array regardless of input sparsity — at
N=100 ($n=13$ qubits), that's $4^{13} \approx 67$M coefficients
computed to find ~20.3M nonzero ones. Exploiting the Hamiltonian's
actual sparsity (rather than computing-then-filtering) is the natural
next optimization target once profiling (Phase 2) confirms where the
time actually goes.


## 4. Tooling and prior-work inventory

Surveyed this session, informing methodology (not code reuse):

- A prior, unrelated private project (not named or linked here) informs
  two general engineering lessons applied to this plan, without any of
  its code being reused:
  - A profiled example of a custom memory-pool allocator where 99%+ of
    runtime went to allocator-internal bookkeeping rather than the
    intended computation — a cautionary reminder that an "optimized"
    data structure (e.g. segregated free lists) can itself become the
    bottleneck if the underlying search isn't actually efficient under
    the real allocation/free churn pattern. Worth keeping in mind once
    this project reaches its own C phase.
  - A survey of Python/C bridging techniques and their trade-offs —
    SWIG (mature, heaviest boilerplate), CFFI (dynamic loading, extra
    effort for structs/threads), ctypes (stdlib-only, weakest
    ergonomics for struct-heavy APIs), Cython (best raw performance,
    three-language complexity cost), Numba (JIT, limited type support,
    doesn't wrap existing C). Informs Phase 2's ordering (Cython →
    CFFI → ctypes → SWIG); the C code itself, when we get there, will
    be original to this project.
- **oneTBB** (`onetbb` package, already installed on this machine) —
  Intel's C++ template library for task-based parallelism with a
  work-stealing scheduler. A strong fit for FWHT's embarrassingly
  parallel per-row/per-term structure; the published FWHT paper's own
  ~7x/8-core speedup used a similar parallelization strategy.
- **Intel oneMKL** — investigated and ruled out as directly relevant:
  it's a dense/sparse *linear-algebra* library (BLAS, sparse solvers,
  QR/PARDISO), not a fit for this *symbolic-to-Pauli-string*
  decomposition problem. oneTBB (parallelism, not linear algebra) is
  the actually-relevant Intel technology here.
- **MIT 6.172 Performance Engineering of Software Systems** course
  materials (`~/studies/performance engineering/`) — Bentley's Rules
  (a concrete, numbered checklist of code transformations: data
  structure augmentation, precomputation/caching, loop-invariant
  code motion, strength reduction, inlining, special-case fast paths,
  etc.), execution-strategy surveys (interpreter vs. AOT vs. JIT),
  and deep dives on pthread, OpenMP, Intel TBB, and Cilk. This
  informs the general engineering discipline applied here: profile
  before optimizing, understand the real cost model, focus on
  confirmed hot spots rather than guessing.


## 5. Phased plan

### Phase 0 — Scaffolding (this document + directory structure)
- `tools/paulikit/README.md` — package overview.
- `tools/paulikit/PLAN.md` — this document.
- `tools/paulikit/pyproject.toml` — PEP 621
  package metadata; `paulikit` is installable (`pip install -e .`)
  with a `paulikit` console-script entry point, separate from the
  parent repository's plain `requirements.txt` approach used by the
  tutorial notebooks. Package restructured (2026-08-04) from an
  initial flat `pauli_perf/` script collection into a proper
  `src/paulikit/` layout with `algorithms/` and `testing/`
  subpackages, once it became clear more than one decomposition
  algorithm was in scope (see Section 5, Phase 1 note below) and that
  release to PyPI was a real possibility, not just an internal tool.

### Phase 1 — Original pure-Python FWHT implementation
- Correctness fixtures first, independent of implementation: N=2 (hand
  cross-checked this session against PennyLane and the repo's own
  documented 12-term claim) and N=4 expected decompositions.
- Implement the FWHT-based decomposition from scratch in NumPy,
  following the 3-step structure from the literature (XOR-index
  permutation, FWHT per row, phase-factor multiplication) — an
  original implementation, not a call into PennyLane or any existing
  Pauli-decomposition library.
- Validate against the correctness fixtures and cross-check against
  the PennyLane reference table in Section 3.
- Benchmark at matched N values (16, 30, 50, 100) against Section 3.2's
  numbers; record results in the module README.

### Phase 2 — Profiling
- `cProfile` + `snakeviz` for an initial flame-graph pass across the
  full decomposition pipeline.
- `line_profiler` for granular per-line timing inside whatever
  function(s) cProfile identifies as hot.
- `py-spy` as a no-instrumentation sampling cross-check (catches
  anything the instrumented profilers might distort via overhead).
- Document actual hot spots with real numbers before deciding what (if
  anything) needs a native port — no guessing.

### Phase 3 — C porting experiment (only for confirmed hot loops)

Phase 2 profiling (`tools/paulikit/profiling/README.md`, task #16,
2026-08-16) found two distinct, additive costs, addressed as two
sequenced sub-phases rather than one conflated "port the hot loop"
exercise:

#### Phase 3a — `pauli_label` C port (scoped 2026-08-16, COMPLETE 2026-08-16)
The confirmed bottleneck at N=50: `pauli_label`'s per-term,
per-qubit Python loop is ~60% of `fwht_pauli_terms`'s cumulative
time (4.9s self time out of 11.7s total), called once per nonzero
coefficient. Smallest, most self-contained kernel to port — pure
integer bit-twiddling + string building, no NumPy/array semantics to
bridge, making it a clean first exercise for the binding-technique
comparison.

**Results summary** (full detail in `bindings/README.md`):
- C kernel (`src/paulikit/_native/pauli_label.c`): both single-term
  and batch entry points, verified exhaustively (n_qubits 1-4) and
  against 100K+ random cases at production n_qubits (5-13) — zero
  mismatches against the Python reference throughout.
- All four bindings (Cython, CFFI, ctypes, SWIG) built, verified, and
  benchmarked at matched N (16/30/50/100). **Cython won on both raw
  speed (26.2x label-gen speedup at N=50, vs. 6.5-11.1x for the
  other three) and binding effort (lowest of the four; SWIG needed
  hand-written typemaps for the buffer-argument signatures, the
  highest-effort of the four as PLAN.md's Section 4 survey
  predicted)** — retained as the binding paulikit ships with.
- End-to-end impact at N=100: swapping in the Cython batch label
  call cuts `fwht_pauli_terms`-equivalent time from the all-Python
  126.3s baseline to ~40.2s (3.1x), leaving
  `fwht_pauli_coefficients`'s dense-array computation (38.6s) as the
  new dominant cost — exactly Phase 3b's scope below.
- oneTBB parallelization added (`pauli_label_batch_parallel`,
  `tbb::parallel_for` over independent terms). Standalone C++
  benchmark: 3.9-4.1x on 8 cores (below the ~7x/8-core reference
  point below, plausibly because this kernel is memory-bandwidth-
  rather than compute-bound). **Through the actual Python boundary,
  parallelization barely helps (1.1-1.25x)** — isolated to the
  ~1.26M-element Python list-of-`str` construction already
  dominating wall-clock time, confirming (one level further down the
  stack) the same per-term Python-object-construction cost Phase 2
  profiling originally found. Next lever, if pursued, is avoiding
  per-term `str` construction entirely (e.g. a NumPy fixed-width
  string array), not further parallelizing the C loop.
- **Out of scope for 3a, as planned:** the dense-array-vs-sparsity
  issue below (Phase 3b) was not touched — kept in a separate kernel
  as scoped.

Original scope notes (kernel signature, batch entry point, binding
order, correctness gate) are preserved in git history (see the
Phase 3a commits) rather than duplicated here now that the phase is
complete.

#### Phase 3b — Sparsity-aware `fwht_pauli_coefficients` (scoped 2026-08-16, COMPLETE 2026-08-18)
Section 3's benchmark table shows `paulikit`'s own N=100 time
(126.3s) is disproportionately worse than its N=16/N=30 speedup
ratio would predict, because `fwht_pauli_coefficients` always
computes the *full* dense `2**n x 2**n` coefficient array — at
N=100 (13 qubits), ~67M coefficients computed to find ~20.3M nonzero
(30% density, consistent with the N=50 measurement:
4,194,304 computed vs. 1,261,568 nonzero).

**Profiling first (task #30):** measured, on the real
`build_hamiltonian()` output, what fraction of the FWHT's `dim` rows
are entirely zero and how many nonzero entries an active row has.
Result: the operator itself is sparse (O(N) nonzeros), but the WHT
*rows* are not uniformly sparse — 47-86% of rows have at least one
nonzero entry, depending on N. This ruled out the initially expected
"headline" fix (replacing the O(dim log dim) WHT butterfly per row
with an O(k*dim) sparse-impulse identity, k = nonzeros/row =~4) since
active-row count stays too close to `dim` for that trade to win; full
exploration (8 attempted variants, several regressions) is recorded
in `phase3b/README.md` and `phase3b/explore/`.

**Implemented fix** (`src/paulikit/algorithms/fwht.py`): avoid all
O(dim^2) dense-array construction that the original implementation
did unconditionally regardless of sparsity:
- Skip the O(dim^2) full gather (`operator[p_indices, q_indices]`)
  entirely; scatter the operator's actual nonzero entries directly
  into an `(n_active_rows, dim)` array instead of a `(dim, dim)` one.
- Run the existing dense WHT butterfly only on active rows (all-zero
  rows are skipped, not computed-then-discarded).
- Compute the phase factor only for active rows.
- Replace `_popcount_array`'s bit-serial Python loop with an 8-bit
  lookup-table popcount — a general win, independent of sparsity,
  that also benefits any future dense-input use case.

Exact algorithm (not an approximation): verified against the full
existing test suite (25/25 passing, `tests/test_fwht.py` +
`tests/test_fixtures.py`) and against fixture/PennyLane cross-checks
at N=2/4/16/30 with no change to the public API or return contract.

**Results** (full detail in `phase3b/README.md` Section 5):

| N (oscillators) | `fwht_pauli_coefficients` (old dense) | (new) | speedup | `fwht_pauli_terms` end-to-end (old) | (new) | speedup |
|---|---|---|---|---|---|---|
| 50  | 1.971s  | 0.636s  | 3.1x  | 6.2213s   | 5.4957s   | 1.13x |
| 100 | 35.47s  | 17.56s  | 2.0x  | 126.3250s | 107.2403s | 1.18x |

Term counts match Section 3.4's PennyLane-cross-checked figures
exactly at every N (15360/112384/1261568/20299776) — a correctness
confirmation, not just a performance measurement.

**Honest scope note (resolved by Phase 3c below, 2026-08-18):** the
end-to-end `fwht_pauli_terms` speedup (~1.13-1.18x) was smaller than
the coefficients-only speedup (2.0-3.3x) because `fwht_pauli_terms`
still used the pure-Python `pauli_label` loop — it was not wired to
Phase 3a's Cython kernel. That integration is Phase 3c.

#### Phase 3c — Wire the native kernel into `fwht_pauli_terms`, and adopt meson-python (2026-08-18, COMPLETE)

Two changes, done together since the packaging decision blocked the
integration decision:

**1. Build-system migration: setuptools -> meson-python.** paulikit
now uses the same build backend NumPy and SciPy use (verified via
their current docs/source, not assumed). The Cython `pauli_label`
kernel from Phase 3a (previously a standalone comparison artifact
under `bindings/cython/`) is now packaged inside the library itself
as `paulikit._native.pauli_label_native`, built via `meson.build` +
a Meson `native` feature option (`meson.options`): `auto` (default,
builds it if a C++ toolchain, Cython, and oneTBB are all found),
`enabled` (fail the build if they're missing), or `disabled` (force
pure-Python-only). `pyproject.toml`'s `[build-system] requires` now
lists `meson-python`, `Cython`, and `numpy` unconditionally — PEP
517/518's `requires` list has no standard mechanism for conditional
build dependencies, so this is a deliberate, accepted trade: Cython
and NumPy (both pure-Python-installable, no C toolchain needed just
to have them) are now always required to build from source, while
the actual heavy requirement — a C++ compiler and oneTBB — stays
gated by the `native` feature option. Verified both configurations
build and pass the full test suite: default (`auto`, extension
built and importable) and `-Dnative=disabled` (pure-Python-only,
extension correctly reports as unavailable).

This is a deliberate step toward, but not yet completion of, the
NumPy/SciPy packaging model — the extension is still optional with a
pure-Python fallback, not a hard requirement, because paulikit has no
prebuilt-wheel CI yet (a plain `pip install` from source would
otherwise require a C++ toolchain + oneTBB for every user). **This
is explicitly a temporary compromise, tracked for near-term
resolution, not indefinitely deferred** — adopting prebuilt wheels
(cibuildwheel/manylinux-style CI) so the extension can become a hard
requirement is the next packaging milestone once there's bandwidth
for the CI investment.

**2. Wiring:** `fwht_pauli_terms` now calls a new internal
`_pauli_label_batch` helper that uses
`paulikit._native.pauli_label_native.pauli_label_batch` when the
compiled extension is importable, falling back to the original
per-term pure-Python `pauli_label` loop otherwise. The fallback is
**not silent**: since paulikit's entire purpose is fast Pauli
decomposition, running the slow path unknowingly would defeat the
point of the package, so a `UserWarning` fires (once per process,
not once per call) the first time the fallback path is actually
used, naming the rebuild steps needed to get the fast path.

**Results** (matched N, same synthetic Hamiltonian generator as
Phase 1/3a/3b):

| N (oscillators) | `fwht_pauli_terms` end-to-end: Phase 1 baseline | Phase 3b (sparse coeffs, Python labels) | Phase 3c (native labels) | speedup vs. Phase 1 | speedup vs. Phase 3b |
|---|---|---|---|---|---|
| 50  | 6.2213s   | 5.4957s   | 2.1535s | 2.9x | 2.6x |
| 100 | 126.3250s | 107.2403s | 43.5629s | 2.9x | 2.5x |

Term counts match exactly at every N (15360/112384/1261568/20299776)
across all three implementations — a correctness confirmation, not
just a performance measurement. Full existing test suite (25/25)
passes with the native kernel wired in and, separately, with it
disabled (fallback path).

**Honest scope note:** this closes most, but not all, of the gap
between the coefficients-only speedup and the end-to-end speedup —
`fwht_pauli_coefficients` itself (Phase 3b's scope) is still 2.0-3.1x
faster than its own original baseline, while end-to-end is now
2.9x faster than the ORIGINAL Phase 1 baseline (a fair comparison,
since both label generation and coefficient computation improved).
Remaining end-to-end cost breakdown at N=100 (43.6s total) not yet
re-profiled after this change — a natural next profiling target if
further speedup is pursued, rather than assuming where the new
bottleneck is.

### Phase 4 — Comparison and write-up
- Assemble a final results table/plot: naive SymPy (small N only) vs.
  pure-Python FWHT vs. each C-binding variant vs. PennyLane reference,
  at matched N values, covering both correctness and wall-clock time.
- Update this document and the module README with final findings.
- Feed conclusions back into the main `coupled_harmonic_oscillators`
  module once a production-ready decomposition function exists (a
  separate, later integration step — not assumed as part of this
  plan).

### Phase 5 — Prebuilt wheels, then make the native extension a hard requirement (scoped 2026-08-19, not started)

Phase 3c's optional-extension-with-fallback model was explicitly
scoped as a **temporary compromise**, not the end state: paulikit's
whole purpose is to be a fast Pauli decomposer, so shipping a pure-
Python fallback as a permanent default undercuts that. The NumPy/SciPy
model — the compiled extension is a hard requirement, made viable by
prebuilt binary wheels so `pip install` never needs a local C++
toolchain — is the actual target. This phase is the CI investment that
was deferred in Phase 3c pending "bandwidth."

**Why this wasn't trivial to just do in Phase 3c:** the extension
links `pauli_label_parallel.cpp` unconditionally
(`src/paulikit/_native/meson.build`), so oneTBB is a hard link-time
dependency of the compiled `.so`/`.pyd`, not something that can be
quietly dropped. None of manylinux, macOS, or Windows GitHub/Codeberg
runners ship oneTBB preinstalled, so building real wheels means
fetching it per-platform as part of the build, not just running
`cibuildwheel` out of the box.

**Scope decision (2026-08-19):** build the full cross-platform matrix
now rather than Linux-only — Linux (manylinux, oneTBB via the
container's package manager), macOS (Homebrew), Windows (vcpkg) — across
the Python versions this package already claims to support (3.10–3.13,
per `pyproject.toml`'s classifiers). This is more CI surface area to
debug than a Linux-only first cut, but matches the actual target model
and avoids doing this scoping exercise twice.

**Plan:**
1. **`before-all`/`before-build` provisioning per platform** in a
   `cibuildwheel` config (`[tool.cibuildwheel]` in `pyproject.toml`):
   - Linux (manylinux2014/manylinux_2_28): `yum install tbb-devel` (or
     equivalent) inside the container before the build step.
   - macOS: `brew install tbb`, with `TBB_ROOT`/`CMAKE_PREFIX_PATH`
     (or the meson equivalent — `PKG_CONFIG_PATH`) pointed at the
     Homebrew prefix so `dependency('tbb')` resolves.
   - Windows: `vcpkg install tbb` (or the oneAPI base toolkit's TBB
     component), with `CMAKE_TOOLCHAIN_FILE`/include-lib paths wired
     into the meson cross/native file cibuildwheel generates.
2. **CI workflow files, both platforms** (per the "both GitHub Actions
   + Codeberg" decision):
   - `.github/workflows/paulikit-wheels.yml` (done, commit `d11f61c`)
     — the standard `pypa/cibuildwheel` GitHub Action, matrixed over
     `{ubuntu-latest, macos-latest, windows-latest}`, triggered on tags
     and manually (`workflow_dispatch`) so wheel builds aren't run on
     every push.
   - **Codeberg side (investigated 2026-08-21): full matrix is not
     possible, by design, not just "not yet configured."** Confirmed
     via Codeberg's own docs (`docs.codeberg.org/ci/actions/` and
     `codeberg.org/actions/meta`): Codeberg's hosted Forgejo Actions
     runners are **Linux amd64 only** (`codeberg-tiny/small/medium`,
     2/5/10 min max runtime respectively, plus `-lazy` variants for
     delay-tolerant jobs). This isn't a resourcing gap to wait out —
     upstream Forgejo Runner itself has no official macOS or Windows
     support at all, for a stated project-philosophy reason (Forgejo
     commits to free/libre software only; officially supporting a
     proprietary OS runner would require the project itself to run and
     test against that OS). A community Windows-runner port exists but
     is unofficial/unsupported. Separately, even the 10-minute cap on
     Codeberg's largest hosted tier is likely too tight for a real
     `cibuildwheel` build (compiling a C++ extension against oneTBB
     across cp310-cp313) regardless of the OS question.
     **Decision: Codeberg gets a Linux-only wheel-build workflow**
     (`codeberg-medium` runner, manylinux via `cibuildwheel` same as
     the Linux leg of the GitHub matrix), and this asymmetry is
     documented plainly in the README/release process rather than
     worked around — macOS/Windows wheels are GitHub-only. Revisit
     only if Forgejo Runner ships official macOS/Windows support
     upstream, or if Codeberg's hosted runtime caps are raised enough
     to make it moot.
     **Done:** `.forgejo/workflows/paulikit-wheels.yml`. Two further
     adjustments required for the Codeberg environment specifically,
     beyond just dropping macOS/Windows: (a) `codeberg-medium`'s
     10-minute cap makes a single `cibuildwheel` job covering all of
     cp310-cp313 risky, so the workflow matrixes one job per Python
     version (`CIBW_BUILD` pinned per job) instead of the GitHub
     workflow's single combined job; (b) actions are referenced via
     fully-qualified `https://data.forgejo.org/actions/...@vN` URLs
     (Forgejo's own recommendation over short-form refs, since a
     relying admin-configurable default mirror URL is a real footgun),
     and `cibuildwheel` itself is invoked as a plain `pip install` +
     CLI step rather than via `pypa/cibuildwheel`'s marketplace action,
     since that action's availability on Forgejo's action mirror
     wasn't confirmed and installing via pip sidesteps the question
     entirely. Genuinely unverified until a real run: no tag pushed,
     no manual dispatch triggered yet — same caveat as the GitHub side.
3. **Verify wheels are correct**, not just "the build didn't error":
   run the full test suite (`pytest`, not just import-and-exit) inside
   each built wheel via `cibuildwheel`'s `test-command`/`test-requires`
   hooks, on each platform — the existing 25-test suite plus a direct
   check that `paulikit._native.pauli_label_native` actually imports
   (catching a wheel that "succeeds" but silently fell back to no
   extension, which would be a much worse regression than today's
   honest optional-fallback model).
4. **Only after wheels build and verify successfully on all targeted
   platforms**, flip the packaging model:
   - `pyproject.toml`: remove `native` from being a `meson.options`
     *feature* with an `auto`/`disabled` escape hatch for end users —
     keep the option for from-source builds/development, but stop
     treating "falls back silently" as acceptable for a `pip install
     paulikit` from PyPI (i.e. PyPI-published wheels always include the
     extension; source builds may still opt out via `-Dnative=disabled`
     for development/unsupported-platform cases, matching how NumPy
     itself still allows some source-build flexibility even though its
     wheels are unconditional).
   - `src/paulikit/algorithms/fwht.py`'s `_pauli_label_batch`: decide
     whether the pure-Python fallback path is removed entirely once
     wheels are the primary install path, or kept as a documented
     escape hatch for unsupported platforms/source builds — **an open
     question for this phase, not decided yet**; leaning toward keeping
     a fallback for source builds specifically (some platform will
     always be unsupported by the wheel matrix) while making the
     `UserWarning` more prominent, rather than a hard `ImportError`,
     since a working-but-slow package is still more useful to a
     resource-constrained researcher than a package that refuses to
     install at all.
   - Update `README.md`'s "Native extension" section, `docs/tutorial.md`,
     and `docs/index.md`/toctree accordingly once the model actually
     changes — do not describe the hard-requirement model in the docs
     before the wheels exist and are verified (matches this project's
     "verify, don't guess" / no-overselling discipline established in
     Phase 3b).
5. **PyPI publishing** itself (`twine upload` / trusted publishing via
   GitHub Actions `pypa/gh-action-pypi-publish`) is a separate decision
   from wheel-building CI and is explicitly **not** bundled into this
   phase — building and testing wheels in CI doesn't obligate
   publishing them anywhere; that's a distinct, later call once wheels
   are proven reliable across the matrix.

**Explicitly out of scope for this phase:** ARM/aarch64 wheel variants
(cibuildwheel supports them, but they add QEMU-emulation build time and
aren't needed yet since no current use case has been raised for
non-x86_64 deployment); free-threaded (3.13t) wheel variants (numpy/
Cython ecosystem support for free-threading is still maturing as of
this writing).


### Phase 6 — Sparse output for `fwht_pauli_coefficients` (scoped 2026-08-25, not started)

**Motivation.** A fundamental (not TBB-limited) cache-locality
investigation, per the user's explicit direction, was carried out in
`profiling/cache_locality/` (see that directory's `README.md` for the
full trail - eight findings docs, all with reproducible, checked-in
scripts). It identified a real, measured, robustness-relevant bug,
independent of TBB/compiler flags/threading:

- `fwht_pauli_coefficients` (`src/paulikit/algorithms/fwht.py`,
  currently lines 202-218) computes its result sparsely (only active
  rows, per Phase 3b's optimization - genuinely preserved, not
  regressed by anything below) but then **materializes a full dense
  `(dim, dim)` complex128 array** (`coefficients = np.zeros((dim,
  dim), ...)`, then scatters the sparse rows into it).
- `fwht_pauli_terms` immediately **re-scans that entire dense array**
  (`np.nonzero(np.abs(coefficients) > atol)`) to find the same
  nonzero rows `fwht_pauli_coefficients` already knew about via
  `active_x`/`n_active` one function call earlier.
- Measured impact: cache-miss ratio and memory-stall cycles both
  scale cleanly with how far this dense array exceeds cache size
  (`profiling/cache_locality/steady_state_scaling_findings.md`:
  ~23% at N=25 where it fits in L3, up to ~58.5% at N=100 where it's
  128x over). At N=150, this **OOM-killed the investigation's dev
  machine** (15 GiB RAM) on the current, unmodified code, without
  even completing one decompose call
  (`profiling/cache_locality/n150_oom_finding.md`) - this is not
  merely a performance nicety, it's a real crash risk at realistic
  problem sizes.

**Prior art - this was already anticipated, not a new idea.** Phase
3b's own exploration work already prototyped this exact fix and
explicitly deferred it. `phase3b/explore/08_v5_active_rows_only.py`'s
docstring (2026-08-18, predating this investigation by a week):
*"Returns a dense (dim,dim) array for API compat with the existing
dense function (Phase 3b step: keep the output contract identical for
now; a later step could return the sparse form directly to
`fwht_pauli_terms` without ever densifying)."* That "later step" is
this phase.

**The explicit tension to resolve, not assume away (user's direct
instruction, 2026-08-25):** there is a real, unproven risk that
trading the dense array for sparse iteration could *move* a cost
rather than remove it - e.g. if the sparse representation forces
scattered, irregular memory access in `fwht_pauli_terms`'s label-
generation/dict-construction step where the dense version currently
gets contiguous, vectorizable `np.nonzero`/fancy-indexing access. Any
fix must be measured against both (a) the cache-locality metrics in
`profiling/cache_locality/` and (b) Phase 3b's own sparsity-of-
*computation* gains (2.0-3.1x on `fwht_pauli_coefficients`, see
`phase3b/README.md`) - it is not acceptable to "fix" cache locality by
quietly re-introducing dense, O(dim²) work somewhere else in the
pipeline, or by degrading Phase 3b's already-measured win.

**Compatibility surface (checked directly, not assumed):**
`fwht_pauli_coefficients` is called by: `fwht_pauli_terms` (the only
non-test, non-exploration caller); `tests/test_fwht.py` (6 call
sites) - two of which
(`test_fwht_pauli_coefficients_handles_non_hermitian_matrices`,
`test_fwht_pauli_coefficients_reconstructs_non_hermitian_matrix`)
treat the return value as a **dense, arbitrarily-indexable `(dim,
dim)` array** (`fast.imag`, `coefficients[x, z]` for arbitrary `x,
z`) - this is a real part of the function's tested public contract,
not just an internal implementation detail, and changing it is a
breaking API change, not a private refactor.
`phase3b/explore/*.py` (6 files) call it only as a correctness
reference (`ref = fwht_pauli_coefficients(padded)`) for comparing
against exploratory sparse variants - these are historical artifacts
(see `feedback_document_exploration_scripts` project convention) and
not required to keep working post-change, but breaking them silently
without a note would be sloppy.

**Plan:**
1. **Decide the API shape** before writing any implementation code:
   - Option A: keep `fwht_pauli_coefficients`'s signature/return type
     exactly as-is (dense array), fix *only* `fwht_pauli_terms`'s
     redundant re-scan (it can use `active_x`/`active_coefficients`
     directly if `fwht_pauli_coefficients` is refactored to expose
     them, e.g. via a private helper or an optional return mode).
     Smaller blast radius, keeps the tested dense-array contract for
     `fwht_pauli_coefficients` intact, but leaves the *bigger* half of
     the problem (dense array construction itself, not just the
     re-scan) unfixed for any *other* caller of
     `fwht_pauli_coefficients` and for the two dense-contract tests'
     own dense-array construction cost.
   - Option B: change `fwht_pauli_coefficients` to return a sparse
     representation by default (breaking change - update its two
     dense-contract tests and its docstring/type signature), with
     `fwht_pauli_terms` consuming that sparse form directly. Larger
     blast radius (breaking API change, PLAN.md/README.md doc
     updates, deciding on a concrete sparse return type - e.g. a
     `(x_indices, z_indices... )`... no, likely `(active_x,
     active_coefficients)` matching the existing internal naming, or
     a `scipy.sparse` type), but actually removes the dense array
     everywhere, not just in the one call site currently profiled.
   - This decision is **not yet made** - it should be settled with
     the user before implementation, weighing "smaller, safer patch"
     (A) against "actually fixes the root cause everywhere" (B). Not
     assumed here.
2. **Prototype and measure both real candidates** (not just reason
   about them) using `profiling/cache_locality/`'s existing
   infrastructure - specifically `steady_state_decompose.py` and
   `run_steady_state_sweep.sh`/`run_openblas_comparison.sh`'s
   conventions (in-process warm-up timing, `OPENBLAS_NUM_THREADS=1`
   to avoid the noise documented in `stall_floor_mystery_solved.md`) -
   before picking one. This directly addresses the
   densification-vs-locality tension above: measure whether the
   candidate actually reduces cache-miss ratio/mem-stall *and*
   preserves or improves wall-clock time at N=25/50/100(/150 if
   memory-safe post-fix), not just assume a "sparse is obviously
   better" story.
3. **Update the two dense-contract tests** in `tests/test_fwht.py`
   (`test_fwht_pauli_coefficients_handles_non_hermitian_matrices`,
   `test_fwht_pauli_coefficients_reconstructs_non_hermitian_matrix`)
   to match whichever API shape is chosen - these are the only tests
   that exercise `fwht_pauli_coefficients`'s dense-array contract
   directly and must not be silently weakened or deleted just to make
   the change easier.
4. **Re-run the full `profiling/cache_locality/` sweep post-fix**
   (N=25/50/100/150) and add a new findings doc comparing before/after
   - honestly, including if the improvement is smaller than the
   cache-miss-ratio numbers alone might suggest (per
   `stall_cycles_n50_findings.md`'s and
   `steady_state_scaling_findings.md`'s existing caution about
   overclaiming from ratio metrics alone). Specifically verify N=150
   no longer OOMs, since that's the concrete robustness claim this
   phase is meant to fix.
5. **Update `README.md`'s "Native extension"/benchmark sections and
   `PLAN.md`'s status** once the fix is measured and merged - do not
   describe the fix as done in any doc before it's implemented and
   verified, matching this project's established no-overselling
   discipline.

**Follow-up items, tracked here explicitly so they aren't lost (not
started, not blocking Phase 6's own implementation):**

- **Statistical rigor for `perf`-based measurements.** Every sweep in
  `profiling/cache_locality/` so far reports a bare mean over 3 runs,
  with no variance/confidence-interval reporting and no systematic
  outlier handling. One concrete unexplained data point surfaced
  during an earlier (superseded, non-committed) TBB comparison attempt
  at N=25: a single run showed ~2.7x the cycles and ~5.9x the stall
  cycles of its two sibling runs, silently excluded rather than
  investigated at the time. Before or during Phase 6's "prototype and
  measure" step, add: (a) reporting min/max or stddev alongside the
  mean in findings docs, not just the mean; (b) an explicit rule for
  what counts as an outlier and what to do with one (investigate and
  report the cause, don't just silently average it in or drop it);
  (c) consider whether 3 runs is enough given the variance actually
  observed, or whether the default should increase (`run_*.sh`
  scripts already accept a runs-per-N argument, so this is a
  documentation/convention change, not a script rewrite). This
  strengthens every finding already on record, not just Phase 6's
  new ones - worth doing as its own small pass, not deferred
  indefinitely.
- **Revisit TBB parallelization if Phase 6's redesign changes the
  work shape.** `tbb_evaluation_findings.md` (2026-08-25) found no
  effect at the *current* pipeline structure - TBB parallelizes label-
  string construction, which isn't where the dense-array cache misses
  live. If Phase 6's sparse-representation redesign changes how much
  work happens in or near label generation (e.g. if the sparse mode
  processes far more active terms per call than the dense mode's
  re-scan currently surfaces), re-run `run_tbb_comparison.sh`'s
  methodology against the *new* pipeline shape as part of Phase 6's
  own "prototype and measure" step (step 2 above) - a fair question
  to re-open then, not a settled "TBB is irrelevant forever" verdict.

**Explicitly out of scope for this phase:** re-profiling or
re-optimizing the native Cython/C++/TBB kernel itself
(`pauli_label_native`/`pauli_label_parallel.cpp`) -
`profiling/cache_locality/perf_record_n50_findings.md` and
`compiler_flags_findings.md` both found the current cache-locality
bottleneck lives in `fwht.py`'s Python-level dense-array handling and
NumPy's own ufunc code, not the native kernel. Stronger than "not the
bottleneck": `tbb_not_actually_used_finding.md` confirmed the TBB-
parallelized entry point (`pauli_label_batch_parallel`) isn't even
called by `fwht_pauli_terms` today - `_pauli_label_batch` calls the
serial `pauli_label_batch` kernel (Phase 3a found parallelizing this
specific loop barely helped, since Python string construction
dominated wall-clock time, not the C loop). Touching the native/TBB
kernel is not motivated by any finding in this investigation and
would be scope creep.

**Update 2026-08-25 - TBB directly measured, not just inferred:**
per explicit instruction, the TBB-parallel kernel was tested end-to-end
with the full cache-locality methodology (N=25/50/100) *before*
starting this phase's implementation work, rather than leaving the
"would TBB help a redesigned pipeline" question open. Result: no
measurable effect on wall time, cache-miss ratio, LLC-miss ratio, or
stall percentages at any N - see
`profiling/cache_locality/tbb_evaluation_findings.md`. This closes the
question for the *current* pipeline structure. It remains a fair
question to revisit only if Phase 6's own prototyping surfaces a new
hot loop TBB could plausibly parallelize - not as a default
assumption, and not motivated by anything found so far. Also out of
scope: addressing the OpenBLAS thread-pool noise itself
(`stall_floor_mystery_solved.md`) - that's an
environment/measurement-methodology concern, not a paulikit code
change.

**API-shape decision (user, 2026-08-25):** Option B (change
`fwht_pauli_coefficients`'s return type - see step 1 above), but not
as a one-way breaking replacement. The user was explicit: *"since I am
extremely skeptical that going for sparse matrix instead of dense
could potentially degrade performance dramatically, let's do it in a
way that it is easily reversible or even create it as an option! we
could do the calcs either with dense or with sparse matrices. I am
more inclined towards the optional deployment."* This modifies step 1
above: the implementation should expose both a dense and a sparse
output mode (e.g. a keyword argument, not two separate functions,
to keep one code path to maintain and test), with the sparse mode
becoming Phase 6's actual fix and the dense mode preserved verbatim
for the two existing dense-contract tests and any caller not yet
migrated. Not yet designed in detail - the concrete parameter
name/shape and default value are open, to be settled at
implementation time, not assumed here.


### Phase 7 — Items from the 2026-08-25 Gemini-transcript review (scoped 2026-08-25, not started)

**Context.** The user shared a Google AI Mode chat about paulikit
(exported to PDF, no export/copy method existed) that raised five
distinct technical threads. All five were explicitly agreed as
separate work items requiring individual research-then-plan-then-
approve-then-execute treatment, not a single blended effort. Item 3
(cache locality) was prioritized first and fully investigated - see
`profiling/cache_locality/README.md` and this document's Phase 6. The
other four are tracked here so they aren't silently dropped the way
items 7.1/7.2 initially were (caught only when the user asked directly
"do you remember those items? are all in memory/plan?").

**7.1 — False sharing in the TBB-parallel kernel
(`pauli_label_parallel.cpp`).** Whether concurrent worker threads
write to adjacent memory that shares a cache line, causing invisible
cross-thread invalidation traffic. **Explicitly conditional on TBB
actually being in the hot path** - `tbb_not_actually_used_finding.md`
and `tbb_evaluation_findings.md` (Phase 6) both confirm it currently
is not, and measured no cache-locality effect either way at the
current pipeline structure. Investigating false sharing in dead code
would be premature optimization of something that doesn't run. Revisit
only if Phase 6's redesign (or any future change) actually wires TBB
back into `fwht_pauli_terms`'s call path - at that point, this becomes
a real, checkable question (e.g. via `perf c2c` or manual cache-line-
padding experiments on the per-thread output buffers in
`pauli_label_parallel.cpp`), not before.

**7.2 — TBB partitioner/grain-size choice for sparse, uneven
workloads.** Whether the current partitioner (default, unexamined) is
well-suited to Phase 3b's sparsity-aware row-skipping, which makes the
per-row workload uneven (some rows short-circuit, some don't). Same
conditionality as 7.1: not worth tuning a partitioner for a kernel
that isn't called. If Phase 6 or a later phase reintroduces TBB to a
genuinely uneven workload, benchmark `simple_partitioner` vs.
`auto_partitioner` vs. explicit grain sizes against
`profiling/cache_locality/`'s existing methodology before picking one
- don't assume the default is right just because it wasn't wrong when
nothing depended on it.

**7.3 — Statistical rigor for `perf`-based measurements.** Already
tracked under Phase 6's "Follow-up items" above (min/max or stddev
reporting, an explicit outlier-handling rule, reconsidering the
3-runs-per-N default) - not duplicated here, but this is where that
item's origin (item 4 of the five) is recorded for traceability. This
one is NOT conditional on anything else - it strengthens every finding
already on record and should be picked up opportunistically, e.g.
alongside Phase 6's own "prototype and measure" step, rather than
waiting for a dedicated session.

**7.4 — PyPI publishing and write-up strategy.** Currently only a
one-line deferral in Phase 5 ("PyPI publishing itself... is a separate
decision"). Needs actual scoping as its own decision point before
Phase 5 is considered complete: concrete sub-questions not yet
answered - (a) trusted publishing via GitHub Actions vs. manual
`twine upload`; (b) version-numbering scheme and whether `0.1.0` ships
as the first PyPI release or whether a pre-1.0 milestone is set first;
(c) whether publishing happens per-tag automatically or is a manual,
reviewed step each time; (d) what "ready to publish" means concretely
(Phase 4's write-up done? Phase 5's wheels validated on a real CI run,
not just written? Phase 6's cache-locality fix merged, given N=150
currently OOMs?) - i.e. should Phase 7.4 itself wait until Phases 4-6
are actually done, not just scoped. This item is explicitly a
**planning** task, not an execution task, until those sub-questions
are answered with the user.

**Suggested order, not a hard sequence:** Phase 6 (already in
progress, cache locality was item 3 and prioritized first) → Phase
7.3 (cheap, strengthens existing findings, no dependencies) → Phase
7.4's planning sub-questions (needed before Phase 5 can be called
done regardless of 7.1/7.2) → Phase 7.1/7.2 only if/when TBB
re-enters the hot path (currently no known trigger for this - don't
schedule work against a hypothetical). Per the user's stated
preference, items can be tackled strictly in order or interleaved as
each session's context allows - same approach as folding the TBB
evaluation into Phase 6 rather than treating phases as rigidly
sequential.


## 6. Explicitly out of scope

- Direct reuse of any private prior project's code. Only general
  lessons (e.g. the allocator bottleneck pattern discussed in
  Section 4) inform this work.
- Fixing any private prior project's own performance issues — out of
  scope for this repo.
- Committing to Intel oneMKL specifically — investigated and ruled out
  (wrong problem domain; see Section 4).
- Module 05's other open items (sign-loss in `decode_state`, Trotter
  repetitions tied to loop index, the "exponential speedup" claim
  qualification, `suzuki_trotter`'s own Classiq API migration) — these
  remain tracked separately as task #5 in the main repo task list and
  are not part of this performance-engineering effort.


## 7. Process notes

- Work proceeds in small, individually-committed steps ("baby steps"),
  each with its own commit, per this repo's established commit
  discipline: one logical change per commit, no accumulative commits.
- The task list (harness `TaskList`, tasks #12–#16 at time of writing)
  tracks granular progress; this document tracks the durable plan and
  rationale, and should be updated (not just appended to) as phases
  complete or the plan changes.


## 8. References

- [Exponential Quantum Speedup in Simulating Coupled Classical Oscillators](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.041041) — the algorithm this Hamiltonian simulation implements.
- [Pauli decomposition via the fast Walsh-Hadamard transform](https://iopscience.iop.org/article/10.1088/1367-2630/adb44d) — primary algorithm reference for Phase 1.
- [Tensorized Pauli decomposition algorithm](https://arxiv.org/abs/2310.13421) — candidate for later comparison.
- [PennyLane `qml.pauli_decompose` documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli_decompose.html) — reference baseline implementation.
