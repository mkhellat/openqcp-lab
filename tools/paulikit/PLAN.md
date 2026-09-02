# Pauli Decomposition Performance Engineering — Plan

Status: Phases 0-3c complete. Phase 6 (sparse output for
`fwht_pauli_coefficients`, plus two follow-on memory-footprint fixes
and an optional `chunk_size` tiling parameter - see Phase 6's
"Update 2026-08-26" below) is implemented and correctness-verified.
Phases 8, 9, and 10 (sparse Hamiltonian input; chunked-accumulator
space-complexity fix with opt-in checkpoint/resume; and
streaming/generator output, respectively) are all implemented and
correctness-verified (see each phase's own status below). Together
they fully resolve N=150: `fwht_pauli_terms_iter` (Phase 10) streams
all 91,652,096 real terms to completion under a 2 GB memory cap,
down from a design that needed well over 13.5 GB and still failed
before Phase 10 - see `profiling/phase10/phase10_streaming_findings.md`.
N=150 is now a solved, repeatable case, not an open problem. Phase 4
(final write-up), Phase 5 (prebuilt wheels + hard-require the native
extension), and Phase 7 (remaining items from the 2026-08-25
Gemini-transcript review: TBB false sharing/partitioner tuning -
conditional on TBB re-entering the hot path, statistical rigor for
perf measurements, PyPI publishing strategy) are all scoped but not
started. Phase 5 scoped 2026-08-19; Phase 6 and Phase 7 scoped
2026-08-25; Phase 8 scoped 2026-08-26, implemented 2026-08-27; Phase
9 and Phase 10 both scoped and implemented 2026-08-27.
Last updated: 2026-08-27.


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

### Phase 0.5 — `configure` / per-tool build system (2026-08-28)

**Motivation:** the repo-wide `bootstrap` script is scoped to
provisioning Python 3.12 for `classiq` compatibility (the tutorials'
constraint) — irrelevant to paulikit, which has its own build system
entirely (meson-python, Cython, optional C++/oneTBB native
extension). This surfaced concretely while setting up an environment
to cross-validate paulikit against PennyLane: three separate
build-isolation bugs (missing cython on PATH during the editable
rebuild-on-import hook, a stale numpy include path baked in from an
ephemeral pip build-isolation env deleted right after install, meson
rejecting an absolute in-tree numpy include path when a venv is
created inside `tools/paulikit/`) had to be fought by hand before
PennyLane could even be installed. Per-tool build systems, scoped to
each tool's actual needs, are more practical than one universal
bootstrap.

**Design, in the spirit of GNU-style `configure`/`Makefile.in` but
hand-written POSIX `/bin/sh`** (no autotools/autoreconf/M4
dependency, portable to non-GNU userlands): `./configure` runs a
battery of independent diagnostic compile-tests (musl/zlib-style —
write a tiny source file, invoke the compiler, check the result) and
prints an itemized capability report, covers venv creation, then
generates `Makefile` from `Makefile.in` via literal-value shell
substitution (no GNU-make-only or BSD-make-only conditional syntax
needed in the emitted file, since every value is already resolved by
configure itself before writing it out).

**Explicitly does NOT duplicate meson-python's own build-gating
compiler/dependency detection** — that stays solely meson's job,
invoked from the Makefile's `build` target. `configure`'s checks are
diagnostic/observability, not build-gating: the point, per direct
user instruction, is that "the developer should be fully aware of
... what exactly are needed in terms of compiler features and system
capabilities for what is needed under the hood for different
features of paulikit," warned explicitly when a feature won't be
available on this machine — not a second, potentially-drifting source
of build-gating truth.

**Scope, split into v1 / v2 (both now implemented)**
after 4 rounds of critical review (profiling/ read in full across 3
rounds, PLAN.md/pyproject.toml/src/ cross-checked, a design-readiness
pass covering implementation ordering and failure-mode philosophy,
plus a PEP 405/668/517/518 citation check to settle whether
Python-dependent checks should run against system Python or venv
Python — settled: PEP 405 makes interpreter-selection-before-venv-
creation definitional, not a preference, but "which Python should
diagnostics run against" is genuinely unaddressed by any PEP; the
tool-design choice made here is venv Python, since that's the
environment paulikit will actually run in):

- **v1 (implemented, `tools/paulikit/configure` +
  `tools/paulikit/Makefile.in`):** Python interpreter + venv (default
  `~/.venvs/paulikit`, outside the source tree — an established prior
  convention confirmed from historical `perf stat` command lines, and
  necessary since meson rejects an absolute in-tree numpy include
  path), C++ compiler + C++17 compile-test, Cython >=3.0, meson
  >=1.1.0 (plus reporting the existing `-Dnative` feature option),
  oneTBB (compile-test against `<tbb/tbb.h>`, checking
  `TBB_ROOT`/`CMAKE_PREFIX_PATH` on macOS, `CMAKE_TOOLCHAIN_FILE` on
  Windows, per the committed Phase 5 wheel-matrix policy), cache
  hierarchy L1/L2/L3 via `getconf LEVEL*_*CACHE_SIZE` (POSIX-standard,
  verified working; note `getconf` reports per-core sizes, `lscpu`
  reports per-socket — not directly comparable without dividing by
  instance count; feeds Phase 12's chunk_size auto-tuner directly),
  NumPy's linked BLAS backend + thread count (`threadpoolctl`
  preferred, falls back to `numpy.show_config()` — surfaces the
  OpenBLAS idle-worker-pool confound from
  `profiling/cache_locality/stall_floor_mystery_solved.md`), Python
  version vs. `requires-python` (warns "untested" not "violation" —
  no declared upper bound exists), git commit + working-tree
  clean/dirty state, `lscpu` output including its own
  `Vulnerabilities:` section. Failure-mode philosophy: only Python/venv
  and the C++ compiler are allowed to partially-abort (skip the
  dependent report section, never kill the whole script — paulikit's
  pure-Python fallback works with zero compiler present); every other
  check degrades to "not found"/"unknown" and the report continues.
- **v2 (implemented 2026-08-29)** — each item's value is either
  forward-looking (nothing in paulikit consumes it yet) or only
  matters during an active profiling session, not routine
  build/install: CPU SIMD AVX2/AVX-512 via real
  compile-and-execute micro-tests (diagnostic only, must never
  auto-apply `-march=native` — known SIGILL wheel-portability hazard
  per `profiling/`'s `compiler_flags_findings.md`), GPU/CUDA presence
  (`nvidia-smi`/`nvcc`), MPI toolchain presence (`mpicc`/`mpirun`),
  `psutil` presence (Phase 12 open question #4), container/VM
  detection (`systemd-detect-virt` — cgroup limits differ from
  bare-metal `ulimit -v`, and every N=150 memory-cap finding this
  session assumed bare metal), `/proc/sys/kernel/perf_event_paranoid`
  (currently copy-pasted identically in 6 separate
  `profiling/*/run_*.sh` scripts — centralize here instead), CPU
  frequency scaling/governor state (confirmed dynamic and never
  checked in any measurement this session — a real gap), total
  RAM/swap presence, current system load average, actual available
  `perf list` event set (`cycle_activity.stalls_*` confirmed
  Skylake-family-specific, not universal), NumPy's own runtime SIMD
  dispatch tier (distinct from paulikit's native-extension build
  flags), musl-host awareness (`skip = "*-musllinux*"` is deliberate
  cibuildwheel policy, not an oversight — report "no wheel target
  exists here by design" rather than implying unverified support),
  disk I/O type (`lsblk -d -o rota` — relevant only to
  `checkpoint_path`'s write throughput, lower priority than the rest).

**Real bugs found and fixed while integration-testing v2** (run to
completion, not just written — `./configure` executed end-to-end,
`make build && make check`, 79/79 tests passing):
1. The AVX-512 compile-and-execute test correctly detected the
   feature as unsupported on this machine (an AVX2-only i7-8550U),
   but bash's own job-control notice for the SIGILL-killed test
   binary (`Illegal instruction (core dumped)`) leaked into
   `config.log` regardless of the command's own `2>&1` redirection —
   a `set -eu` interaction, not fixable via the command's own
   redirection alone. Fixed by wrapping each SIMD compile-test
   statement in `{ ...; } 2>/dev/null`, which suppresses bash's own
   job-completion notice at the enclosing-block level.
2. **Pre-existing v1 bug, found incidentally while verifying v2's new
   sections didn't disturb report ordering**: the `lscpu` check wrote
   its output via its own explicit `>>"$REPORT_TMP"` append-redirect,
   nested inside the outer report block's own `{ ... } >"$REPORT_TMP"
   2>&1` redirection — two independent writers to the same file
   caused `lscpu`'s content to land at the wrong offset relative to
   the rest of the report (confirmed via `config.log`: the `== CPU
   (lscpu) ==` header printed with no content directly beneath it,
   while stray `lscpu` output appeared much later, interleaved with
   the final Makefile-generation message). Fixed by removing the
   redundant explicit redirect — `lscpu`'s stdout was already
   correctly captured by the enclosing block once written as a plain
   statement.

**Real bugs found and fixed while integration-testing the generated
Makefile** (not just designed — actually run to completion,
`make build && make test`, 71/71 passing with the native TBB
extension loaded, not the pure-Python fallback):
1. The stale-build-isolation-path bug reproduced exactly as
   diagnosed by hand earlier this session. Fix: `make build`
   pre-installs `numpy`/`meson-python`/`cython`/`ninja` into the venv
   itself, then installs paulikit with `--no-build-isolation`, so the
   paths meson records in `build.ninja` are the venv's own permanent
   ones, not an ephemeral temp dir deleted right after install.
2. Even with `--no-build-isolation`, meson's own compiler-detection
   subprocess looks up `cython`/`cython3` via a bare `PATH` search,
   and `pip` does not activate the venv just because
   `--no-build-isolation` was passed — fixed by prepending the venv's
   own `bin/` onto `PATH` for both the `build` and `test` targets.
3. **A generated `Makefile` almost silently overwrote the
   pre-existing, committed, hand-written `tools/paulikit/Makefile`**
   (targets `install`/`test`/`test-native`/`docs`/`clean`), which the
   *repo-root* Makefile's `test-tools`/`docs-tools`/`clean` targets
   depend on via a specific calling convention (`make -C
   tools/paulikit test PYTHON=<repo-shared-venv-python> PIP="... -m
   pip" PYTEST=<repo-shared-venv-pytest>`, using the repo-wide
   `bootstrap`-provisioned venv, not paulikit's own dedicated one).
   Caught before committing (via `git status` showing "Modified" not
   "new file"). Resolved by making `Makefile.in` support both calling
   conventions simultaneously: `PYTHON`/`PIP`/`PYTEST` stay
   `?=`-overridable (honoring the repo-root's override), defaulting
   to this package's own dedicated venv when invoked standalone; the
   old `test-native` target (standalone C/C++ kernel checks, no
   Python binding) was carried over unchanged. `Makefile`/`config.log`
   are now gitignored (generated artifacts, autotools convention) and
   the repo-root Makefile's `clean`/`test-tools`/`docs-tools` loops
   now run `./configure` automatically in any `tools/*/` directory
   that has a `configure` script but no `Makefile` yet, so a fresh
   clone doesn't silently skip paulikit.

**GNU Coding Standards compliance pass (2026-08-28, same day):**
direct user question ("we had to abide by GNU coding and formatting
conventions — do you think configure abides by that?!") prompted a
dedicated audit against the actual GNU Coding Standards
(https://www.gnu.org/prep/standards/), not just the musl/zlib-style
compile-test *technique* the diagnostic checks were originally
modeled on — these are different standards serving different
purposes, and the first `configure` draft only satisfied the latter.
Audit found one real bug and several defensible-but-real scope gaps;
user's explicit instruction was full compliance now, not deferral:

- **Fixed: `all` as the default target.** GCS requires the first
  rule in the Makefile be `all`, and bare `make` build by default.
  The first draft's first target was `help`, meaning bare `make`
  printed help instead of building — a genuine bug against the
  standard, now fixed (`all: build`).
- **Added the full standard GNU directory-variable option set**
  (`--prefix`, `--exec-prefix`, `--bindir`, `--libdir`,
  `--sysconfdir`, `--datadir`, `--includedir`, `--mandir`,
  `--infodir`, `--docdir`, `--htmldir`, `--dvidir`, `--pdfdir`,
  `--psdir`, etc.) — resolved the one real design ambiguity first:
  most of these have no real target for a pip/meson-python-installed
  package (pip already owns all binary/library placement inside the
  venv), so only `--prefix` (install-root concept) and `--docdir`
  (where `make docs` output lands) actually change behavior; the rest
  are accepted, explicitly reported in `config.log` as
  accepted-but-inert, and documented in `--help` — never silently
  dropped, never a hard error either (a user scripting a generic
  GNU-style build invocation shouldn't get surprised).
- **Added `VAR=value` positional-argument overrides**
  (`./configure CXX=clang++`), the GCS convention, alongside the
  pre-existing `CXX` environment-variable pickup.
- **Added `--srcdir`** — accepted, but out-of-tree builds are
  honestly reported as unsupported (every value `configure` emits is
  already a resolved literal, not a build-tree-relative path; adding
  real VPATH support was judged out of scope for this pass).
- **Added `config.status`** — a minimal re-runnable record of the
  exact `./configure` invocation (not full autoconf-style
  `--recheck`/cache machinery, just enough to replay the same flags
  without having to remember them).
- **Added missing standard targets**: `install-strip` (alias — no
  compiled binaries land outside the venv to strip), `installdirs`,
  `installcheck` (caught and fixed a real bug while testing: the
  first draft ran `pytest --pyargs paulikit`, which collects **zero**
  tests, since paulikit's tests live in `tests/`, outside the
  package itself, not inside `src/paulikit/` — `--pyargs` only
  discovers package-internal tests; fixed to a plain `$(PYTEST)`
  run), `mostlyclean` (deliberately preserves `build/`'s compiled
  native extension — genuinely expensive to relink against TBB,
  unlike the cheap-to-regenerate cache/egg-info clutter `clean`
  removes), `maintainer-clean`, `dist` (`python -m build`, sdist +
  wheel), `TAGS` (ctags preferred, etags fallback).
- **Added the `Makefile: Makefile.in configure` self-regeneration
  rule** GCS requires — verified it actually fires (touched
  `Makefile.in`, confirmed the next `make` invocation reran
  `./configure` automatically before proceeding).
- **Confirmed already-correct**: `#!/bin/sh` + POSIX-only constructs
  throughout, the `Makefile.in` generated-file header comment, and
  the one hard error message (`configure: unrecognized option
  '$arg'`) already matched GCS's `program: message` format exactly.

Every new/changed target was actually run to completion during this
pass, not just written — `make`/`make check`/`make installcheck`/
`make installdirs`/`make mostlyclean`/`make maintainer-clean`/
`make TAGS`/`make dist`, plus the self-regeneration rule, all
verified working (71/71 tests passing throughout).

**Update (2026-08-29): comprehensive correctness batteries + output
format overhaul.** A comprehensive-sanity/bonus-tests review found the
v1/v2 diagnostic checks were mostly presence/version checks, not
correctness checks — e.g. the Cython check only verified `import
Cython; print(version)`, never that Cython could actually build
anything. Closed for every toolchain layer, each verified to catch
real breakage via a deliberately-broken stub before being trusted:

- **C++**: `-O1`/`-O2`/`-O3` numeric-correctness (not just "compiles"
  — a real miscompilation bug would show as a wrong sum, not a compile
  error), `-Wall -Wextra` warnings actually fire, `-lm` math-library
  correctness, exception handling across a call boundary, a
  project-header battery matching `_native/pauli_label_parallel.cpp`'s
  real includes exactly (`<stdint.h>`, `<vector>`, `<chrono>`,
  `<random>` — `std::mt19937(42)`'s first output checked against the
  C++-standard-mandated exact value), and a known-quirk check for
  `-march=native` leaking in via `CXXFLAGS`/`CPPFLAGS` (the real,
  documented SIGILL wheel-portability hazard from
  `profiling/cache_locality/compiler_flags_findings.md`).
- **Python**: IEEE754 float correctness, ctypes/dynamic-linker
  sanity, json round-trip correctness, and a full replication of
  `fwht.py`'s real checkpoint I/O mechanism (`_append_checkpoint_chunk`/
  `_load_checkpoint`'s exact Path+append-mode+progress-marker pattern).
- **Cython**: a real `cython --cplus` transpile → C++ compile → link →
  import → execute pipeline exercising `cimport numpy`, C-level
  `libc.stdint` typing, and typed `ndarray` buffer access — the actual
  capabilities `pauli_label_native.pyx` depends on.
- **meson**: fixed a real pre-existing bug (`mesonbuild.coredata`
  isn't auto-imported, so meson's version was silently always
  "unknown"), and added a real `meson setup` run in a throwaway temp
  dir — the same command `make build` itself depends on — checking
  meson's own reported findings for the `cython` language module and
  `tbb` dependency resolution.
- **oneTBB**: replaced the link-only test with a real
  `tbb::parallel_for`/`tbb::blocked_range<int64_t>` correctness check
  matching `pauli_label_parallel.cpp`'s exact API usage, plus a
  `tbb::info::numa_nodes()` NUMA-awareness check.
- **NumPy/BLAS**: found that paulikit's actual FWHT
  (`_walsh_hadamard_transform_rows`) is pure elementwise complex
  add/subtract on reshaped views — no BLAS-backed call anywhere — so
  BLAS's own correctness isn't load-bearing for paulikit's math. Added
  two separate, honestly-labeled checks instead: the real FWHT
  butterfly pattern verified against known Walsh-Hadamard-transform
  identities, and a generic BLAS matmul check explicitly labeled "not
  currently used by paulikit."

**CPU/cache/NUMA moved right after OS/arch detection** (previously
scattered across the file), matching the same "machine-identity fact"
category as OS detection. Added a `getconf`-vs-sysfs cache-size
cross-check (two independent OS sources, previously never compared),
a real empirical cache-latency probe (`--probe-cache-latency`, opt-in
since ~3s vs. <1s for every other check) written in raw x86_64
assembly — deliberately not compiled C++, since this section runs
before C++ toolchain detection populates `$CXX` later in the file;
assembled/linked directly with `as`/`ld`, zero compiler dependency,
raw `mmap`/`munmap`/`write`/`exit` syscalls — and Slurm/PBS-Torque/LSF
job-scheduler-context detection (honest about the real limitation:
`configure` runs once on one node and cannot see a whole
supercomputer's cross-node topology, which lives in the scheduler).

**Report format overhaul**: replaced the original post-hoc itemized-
table style with the classic incremental GNU `checking for X...
yes/no` format (live, one line at a time, matching real
autoconf-generated scripts), `--verbose` now shows the underlying
probe/command for every check (not just the compile-tests), and a
full manual line-length sweep brought the whole script to real GNU
~79-column discipline (verified against the actual `config.guess`
source, not assumed) — caught 2 real orphaned-output bugs along the
way (unknown-OS fallback branches missing their `checking()` call).

**Update (2026-08-30): ARM64 + RISC-V assembly cache-latency probes
DONE.** Both architectures added as new `case "$OS_ARCH"` arms in the
`--probe-cache-latency` block, mirroring the x86_64 raw-assembly
probe's structure exactly (same mmap/munmap/write/exit syscalls, same
scrambled-stride pointer-chase, same awk boundary-detection). Real
syscall numbers verified from `/usr/include/asm-generic/unistd.h`
(aarch64 and riscv64 share the SAME generic syscall table — unlike
x86_64's own unique table): write=64, exit=93, munmap=215, mmap=222.
Timing source per architecture: aarch64 uses `CNTVCT_EL0` via `MRS`
(EL0-accessible with no privileged setup, unlike `PMCCNTR_EL0` which
needs `PMUSERENR_EL0` enabled first); riscv64 uses the `rdtime`
pseudo-instruction (a CSR read of the `time` counter). Both probes
were built, then live-verified end-to-end through the real `configure`
script (not a standalone harness) via cross-assembling with
`aarch64-linux-gnu-as`/`-ld` and `riscv64-linux-gnu-as`/`-ld` and
executing under `qemu-aarch64`/`qemu-riscv64` (user-mode emulation) —
confirmed via `OS_ARCH` override testing that both arms correctly
assemble, link, run, and parse output. A deliberately-broken fake
`aarch64-linux-gnu-as` was used to confirm the check fails honestly
("skipped") rather than false-passing. `make check` still 79/79,
`shellcheck -s sh` clean, no shell-line-length regressions.

**Explicit caveat baked into both probes' own result output** (not
glossed over): QEMU user-mode emulation validates assembly/syscall
*correctness* only — it has no real cache hierarchy to time against,
so the reported cycle numbers are not a trustworthy latency curve on
real hardware. For riscv64 specifically there is a second caveat: a
clean `rdtime` read under QEMU does not by itself prove the `time` CSR
is user-mode-accessible on real silicon, since real cores commonly
gate it behind `mcounteren`/`scounteren` and Linux often traps+
emulates `rdtime` in the kernel — QEMU's linux-user emulation may be
more permissive than real hardware in this respect.

PowerPC remains explicitly out of scope (legacy Summit/Sierra-class
systems, not a current new-build target).

**Update (2026-08-30, continued): aarch64/riscv64 SIMD battery DONE.**
Extended the existing x86_64 AVX2/AVX-512 compile-and-execute block
with cross-arch counterparts — NEON for aarch64 (`<arm_neon.h>`,
`vaddq_s32`) and RVV for riscv64 (`<riscv_vector.h>`,
`__riscv_vadd_vv_i32m1`, requires `-march=rv64gcv` — confirmed the
compile genuinely fails without that flag, so it's a real gate, not a
no-op). Cross-compiled with `aarch64-linux-gnu-g++`/
`riscv64-linux-gnu-g++` (`-static`, to avoid a cross-sysroot `-L`
dependency at run time — confirmed the dynamic-linked build needs
`-L /usr/aarch64-linux-gnu` while `-static` needs nothing extra) and
executed under `qemu-aarch64`/`qemu-riscv64`, same pattern as the
cache-latency probes. NEON is baseline-mandatory on real aarch64
hardware (so this is mainly a toolchain-support check); RVV is
genuinely optional/uneven across real RISC-V hardware (so it's a real
capability probe, closer in spirit to the AVX tests) — both
distinctions stated explicitly in the checks' own result text, along
with the standard QEMU-can't-prove-real-hardware-support caveat (RVV
especially, since QEMU's default vector model/VLEN may not match any
real target's actual configuration). Verified live through the real
`configure` script, confirmed to fail honestly via a deliberately-
broken fake `aarch64-linux-gnu-g++`. `make check` 79/79, shellcheck
clean, no line-length regressions, memory monitored (no leak).

### Phase 0.6 — Exhaustive correctness verification (closed 2026-08-28)

**Problem.** `tests/test_benchmark_reference.py`'s existing PennyLane
comparison (Section 3.4) only checked term *counts* at large N
(N=50/100), not coefficient values — a real correctness gap, since two
decompositions could have the same term count but wrong coefficients.
Non-Hermitian input had zero PennyLane cross-check at any N.
Strengthening the existing test to check coefficients directly hits a
hard wall: `qml.pauli_decompose` is O(n·4^n) **regardless of input
sparsity** — confirmed by reading its own source (its sparse-input
support only avoids densifying the *input*, not the exponential
*output*). Measured directly: PennyLane took 34s at N=25 and 51s at
N=30 on a genuinely sparse Hamiltonian; paulikit computes 91.65
million terms at N=150 in ~100s. PennyLane can never verify paulikit
at its target scale — a mathematical ceiling, not an inefficiency to
optimize away.

**Design.** Two independent oracles, N-scoped by tractability:
- **Small N (≤20, where PennyLane is tractable)**: full PennyLane
  decomposition (`check_hermitian=False`, confirmed to correctly
  handle non-Hermitian operators too) AND a second, differently
  derived method — direct per-term projection — both required to
  match paulikit exactly, every term.
- **Large N (50 through 150+, where PennyLane cannot finish)**:
  PennyLane is not invoked. Every single term paulikit outputs (true
  100% coverage, not a sample — required per explicit direction) is
  individually checked via the same projection method.

**The projection method**: for a Pauli label with symplectic bitmasks
`(x_mask, z_mask)`, its coefficient is `Tr(H @ P_label^dagger) / dim`,
computed without ever materializing the full Pauli matrix, by
exploiting that `H[r,c]` contributes only where `r XOR c == x_mask`
with a closed-form sign/phase from `popcount`. This is mathematically
independent of paulikit's own FWHT algorithm — a direct definitional
check, not a re-derivation of the same shortcut. Fully vectorized in
NumPy (sort-then-`searchsorted`-then-grouped-broadcast, no
Python-level loop over labels or nonzeros) after 3 slower iterations
were tried and rejected (naive per-label matrix rebuild, per-label
Python loop, dict-bucketed loop — the last one looked fast at N=50 but
did not scale to N=80).

**Real measured results** (all exact to float64 noise floor, every
term individually verified — see
`verification/FINDINGS.md` and `verification/results/*.json` for full
provenance):

| N | terms | wall time | max abs error |
|---|---|---|---|
| 20 (Hermitian + non-Hermitian, dual-oracle w/ PennyLane) | 24,448 / 49,024 | 0.05–0.10s | 1.39e-17 |
| 50 | 1,261,568 | 5.5s | 3.47e-18 |
| 80 | 6,473,728 | 30.3s | 1.04e-17 |
| 100 | 20,299,776 | 124.8s | 2.60e-18 |
| **150** | **91,652,096** | **369.7s** | **8.67e-19** |

**N=150 required a fix, not just patience.** The first 3 attempts
OOM-killed the machine (once observed live via swap usage). Root
cause: calling the dict-returning `fwht_pauli_terms` API, which
Phase 9/10's own findings already document as unable to complete at
N=150 *regardless of available RAM* (it re-fuses all ~91.65M terms
into one dict before returning). Fixed by switching to the
already-built streaming `fwht_pauli_terms_iter` API and verifying each
chunk as it arrives, never holding the combined term set — exactly
Phase 10's own streaming design, just not the API reached for first.
Diagnosed via `ulimit -v <cap>` for a clean `MemoryError` instead of
further swap-thrashing; see `verification/FINDINGS.md` §6 for the full
debugging narrative.

**Deliverables** (committed, not scratch — commit `05f1835`):
- `verification/exhaustive_projection.py` — the method itself,
  reusable (`project_labels`, `verify_terms`, `verify_terms_streaming`).
- `verification/run_verification.py` — single reproducible CLI entry
  point; every result JSON records its exact command, git commit, and
  dependency versions.
- `verification/results/*.json` — committed result files.
- `verification/FINDINGS.md` / `verification/README.md` — full
  writeup and reproduction instructions.
- `tests/test_exhaustive_verification.py` — real pytest integration
  (fast N≤20 cases in CI, N=50/80/100/150 marked `slow`).

**Status: closed.** No open items remain in this plan; any future
verification at a new N or Hamiltonian family should reuse
`run_verification.py` rather than re-deriving the method.

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

Phase 2 profiling (`tools/paulikit/profiling/phase2/README.md`, task #16,
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
in `profiling/phase3b/README.md` and `profiling/phase3b/explore/`.

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

**Results** (full detail in `profiling/phase3b/README.md` Section 5):

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


### Phase 6 — Sparse output for `fwht_pauli_coefficients` (scoped 2026-08-25, implemented and shipped 2026-08-26)

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
explicitly deferred it. `profiling/phase3b/explore/08_v5_active_rows_only.py`'s
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
`profiling/phase3b/README.md`) - it is not acceptable to "fix" cache locality by
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
`profiling/phase3b/explore/*.py` (6 files) call it only as a correctness
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

**Update 2026-08-26: `sparse` implemented and shipped; N=150 dug into
further, two more real fixes made, and the actual remaining ceiling
identified.**

`fwht_pauli_coefficients(operator, sparse=False)` and
`fwht_pauli_terms` are implemented as designed above (commit history
has the details); all pre-existing tests pass unmodified, and
bit-for-bit correctness against the old dense path was re-verified
after every change described below, down to `chunk_size=1`.

Attempting the actual N=150 measurement this fix was meant to unblock
surfaced that the dense-output scatter wasn't the only large
allocation in the hot path:

1. `_walsh_hadamard_transform_rows` unconditionally copied its input
   array before transforming (`transformed = array.copy()`), and its
   per-stage butterfly loop copied both halves (`left`, `right`)
   before combining them. Neither copy was needed by
   `fwht_pauli_coefficients`'s only caller of this function, since it
   never reads `gathered_active` again afterward. Fixed via a new
   `overwrite_input` flag (default `False`, safe for any other
   caller) and view-based (rather than copy-based) `left`/`right`
   handling. Removed roughly 5.5 GiB of transient allocation at
   N=150.
2. Even with (1), one `(n_active, dim)` array was still live in memory
   at once - fine at N≤100, tight at N=150. Following the MIT 6.172
   lecture 1 tiling technique (loop-order/tiling slides, as suggested
   by the user, applied here for memory-footprint reduction rather
   than cache reuse - each row transforms independently with no
   cross-row reduction to preserve), added an optional `chunk_size`
   parameter to both `fwht_pauli_coefficients` and `fwht_pauli_terms`:
   active rows are processed in bounded row-blocks, bounding peak
   memory to roughly `chunk_size * dim` complex entries instead of
   `n_active * dim`. Exposed via `--chunk-size` on the `decompose` and
   `benchmark` CLI subcommands. Verified bit-for-bit identical to the
   unchunked path at multiple chunk sizes including the degenerate
   `chunk_size=1`.

**The actual N=150 ceiling, found while testing (1) and (2) together:**
none of the above was the dominant cost. The coupled-oscillator
Hamiltonian (`paulikit.hamiltonian.build_hamiltonian`) is built and
stored as a dense `numpy.ndarray` despite being extremely sparse in
practice - at N=150 it is `11475x11475` with only 45,000 nonzero
entries (0.034% density), yet costs 1.05 GiB dense. Padded to
`16384x16384` and upcast to `complex128` inside
`fwht_pauli_coefficients` (`operator = np.asarray(operator,
dtype=complex)`), this becomes a ~4 GiB allocation that happens
*before* any of the Pauli-decomposition algorithm runs - dwarfing
everything fixed in this update and making chunking's benefit moot at
N=150 until this is addressed.

**Not yet done, scoped as follow-up work (not started):**

- **Sparse-Hamiltonian redesign.** `build_hamiltonian`/
  `pad_to_power_of_two` would need to produce (or
  `fwht_pauli_coefficients` would need to directly accept) a
  `scipy.sparse` matrix, so the operator never gets densified in the
  first place. This is a real architectural change - it touches the
  public shape of `paulikit.hamiltonian`'s API and
  `fwht_pauli_coefficients`'s input contract, not just an internal
  optimization - and needs its own scoping/design pass before
  implementation, not a quick patch. This is the actual fix for the
  N=150 OOM; everything landed in this update is real and worth
  keeping, but insufficient on its own to unblock N=150.
- **Auto-tuned `chunk_size`.** Currently a manual, undocumented-default
  knob (`None` = no chunking). Picking a good automatic default (e.g.
  targeting a fixed memory budget per chunk) is deliberately deferred
  until the sparse-Hamiltonian redesign above removes the 4 GiB
  operator-densification wall - tuning `chunk_size` against today's
  pipeline would be tuning around a bottleneck that's about to be
  removed, producing a heuristic calibrated against the wrong
  problem.
- N=150 dense-vs-sparse cache-locality write-up (N=25/50/100 findings
  already measured, not yet written up as a formal findings doc - see
  the "Follow-up items" list below, unchanged from before this
  update).


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


### Phase 8 — Sparse-Hamiltonian construction and input (scoped 2026-08-26, implemented and shipped 2026-08-26)

**Motivation.** Phase 6's two follow-on memory-footprint fixes
(overwrite-in-place WHT, `chunk_size` row-tiling - see Phase 6's
"Update 2026-08-26") surfaced the actual N=150 memory ceiling, and it
is upstream of everything Phase 6 or its follow-ons touch:
`paulikit.hamiltonian.build_hamiltonian` constructs and returns a
dense `numpy.ndarray`, even though the coupled-oscillator Hamiltonian
is extremely sparse in practice - at N=150, `11475x11475` with only
45,000 nonzero entries (0.034% density), yet costs 1.05 GiB dense.
`pad_to_power_of_two` preserves that density (padding a dense array
with more dense zeros), and `fwht_pauli_coefficients` then upcasts the
padded `16384x16384` array to `complex128`
(`operator = np.asarray(operator, dtype=complex)`) - a ~4 GiB
allocation that happens before any Pauli-decomposition algorithm code
runs, dwarfing every fix made in Phase 6 and its follow-ons. Chunking
`fwht_pauli_coefficients`'s internal computation (Phase 6's
`chunk_size`) cannot help with this, because the bottleneck is the
*input* to that function, not anything inside it.

**Why this is a real architectural change, not a quick patch.** Unlike
Phase 6 (which changed only `fwht_pauli_coefficients`'s internals and
added an optional return-shape/knob), fixing this means the operator
must stay sparse from construction (`build_hamiltonian`) through
padding (`pad_to_power_of_two`) to the point where
`fwht_pauli_coefficients` reads it - three public functions across two
modules, all of which currently have a `numpy.ndarray`-only contract.
This is not internal-only refactoring; it changes what callers
construct, pass around, and receive.

**Design questions - resolved 2026-08-26 (questions 1-2), 3-5 still
open:**

1. **RESOLVED: `scipy.sparse` stays optional, via a new `sparse`
   install extra, not a hard dependency.** `pip install paulikit`
   keeps today's behavior (dense-only, no `scipy` pulled in); `pip
   install paulikit[sparse]` (a new `[project.optional-dependencies]`
   entry in `pyproject.toml`, alongside the existing `test`/`dev`/
   `docs` extras) installs `scipy` for callers who want
   `sparse=True`. Mirrors the existing `_native` extension's
   `try: import ... / except ImportError` pattern in `fwht.py` (lines
   77-80): `paulikit.hamiltonian` gets its own `_HAVE_SCIPY` flag set
   the same way. Unlike the native extension's silent-fallback-to-
   pure-Python behavior, calling `sparse=True` without `scipy`
   installed must raise a clear, actionable error (e.g.
   `ImportError("scipy is required for sparse=True - install with "
   "'pip install paulikit[sparse]'")`) rather than silently falling
   back to dense - the caller explicitly asked for sparse and
   silently ignoring that request would be a correctness surprise
   Phase 6's own `sparse=True/False` precedent doesn't have (Phase
   6's default is `sparse=False`, so nothing there is a request that
   can go silently unfulfilled). The `sparse` extra itself, and the
   `_HAVE_SCIPY`/error-message pattern, need to be documented in
   README's Installation section alongside the existing `test`/`dev`/
   `docs` extras, not left implicit.
2. **RESOLVED: `sparse: bool` parameter, mirroring Phase 6's own
   `fwht_pauli_coefficients(..., sparse=...)` pattern.**
   `build_hamiltonian(..., sparse=False)` and
   `pad_to_power_of_two(..., sparse=False)` - default unchanged
   (`False`, current dense behavior), optional and reversible, no
   breaking change for any existing caller (`tests/test_fwht.py`,
   `tests/test_benchmark_reference.py`, `profiling/`'s driver scripts,
   `cli.py` all keep working with zero changes). Consistent with this
   project's established convention since Phase 6's own API-shape
   decision (driven by the user's explicit preference for optional,
   reversible deployment over one-way breaking changes). Rejected: (a)
   always returning `scipy.sparse` - breaking, inconsistent with
   Phase 6's precedent; (c) a separate `build_hamiltonian_sparse`
   function - a third construction function to keep correct/tested
   alongside the existing one, more maintenance surface than a single
   parameter for no clear benefit here.
3. **RESOLVED: reconstruct at the padded shape via COO coordinates,
   not `csr_matrix.resize`.** Verified directly (2026-08-26): both
   approaches avoid densifying and give identical results
   (`sp.coo_matrix((coo.data, (coo.row, coo.col)), shape=padded_shape)`
   vs. `matrix.resize(padded_shape)`), but `.resize()` **mutates its
   input in place** (confirmed: resizing one reference changes what a
   second reference to the same object sees) - a real behavioral
   mismatch with the existing dense `pad_to_power_of_two`, which
   returns a new array and never touches its input. The COO
   reconstruction is naturally non-mutating (builds a new object) and
   preserves that contract with no extra `.copy()` needed. Both scale
   with `nnz`, not `dim**2` - confirmed on a synthetic sparse matrix
   sized like N=150's padded operator.
4. **RESOLVED, with a real compatibility gap found, not just a
   mechanical change.** Step 1's `p_nz, q_nz = np.nonzero(operator)`
   already works correctly and cheaply on `scipy.sparse` input with
   zero changes - verified directly: `np.nonzero()` dispatches to
   scipy's own sparse-native `.nonzero()`, confirmed via timing on a
   synthetic N=150-sized sparse matrix (0.0004s vs. what would be a
   4.3 GiB densify). Two lines do need real changes: (a) the
   `operator = np.asarray(operator, dtype=complex)` cast (current line
   239, the actual ~4 GiB N=150 allocation) must become
   `operator.astype(complex)` for sparse input, which scipy supports
   without densifying (verified directly) - this branch needs an
   `if sparse:`/`isinstance` check, not a single line that works for
   both; (b) **a genuine compatibility gap**: sparse fancy-indexing
   (`operator[p_nz, q_nz]`, used to build `gathered_active` at what is
   currently line 241) returns a `numpy.matrix` of shape `(1, nnz)`,
   not a flat `ndarray` of shape `(nnz,)` like dense indexing does -
   confirmed directly, this is not a hypothetical edge case. Assigning
   that shape-mismatched result into `gathered_active[inverse, q_nz]
   = ...` needs an explicit `np.asarray(...).ravel()` (verified this
   produces the numerically correct result; without it, NumPy's
   broadcasting silently "succeeds" but was not checked for
   correctness beyond not raising - do not assume broadcasting alone
   is safe here without this fix and its own test).
5. **RESOLVED: neither existing test file needs changes, confirmed by
   reading both directly (2026-08-26), not assumed.**
   `tests/test_fwht.py`'s two dense-contract tests
   (`test_fwht_pauli_coefficients_handles_non_hermitian_matrices`,
   `test_fwht_pauli_coefficients_reconstructs_non_hermitian_matrix`)
   construct small dense matrices directly and call
   `pad_to_power_of_two` with no `sparse` argument - the new
   parameter's `False` default means these keep passing completely
   unmodified, matching Phase 6's own precedent.
   `tests/test_benchmark_reference.py`'s `_build_padded` helper
   likewise calls `pad_to_power_of_two` with no `sparse` argument
   (stays dense, unaffected), and separately builds
   `padded_sparse = sp.csr_matrix(padded)` purely to feed PennyLane
   its preferred sparse input format - this has nothing to do with
   paulikit's own internal sparse-input question and was a false
   pattern-match in the original framing of this question; it is not
   "made redundant" by Phase 8 and needs no changes. **New tests are
   still needed**, not yet written: correctness parity between
   `sparse=True` and `sparse=False` for `build_hamiltonian`/
   `pad_to_power_of_two`/`fwht_pauli_coefficients` (mirroring Phase
   6's own bit-for-bit verification discipline), and a test for the
   `ImportError` path when `sparse=True` is requested without `scipy`
   installed (question 1) - these are implementation-time work, not
   further design questions.

**Explicitly out of scope for this phase's initial implementation:**
exploiting sparsity *within* `fwht_pauli_coefficients`'s WHT step
itself (i.e. a sparse-matrix-aware Walsh-Hadamard Transform algorithm,
rather than a dense butterfly applied to a small number of gathered
rows) - Phase 3b already found the *active-row* count (not the raw
nonzero-entry count) is the real constraint on how sparse the WHT
step's own computation can be (see `profiling/phase3b/README.md`
Section 1: 47-86% of rows are active even for genuinely sparse
Hamiltonians), so a sparse-*input* fix (this phase) and a
sparse-*algorithm* fix (a separate, larger research question, not
scoped here) are different problems with different payoffs. This
phase targets the measured N=150 ceiling (input densification, ~4
GiB), not a further algorithmic redesign of the WHT step.

**Status: implemented and correctness-verified 2026-08-27.**
`build_hamiltonian(..., sparse=True)` and `pad_to_power_of_two(...,
sparse=True)` added to `hamiltonian.py`, gated behind a new
`try: import scipy.sparse` (hard `ImportError` with an actionable
message when `sparse=True` is requested without the `sparse` extra
installed, per question 1 - never a silent dense fallback).
`fwht_pauli_coefficients` now accepts a `scipy.sparse` operator
directly (converted to CSR internally for fancy-indexing support,
since COO - the format `pad_to_power_of_two` returns - is not
fancy-indexable), with the `.astype(complex)` and
`np.asarray(...).ravel()` fixes from question 4 applied at both
fancy-indexing gather sites (the unchunked path and the chunked
path). 15 new tests added
(`tests/test_sparse_hamiltonian.py`): sparse/dense parity for all
three functions (bit-for-bit, `max_error == 0.0`, not just
`pytest.approx`), the non-mutation contract for
`pad_to_power_of_two(..., sparse=True)`, sparse-input combined with
the existing `sparse=True` *output* mode and with `chunk_size`
chunking, and both `ImportError` paths (via `monkeypatch` on the
module's `_sp` name). All 40 tests pass (25 pre-existing + 15 new).

A real N=150 attempt (`build_hamiltonian` → `pad_to_power_of_two` →
`fwht_pauli_terms`, all with `sparse=True`, `chunk_size=256`, under a
6 GB `ulimit -v` cap) confirms this phase's targeted ceiling is
cleared: `build_hamiltonian(sparse=True)` takes 0.04s (was the ~4 GiB
densify-and-upcast crash point), `pad_to_power_of_two(sparse=True)`
is instant, and the WHT transform itself completes. The run still
crashed, but one step later, inside `fwht_pauli_terms`'s own nonzero
extraction (`active_coefficients[row_idx, z_nonzero]`) - a **new,
distinct** bottleneck unrelated to Hamiltonian sparsity, now scoped
as Phase 9. This phase's own scope (Hamiltonian construction/
storage/input densification) is fully closed.

### Phase 9 — `fwht_pauli_coefficients`/`fwht_pauli_terms` output-accumulator memory (scoped 2026-08-27, not yet designed)

**Motivation.** Phase 8 cleared N=150's original ceiling (dense
Hamiltonian construction, ~4 GiB), confirmed by a real run under a
6 GB `ulimit -v` cap. That run got much further than any previous
attempt - Hamiltonian construction, padding, and the WHT transform
itself (with `chunk_size=256` bounding the *transient* per-chunk
working set) all completed - but then crashed one step later, inside
`fwht_pauli_terms`'s nonzero-extraction line:
```
active_coefficients[row_idx, z_nonzero]
```
with `numpy._core._exceptions._ArrayMemoryError: Unable to allocate
1.37 GiB for an array with shape (91652096,)`.

**Root cause, confirmed by direct calculation, not guesswork.** At
N=150, `n_active` (the number of x-values with at least one nonzero
gathered entry) is 45,000 and `dim` is 16,384. `chunk_size` only
bounds the *transient* per-chunk array
(`gathered_chunk`/`transformed_chunk`, shape `(chunk_size, dim)`)
inside `fwht_pauli_coefficients`'s chunked loop - but the loop still
writes every chunk's result into one `active_coefficients =
np.empty((n_active, dim), dtype=complex)` accumulator allocated
up front (see `fwht.py`, the line right before the chunking loop).
At N=150 that's `45000 * 16384 * 16 bytes ≈ 11.8 GiB` - already the
real ceiling, independent of `chunk_size`'s value, and independent
of Hamiltonian sparsity (this is downstream of the WHT step, not
upstream of it - Phase 8 is unrelated). `fwht_pauli_terms` then needs
a second full-size gather (`active_coefficients[row_idx, z_nonzero]`)
on top of that, which is what actually raised the exception this run
(91.6M of the 737M possible entries survive the `atol` threshold -
~12%, roughly consistent with Phase 3b's 47-86%-active-*rows* finding,
just applied to entries here instead of rows).

**Why this is a real, distinct problem from Phase 6/8.** Phase 6's
`chunk_size` was designed to bound *transient* working memory during
the WHT computation itself (per row-block). It was never designed to
bound the *output* - the full set of (x, z) coefficients above
`atol` is data the caller (eventually) needs, and at N=150 there is
a lot of it (~91.6M complex entries, if none are filtered further).
This is not a "just chunk it more" fix in the way Phase 6's was;
returning ~92M terms as a Python dict (`fwht_pauli_terms`'s own
return type) is itself questionable at that scale, both for peak
memory and for whether a 92M-entry `dict[str, float]` is a usable
result for any downstream caller.

**Design questions for next session (none yet answered):**
1. Is a 92M-term result at N=150 actually useful to any real caller,
   or does this indicate the *problem itself* (not just the
   implementation) needs bounding - e.g. a coefficient-magnitude cutoff
   tighter than the current default `atol=1e-10`, or a top-k/truncated
   result mode? This is a physics/use-case question as much as an
   engineering one and should not be answered by engineering guesswork
   alone.
2. If the full term set genuinely is needed, should
   `fwht_pauli_terms`'s extraction become a generator/streaming API
   (yielding `(label, coefficient)` pairs, or writing directly to a
   file/sparse-on-disk format) rather than building one dict in
   memory - a bigger API-shape change than anything in Phase 6/8, so
   needs explicit user sign-off before implementation, not silent
   scoping.
3. Should `fwht_pauli_coefficients`'s chunked path write its per-chunk
   output directly into a **sparse** accumulator (e.g. incrementally
   building COO triples per chunk, only for entries above `atol`)
   instead of a dense `(n_active, dim)` array - moving the
   thresholding *into* the chunk loop rather than after it, so the
   ~92M *entries* never coexist as a single dense block? This looks
   like the most promising direction (mirrors Phase 6/8's own
   dense-to-sparse pattern) but needs the same measurement discipline
   before being trusted - unverified as of this scoping.
4. Does `chunk_size`'s existing docstring/contract need updating once
   Phase 9 lands, to clarify what it does and does not bound (it never
   claimed to bound the output accumulator, but the N=150 crash shows
   that distinction matters in practice and should be stated
   explicitly)?

**Status: implemented and verified 2026-08-27.** Resolved with the
user directly (not engineering guesswork): tackled as a pure
space-complexity/scalability engineering problem, prioritized in this
explicit order - (1) time complexity (cache-locality, parallelism,
vectorization), (2) space complexity/scalability (reaching the
highest possible problem sizes), (3) resumability (only if it doesn't
meaningfully cost (1) or (2)). This resolved design question 3 above
in favor of the sparse-accumulator direction, and added checkpointing
as a genuinely cheap addition (verified: ~59M butterfly ops per chunk
vs. one small file append - negligible).

Implementation: `fwht_pauli_coefficients`'s chunked path now
thresholds each chunk's output against `atol` immediately (moved atol
into `fwht_pauli_coefficients` per design question 4's resolution -
only when `chunk_size` is set; the dense-block `chunk_size=None`
modes are unaffected) and accumulates only surviving `(x, z,
coefficient)` triples into a new `_GrowableArray` (amortized-doubling
growable array - O(total survivors) amortized append cost, not
O(chunks) reallocations or O(n_terms) Python-object overhead) instead
of one dense `(n_active, dim)` block. Returns a COO-style triple
`(x_out, z_out, coeff_out)` when chunked (a new, distinct return shape
from the unchunked `sparse=True` mode's dense-block form - documented
in the docstring, not silently overloaded). An optional
`checkpoint_path` parameter appends each completed chunk's triples to
a newline-delimited JSON file plus a small progress marker, enabling
resume after a crash/interruption. `fwht_pauli_terms` gained a
matching `checkpoint_path` parameter and now calls the chunked COO
path directly (skipping its own redundant re-threshold) when
`chunk_size` is set. 15 new tests in `tests/test_chunked_accumulator.py`
- dense/chunked parity across multiple chunk sizes and fixtures,
amortized-growth correctness at odd chunk-count boundaries, a full
checkpoint-interrupt-and-resume round trip, and a hand-truncated
progress-file crash simulation. `tests/test_sparse_hamiltonian.py`'s
chunking test updated to match the new atol-aware COO return shape.
Full suite: 55/55 passing.

**Real N=150 result, confirmed via direct measurement (see
`profiling/phase9/phase9_findings.md` for the full run log and
`profiling/phase9/n150_chunked_accumulator_test.py` for the script
used):** the fix works exactly as designed - the artificial `n_active
* dim` (11.8 GiB) ceiling is gone, confirmed by a 10 GB `ulimit -v`
run completing the *entire* chunked accumulation of ~134M survivor
terms, something the pre-Phase-9 code could never do at any memory
size. What remains is the genuine, non-artificial result size: ~134M
terms at `atol=1e-10` (~4.3 GiB for the raw triple alone), plus a
**third**, still-separate O(n_terms) allocation inside
`_pauli_label_batch` (label-string generation) that this phase's fix
does not address, since it runs after `fwht_pauli_coefficients`
returns. A 13.5 GB attempt was killed by the safety-monitoring loop
itself after real system memory (not the `ulimit` cap) dropped to
630 MiB - decisive confirmation that materializing the full
`dict[str, float]` result genuinely does not fit in this machine's
~11 GiB available RAM, not an artifact of an overly strict test cap.

**Conclusion, confirmed with the user 2026-08-27:** no fixed-RAM fix
survives arbitrary N, since term count grows with N and any given
machine's RAM is finite. `fwht_pauli_terms`'s fully-materialized
`dict` return contract has an inherent ceiling that raising memory
limits cannot remove in general - only a streaming/generator output
(never holding the full result at once) removes the ceiling itself.
The user explicitly decided this is **mandatory follow-up work, not
optional** - scoped as Phase 10 below.


### Phase 10 — Streaming/generator output for `fwht_pauli_terms` (scoped 2026-08-27, implemented and verified 2026-08-27)

**Motivation.** Phase 9 proved the chunked/checkpointed COO
accumulator itself is correct and scales past the old artificial
ceiling - but `fwht_pauli_terms`'s contract (materialize every
surviving term as Python label strings, then build one `dict`) still
requires O(n_terms) memory for the *final* answer, and N=150 already
produces ~134M terms - more than this machine's available RAM can
hold once label strings and dict overhead are added on top of the
raw triple (see Phase 9's findings). Mandatory follow-up per the
user's explicit direction ("we MUST do the streaming"), not optional
hardening.

**Direction (not yet designed in detail).** The chunked accumulator
from Phase 9 is the natural foundation: each chunk's surviving
triples already exist independently and briefly before being folded
into the growable arrays. A streaming mode should instead convert
each chunk's triples to labels and yield `(label, coefficient)` pairs
immediately - or write them incrementally to the existing
`checkpoint_path` mechanism - so no more than one chunk's worth of
labels is ever held in memory, and the final result is never required
to exist as a single in-memory `dict`.

**Design reframing (2026-08-27, direct discussion with the user before
implementation).** The user explicitly redirected the framing: the
MIT 6.172 tiling example (studied at the user's request earlier this
project) was meant to teach the *general principle* of
divide-and-conquer, not to be copy-pasted as a chunking recipe -
"designing/implementing the right strategy for divide-and-conquer is
on us." Reasoned through properly: each chunk of active rows is
already a fully independent sub-problem - there is no cross-chunk
combination step anywhere in the underlying math (unlike tiled matrix
multiply, which requires *summing* blocks - a real combination step).
`fwht_pauli_terms`'s actual defect was never really about chunk
size; it was re-fusing every chunk's result into one combined `dict`
before the caller ever saw it - an artificial recombination the math
does not require, forced only by the caller's contract. The correct
divide-and-conquer strategy is therefore to keep each tile a tile all
the way out to the caller, not to bolt a generic generator/chunking
mechanism onto the existing accumulate-then-return shape. See
[[feedback_divide_and_conquer_strategy]] (memory) for this principle
recorded for future similar problems, and
[[feedback_perf_priority_order]] for the refined priority order
established alongside it: **(1) performance/time complexity
(cache-locality, vectorization, parallelism), (2) reliability,
(3) scalability/space complexity, (4) resumability only if
essentially free.**

**Resolved design questions:**
1. **New sibling function**, `fwht_pauli_terms_iter` - not a `stream:
   bool` flag on `fwht_pauli_terms` (avoids an awkward
   conditional-return-type API).
2. **`chunk_size` is required** (no default, no whole-array streaming
   mode) - streaming without chunking is not a meaningful combination;
   a caller with no chunking need should call `fwht_pauli_terms`
   directly.
3. Primary target is an **in-process generator** yielding one `dict`
   per chunk; the existing `checkpoint_path` file (newline-delimited
   JSON) remains a secondary, opt-in resumability mechanism (passed
   through to the same `fwht_pauli_coefficients`/`_iter_chunked_coefficients`
   internals as the non-streaming chunked path), not the primary
   streamed product.
4. The Hermiticity `ValueError` **does** raise mid-stream (after zero
   or more prior chunks have already been yielded) rather than
   deferring to an all-or-nothing check - documented explicitly as a
   real behavior difference from `fwht_pauli_terms`, not silently
   changed.

**Implementation.** `fwht_pauli_terms_iter(operator, chunk_size, atol,
assume_hermitian, checkpoint_path, parallel_labels)` added to
`fwht.py`. Internally, the chunked path's per-chunk body (previously
inline in `fwht_pauli_coefficients`'s `chunk_size` branch) was
factored into a shared generator, `_iter_chunked_coefficients`,
consumed by both `fwht_pauli_coefficients` (which still accumulates
every tile into one COO triple, unchanged behavior) and
`fwht_pauli_terms_iter` (which converts each tile to labels and
yields it immediately, never accumulating). Shared
operator-validation/setup logic (shape checks, sparse-input CSR
conversion, the XOR-index gather) was similarly factored into
`_prepare_operator_for_fwht`, used by both `fwht_pauli_coefficients`
and `fwht_pauli_terms_iter` (previously duplicated). Along the way,
`_pauli_label_batch` gained a `parallel: bool` parameter, dispatching
to the existing (previously unused-in-production) oneTBB
`pauli_label_batch_parallel` native kernel - re-measured first at
N=150-representative scale (see
`profiling/phase10/tbb_labeling_n150_findings.md`): a real 1.1-1.4x
wall-clock win, at a modest, explicitly-recorded cache-locality cost
(cache-miss/stall percentages each rise a few points) - adopted per
direct user decision, with cache-locality behavior at real N=150+
scale to be **re-investigated iteratively**, not treated as closed.
CLI (`cli.py`): `decompose` gained `--stream` (requires `--chunk-size`),
`--parallel-labels`, and `--checkpoint-path` (also wired into the
existing non-streaming `fwht_pauli_terms`/`fwht_pauli_coefficients`
call sites for consistency). 16 new tests in `tests/test_streaming.py`
- combined-output parity against `fwht_pauli_terms` across chunk
sizes/fixtures, disjoint-labels-across-chunks, empty-chunk handling,
parallel-vs-serial label parity, a full checkpoint-interrupt-and-resume
round trip through the generator interface, and both the mid-stream
and immediate-validation `ValueError` paths. Full suite: 71/71
passing. Sphinx docs build clean.

**Real N=150 result, confirmed via direct measurement (see
`profiling/phase10/phase10_streaming_findings.md` for the full run log
and `profiling/phase10/n150_streaming_test.py` for the script used):**
N=150 now **completes fully** end to end - 91,652,096 terms streamed
successfully under both a 4 GB and a 2 GB `ulimit -v` cap (~100s each,
identical term counts both runs), compared to Phase 9's finding that
the non-streaming accumulator needed at least 10 GB just to finish
accumulation, and never completed even at 13.5 GB once label
generation and dict construction were included. This is the
qualitative result the divide-and-conquer reframing predicted: peak
memory is now decoupled from total term count, not merely raised to a
higher fixed ceiling - no `ulimit` value fixed N=150 before this
phase; only decomposing the problem differently did.

**Status: implemented and verified 2026-08-27.** N=150 is now a
solved, repeatable case on this machine.

**Full-pipeline cache-locality re-investigation (2026-08-27, per
direct user request - see
`profiling/phase10/full_pipeline_n150_findings.md`).** The
`--parallel-labels` recommendation above was based on
`pauli_label_batch_parallel` measured *in isolation* against a
synthetic 40M-pair benchmark. Re-measured embedded in the real
pipeline at N=150: a per-stage wall-clock breakdown
(gather/WHT/threshold/label/dict-construction) found **dict
construction - not labeling, not the WHT butterfly - dominates at
~60% of total pipeline time** (vs. ~21% WHT, ~7% labeling), a cost no
prior measurement had isolated. At that proportion, labeling's
isolated 1.1-1.4x win becomes noise-level at the whole-pipeline scale:
whole-pipeline `perf stat` shows `--parallel-labels` **slightly
slower** end-to-end (97.97s vs. 96.96s) with cache-miss/LLC-miss/stall
percentages all flat to within a point - the isolated kernel's
cache-locality cost does not reproduce as a measurable system-level
effect either, simply because labeling is too small a share of total
time for its cost or benefit to matter much at this scale.
**Correction to Phase 10's original recommendation:**
`--parallel-labels` should not be treated as a default optimization
based on the isolated measurement; it delivers no measurable
full-pipeline benefit at N=150's real proportions (still available as
an opt-in flag, since it is not harmful, just not helpful here).
**New finding, not yet scoped as a phase:** dict construction is now
the highest-leverage remaining optimization target in this pipeline -
this directly parallels Phase 3's original N=50 finding
(`pauli_label`'s per-term Python loop dominating the vectorized FWHT
core), with the specific bottleneck having moved from label-string
formatting to dict construction now that labels are TBB/Cython-fast,
but the general shape of the problem (pure-Python per-term
bookkeeping dominating a vectorized numeric core) unchanged.

### Phase 11 — `dict_build` optimization (scoped 2026-08-27, not yet implemented)

**Motivation.** `full_pipeline_n150_findings.md` found `dict_build` -
the per-chunk Python loop in `fwht_pauli_terms`/`fwht_pauli_terms_iter`
that converts each chunk's `(label, coefficient)` pairs into a
`dict` - accounts for ~60% of total pipeline time at N=150, dwarfing
the WHT butterfly (~21%) and label generation (~7%).
`n_scaling_streaming_findings.md`'s N=25/50/100/150 timing table
confirms the resulting behavior: total time scales essentially
**linearly** with term count (both grow ~4.5x from N=100 to N=150) -
exactly what's expected when a per-term Python cost dominates, and
the signature this phase should look for gone once fixed.

**Divide-and-conquer decomposition of `dict_build` itself.** Before
optimizing, the loop's body was broken into its actual sub-costs via
a standalone microbenchmark (1M and 10M synthetic terms, not yet
committed as a checked-in script - see "Not yet done" below) to see
which part dominates, rather than optimizing by assumption:

1. `chunk_coeff.tolist()` - converts the chunk's complex128 NumPy
   array to a list of Python `complex` objects. Present in every
   variant tested; not differentiating.
2. **The per-term Hermiticity check** (`abs(c.imag) > max(atol, 1e-6 *
   abs(c))`, evaluated inside the Python loop, once per term) - this
   is a fully vectorizable NumPy operation (`np.abs`, `np.maximum`,
   one array comparison) currently being paid for one Python-level
   `abs()`/`max()`/comparison at a time. **This is the single largest
   sub-cost measured**: removing it (replaced by one vectorized check
   before the loop) alone produced a 3.76x speedup at 1M terms and
   2.58x at 10M terms in isolation (both numbers report honestly, not
   averaged/oversold - the effect is real but its exact magnitude
   shrinks somewhat at larger scale, plausibly as `.tolist()`'s own
   cost becomes proportionally larger).
3. **The dict construction itself** (`real_terms[label] = ...` inside
   an explicit loop) - even with the check removed, an explicit
   per-item loop is measurably slower than `dict(zip(labels,
   values))`'s C-level constructor (a further ~30-40% within the
   remaining cost, per the isolated microbenchmark).

**Two genuinely different problems, not one:** the Hermiticity check
(only present in the `assume_hermitian=True` branch) is a validation
step masquerading as a per-item Python cost - it should never have
been inside the per-term loop. Dict construction is an inherent
requirement of the API's return type (a `dict[str, float]`), not
removable, but its *construction method* still has a faster and a
slower version. The `assume_hermitian=False` branch already has no
per-term check (just a dict comprehension), so it only needs the
construction-method fix, not the vectorized-check fix - these two
branches need different treatment, confirmed by reading the code, not
assumed symmetric.

**Design questions for implementation (not yet answered):**
1. Vectorizing the Hermiticity check loses the current error
   message's per-term specificity (naming the exact offending
   `label` and its `imag` value) if done naively (a single `.any()`
   check tells you *that* something failed, not *which* term). Fix:
   on violation, find the first (or all) violating index/indices via
   `np.nonzero` *only in the error path* (which is rare and doesn't
   need to be fast) and reconstruct the same error message from that
   index - preserves today's diagnostic quality without paying its
   cost on the common, non-error path. This must be verified to
   produce byte-identical error messages to today's before being
   trusted (per this project's own established discipline - Phase 6/8/9's
   own verification bar - not just "looks equivalent").
2. Should the vectorized check + `dict(zip(...))` construction replace
   the loop in both `fwht_pauli_terms` (dense, non-chunked path) and
   `fwht_pauli_terms_iter`'s per-chunk loop, or only the latter (since
   the streaming path is where the ~60%-of-time finding was measured,
   at N=150 scale)? Likely both, since the same sub-cost structure
   applies to `fwht_pauli_terms`'s single, larger loop too - not yet
   separately measured at non-streaming scale to confirm.
3. `atol`/Hermiticity-tolerance semantics: the current per-term check
   uses `max(atol, 1e-6 * abs(c))` - a per-term *relative* tolerance
   floor. The vectorized form needs `np.maximum(atol, 1e-6 *
   np.abs(coeffs.real))` (broadcasting), which should be mathematically
   identical - needs bit-for-bit verification against the scalar
   per-term form, not assumed equivalent from reading the formulas.
4. Does this change interact with `chunk_size=None`'s dense path in
   `fwht_pauli_terms` (which currently reuses the same per-term loop
   structure, per question 2)? If chunk_size is set,
   `fwht_pauli_terms_iter`'s chunk boundaries already provide a
   natural vectorization unit (one chunk's `chunk_coeff` array); the
   non-chunked path's `coefficient_values` array would need the same
   treatment applied to the whole array at once.

**Priority-order framing** (per
[[feedback_perf_priority_order]]): this is squarely a **performance**
fix (priority 1) - it does not touch reliability, scalability, or
resumability, all of which stay exactly as Phase 8/9/10 left them.
The one reliability-adjacent detail (design question 1, preserving
error-message specificity) must be resolved as part of the
implementation, not treated as acceptable collateral damage for
speed - matches the priority order's "performance first, but not at
the cost of reliability" framing already established for this
project.

**Update (2026-08-31): implemented.** `dict_build_microbenchmark.py`
was committed (per this project's "document exploration scripts"
convention) alongside the scoping doc. Before implementing in
production, re-running it surfaced a real formula bug in its own
Variant B: the tolerance floor used `np.abs(coeffs.real)` instead of
the production formula's `abs(c)` (full complex magnitude) - fixed in
the script (see `phase11_dict_build_scoping_findings.md`'s 2026-08-31
correction) and re-run; the 2.7-3.2x speedup conclusion was unaffected
(same operation count either way, only the formula's correctness was
in question).

Implementation: a new shared helper, `_build_real_terms` (in
`fwht.py`, used by both `fwht_pauli_terms` and
`fwht_pauli_terms_iter`'s `assume_hermitian=True` branch), replaces
the per-term Python loop with one vectorized NumPy check
(`np.abs`/`np.maximum`/comparison, using `abs(c)` correctly) plus
`dict(zip(labels, coefficient_values.real.tolist()))`. Design question
2 is resolved as "both" (the same helper serves both call sites, since
both had the identical loop pattern). Design question 1 (error-message
specificity) is preserved: on violation, `np.nonzero` finds the first
offending index only on that rare path and reconstructs the exact same
message format as the original per-term loop - verified by a new test
asserting the raised message names the real violating label and its
exact imaginary part (`test_fwht_pauli_terms_hermiticity_error_names_a_real_violating_term`).

Design question 3 (bit-for-bit tolerance-floor equivalence) turned out
to be **not testable behaviorally**: analysis while writing its
regression test showed `abs(c)` and `abs(c.real)` agree to leading
order for any term whose imaginary part sits near either floor (their
difference is `O(imag**2 / real)`, negligible exactly where it would
flip a pass/fail verdict) - no single term's Hermiticity outcome can
distinguish the two formulas, which is also why the mistaken
`abs(c.real)` substitution above was never caught by any prior
behavioral test. Covered instead by a source-inspection test
(`test_build_real_terms_tolerance_floor_uses_full_complex_magnitude`)
pinning the exact formula used, confirmed via deliberate mutation
testing to fail if the substitution is reintroduced.

All 81 tests pass (`make check`), including two new ones covering
error-message fidelity and the tolerance-floor formula. Both
`fwht_pauli_terms` and `fwht_pauli_terms_iter` use the shared helper;
the non-chunked (`chunk_size=None`) dense path in `fwht_pauli_terms`
is covered since it also calls `_build_real_terms`.

**Status: implemented 2026-08-31.**

**Update (2026-08-31, same day): re-measured against the real N=150
pipeline**, prompted directly by the user asking whether GPU
acceleration was now worth pursuing for the WHT butterfly stage. See
`profiling/phase11/n150_post_implementation_findings.md`. Real result:
dict construction 64.64s → 27.64s (2.34x, short of the synthetic
2.7-3.2x estimate - real coefficient-array `.tolist()`/layout costs
differ from the synthetic benchmark), total pipeline 107.48s → 70.77s
(34.2% faster). Because dict_build shrank, every other stage's
*relative* share rose - most notably the WHT butterfly, from 21.1% to
**31.9%** of total time, now close to dict_build's remaining 39.1%
rather than dwarfed by it (roughly 1.2:1 instead of 3:1).

**This changes the GPU-worth-it answer** from "clearly not" (21% of a
Python-loop-dominated pipeline) to "genuinely marginal" (31.9% of a
now-smaller total, competitive with the remaining dominant stage).

**Update (2026-08-31, same day, follow-up): checked dict_build for
further headroom first, per the recommendation above - none found.**
See `profiling/phase11/phase11b_remaining_dict_build_scoping.md`.
Breaking `_build_real_terms` down further shows the Phase 11
vectorized check is now under 2% of its own cost (a complete reversal
from before Phase 11); `dict(zip(labels, real_list))` is now **~90%**
of what remains, and ~56% of *that* is irreducible Python
string-hashing + dict-entry insertion with no NumPy-vectorizable
equivalent (unlike the arithmetic check Phase 11 fixed). Confirmed
`dict(zip(...))` is already at or near CPython's practical floor for
this pattern (a dict comprehension is statistically indistinguishable;
an explicit insert loop is ~1.9x worse). A further cut would require
either a breaking API change (returning something other than
`dict[str, float]`) or a C/Cython dict-construction kernel of unclear
realistic upside versus CPython's own optimized `dict` builtin -
neither scoped as a phase here, both meaningfully bigger than a
Phase-11-shaped fix.

**Net effect: the GPU-worth-it question now rests entirely on a real
port cost/benefit estimate** (host<->device transfer overhead for the
coefficient arrays, new toolchain/dependency cost, portability - this
project has no existing GPU dependency), not yet done. dict_build and
the WHT butterfly are both, independently, about as optimized as they
can get without either a GPU port or a breaking API-contract change.

### Phase 12 — auto-tuned `chunk_size` and streaming-vs-dense decision, with manual override always available (scoped 2026-08-27, not yet designed in detail)

**Motivation.** Prompted directly by the user questioning whether
`chunk_size=256` (used as an example default throughout Phase 9/10's
own docs) is actually well-chosen, rather than accepting it as tuned.
It is not: a controlled sweep across `chunk_size` at N=25/50/100
(`profiling/phase12/chunk_size_cache_locality_findings.md`) found
`chunk_size=256` is measurably suboptimal at **every** N tested -
never the fastest option in any comparison. `perf stat` at N=100
confirmed the mechanism: cache-miss ratio scales cleanly with
`chunk_size`'s working-set size (`chunk_size * dim * 16 bytes`)
relative to this machine's actual cache hierarchy (L2 1 MiB, L3 8 MiB
shared) - 7.3% miss ratio at `chunk_size=4` (512 KiB, fits in L2),
21.3% at `chunk_size=32` (4 MiB, fits in L3), 44.6% at `chunk_size=256`
(32 MiB, 4x over L3). `chunk_size` was designed (Phase 6/9) purely as
a memory-footprint bound; this reveals it is independently, and often
more impactfully at N≤100 scale, a **cache-locality lever** - the two
considerations point the same direction (smaller is generally better)
but for different reasons, and neither alone justified today's
example value.

**Two distinct auto-decisions in scope, per the user's explicit
framing** (not one general "make streaming smart" instruction):
1. **Auto-pick `chunk_size`** when the caller doesn't specify one -
   `fwht_pauli_terms_iter` currently requires `chunk_size` with no
   default at all; `fwht_pauli_coefficients`'s `chunk_size=None` means
   "don't chunk," not "pick one." A computed default removes the
   burden of the caller having to guess a value from first principles.
2. **Auto-pick streaming vs. dense** - a higher-level decision (should
   `fwht_pauli_terms` vs. `fwht_pauli_terms_iter` even be chosen) that
   the user separately confirmed is in scope, not deferred.

**Non-negotiable constraint, stated explicitly by the user**: both
auto-picks must remain **manually overridable** - "we should allow
the user to manually toggle them on/off and control their params as
well." Auto-tuning is the default behavior when a parameter is
unset, never a forced decision a caller cannot bypass. This directly
continues [[feedback_divide_and_conquer_strategy]]'s discipline
(reasoned strategy, not a black box) and
[[feedback_perf_priority_order]]'s ordering (this is a performance fix,
priority 1 - it must not remove existing manual control, which would
be a reliability/usability regression, priority 2).

**Design questions for implementation (not yet answered):**
1. What is the right target cache level for the `chunk_size` heuristic
   - L2 (1 MiB on this machine, per `chunk_size=4`'s strong result) or
   L3 (8 MiB, per `chunk_size=32`'s still-good but less extreme
   result)? Only 3 points were measured at one N; a proper sweep
   (more chunk_size values, more N, ideally more than one machine) is
   needed before trusting a specific target - not yet done.
2. How should `dim` (which the heuristic needs to compute a working-set
   size) interact with the *lower* bound found at N=25's
   `chunk_size=1` (57% slower than dense, since too little work per
   chunk lets fixed per-chunk overhead dominate)? The heuristic likely
   needs both a cache-driven upper bound on working-set size AND a
   floor on absolute chunk_size (or total chunk count) to avoid this
   failure mode - not yet formalized.
3. How to query the machine's actual cache sizes portably? `lscpu` was
   used for this scoping's own measurement but is Linux-specific and
   not meant to be shelled out to at runtime; `os.sysconf` has some
   cache-size constants on some platforms but portability across
   Linux/macOS/Windows is unverified - needs research before
   implementation, not assumed to just work.
4. What triggers the streaming-vs-dense decision (auto-pick #2)?
   Candidates: estimated term count (requires either a cheap
   pre-pass or a heuristic from `n_active`/sparsity), available
   system memory (queryable, e.g. via `psutil` - a new dependency
   question, mirroring Phase 8's `scipy` optionality precedent), or a
   fixed N/dim threshold (simpler, less adaptive to the actual
   machine's resources). Not yet decided.
5. API shape for manual override: keep today's explicit `chunk_size`
   parameter (a caller passing a value bypasses auto-tuning
   entirely, no new flag needed) plus a new top-level convenience
   function/parameter for auto-picking streaming-vs-dense (since that
   decision currently requires the caller to choose between two
   different function names, `fwht_pauli_terms` vs.
   `fwht_pauli_terms_iter`)? Not yet designed - this is a bigger API
   surface question than `chunk_size` alone.

**Status: scoped 2026-08-27, design finalized 2026-09-01, not yet
implemented.** The cache-locality mechanism behind auto-decision #1 is
confirmed by direct measurement; all five design questions above are
now resolved as follows.

**Design decisions (2026-09-01):**

1. **Cache-boundary target**: resolved by first discovering the
   original scoping's own `lscpu`-derived "L2 1 MiB" figure was wrong
   - `lscpu` reports the *aggregate* across all 4 cores' L2 instances;
   `/sys/devices/system/cpu/cpu0/cache/index2/size` (per-core, what a
   single-threaded chunk computation actually sees) reads 256 KiB on
   this machine, 4x smaller. Rather than trust either declared-topology
   source, the auto-tuner uses an **empirical pointer-chase latency
   probe** (the user's proposal, directly reusing the mechanism already
   built for `configure --probe-cache-latency`: scrambled-stride
   pointer chase over doubling buffer sizes, timed, boundary detected
   where the cycles-per-access ratio between consecutive sizes exceeds
   a threshold) instead of parsing any declared cache-size source at
   all. This sidesteps the aggregate-vs-per-core ambiguity entirely by
   measuring what the hardware actually does, not what it reports.
2. **Small-chunk floor**: not yet re-derived with a fresh sweep: the
   original N=25 `chunk_size=1` result (57% slower than dense) still
   establishes that a floor is needed; the exact formula for it is an
   implementation-time detail (e.g. a minimum absolute chunk_size,
   separate from the cache-driven upper bound) rather than a fresh
   open question.
3. **Portable cache-size query**: resolved as above - an empirical
   probe sidesteps portability questions about declared-size sources
   entirely (no `/sys`, `lscpu`, or `os.sysconf` cache constants
   needed for the primary path). Implementation: a new **standalone
   Cython extension** (`src/paulikit/_native/cache_probe.pyx` + a
   plain-C pointer-chase kernel), gated only on Cython >= 3.0 - not
   folded into the existing `pauli_label_native` extension, since that
   extension is additionally gated on oneTBB, which the probe has no
   relationship to; bundling them would wrongly make chunk_size
   auto-tuning unavailable whenever oneTBB specifically is missing.
   Falls back to declared-size detection (`/sys/devices/system/cpu/cpu*/cache/`
   parsing, POSIX/Linux-only) with a visible warning if the compiled
   probe extension isn't available, mirroring `pauli_label_native`'s
   existing optional-extension pattern. Probe result is cached
   **per-process** (module-level, lazy on first auto-tuned call) - not
   persisted to disk, since a persistent cache raises staleness
   questions the moment the process runs on different hardware (a real
   risk in HPC/container contexts, not a hypothetical - see below).
4. **Streaming-vs-dense trigger**: available-memory-based, computed
   with **zero third-party dependencies** per explicit instruction to
   prefer POSIX-compliant/self-contained probing over `psutil` or
   shelling out to non-portable commands. Two layers, both required
   for correctness on shared HPC nodes (explicitly a target
   environment, not just a workstation):
   - **Physical availability**: parse `/proc/meminfo`'s `MemAvailable`
     field when present (Linux; matches what `free -h`'s "available"
     column reports, correctly counting reclaimable page cache/buffers
     as usable - confirmed on this machine: `SC_AVPHYS_PAGES` alone
     read 1.5 GiB "free" while `MemAvailable` correctly read ~10 GiB
     "available"). Falls back to the POSIX-standard
     `os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGESIZE')` on
     non-Linux/no-`/proc` systems, accepting its more conservative
     "free-not-available" bias there.
   - **Cgroup limit**: on a shared HPC node, a job is frequently
     memory-capped below the node's physical RAM by the scheduler
     (Slurm/PBS) via a cgroup, which the above alone would not see -
     read `/sys/fs/cgroup/memory.max` (cgroup v2; the literal string
     `"max"` means unlimited) falling back to
     `/sys/fs/cgroup/memory/memory.limit_in_bytes` (cgroup v1; often a
     huge sentinel value like `9223372036854771712` when effectively
     unset, must be clamped against the physical-RAM figure rather
     than trusted directly). The auto-decision uses **whichever of the
     physical-availability and cgroup-limit figures is smaller**.
     Explicitly checked and confirmed correct to build now (not
     deferred) per direct instruction: "these should work for
     supercomputers with nodes."
5. **API shape**: rejected the "existing functions change return type
   based on runtime machine state" option on real reliability/modularity
   grounds - it would make `fwht_pauli_terms` nondeterministic across
   identical calls (same operator, different code path depending on
   available memory at call time) and silently change
   `assume_hermitian`'s already-documented validation-contract
   difference (all-or-nothing vs. partial-yield-then-error) out from
   under a caller who never asked for that. Instead: a **new top-level
   function**, `auto_decompose()`, whose name itself documents that its
   return type (`dict` or `Iterator[dict]`) is a runtime decision.
   `fwht_pauli_terms`/`fwht_pauli_terms_iter` keep their exact current,
   fixed contracts unchanged - `auto_decompose()` is purely additive.

**HPC/supercomputer-node applicability (explicit design target, not
an afterthought):** the cache probe's per-process-only caching means
it naturally re-measures on each node a job lands on, rather than
trusting a value that might be stale on a heterogeneous cluster
(different node generations/partitions can have genuinely different
cache hierarchies). The cgroup-aware memory budget (above) is the
correctness-critical piece for shared/multi-tenant HPC nodes
specifically - a physical-memory-only check would be actively unsafe
there, not just suboptimal, since it could greenlight the dense path
inside a job that is actually capped well below full-node RAM.
Checkpoint I/O (`checkpoint_path`, already opt-in, Phase 9) is noted
here only as a caveat, not addressed by this phase: shared HPC
filesystems (Lustre/GPFS) can behave very differently than local disk
under many small per-chunk appends - not re-scoped, since checkpointing
remains off by default.

**Implemented 2026-09-01.** Both auto-decisions are live:

- `src/paulikit/algorithms/autotune.py` - `recommended_chunk_size(dim)`
  (empirical-probe-first, declared-`/sys`-size fallback, fixed-32
  last-resort, per-process cached) and `available_memory_bytes()`
  (`/proc/meminfo` `MemAvailable`-first, POSIX `SC_AVPHYS_PAGES`
  fallback, minimum against any cgroup v2/v1 memory cap, per-process
  cached).
- `fwht.auto_decompose(operator, ...)` - the new top-level function per
  design decision 5 above. Estimates the dense path's worst-case
  footprint as `dim**2 * 16` bytes (a safe upper bound - see its own
  docstring for why this doesn't pre-scan the operator for the real
  `n_active`), compares against half the available memory budget (a
  deliberate safety margin, `_DENSE_MEMORY_SAFETY_FRACTION = 0.5`, not
  yet independently tuned), and calls `fwht_pauli_terms` (dense) or
  `fwht_pauli_terms_iter` with an auto-tuned `chunk_size` (streaming)
  accordingly. `fwht_pauli_terms`/`fwht_pauli_terms_iter` themselves
  are completely unchanged - `auto_decompose` is purely additive, per
  design decision 5.
- Tests: `tests/test_autotune.py` (14 tests - memory-budget formula,
  cgroup-vs-physical minimum logic, chunk_size formula and its floor/
  fallback, per-process caching, and `auto_decompose`'s dense/streaming
  correctness against `fwht_pauli_terms` as ground truth, both paths
  forced via monkeypatching rather than depending on this machine's
  actual memory state).

**Real N=100/N=150 re-measurement, 2026-09-01 (same day):** see
`profiling/phase12/n100_n150_autotuning_remeasurement_findings.md`.
The auto-tuned `chunk_size` is confirmed as a genuine win: **2.32x
faster at N=100** (16.17s -> 6.96s), **2.04x faster at N=150** (69.32s
-> 33.90s), against the old fixed `chunk_size=256`, with identical
term counts confirming correctness. `recommended_chunk_size` returned
the floor value (8) at both N on this machine, not yet exercising the
cache-boundary-targeting branch of the formula at meaningful scale.
`auto_decompose()`'s dense-path correctness was independently
confirmed bit-identical to `fwht_pauli_terms` at N=25/50.

**Two real bugs found by this re-measurement:**
1. **FIXED 2026-09-01 (same day, follow-up).**
   `auto_decompose()`'s dense-path memory estimate (`dim**2 * 16`
   bytes) underestimated real peak usage by roughly 3x, in the unsafe
   direction. At N=150 the estimate was 4.00 GiB; the real dense path
   failed to complete under a ~7.6 GiB cap, an ~11.4 GiB cap, **and,
   during the fix's own re-verification, an ~18 GiB cap** (clean
   Python exceptions each time, not a crash, but real system memory
   was observed dropping as low as 177 MiB / 593 MiB free during two
   separate runs before recovering). Root-caused by hand-tracing every
   array concurrently live in the dense code path
   (`gathered_active`/`transformed_active`, `xz_and`,
   `_popcount_array`'s temporaries, `phase`, `active_coefficients`,
   the `np.abs()` boolean-mask rescan, label/dict construction) - none
   individually huge, but their sum is 4-6x the naive single-array
   estimate. Confirmed with a real `resource.getrusage` peak-RSS sweep
   at N=50/75/100 (chunk_size=None, dense path): measured ratios of
   5.63x, 6.47x, 5.27x respectively (N=25's 18.20x excluded as a
   small-N process-baseline artifact) - consistently clustering around
   6x. **Fix**: `estimated_dense_bytes` now multiplies the naive
   `dim**2*16` figure by a new `_DENSE_MEMORY_MULTIPLIER = 6.0`
   constant, AND `_DENSE_MEMORY_SAFETY_FRACTION` was independently
   tightened from `0.5` to `0.2` (both changes together, not either
   alone, since the true N=150 ratio was not fully bounded above even
   at 18 GiB / 4.5x - see `fwht.py`'s own comments on both constants
   for the full derivation). Deliberately erred conservative: this
   means `auto_decompose` will now sometimes choose streaming at sizes
   where dense would in fact have fit and been faster (confirmed via
   the same N=100 measurement: real peak 5.27 GiB fits easily under an
   11 GiB budget, but the new 6x-inflated estimate exceeds the
   tightened 0.2-fraction threshold there) - an intentional tradeoff,
   not an oversight, for a safety-critical decision. Regression tests
   added (`tests/test_autotune.py`, 2 new) pinning both constants and
   the N=150 real-numbers outcome.
2. **Investigated and resolved 2026-09-01 (same day, follow-up)** —
   see `profiling/phase12/cache_probe_idempotency_investigation_findings.md`.
   Root-caused precisely (not just probabilistically): a second
   `probe_cache_boundaries()` call's small-buffer readings are
   inflated specifically when a prior call already walked large
   (multi-MiB) buffer sizes — stale cache/TLB state that each size's
   own `3x`-element warm-up doesn't fully clear. A larger warm-up
   floor was tried and found to genuinely help (from ~1-in-5 wrong to
   ~1-in-8 at a 16M-access floor) but never fully eliminate the
   effect, at rising runtime cost — then **reverted**, once it was
   confirmed this scenario cannot occur in shipped code at all:
   `recommended_chunk_size()` calls the probe at most once per
   process (its own cache), so every real invocation is a first,
   isolated call — already independently confirmed reliable (10/10
   fresh-process trials). Explicitly re-checked whether the
   per-process cache holds under real parallelism (multi-process HPC
   jobs, `os.fork()`) per direct instruction not to assume this
   without verifying — both confirmed safe by construction (each
   process/fork has its own independent cache). **One real gap found
   and fixed**: the cache was not thread-safe — concurrent callers
   from multiple *threads* in one process (a real heavy-parallelism
   pattern) could race to both invoke the underlying probe/memory
   detection at once. Fixed with a `threading.Lock` guarding both
   caches' population (double-checked locking, common cached path
   stays lock-free). Verified via mutation testing (temporarily
   removing the lock reproduces the race in the new regression tests,
   confirming they actually catch it) — 2 new tests
   (`tests/test_autotune.py`), 103 tests total, all pass.

**Post-bugfix re-verification, 2026-09-01 (same day, follow-up)** —
see `profiling/phase12/post_bugfix_reverification_findings.md`. Both
headline claims above were originally measured *before* either bug
fix landed; re-run fresh afterward (Bug 1's fix in particular changes
`auto_decompose()`'s actual routing decision, so this was not just
due diligence for its own sake). Confirmed for real, not by hand:
correctness holds bit-exact (`max_coefficient_error=0.000e+00`) at
N=25/50/100 with the fixed formula correctly routing N=25/50 to dense
and **N=100 to streaming** (a real, observed behavior change from
before the fix); the chunk_size speedup persists at 2.09x at N=100 and
1.84x at N=150 (both within normal run-to-run variance of the
original 2.32x/2.04x numbers, not a regression from either fix,
since neither touched `fwht_pauli_terms_iter` or the chunk_size
formula itself); `auto_decompose()` correctly picks streaming at both
N=100 and N=150 with wall-clock closely matching the direct
`fwht_pauli_terms_iter` call. The N=150 run stayed at 10-11 GiB
available memory throughout (`free -h` polled every 5s) - a stark
contrast to the pre-fix bug-hunting runs that dropped to 177-593 MiB
free, itself confirmation the fix is working as intended.

**Known gaps:**
- **Investigated 2026-09-01, FIXED 2026-09-02** - see
  `profiling/phase12/chunk_size_floor_scale_dependence_findings.md`.
  Design question 2 (the small-chunk-size floor) prompted by direct
  user suspicion after noticing streaming's abundant unused memory
  during the re-measurement runs. Real N=150/N=200 chunk_size sweeps
  (single-run then repeated-run then `perf stat`-confirmed) found the
  old static floor (8) was measurably suboptimal at both scales -
  `chunk_size=2` wins at N=150 (11% faster, mechanism confirmed via
  cache-miss ratio), `chunk_size=1` wins at N=200 (22% faster) - while
  a direct re-check found `chunk_size=8` is still clearly best at
  N=25/50 (2 would be 12-57% slower there). No single static floor
  constant was right across N=25-200 - best `chunk_size` decreases
  monotonically as `dim` grows (8 at dim=512/2048, 2 at dim=16384, 1
  at dim=32768), and even the "best working-set size" doesn't resolve
  to one fixed cache-level target (fits L2 at small N, spills into L3
  at N=150/200) - a single closed-form fit to those 4 sparse points
  would have been fabricated, not derived. **Fix**: `_min_chunk_size_
  floor(dim)` now takes `dim` and returns a value derived by
  log-log-interpolating between the 4 real measured anchor points
  (`_FLOOR_ANCHORS_DIM_TO_CHUNK_SIZE` in `autotune.py`), clamped to the
  endpoint values outside the measured dim range rather than
  extrapolating past real data. The dim=2048..16384 gap between
  anchors (no direct measurement there - see the findings doc's own
  "does NOT show" section) is bridged by interpolation, a deliberate,
  documented compromise for that specific range, not a re-verified
  value - a candidate for a future N=75/100 repeated-run sweep to
  narrow. 2 new regression tests pin the exact anchor values and the
  interpolation's monotonicity. On this dev machine,
  `recommended_chunk_size` now returns exactly the measured optima at
  dim=16384 (2) and dim=32768 (1), where it previously returned the
  stale floor of 8 at N=150 and clamped there for N=200 as well before
  the cache-target term brought it down. Also confirmed directly
  during this investigation: real system memory usage stayed flat
  (~2.6-2.9 GiB, process RSS 88-106 MiB) throughout every N=150/200
  run regardless of `chunk_size`, confirming Phase 9's streaming
  design bounds memory correctly (`O(chunk_size * dim)`, decoupled
  from total term count) - the abundant headroom the user noticed is
  real and expected, not itself a bug, even though it correctly
  prompted checking whether chunk_size was tuned well.
- **Fixed 2026-09-01**: `configure`'s RAM/swap diagnostic section
  (Linux path) now additionally reports `/proc/meminfo`'s
  `MemAvailable` and any cgroup v2/v1 memory cap, matching exactly
  what `autotune.available_memory_bytes()` reads at runtime (same
  `MemAvailable`-first, sentinel-rejecting-v1-fallback logic, ported
  to POSIX shell) - the diagnostic no longer silently drifts from the
  shipped auto-tuning code's actual inputs. Verified on this machine
  (10336 MiB reported, matching `autotune.available_memory_bytes()`'s
  own reading within run-to-run cache-reclaim noise) and the cgroup
  v2/v1 detection logic verified in isolation against synthetic
  `memory.max`/`memory.limit_in_bytes` values (a real `systemd-run
  --user --scope -p MemoryMax=` did not delegate the memory
  controller down to the scope's own leaf cgroup on this host, so a
  live end-to-end cgroup-capped `configure` run was not achieved -
  the shell logic itself was verified directly instead). macOS/BSD
  branches unchanged (no cgroup concept there).


### Phase 13 — multi-core / multi-node chunk parallelism (scoped 2026-09-02, not yet implemented)

**Motivation.** Raised directly by the user during Phase 12's
chunk_size-floor investigation: if this pipeline cannot use more than
one core, what is the point of running it on a supercomputer node or a
machine with a lot of memory? Confirmed via direct code search: zero
multiprocessing/MPI exists anywhere in `paulikit` today - every
speedup so far (Phases 3, 9, 10, 11, 12) came from making one core's
work cheaper, never from a second core. `_iter_chunked_coefficients`
already documents the key enabling property in its own docstring: each
chunk is a fully independent sub-problem, with no cross-chunk
combination step in the underlying math - concatenation, not
reduction.

**Full scoping**: `profiling/phase13/scoping.md`. Summary:
- Split into two sub-phases: **13a** (multi-core, single-node,
  `ProcessPoolExecutor` over chunks - self-contained, no cluster
  dependency) first; **13b** (multi-node, needs a job-launcher/MPI
  mechanism) deliberately deferred until 13a's real measurements are
  in hand.
- **A real, unresolved tension flagged, not assumed either way**:
  Phase 12's `chunk_size` floor and memory-budget formulas were
  measured on a single lone process. Multi-core execution means N
  worker processes competing for the *same shared* LLC and dividing
  the *same* memory budget - `available_memory_bytes()` must become
  `available_memory_bytes() / n_workers` per worker, and Phase 12's
  own cache-locality tuning needs re-measurement under real concurrent
  load (a single process today never contends with itself for cache).
- Worker-count auto-detection must use
  `len(os.sched_getaffinity(0))`, not `os.cpu_count()` - the same
  cpuset-awareness bug class Phase 12 already fixed for memory
  (`available_memory_bytes` vs. raw `MemTotal`), now for core count.
- API shape leans toward a new top-level function (parallel to
  `auto_decompose`'s own precedent - keep `fwht_pauli_terms`/
  `fwht_pauli_terms_iter`'s existing contracts untouched), not yet
  finalized - a design-question round with the user is expected before
  implementation starts, same as Phase 12's own process.
- Checkpoint/resume (Phase 9) assumes strictly sequential chunk
  completion order; parallel workers complete out of order - leaning
  toward disabling checkpointing in the parallel path initially rather
  than redesigning the checkpoint format ahead of a real need.


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
