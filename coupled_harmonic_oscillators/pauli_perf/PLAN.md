# Pauli Decomposition Performance Engineering — Plan

Status: draft, scaffolding phase. Last updated: 2026-08-04.


## 1. Problem statement

`coupled_harmonic_oscillators/N_coupled_harmonic_oscillators_1_D.ipynb`
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
- `coupled_harmonic_oscillators/pauli_perf/README.md` — module overview.
- `coupled_harmonic_oscillators/pauli_perf/PLAN.md` — this document.
- `requirements-dev.txt` (repo root) — profiling tools, kept separate
  from the main pinned `requirements.txt` so tutorial users don't need
  to install them.

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
- Port the smallest correct C kernel needed to address the profiled
  bottleneck.
- Bind it back to Python via, in this order, as a structured
  comparison exercise: Cython, then CFFI, then ctypes, then SWIG —
  building the same kernel each time and comparing raw performance,
  binding overhead, and build/tooling complexity.
- Parallelize the C kernel with oneTBB once a correct serial version
  exists, targeting the ~7x/8-core speedup reported in the FWHT paper.

### Phase 4 — Comparison and write-up
- Assemble a final results table/plot: naive SymPy (small N only) vs.
  pure-Python FWHT vs. each C-binding variant vs. PennyLane reference,
  at matched N values, covering both correctness and wall-clock time.
- Update this document and the module README with final findings.
- Feed conclusions back into the main `coupled_harmonic_oscillators`
  module once a production-ready decomposition function exists (a
  separate, later integration step — not assumed as part of this
  plan).


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
