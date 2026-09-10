# Auditing the pauli_lcu paper's claims

2026-09-10. Source: Georges T N, Berntson B K, Sünderhauf C, Ivanov
A V, *Pauli decomposition via the fast Walsh-Hadamard transform*,
**New J. Phys. 27 (2025) 033004**, doi:10.1088/1367-2630/adb44d.
Received 10 Oct 2024, accepted 10 Feb 2025, published 28 Feb 2025.
Open access, CC-BY.

Local copy: `notes/(Georges-2025)--Pauli decomposition via the fast
Walsh-Hadamard transform.pdf`.

**Note on citation year.** The `pauli_lcu` README cites this as
"(2024)" via arXiv:2408.06206. The published version is 2025. Both
refer to the same work; cite the NJP version.

This audit is deliberately adversarial - the point is to find what
does not hold up. Several claims **do** hold up, and are recorded as
such.

---

## Claim 1: "O(1) additional memory" — TRUE AS STATED, but the
## framing hides the binding constraint

**What the abstract says:** "calculate all Pauli decomposition
coefficients in O(N² log N) time and using O(1) additional memory,
for an N × N matrix."

**Verified empirically.** Peak RSS measured before allocation, after
allocating the input, and after the call, so the final delta is
auxiliary space alone:

| n | dim | matrix | RSS after alloc | RSS after call | **auxiliary** |
|---|---|---|---|---|---|
| 10 | 1024 | 16 MiB | 69.6 MiB | 69.6 MiB | **+0.0** |
| 11 | 2048 | 64 MiB | 165.8 MiB | 165.8 MiB | **+0.0** |
| 12 | 4096 | 256 MiB | 549.7 MiB | 549.7 MiB | **+0.0** |
| 13 | 8192 | 1024 MiB | 2085.8 MiB | 2085.8 MiB | **+0.0** |

The claim is **literally true**: zero additional bytes at every size.
`pauli_coefficients` overwrites the caller's buffer in place. Credit
where due - this is a real and well-executed property, and our own
earlier characterisation of it was correct.

**What the framing omits.** "O(1) *additional*" is auxiliary space on
top of an input the caller must already hold. That input is
Θ(4ⁿ) and is not optional:

| n | dim | dense complex128 input |
|---|---|---|
| 13 | 8192 | 1.0 GiB |
| 14 | 16384 | 4.0 GiB |
| 15 | 32768 | 16.0 GiB |
| 16 | 65536 | 64.0 GiB |

So the *algorithm's* space complexity is O(1), while the *problem as
posed* is Θ(4ⁿ). An abstract reader who takes "O(1) memory" at face
value will mis-estimate the hardware requirement by orders of
magnitude.

**The paper never states this.** Checked directly: no passage
acknowledges that the caller must hold the full dense 2ⁿ × 2ⁿ matrix,
and there is no discussion of what happens when it does not fit in
RAM. The only memory-capacity discussion in the paper is a "memory
error" entry recorded for a *competitor's* implementation (Hamaguchi
et al) at the largest size in both Table 1 and Table 2. The
hardware note for figure 3 mentions "386 GB RAM" as a spec, not as a
constraint being discussed.

This is not an error - it is a scope choice, and a defensible one for
a paper about a transform. It is, however, exactly the axis on which
a streaming implementation differs, and it is unaddressed.

---

## Claim 2: parallelism — THE PAPER CLAIMS IT; THE RELEASED CODE DOES
## NOT CONTAIN IT

**What the paper says** (section 3, page 8, verbatim):

- "our algorithm allows for straightforward parallelization. First,
  the XOR transformation (Step 1 in figure 1) is applied to each
  column, then Hadamard transformation and phase factor
  multiplication are applied to each row independently (Steps 2-3 in
  figure 1)."
- "**We have implemented this parallelization with OpenMP** and
  tested it on the largest example from table 2 (2¹⁵ × 2¹⁵ matrix)."
- "we observe nearly ideal speedup for 2 and 4 cores. For 8 cores,
  the wall time is reduced by a factor of 7 and the Pauli
  coefficients can be calculated in around 2.5 s as compared to
  17.2 s on a single AMD EPYC 7763 core."
- Figure 3 plots this speedup.

**What ships.** The released `pauli_lcu` 1.0.1 - the implementation
the paper points readers to (reference [24], `pip install pauli_lcu`)
- contains **no OpenMP at all**. Verified three independent ways:

1. **Source distribution** (`pip download --no-binary`): zero matches
   for `pragma omp`, `_OPENMP`, `omp_get`, or `fopenmp` anywhere in
   the package.
2. **Build configuration**: `setup.py` sets
   `xoptions = ['-O3']` and passes it as the only
   `extra_compile_args`. No `-fopenmp`.
3. **Compiled binary** (`pauli_lcu_module.cpython-312-*.so`, from the
   published manylinux wheel): `nm -D -u` shows no OpenMP symbols,
   `objdump -d` shows no packed-double SIMD, and measured
   CPU-time/wall over the decomposition region alone is **0.99** -
   strictly one core.

The shipped README makes no mention of parallelism either.

**The archived reproduction package does not contain it either.**
The paper deposits its benchmark materials on Zenodo (reference [31],
doi:10.5281/zenodo.14905815, *"Pauli Decomposition via the Fast
Walsh-Hadamard Transform v3"*, published 2025-02-21). Fetched and
inspected directly:

- `README.txt` directs readers to `pip install pauli_lcu` or the
  GitHub repo - i.e. to the serial release - and additionally pins
  the two competitor sources (Hamaguchi's `paulidecomp`, and Qiskit
  PR #11557 at commit `9ad36a3da`, both "as of 26 Sept 2024").
- `run.py`, the benchmark driver, contains **no OpenMP, thread, or
  core references whatsoever** (grepped for `omp`, `thread`, `core`,
  `parallel`).
- The archived results are `results_coeff.txt`,
  `results_zx_phase.txt`, `results_qiskit.txt` and
  `results_hamaguchi.txt` - **exactly the four single-core series of
  figure 2**. There is no fifth series, and no script, for figure 3's
  OpenMP scaling.
- The record declares no sibling versions or related identifiers, so
  there is no separate parallel deposit.

So figure 3's result - the 7x on 8 cores - is **not reproducible from
any artifact the authors published**: not the PyPI package, not the
GitHub repo, and not the Zenodo deposit that exists precisely to make
the paper's numbers reproducible.

**How to state this fairly - and the distinction that matters.** The
paper does not claim the *released package* is parallel. It says "we
have implemented this parallelization", a claim about work the authors
did, and there is no reason to doubt they did it.

But "the algorithm parallelises" and "the system delivers that
speedup" are **different claims about different artifacts**, and only
the first is established here. A decomposition into independent rows
is a mathematical property; turning it into sustained multi-core
throughput is an engineering problem with its own failure modes, none
of which the row-independence argument addresses:

- Where does the result go? Independent rows still have to be
  collected. Phase 13 measured a 25 GiB blowup from unbounded result
  queuing that no amount of algorithmic independence prevented.
- Does the working set survive concurrency? Phase 14 measured total
  CPU rising 77% from 1 to 4 workers on identical work - shared-L3
  contention that only a cache-blocked kernel reduced (to +30%).
- Is the per-unit work large enough to amortise the coordination?
  Phase 14 found it was not: once the kernels made chunks cheap, the
  same pool that had bought 2.95x became a net loss.
- What is the serial fraction *in the implementation*, as opposed to
  the algorithm? Measured here at f = 0.0336 - small, but it had to
  be measured, not assumed.

Each of those is invisible from the algorithm and decisive for the
system. This project's own history is the evidence: the row
independence was never in doubt, and it still took Phases 13-15 -
bounded submission, an array-yielding API, two C kernels, and a
falsified drain-loop fix - to convert it into real throughput.

So the criticism is not about honesty, and it is not only about
reproducibility. It is that **a reported speedup is doing the work of
a delivered capability**. Figure 3 is presented as what the method
achieves, while every artifact offered to readers - PyPI package,
GitHub repo, and the Zenodo deposit that exists precisely to make the
numbers reproducible - is serial. A reader cannot verify the figure,
cannot build on the parallel version, and cannot obtain the speedup
the paper reports. The gap between "we implemented it" and "you can
run it" is exactly the engineering, and it is the part that is
missing.

**For our own write-up:** state this plainly but without insinuation.
We are not alleging the measurement is wrong. We are pointing out
that a parallel result which ships in no artifact is a claim about
the authors' private build, and that the distance between a
parallelisable algorithm and a parallel system is the substance of
the work - not a detail to be assumed away.

**Consequence for any comparison we publish.** Two things follow, and
both matter:

1. A single-core comparison against `pauli_lcu` **is** a fair
   comparison against the released artifact, and should be described
   in exactly those terms - "pauli_lcu 1.0.1 as released", never
   "pauli_lcu is single-threaded" as though that were a property of
   the algorithm. It is not; the paper shows otherwise.
2. We must not claim a *parallelism* advantage over their method on
   the strength of comparing against a serial release. Their rows are
   independent exactly as ours are, and they report 7x on 8 cores. If
   we ever claim to parallelise better, it has to be against a
   comparably parallel build of theirs - which does not exist
   publicly, so that comparison currently cannot be made by anyone,
   us included.

   What we *can* claim, and should, is on the axis where artifacts
   can actually be compared: what a user obtains and runs. On that
   axis paulikit ships a working parallel path, a streaming
   bounded-memory path, and measured numbers reproducible from this
   repository. That is an engineering claim, and it is the honest
   one - it does not require asserting anything about the quality of
   their unpublished parallel build.

---

## Claim 3: priority over FWHT for Pauli decomposition — THE PAPER
## DOES NOT CLAIM IT, AND CONCEDES PRIOR ART

Worth stating plainly because it is easy to assume otherwise from the
title: **the paper does not claim to have invented the use of FWHT
for Pauli decomposition.** Its own "Relation to previous work"
(section 1) concedes:

- "The most similar algorithm is that of Gidney [15] which we learnt
  about **after posting this work online**. It was proposed in
  Stack-Overflow comment [15] without though an explanation of how
  this algorithm was derived. **Gidney's code then was also
  incorporated in Pennylane** [19]."
- "Hamaguchi, Hamada and Yoshioka proposed an algorithm [16] which is
  based on the Fast Walsh-Hadamard transformation. While the central
  primitive/subroutine in their algorithm is also the Fast
  Walsh-Hadamard transformation, we observe that our implementation
  is around 3x faster."
- "the (Fast) Walsh-Hadamard transform is well understood and its
  fast implementations are known [7, 20]."
- FWHT is "used in other areas of quantum computation ... in the
  context of Pauli channels ... and its fast implementation is used
  in several papers [21, 22, 23]."

**What they actually claim as novel:** "To the best of our knowledge,
the explicit equations (8) and (9) and their proofs have not been
published before." That is a claim about the closed-form expression
and its proof, plus a fully in-place implementation - **not** about
the FWHT approach itself.

This is honest scholarship and should be represented as such. Any
critique that reads the paper as claiming to have originated
FWHT-based Pauli decomposition would be attacking a position they do
not hold.

The `pauli_lcu` README's phrasing - "according to the algorithm
presented in [the paper]" - is looser than the paper itself, but that
is a README, not a claim in the literature.

---

## Claim 4: "outperforms currently available solutions"

**Their benchmarks, all on a single AMD EPYC 7763 core at 2.4 GHz**
(stated explicitly for figure 2, table 1 and table 2), and timed
**through the Python bindings, not raw C** ("we ... time our Python
bindings rather than C code throughout").

Compared against: Qiskit's `decompose_dense` implementing the TPD
algorithm of Hantzko et al [14] (an unpublished dev version, [17]),
and Hamaguchi et al's C++ FWHT implementation [16, 26].

Reported: ≈1.4x max over Qiskit on random Hermitian matrices
(figure 2, 2-14 qubits, 100 runs per point); 3.6x max over Hamaguchi
et al; 2.78x total over Qiskit-dev TPD on Table 1's largest case;
4.12x total over Qiskit-dev TPD on Table 2's largest case.

**Assessment: the methodology here is good.** 100 runs per data point,
hardware stated, single-core stated, the Python-binding caveat stated
explicitly. The speedups are modest and honestly reported - 1.4x, not
an order of magnitude. Nothing here is overclaimed.

**One gap:** the comparison set is Qiskit-dev TPD and Hamaguchi et al.
PennyLane is cited [19] and is noted as carrying Gidney's code, but is
not benchmarked. Neither is any streaming or sparsity-aware approach -
reasonably, since the paper's scope is dense matrices.

---

## Still to check

- Refs [8]-[16] individually: do any predate the paper with an
  FWHT-based method beyond Gidney and Hamaguchi? Hantzko et al [14]
  (TPD) and Gunlycke et al [13] are the ones to read.
- **Classiq**: `matrix_to_hamiltonian` uses FWHT explicitly and
  supports non-Hermitian input via `is_hermitian=False`. Dating this
  against the paper's 10 Oct 2024 submission is open - see
  `classiq_fwht_history.md`.
