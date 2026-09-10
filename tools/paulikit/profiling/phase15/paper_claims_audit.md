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

**How to state this fairly.** The paper does not claim the *released
package* is parallel. It says "we have implemented this
parallelization" - which is a claim about work the authors did, and
there is no reason to doubt they did it. The gap is between the
paper's reported capability and the artifact readers can obtain.

**Consequence for any comparison we publish.** Two things follow, and
both matter:

1. A single-core comparison against `pauli_lcu` **is** a fair
   comparison against the released artifact, and should be described
   in exactly those terms - "pauli_lcu 1.0.1 as released", never
   "pauli_lcu is single-threaded" as though that were a property of
   the algorithm. It is not; the paper shows otherwise.
2. Their algorithm parallelises well - 7x on 8 cores by their own
   measurement, on rows that are independent exactly as ours are.
   Any claim of a parallelism advantage on our side must be measured
   against a comparably parallel build of theirs, not against the
   serial release. We have not done that.

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
