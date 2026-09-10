# Non-Hermitian Pauli decomposition, and Classiq's FWHT

2026-09-10.

## The folk belief, and why it is wrong

There is a widespread assumption that Pauli decomposition is limited
to Hermitian operators. It is not, and none of the three
implementations examined here holds that limitation.

The confusion is understandable and is stated correctly in the user's
own tutorial (`tutorials/coupled_harmonic_oscillators/
N_coupled_harmonic_oscillators_1_D.ipynb`, cell 11):

> Pauli terms decomposition of Hermitian operators would have real
> coefficients; consequently, it could only involve terms with an
> even number of Y operators if the decomposition bases are
> {I, X, Y, Z}. For a non-Hermitian Hamiltonian, one could use
> Classiq's built-in function `matrix_to_hamiltonian`.

That is the actual content of the restriction: **Hermiticity
constrains the coefficients to be real** (and hence constrains which
Pauli strings can appear), it does not constrain whether the
decomposition exists. Every complex 2ⁿ × 2ⁿ matrix has a unique Pauli
expansion, because the 4ⁿ Pauli strings form a basis for ℂ^(2ⁿ×2ⁿ)
under the Hilbert-Schmidt inner product. Non-Hermitian input simply
yields complex coefficients.

## Verified directly, all three implementations

**The paper.** Georges et al state the general case from the outset -
equation (2) is for "a complex matrix, A ∈ ℂ^(2ⁿ×2ⁿ)". Hermiticity
appears only as a *special case*: section 2 states three corollaries
for when the matrix is (i) Hermitian, (ii) real symmetric or (iii)
complex symmetric. So the paper is explicitly general and treats
Hermitian as an optimisation opportunity, not a precondition.

**paulikit and pauli_lcu, measured.** Decomposing a deliberately
non-Hermitian complex random matrix (verified `A != A†`):

| n_qubits | paulikit reconstruction max abs error | terms | pauli_lcu |
|---|---|---|---|
| 2 | 2.48e-16 | 16 | ran, sum abs(c) = 8.2090 |
| 3 | 4.58e-16 | 64 | ran, sum abs(c) = 28.8730 |
| 4 | 6.66e-16 | 256 | ran, sum abs(c) = 81.9200 |

paulikit reconstructs to machine precision via
`fwht_pauli_terms(A, assume_hermitian=False)`. Note all 4ⁿ terms
survive - a dense random matrix has no structural zeros, as expected.

**Classiq.** `matrix_to_hamiltonian(mat, tol=ATOL, is_hermitian=True)`
takes an explicit `is_hermitian` flag; passing `False` skips the
Hermiticity assertion and changes the coefficient sign handling
(`_get_signed_coefficient(coef[i], k, i, is_hermitian)`). So
non-Hermitian support is a deliberate, documented feature, not an
accident.

## Classiq uses FWHT, and says so

Captured from `inspect.getsource(matrix_to_hamiltonian)` (saved in
`notes/fwht.ipynb`'s cell output, classiq pinned at 1.23.0 in this
repo's `requirements.txt`):

```python
def matrix_to_hamiltonian(
    mat: np.ndarray, tol: float = ATOL, is_hermitian: bool = True
) -> List[PauliTerm]:
    """
    The decomposition per set is done by the Walsh-Hadamard transform,
    since the transformation between {e_0,e_3} ({e_1,e_2}) to {I,Z}
    ({X,iY}) is the Hadamard matrix.
    """
    ...
    for k in range(2**num_qubits):
        coef = fwht(_coefficents_for_set(mat, k))
        hamiltonian += [
            PauliTerm(
                pauli=_get_pauli_string(k, i, num_qubits),
                coefficient=_get_signed_coefficient(coef[i], k, i, is_hermitian),
            )
            for i in range(2**num_qubits)
            if abs(coef[i]) > tol
        ]
```

The docstring states the same structural insight the paper's theorem 1
formalises: the change of basis from matrix-unit pairs to Pauli
operators **is** a Hadamard matrix, so the coefficients follow from a
Walsh-Hadamard transform.

Structural differences from `pauli_lcu` worth noting:

- Classiq loops `k` over 2ⁿ "sets" and calls `fwht` per set, rather
  than doing one in-place pass over the whole matrix. Not in place.
- It thresholds against `tol` and returns a list of `PauliTerm`
  objects - i.e. a **sparse, labelled** output, closer to paulikit's
  COO triples than to pauli_lcu's positional dense array.
- The phase/sign handling is factored into
  `_get_signed_coefficient(..., is_hermitian)` rather than the
  `i**popcount(x & z)` form.

## Dating the prior art — VERIFIED against primary sources

Reference dates for the paper: **arXiv:2408.06206 posted 12 Aug
2024**; **NJP submission 10 Oct 2024**; published 28 Feb 2025.

### PennyLane — 10 August 2023, over a year earlier

PR #4395, *"Faster, better and differentiable Pauli decompose"*,
merged **2023-08-10** (commit `b5789db`), rewrote
`qml.pauli_decompose` to use the Fast Walsh-Hadamard Transform. The
source at that merge commit imports
`_walsh_hadamard_transform` and carries the comment
`# https://quantumcomputing.stackexchange.com/a/31790` - Gidney's
answer - from that commit onward.

The PR description states it "removes restriction for matrix being
square or Hermitian", so **non-Hermitian support arrived in the same
PR**.

This is the same lineage the paper itself acknowledges ("Gidney's code
then was also incorporated in Pennylane"), now with a date.

### Classiq — 5 August 2024, seven days before the arXiv posting

Verified independently against primary sources:

- **PyPI**: `classiq` 0.44.0, `classiq-0.44.0-py3-none-any.whl`,
  upload time **2024-08-05T10:33:10.010714Z** (PyPI JSON API).
- **The wheel's own source**, downloaded and read directly -
  `classiq/applications/hamiltonian/pauli_decomposition.py`, which
  ships as **plain readable Python**, not obfuscated:
  - line 4: `from sympy import fwht`
  - `matrix_to_hamiltonian(mat, tol=ATOL, is_hermitian=True)` with the
    Walsh-Hadamard docstring quoted above
  - `_get_signed_coefficient` computing
    **`(1j) ** ((i & k).bit_count())`**
- **Classiq's public library repo**: the matching notebook usage
  landed at commit `ef365378`, **2024-08-05T11:14:37Z** ("Updates for
  0.44.0"), the same day. The two prior commits touching that
  notebook (Feb-Jun 2024) do not contain `matrix_to_hamiltonian`.

That phase factor deserves emphasis. Classiq's
`(1j) ** ((i & k).bit_count())` is **the same expression** as the
paper's equation (8) factor `i^(-|r∧s|)` and as `pauli_lcu`'s
`__builtin_popcount(i & j) & 3` switch - the same popcount-of-AND
phase correction, shipped 2024-08-05.

**Caveat, stated because it bounds the claim:** version 0.44.0 is the
earliest release *verified* to contain it. Versions before 0.42.2
(2024-06-17) were not diffed, so the function may exist in an earlier
release without corresponding library-repo usage. The established
claim is "no later than 2024-08-05", not "first appeared then".

### Gidney's StackExchange answer — date NOT established

`quantumcomputing.stackexchange.com/a/31790` could not be fetched in
this environment, and no timestamp was obtained from a primary
source. It **must** predate PennyLane's 2023-08-10 merge that cites
it, but that is inference. **Do not cite a date for it.**

### Summary table

| implementation | FWHT for Pauli decomposition | non-Hermitian | date basis |
|---|---|---|---|
| PennyLane | yes | yes, same PR | **2023-08-10**, merge commit `b5789db` |
| Classiq | yes | yes, `is_hermitian=False` | **2024-08-05**, PyPI upload + wheel source |
| Gidney (SO) | yes | — | predates 2023-08-10 (inferred, not verified) |
| arXiv:2408.06206 | yes | yes | posted 2024-08-12 |
| NJP 27 033004 | yes | yes | submitted 2024-10-10 |

## What this does and does not show

**It does not show the paper claimed false priority.** As recorded in
`paper_claims_audit.md`, the paper explicitly concedes Gidney and
Hamaguchi et al as prior FWHT work and confines its novelty claim to
equations (8)/(9) and their proofs. PennyLane's implementation is
exactly the lineage it names.

**What it adds** is that the prior art is broader and better dated
than the related-work section conveys. Classiq is a *third*
independent FWHT-based implementation, with non-Hermitian support and
the same popcount phase factor, in a public release predating the
arXiv posting by a week - and it is not cited. PennyLane is cited
[19], but as a package carrying Gidney's code rather than as a dated
prior implementation.

**Practical consequence for us:** any claim we publish must not
describe FWHT-based Pauli decomposition as originating with this
paper, and should note that at least three public implementations
(PennyLane 2023-08, Classiq 2024-08, pauli_lcu 2024) predate or
accompany it. Our own contribution is the streaming/COO/bounded-memory
design and the measured performance, not the transform.

The comparison date that matters is **10 October 2024**, the paper's
submission (arXiv:2408.06206 was posted August 2024).

**What the paper itself already concedes.** Its "Relation to previous
work" section is explicit that FWHT-based Pauli decomposition predates
it - Gidney's StackOverflow answer [15] ("which we learnt about after
posting this work online", and whose "code then was also incorporated
in Pennylane"), and Hamaguchi, Hamada and Yoshioka [16], whose
algorithm's "central primitive/subroutine is also the Fast
Walsh-Hadamard transformation". The paper's novelty claim is narrow
and specific: "the explicit equations (8) and (9) and their proofs
have not been published before."

So the interesting question is **not** whether the paper wrongly
claims priority over FWHT - it does not. It is whether the set of
prior FWHT implementations is larger than the paper's related-work
section acknowledges, and Classiq is a concrete candidate for that
set. Establishing its date is what would settle it.
