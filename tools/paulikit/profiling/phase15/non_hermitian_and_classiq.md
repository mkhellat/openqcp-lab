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

## The priority question — what is and is not established

**Established here:** Classiq ships an FWHT-based Pauli decomposition
with explicit non-Hermitian support, and documents the Hadamard
change-of-basis insight in the function's own docstring.

**NOT established here:** *when* Classiq first published it. This
repo pins classiq 1.23.0 and its own history begins 2026-08, so it
cannot date Classiq's release. Dating requires PyPI release history
or Classiq's public repos, and is being traced separately.

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
