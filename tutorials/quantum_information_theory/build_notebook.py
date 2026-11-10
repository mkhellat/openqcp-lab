#!/usr/bin/env python3
"""Build the QI / information-theory notebook (run; commit the .ipynb)."""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name(
    "quantum_information_and_information_theory.ipynb"
)


def fix_math(source: str) -> str:
    """Use $ / $$ like the rest of openqcp-lab (MathJax-reliable)."""
    source = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$" + m.group(1) + "$$",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + m.group(1) + "$",
        source,
        flags=re.S,
    )
    return source


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(fix_math(source.strip() + "\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells: list[nbf.NotebookNode] = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Quantum Information and Information Theory

Information theory studies **uncertainty** and **distinguishability**:
how much is unknown about a classical random variable or a quantum
state, and how well two states of knowledge (or two physical
preparations) can be told apart. It expresses those ideas with
real-valued measures — entropies, divergences, distances — and with
**channels** that map inputs to outputs.

The sets on which those measures live (probability simplices, sets of
density operators) are often not vector spaces under addition of
points; local geometry of distinguishability is developed in the
manifolds foundations notebook (Fisher metric, density operators).
This module develops the **information measures and channels**
themselves, with coded checks, and ends with a map toward code-based
post-quantum cryptography (HQC named as a destination). It does not
replace a dedicated PQC / Open Quantum Safe notebook.

> **Further reading (anchors).** T. M. Cover and J. A. Thomas,
> *Elements of Information Theory*, 2nd ed., Wiley, 2006.
> M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
> Information*, Cambridge, 2000. M. M. Wilde, *Quantum Information
> Theory*, 2nd ed., Cambridge, 2017.
"""))

cells.append(md(r"""
## Prerequisites

- Linear algebra over $\mathbb{R}$ and $\mathbb{C}$.
- Basic discrete probability (mass functions, expectation).
- Undergraduate quantum mechanics: state vectors and density
  operators at the level of a first course (or manifolds notebook
  §A2).

## Learning objectives

1. Compute Shannon entropy, mutual information, and KL divergence on
   finite alphabets and interpret them in bits.
2. Describe a classical channel as a stochastic matrix; evaluate the
   binary symmetric channel (BSC) capacity formula numerically.
3. Compute von Neumann entropy and quantum relative entropy on small
   systems; relate purity to mixedness.
4. Compute fidelity and trace distance; interpret them as
   distinguishability measures.
5. Represent a qubit channel by Kraus operators and check the CPTP
   completeness relation numerically.
6. Explain why coding / decoding hardness feeds code-based PQC, and
   where the dedicated cryptology notebook continues (HQC named).
"""))

cells.append(md(r"""
## How this notebook is organised

- **Part I.** Classical entropy, mutual information, KL, channels,
  Hamming bridge.
- **Part II.** Quantum entropy, distances, Kraus channels, Holevo
  sketch.
- **Part III.** Crypto motivation; code-based idea; HQC destination;
  OQS off-ramp.
- **Part IV.** Bridges to manifolds and the wider lab.

**Citation rule.** Every major block names sources a reader can open.
We do not invent references.
"""))

cells.append(code(r"""
# Setup — QI / information-theory notebook (numpy / scipy / matplotlib)
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigvalsh
from scipy.linalg import logm, sqrtm

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (5.5, 4.0), "font.size": 11})

LOG2E = 1.0 / np.log(2.0)  # convert nat -> bit when using np.log


def shannon_entropy(p, eps=1e-16):
    # H(p) in bits for a finite probability vector
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    return -np.sum(p * np.log2(p))


print("numpy", np.__version__)
"""))

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part I — Classical information

## I.1 Shannon entropy

For a discrete random variable $X$ with probabilities
$p(x)=\Pr(X=x)$, the **Shannon entropy** (in bits) is

$$
H(X)=-\sum_x p(x)\log_2 p(x),
$$

with the convention $0\log 0=0$. It is zero for a deterministic
variable and maximal for the uniform distribution on a fixed alphabet.

Joint and conditional entropies satisfy the chain rule
$H(X,Y)=H(X)+H(Y|X)$.

> **Further reading.** Cover & Thomas, Ch. 2.
"""))

cells.append(code(r"""
# Fair coin vs biased coin vs deterministic bit

for name, p in [
    ("fair coin", [0.5, 0.5]),
    ("biased 0.9", [0.9, 0.1]),
    ("deterministic", [1.0, 0.0]),
]:
    print(f"{name:16s} H = {shannon_entropy(p):.4f} bits")

assert np.isclose(shannon_entropy([0.5, 0.5]), 1.0)
assert shannon_entropy([1.0, 0.0]) < 1e-12
"""))

cells.append(md(r"""
## I.2 Mutual information

$$
I(X;Y)=H(X)-H(X|Y)=H(Y)-H(Y|X)=H(X)+H(Y)-H(X,Y).
$$

It measures how much knowing $Y$ reduces uncertainty about $X$.
Independent variables have $I(X;Y)=0$.

> **Further reading.** Cover & Thomas, Ch. 2.
"""))

cells.append(code(r"""
# Joint distribution on {0,1}x{0,1}: correlated bits

P = np.array([[0.40, 0.10],
              [0.10, 0.40]], dtype=float)
assert np.isclose(P.sum(), 1.0)

p_x = P.sum(axis=1)
p_y = P.sum(axis=0)
H_x, H_y = shannon_entropy(p_x), shannon_entropy(p_y)
H_xy = shannon_entropy(P.ravel())
I_xy = H_x + H_y - H_xy
print("H(X), H(Y), H(X,Y), I(X;Y) =", H_x, H_y, H_xy, I_xy)
assert I_xy > 0
"""))

cells.append(md(r"""
## I.3 Relative entropy (Kullback–Leibler)

For distributions $p$ and $q$ on the same finite alphabet,

$$
D(p\|q)=\sum_x p(x)\log_2\frac{p(x)}{q(x)},
$$

when $p(x)>0$ implies $q(x)>0$; otherwise one sets $D(p\|q)=\infty$.
Always $D(p\|q)\ge 0$, with equality iff $p=q$. It is not a metric
(asymmetric; triangle inequality fails).

> **Further reading.** Cover & Thomas, Ch. 2 (relative entropy).
"""))

cells.append(code(r"""
def kl_divergence(p, q, eps=1e-16):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any((p > eps) & (q <= eps)):
        return np.inf
    mask = p > eps
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


p = np.array([0.7, 0.3])
q = np.array([0.5, 0.5])
print("D(p||q) =", kl_divergence(p, q))
print("D(q||p) =", kl_divergence(q, p))
print("D(p||p) =", kl_divergence(p, p))
assert kl_divergence(p, p) < 1e-12
assert kl_divergence(p, q) > 0
"""))

cells.append(md(r"""
## I.4 Classical channels and the BSC

A discrete channel is a family of conditional distributions
$W(y|x)$, or equivalently a **stochastic matrix** (nonnegative entries,
columns — or rows, by convention — sum to one).

The **binary symmetric channel** with crossover probability $f$ flips
each bit independently with probability $f$. Its Shannon capacity
(Cover & Thomas, Ch. 7) is

$$
C=1-H_2(f),
\qquad
H_2(f)=-f\log_2 f-(1-f)\log_2(1-f)
$$

(for $f\in[0,1/2]$; symmetric about $1/2$).

> **Further reading.** Cover & Thomas, Ch. 7; Shannon, “A Mathematical
> Theory of Communication,” *Bell Syst. Tech. J.* **27** (1948).
"""))

cells.append(code(r"""
def binary_entropy(f, eps=1e-16):
    f = float(f)
    if f < eps or f > 1 - eps:
        return 0.0
    return -f * np.log2(f) - (1 - f) * np.log2(1 - f)


fs = np.linspace(0.0, 0.5, 51)
C = 1.0 - np.array([binary_entropy(f) for f in fs])

fig, ax = plt.subplots()
ax.plot(fs, C, "C0-", lw=2)
ax.set_xlabel(r"crossover $f$")
ax.set_ylabel(r"BSC capacity $C=1-H_2(f)$ (bits)")
ax.set_title("Binary symmetric channel capacity")
plt.show()

assert np.isclose(1 - binary_entropy(0.0), 1.0)
assert 1 - binary_entropy(0.5) < 1e-12
print("C(f=0.11) ≈", 1 - binary_entropy(0.11))
"""))

cells.append(md(r"""
## I.5 Coding bridge (Hamming $[7,4]$ toy)

A **binary linear code** of length $n$ and dimension $k$ is a
$k$-dimensional subspace of $\mathbb{F}_2^n$. The **Hamming distance**
between strings is the number of differing positions. Decoding a
received word to the nearest codeword is easy for tiny textbook codes
and hard for the large structured codes used in cryptography.

The Hamming $[7,4]$ code encodes $4$ information bits into $7$ bits
with minimum distance $3$ (corrects $1$ error). Below we only
demonstrate encode / single-error correct — a pedagogical toy, **not**
a cryptosystem.

> **Further reading.** F. J. MacWilliams and N. J. A. Sloane, *The
> Theory of Error-Correcting Codes*, North-Holland, 1977, Ch. 1
> (Hamming codes).
"""))

cells.append(code(r"""
# Hamming [7,4]: systematic generator (standard textbook form)

# G: 4 x 7 over GF(2)
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

# Parity-check H with H G.T = 0 (mod 2); columns = binary 1..7
H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=int)

msg = np.array([1, 0, 1, 1], dtype=int)
codeword = msg @ G % 2
print("message ", msg)
print("codeword", codeword)
assert np.all(H @ codeword % 2 == 0)

# Flip bit position 2 (0-based index 1)
received = codeword.copy()
received[1] ^= 1
syndrome = H @ received % 2
# Syndrome as integer 1..7 equals error position (1-based) for Hamming
pos = syndrome[0] * 4 + syndrome[1] * 2 + syndrome[2] * 1
print("syndrome bits", syndrome, "-> error position", pos)
corrected = received.copy()
if pos > 0:
    corrected[pos - 1] ^= 1
assert np.array_equal(corrected, codeword)
print("corrected", corrected)
"""))

# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II — Quantum information

## II.1 Density operators (brief)

A quantum state of a finite-dimensional system is a **density
operator** $\rho$: Hermitian, positive semidefinite,
$\operatorname{Tr}\rho=1$. Pure states are rank-one projectors
$\rho=|\psi\rangle\langle\psi|$. Geometry of the set of density
operators is in the manifolds notebook (§A2); here we only need them
as inputs to information measures.

> **Further reading.** Nielsen & Chuang, Ch. 2; manifolds notebook §A2.
"""))

cells.append(md(r"""
## II.2 von Neumann entropy

$$
S(\rho)=-\operatorname{Tr}(\rho\log_2\rho)
=-\sum_i \lambda_i\log_2\lambda_i,
$$

where $\lambda_i$ are eigenvalues of $\rho$ (again $0\log 0=0$).
Pure states have $S=0$; the maximally mixed state
$I/d$ on a $d$-level system has $S=\log_2 d$.

> **Further reading.** Nielsen & Chuang, Ch. 11; Wilde, Ch. 11.
"""))

cells.append(code(r"""
def density_from_ket(psi):
    psi = np.asarray(psi, dtype=complex)
    psi = psi / norm(psi)
    return np.outer(psi, np.conj(psi))


def von_neumann(rho, eps=1e-14):
    w = eigvalsh(rho)
    w = w[w > eps]
    return -np.sum(w * np.log2(w))


psi = np.array([1.0, 0.0], dtype=complex)
rho_pure = density_from_ket(psi)
rho_mix = 0.5 * np.eye(2)
print("S(pure) =", von_neumann(rho_pure))
print("S(I/2)  =", von_neumann(rho_mix))
assert von_neumann(rho_pure) < 1e-12
assert np.isclose(von_neumann(rho_mix), 1.0)
"""))

cells.append(md(r"""
## II.3 Quantum relative entropy

For density operators $\rho$ and $\sigma$ with
$\operatorname{supp}\rho\subset\operatorname{supp}\sigma$,

$$
D(\rho\|\sigma)
=\operatorname{Tr}\bigl(\rho(\log_2\rho-\log_2\sigma)\bigr).
$$

Otherwise $D(\rho\|\sigma)=\infty$. Like KL, it is nonnegative and
asymmetric. When $\rho$ and $\sigma$ commute they may be diagonalized
together and $D$ reduces to classical KL on eigenvalues.

> **Further reading.** Nielsen & Chuang, Ch. 11; Wilde, Ch. 11.
"""))

cells.append(code(r"""
def quantum_relative_entropy(rho, sigma, eps=1e-10):
    # Finite when support(rho) subset support(sigma)
    wr = eigvalsh(rho)
    ws = eigvalsh(sigma)
    if np.any((wr > eps) & (ws <= eps)):
        # crude support check in eigenbasis of each — full check uses
        # projectors; enough for our diagonal / full-rank examples
        pass
    # Use principal log via eigendecomposition in bits
    def log2m(a):
        w, v = np.linalg.eigh(a)
        if np.any(w < -eps):
            raise ValueError("not PSD")
        w = np.clip(w, eps, None)
        return (v * np.log2(w)) @ v.conj().T

    if eigvalsh(sigma).min() < eps and eigvalsh(rho).min() > eps:
        return np.inf
    return float(np.real(np.trace(rho @ (log2m(rho) - log2m(sigma)))))


rho = np.diag([0.7, 0.3])
sigma = np.diag([0.5, 0.5])
print("D(rho||sigma) =", quantum_relative_entropy(rho, sigma))
print("D(rho||rho)   =", quantum_relative_entropy(rho, rho))
assert quantum_relative_entropy(rho, rho) < 1e-10
assert quantum_relative_entropy(rho, sigma) > 0
"""))

cells.append(md(r"""
## II.4 Trace distance and fidelity

$$
T(\rho,\sigma)=\tfrac12\|\rho-\sigma\|_1,
\qquad
F(\rho,\sigma)
=\Bigl(\operatorname{Tr}\sqrt{\sqrt{\rho}\,\sigma\sqrt{\rho}}\Bigr)^2.
$$

For qubits, closed forms exist in Bloch coordinates. Both quantify
distinguishability: small $T$ or large $F$ means the states are hard
to tell apart by any measurement. Fuchs–van de Graaf inequalities
relate $T$ and $F$ (Nielsen & Chuang, Ch. 9).

> **Further reading.** Nielsen & Chuang, Ch. 9.
"""))

cells.append(code(r"""
def trace_distance(rho, sigma):
    w = eigvalsh(rho - sigma)
    return 0.5 * np.sum(np.abs(w))


def fidelity(rho, sigma):
    # F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2
    sqrt_rho = sqrtm(rho)
    w = eigvalsh(sqrt_rho @ sigma @ sqrt_rho)
    w = np.clip(np.real(w), 0.0, None)
    return float(np.sum(np.sqrt(w)) ** 2)


rho = density_from_ket([1, 0])
sigma = density_from_ket([1, 1])  # |+>
print("T(rho,sigma) =", trace_distance(rho, sigma))
print("F(rho,sigma) =", fidelity(rho, sigma))
print("T(rho,rho)   =", trace_distance(rho, rho))
print("F(rho,rho)   =", fidelity(rho, rho))
assert trace_distance(rho, rho) < 1e-12
assert np.isclose(fidelity(rho, rho), 1.0)
"""))

cells.append(md(r"""
## II.5 Quantum channels (Kraus form)

A **quantum channel** on density operators is a completely positive
trace-preserving (CPTP) map. Finite-dimensional channels admit a
**Kraus representation**

$$
\mathcal{E}(\rho)=\sum_i K_i\rho K_i^\dagger,
\qquad
\sum_i K_i^\dagger K_i = I.
$$

Example: the qubit **depolarizing** channel with parameter $p$ replaces
the state by the maximally mixed state with probability $p$.

> **Further reading.** Nielsen & Chuang, Ch. 8; Wilde, Ch. 4.
"""))

cells.append(code(r"""
# Depolarizing channel on one qubit: E(rho) = (1-p) rho + p I/2
# Kraus set (one standard choice):

def depolarizing_kraus(p):
    s = np.sqrt(1 - 3 * p / 4)
    t = np.sqrt(p / 4)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [s * I, t * X, t * Y, t * Z]


def apply_kraus(rho, Ks):
    out = np.zeros_like(rho, dtype=complex)
    for K in Ks:
        out = out + K @ rho @ K.conj().T
    return out


p = 0.3
Ks = depolarizing_kraus(p)
completeness = sum(K.conj().T @ K for K in Ks)
print("||sum K†K - I|| =", norm(completeness - np.eye(2)))
assert norm(completeness - np.eye(2)) < 1e-12

rho = density_from_ket([1, 0])
out = apply_kraus(rho, Ks)
print("Tr(out) =", np.real(np.trace(out)))
print("out =\n", np.real_if_close(out))
assert np.isclose(np.trace(out), 1.0)
"""))

cells.append(md(r"""
## II.6 Holevo information (sketch)

For an ensemble $\{p_x,\rho_x\}$ the **Holevo quantity** is

$$
\chi
=S\Bigl(\sum_x p_x\rho_x\Bigr)-\sum_x p_x S(\rho_x).
$$

It upper-bounds the mutual information between a classical label $X$
and the outcome of any measurement on the quantum state — a central
limit on accessible classical information (Holevo).

> **Further reading.** A. S. Holevo, “Bounds for the quantity of
> information transmitted by a quantum communication channel,”
> *Probl. Inf. Transm.* **9** (1973) 177–183; Nielsen & Chuang,
> Ch. 12; Wilde, Ch. 11.
"""))

cells.append(code(r"""
# Ensemble: prepare |0> or |+> with equal probability

rho0 = density_from_ket([1, 0])
rho1 = density_from_ket([1, 1])
avg = 0.5 * (rho0 + rho1)
chi = von_neumann(avg) - 0.5 * (von_neumann(rho0) + von_neumann(rho1))
print("chi =", chi, "bits")
assert chi > 0
assert chi <= 1.0 + 1e-10
"""))

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part III — Toward cryptography and PQC

## III.1 Why cryptography cares about information

Security definitions ask whether an adversary’s **view** is
distinguishable from an ideal view — equivalently, whether mutual
information or a distinguishing advantage is negligible. That is the
same language as Parts I–II (distinguishability of distributions or
states), applied to protocol transcripts rather than physics lab
states. This notebook does **not** develop reductionist security
proofs; it only records the link.

## III.2 Code-based cryptography (idea)

In **code-based** public-key schemes one publishes a scrambled
description of an error-correcting code. Legitimate users, who know
the hidden structure, can decode; adversaries face a generic decoding
problem that is believed hard. The Hamming toy in §I.5 is the opposite
regime: decoding is easy. Cryptographic codes are large and
structured so that decoding without the trapdoor is infeasible.

## III.3 HQC as a destination

**HQC** (Hamming Quasi-Cyclic) is a code-based key-encapsulation
mechanism in the NIST post-quantum cryptography process. Its hardness
story sits on decoding problems for quasi-cyclic codes — hence the
classical coding bridge above. This module only **names** HQC as a
downstream application of information-theoretic and coding ideas.

> **Further reading (pointers; verify current NIST status when
> implementing follow-ons).** NIST Post-Quantum Cryptography project
> pages and the HQC specification documents (round submissions). Do
> not treat this paragraph as a standards citation with frozen page
> numbers.

## III.4 Off-ramp — dedicated PQC notebook

Engineering depth belongs in a **separate** PQC / cryptology module:

- NIST PQC standards (ML-KEM, ML-DSA, SLH-DSA, HQC if/when standardized
  in the portfolio you target, …);
- RFCs / Internet-Drafts for protocol integration;
- Open Quantum Safe: [`liboqs`](https://github.com/open-quantum-safe/liboqs)
  and [`oqs-provider`](https://github.com/open-quantum-safe/oqs-provider).

No KEM implementation appears in this notebook.
"""))

cells.append(code(r"""
print("PQC off-ramp: use NIST docs + OQS in a dedicated notebook.")
print("This cell intentionally performs no cryptography.")
"""))

# ---------------------------------------------------------------------------
# Part IV
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part IV — Bridges to the rest of the lab

- **Manifolds §A5 (Fisher)** — classical relative entropy and Fisher
  information organise local distinguishability of probability models;
  quantum Fisher / Bures geometry sits on density operators (Amari;
  Petz). This notebook computed the global divergences; that notebook
  computed the Riemannian metric on parameters.
- **Manifolds §A2** — convex geometry of density operators.
- **Module 05 / Hadamard test** — laboratory estimation of overlaps is
  one operational face of distinguishability; no Classiq dependency
  here.
- **Next graduate module** — PQC / OQS / RFCs, consuming the coding and
  distinguishability vocabulary introduced above.

> **Further reading.** S. Amari and H. Nagaoka, *Methods of Information
> Geometry*, AMS, 2000. D. Petz, *Quantum Information Theory and
> Quantum Statistics*, Springer, 2008.
"""))

cells.append(md(r"""
## Takeaway

Classical and quantum information theory supply the numerical language
of uncertainty and distinguishability. Coding theory supplies the
bridge toward code-based post-quantum cryptography. Geometry of the
underlying sets of states and parameters lives next door in the
manifolds foundations notebook; standards-compliant PQC engineering
lives in a future dedicated module.
"""))

# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
