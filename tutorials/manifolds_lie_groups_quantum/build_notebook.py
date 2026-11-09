#!/usr/bin/env python3
"""Build the foundations notebook (run once; commit the .ipynb)."""
from __future__ import annotations

import re

import nbformat as nbf
from pathlib import Path

OUT = Path(__file__).with_name(
    "smooth_manifolds_lie_groups_and_the_exponential_map.ipynb"
)


def fix_math(source: str) -> str:
    """Use $ / $$ like the rest of openqcp-lab (MathJax-reliable)."""
    # Display math first
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
# Title / objectives
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Smooth Manifolds, Lie Groups, and the Exponential Map

Across mathematics and theoretical physics, many fundamental objects
**cannot** be treated as a single global vector space (you cannot
always add two points, or scale a point), yet we still need smooth
curves, functions, and derivatives on them. Those objects are **smooth
manifolds**. When a manifold is also a group of transformations under
composition, it is a **Lie group**, and its tangent space at the
identity — the **Lie algebra** — carries the local linear structure of
the whole group. The **exponential map** sends generators in the Lie
algebra to one-parameter subgroups in the Lie group; for matrices it
is the familiar power series.

This notebook develops that language with pictures and code, then
walks through applications that matter for quantum theory and
computation at three depths (A / B / C). It is a foundations module
for the coupled-oscillator Hamiltonian-simulation tutorials
(module 05), not a replacement for them.
"""))

cells.append(md(r"""
## Prerequisites

- Multivariable calculus (partial derivatives, chain rule).
- Linear algebra over \(\mathbb{R}\) and \(\mathbb{C}\) (matrices,
  eigenvalues, inner products).
- Undergraduate quantum mechanics: state vectors, Hermitian operators,
  unitary time evolution \(U(t)=\exp(-iHt/\hbar)\) at the level of a
  first course.

## Learning objectives

1. Explain, in plain language, what a smooth manifold is and why
   transition maps must be \(C^\infty\).
2. Construct charts on the circle and check transition smoothness in
   code.
3. State what a Lie group and its Lie algebra are, and compute the
   matrix exponential of generators in \(\mathfrak{gl}(n,\mathbb{C})\)
   and \(\mathfrak{u}(n)\).
4. Work through Tier‑A applications (Bloch sphere, projective state
   space, unitaries, symplectic Gaussians, statistical manifolds) with
   executable checks.
5. Read Tier‑B/C vignettes with enough geometry to know *what* the
   manifold is and where to go next (including a PQC/crypto off‑ramp).
"""))

cells.append(md(r"""
## How this notebook is organised

| Part | Content |
|------|---------|
| I | Charts, tangent vectors, Lie groups, \(\mathrm{GL}/\mathfrak{gl}\), exponential map |
| II‑A | Full labs: Bloch / \(\mathbb{CP}^{1}\); \(\mathbb{CP}^{d-1}\) & density operators; \(\mathrm{U}(2^n)\); \(\mathrm{Sp}(2n,\mathbb{R})\); statistical manifolds |
| II‑B | Light labs: Lorentzian Minkowski; \(U(1)\) / Berry phase |
| II‑C | Map + citations: Calabi–Yau; \(G_2\); twistors; elliptic curves & lattices (PQC off‑ramp) |
| III | Bridge to product formulas / module 05 |

**Citation rule.** Every section names textbooks or articles a reader
can open. We do not invent references.
"""))

cells.append(code(r"""
# Setup — foundations notebook (numpy / scipy / matplotlib only)
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, det, eigvalsh
from scipy.linalg import expm, logm

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (5.5, 4.0), "font.size": 11})

print("numpy", np.__version__)
"""))

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part I — Smooth manifolds and Lie groups

## 1. Why a vector space is not enough

In a vector space you can add vectors and scale them. Many spaces we
care about refuse that:

- directions in the plane (rays through the origin) — scaling a ray by
  \(2\) is the *same* ray;
- pure quantum states — \(|\psi\rangle\) and \(\lambda|\psi\rangle\)
  (\(\lambda\neq 0\)) are the same physical state;
- rotations of \(\mathbb{R}^3\) — two rotations do not “add” as vectors;
  they *compose*.

We still want to say that a curve of rotations is smooth, or that a
function on the Bloch sphere has a derivative. A **smooth manifold**
is the structure that makes that precise: the space is covered by
overlapping patches, each looking like an open set in
\(\mathbb{R}^d\), and the dictionaries between overlapping patches are
infinitely differentiable.

> **Further reading.** J. M. Lee, *Introduction to Smooth Manifolds*,
> 2nd ed., Springer, 2013, Ch. 1 (smooth manifolds and smooth maps).
> For a picture-first companion, V. Guillemin and A. Pollack,
> *Differential Topology*, AMS Chelsea, 2010 (reprint), Ch. 1.
"""))

cells.append(md(r"""
## 2. Charts, atlases, and transition maps

Let $M$ be a Hausdorff, second-countable topological space (the usual
mild assumptions so sequences and countable covers behave well). A
**chart** is a pair $(U,\varphi)$ where $U\subset M$ is open and
$\varphi:U\to\varphi(U)\subset\mathbb{R}^d$ is a continuous bijection
with continuous inverse (a homeomorphism) onto an open set of
flat Euclidean space. An **atlas** is a collection of charts whose
domains cover $M$.

Whenever two chart domains overlap, $U_\alpha\cap U_\beta\neq\emptyset$,
the **transition map**

$$
\psi_{\beta\alpha}
  \equiv \varphi_\beta\circ\varphi_\alpha^{-1}
  :\varphi_\alpha(U_\alpha\cap U_\beta)
  \longrightarrow
  \varphi_\beta(U_\alpha\cap U_\beta)
$$

translates coordinates $\mathbf{x}=(x^1,\dots,x^d)$ from one open
subset of $\mathbb{R}^d$ to another via $d$ component functions

$$
\psi_{\beta\alpha}(\mathbf{x})
  =\bigl(\psi_{\beta\alpha}^1(\mathbf{x}),\dots,
         \psi_{\beta\alpha}^d(\mathbf{x})\bigr).
$$

The atlas is **smooth** when every transition map and its inverse are
infinitely differentiable ($C^\infty$). Explicitly: for every order
$k\ge 1$, every component $i\in\{1,\dots,d\}$, and all indices
$j_1,\dots,j_k\in\{1,\dots,d\}$, the partial derivatives

$$
\frac{\partial^k \psi_{\beta\alpha}^i}
     {\partial x^{j_1}\,\partial x^{j_2}\cdots\partial x^{j_k}}
(\mathbf{x})
$$

exist and are continuous on
$\varphi_\alpha(U_\alpha\cap U_\beta)$. By the multivariable chain
rule, this is exactly what makes “$f:M\to\mathbb{R}$ is smooth” a
chart-independent statement: check $f\circ\varphi^{-1}$ in any chart.

We start with the circle $S^1=\{(x,y):x^2+y^2=1\}$, dimension $d=1$,
with two angle charts that each omit one cut point.
"""))

cells.append(code(r"""
# Two charts on S^1: angle from (-pi, pi) omitting (-1,0),
# and angle from (0, 2pi) omitting (1,0).
# Transition on the overlap should be theta |-> theta or theta+2pi.

def chart_east(xy):
    # Angle in (-pi, pi); undefined on the ray x<0, y==0 cut.
    x, y = xy
    return np.arctan2(y, x)  # (-pi, pi]

def chart_west(xy):
    # Angle in (0, 2*pi); shift so the cut is at (+1,0).
    th = np.arctan2(xy[1], xy[0])
    return np.where(th >= 0, th, th + 2 * np.pi)

# Sample the upper and lower arcs (overlap of the two charts)
t = np.linspace(0.2, np.pi - 0.2, 40)
upper = np.stack([np.cos(t), np.sin(t)], axis=1)
lower = np.stack([np.cos(-t), np.sin(-t)], axis=1)

e_u, w_u = chart_east(upper.T), chart_west(upper.T)
e_l, w_l = chart_east(lower.T), chart_west(lower.T)

# On the upper arc, west = east; on the lower, west = east + 2pi
assert np.allclose(w_u, e_u)
assert np.allclose(w_l, e_l + 2 * np.pi)

fig, ax = plt.subplots()
circ = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(circ), np.sin(circ), "k-", lw=1)
ax.plot(upper[:, 0], upper[:, 1], "C0.", label="upper overlap")
ax.plot(lower[:, 0], lower[:, 1], "C1.", label="lower overlap")
ax.set_aspect("equal")
ax.set_title(r"Two angle charts on $S^1$ (cuts at $\pm 1$)")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
plt.show()

print("Transition on upper arc: west - east =", np.max(np.abs(w_u - e_u)))
print("Transition on lower arc: west - east - 2π =",
      np.max(np.abs(w_l - e_l - 2 * np.pi)))
"""))

cells.append(md(r"""
**Takeaway box — what is the manifold here?**  
\(M=S^1\), dimension \(1\). Charts: open arcs with angle coordinates.
Transition maps are \(\theta\mapsto\theta\) or \(\theta\mapsto\theta+2\pi\),
both \(C^\infty\) on their domains. Smoothness of functions on the
circle does not depend on which chart we use.
"""))

cells.append(md(r"""
## 3. Tangent vectors and the exponential map (geometric idea)

Around a point $p\in M$, every chart makes $M$ look like an open set
in $\mathbb{R}^d$. A smooth curve $\gamma:(-\varepsilon,\varepsilon)\to M$
with $\gamma(0)=p$ has an **initial velocity**
$\gamma'(0)\in T_p M$. The **tangent space** $T_p M$ is the real
vector space of all such initial velocities (dimension $d$).

Given a consistent rule that extends each $v\in T_p M$ to a preferred
smooth curve $\gamma_v$ with $\gamma_v(0)=p$ and $\gamma_v'(0)=v$
(geodesic, one-parameter subgroup, …), the **exponential map** at $p$
is

$$
\operatorname{exp}_p:T_p M\supset\mathcal{U}\to M,\qquad
\operatorname{exp}_p(v)=\gamma_v(1),
$$

i.e. travel for one unit of time along that preferred curve. We make
this completely concrete for **matrix Lie groups**, where the preferred
curves are one-parameter subgroups and $\operatorname{exp}$ is the
matrix power series.
"""))

cells.append(md(r"""
## 4. Lie groups and Lie algebras

A **Lie group** $G$ is a smooth manifold that is also a group, with
smooth multiplication $G\times G\to G$ and smooth inversion
$G\to G$. For any fixed transformation $T\in G$, left translation

$$
g\;\longmapsto\; T^{-1}g
$$

is a smooth diffeomorphism sending a neighbourhood of $T$ to a
neighbourhood of the identity $\mathbf{I}$. Therefore the whole group's
local linear structure is encoded in the single vector space
$T_{\mathbf{I}}G$: the **Lie algebra** $\mathfrak{g}=T_{\mathbf{I}}G$.

For matrix groups the Lie **bracket** is the commutator

$$
[\mathbf{X},\mathbf{Y}]=\mathbf{X}\mathbf{Y}-\mathbf{Y}\mathbf{X},
$$

which vanishes if and only if $\mathbf{X}$ and $\mathbf{Y}$ commute.

**One-parameter subgroups** are smooth group homomorphisms
$\gamma:\mathbb{R}\to G$,

$$
\gamma(t+s)=\gamma(t)\,\gamma(s),\qquad \gamma(0)=\mathbf{I}.
$$

Writing $\mathbf{X}=\gamma'(0)\in\mathfrak{g}$, the exponential map of
the Lie group recovers the subgroup as $\gamma(t)=\exp(t\mathbf{X})$.

> **Further reading.** B. C. Hall, *Lie Groups, Lie Algebras, and
> Representations*, 2nd ed., Springer, 2015, Ch. 1–3 (matrix Lie
> groups, matrix exponential, Lie algebras). Hall's notes that became
> the book are also on arXiv: [math-ph/0005032](https://arxiv.org/abs/math-ph/0005032).
"""))

cells.append(md(r"""
## 5. The general linear group $\mathrm{GL}(n,\mathbb{C})$

$\mathrm{GL}(n,\mathbb{C})$ is the set of all invertible
$n\times n$ complex matrices. Matrix multiplication and inversion are
smooth operations on that set, so $\mathrm{GL}(n,\mathbb{C})$ is a Lie
group. It sits as an **open** subset of the flat vector space
$M_n(\mathbb{C})\cong\mathbb{C}^{n^2}$ of *all* matrices, because the
determinant is continuous and
$\mathrm{GL}(n,\mathbb{C})=\{A:\det A\neq 0\}$.

Its Lie algebra $\mathfrak{gl}(n,\mathbb{C})$ is the vector space of
**all** $n\times n$ complex matrices (no invertibility requirement).
In quantum mechanics, Hamiltonians — Hermitian or not, including those
with zero eigenvalues — live in this Lie algebra. Their time evolution
operators are obtained by the exponential map into the Lie group.

Because $\mathrm{GL}$ is open in a flat matrix space, any smooth curve
$\gamma:\mathbb{R}\to\mathrm{GL}(n,\mathbb{C})$ can be differentiated
entry by entry. The exponential map is defined by a curve satisfying

$$
\exp(t\mathbf{X})=\gamma(t),\qquad
\gamma(0)=\mathbf{I},\qquad
\left.\frac{\mathrm{d}\gamma}{\mathrm{d}t}\right|_{t=0}=\mathbf{X}.
$$

A one-parameter subgroup also obeys the homomorphism rule

$$
\gamma(t+s)=\gamma(t)\cdot\gamma(s).
$$

Differentiating both sides (entry by entry) gives

$$
\mathrm{d}\gamma(t+s)
  =\gamma'(t+s)\,(\mathrm{d}t+\mathrm{d}s)
  =\gamma'(t)\cdot\gamma(s)\,\mathrm{d}t
   +\gamma(t)\cdot\gamma'(s)\,\mathrm{d}s.
$$

Holding $t+s$ fixed ($\mathrm{d}t+\mathrm{d}s=0$) forces
$\gamma'(t)\cdot\gamma(s)=\gamma(t)\cdot\gamma'(s)$. Setting $s=0$
and writing $\mathbf{X}=\gamma'(0)$ yields the matrix ODE

$$
\frac{\mathrm{d}\gamma(t)}{\mathrm{d}t}
  =\gamma(t)\cdot\mathbf{X},\qquad
\gamma(0)=\mathbf{I}.
$$

For matrices, the unique entire solution is the power series
(always convergent in any matrix norm)

$$
\exp(t\mathbf{X})
  \equiv\gamma(t)
  =\sum_{m=0}^{\infty}\frac{t^m}{m!}\,\mathbf{X}^m. \tag{1}
$$

Useful identities that follow from the series (or the ODE):

$$
\exp(\mathbf{0})=\mathbf{I},\qquad
\exp(t\mathbf{X})^{-1}=\exp(-t\mathbf{X}),\qquad
\frac{\mathrm{d}}{\mathrm{d}t}\exp(t\mathbf{X})
  =\mathbf{X}\,\exp(t\mathbf{X})
  =\exp(t\mathbf{X})\,\mathbf{X}.
$$

If $\mathbf{A}$ and $\mathbf{B}$ commute ($\mathbf{A}\mathbf{B}=\mathbf{B}\mathbf{A}$),
then $\exp(\mathbf{A}+\mathbf{B})=\exp(\mathbf{A})\exp(\mathbf{B})$.
When they do **not** commute, the product formula / BCH expansion
corrects the difference — that is the bridge to Trotter in Part III.
"""))

cells.append(code(r"""
# Matrix exponential: series vs scipy, and the ODE check d/dt exp(tX) = exp(tX) X

def expm_series(X, terms=40):
    X = np.asarray(X, dtype=complex)
    out = np.zeros_like(X, dtype=complex)
    term = np.eye(X.shape[0], dtype=complex)
    for m in range(terms):
        out += term
        term = term @ X / (m + 1)
    return out

rng = np.random.default_rng(0)
X = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
t = 0.7
E = expm(t * X)
E_ser = expm_series(t * X)
print("||expm - series|| =", norm(E - E_ser))

# Finite-difference ODE check: (exp((t+h)X) - exp(tX))/h ≈ exp(tX) @ X
h = 1e-6
lhs = (expm((t + h) * X) - expm(t * X)) / h
rhs = expm(t * X) @ X
print("||ODE residual|| =", norm(lhs - rhs))

# Invertibility: exp(X) is always invertible, inverse exp(-X)
print("||exp(X) exp(-X) - I|| =", norm(expm(X) @ expm(-X) - np.eye(3)))
"""))

cells.append(md(r"""
**Takeaway box — \(\mathrm{GL}\) / \(\mathfrak{gl}\).**  
Manifold / Lie group: invertible matrices. Lie algebra: all matrices.
Exponential: series (1), always defined; solves the one-parameter
subgroup ODE. Hamiltonians sit in the algebra; time evolution sits in
the group when the exponential is unitary (next section) or, more
generally, invertible.
"""))

# ---------------------------------------------------------------------------
# Part II-A
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II‑A — Application labs (full code)

> **Further reading (QI geometry).** I. Bengtsson and K. Życzkowski,
> *Geometry of Quantum States: An Introduction to Quantum
> Entanglement*, 2nd ed., Cambridge University Press, 2017. Chapter 4
> treats complex projective spaces; the Bloch sphere is \(\mathbb{CP}^1\).
"""))

cells.append(md(r"""
## A1. The Bloch sphere ($\mathbb{CP}^{1}$)

A pure qubit state is a unit vector $|\psi\rangle\in\mathbb{C}^2$ up to
a nonzero complex factor: $|\psi\rangle\sim\lambda|\psi\rangle$ for
$\lambda\in\mathbb{C}\setminus\{0\}$. The space of such rays is the
complex projective line $\mathbb{CP}^{1}$. As a real smooth manifold,

$$
\mathbb{CP}^{1}\;\cong\; S^2
$$

— the **Bloch sphere**. In the computational basis one may write

$$
|\psi\rangle
  =\cos\frac{\theta}{2}\,|0\rangle
  +e^{i\phi}\sin\frac{\theta}{2}\,|1\rangle,
\qquad
\theta\in[0,\pi],\quad \phi\in[0,2\pi),
$$

and the Bloch vector

$$
\mathbf{r}
  =(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta)
  \in S^2\subset\mathbb{R}^3
$$

satisfies $r_i=\langle\psi|\sigma_i|\psi\rangle$ with Pauli matrices
$\sigma_x,\sigma_y,\sigma_z$.

A Hamiltonian $H=\mathbf{n}\cdot\boldsymbol{\sigma}$ with
$|\mathbf{n}|=1$ generates the unitary one-parameter group

$$
U(t)=\exp(-i t H)\in\mathrm{U}(2),
$$

which rotates $\mathbf{r}(t)$ about the axis $\mathbf{n}$ at angular
speed $2$ (in units $\hbar=1$).
"""))

cells.append(code(r"""
# Bloch sphere: state <-> r, and precession under H = n · σ

PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def ket_from_bloch(theta, phi):
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
        dtype=complex,
    )

def bloch_from_ket(psi):
    psi = psi / norm(psi)
    # remove global phase so <0|psi> is real nonnegative when possible
    if abs(psi[0]) > 1e-14:
        psi = psi * np.exp(-1j * np.angle(psi[0]))
    x = 2 * np.real(np.conj(psi[0]) * psi[1])
    y = 2 * np.imag(np.conj(psi[0]) * psi[1])
    z = np.abs(psi[0]) ** 2 - np.abs(psi[1]) ** 2
    return np.array([x, y, z])

def hamiltonian(n):
    n = np.asarray(n, dtype=float)
    n = n / norm(n)
    return n[0] * PAULI["X"] + n[1] * PAULI["Y"] + n[2] * PAULI["Z"]

theta0, phi0 = 0.7, 0.3
r0 = bloch_from_ket(ket_from_bloch(theta0, phi0))
n_axis = np.array([0.0, 0.0, 1.0])  # rotate about z
H = hamiltonian(n_axis)

ts = np.linspace(0, 2 * np.pi, 80)
path = []
for t in ts:
    psi_t = expm(-1j * t * H) @ ket_from_bloch(theta0, phi0)
    path.append(bloch_from_ket(psi_t))
path = np.array(path)

# Analytic: rotation about z leaves z fixed, rotates (x,y)
assert np.allclose(path[:, 2], r0[2], atol=1e-8)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
ax.plot_surface(
    np.cos(u) * np.sin(v),
    np.sin(u) * np.sin(v),
    np.cos(v),
    alpha=0.08,
    color="gray",
)
ax.plot(path[:, 0], path[:, 1], path[:, 2], "C0-", lw=2, label="orbit")
ax.scatter(*r0, color="C3", s=40, label="start")
ax.set_title(r"Bloch precession under $H=Z$")
ax.legend(fontsize=9)
plt.show()

print("start Bloch r =", r0)
print("||r(t)|| stays 1: max deviation",
      np.max(np.abs(norm(path, axis=1) - 1)))
"""))

cells.append(md(r"""
**Takeaway box — A1.**  
Manifold: \(\mathbb{CP}^{1}\cong S^2\) (pure qubit states). Lie group
acting: \(\mathrm{U}(2)\) (or \(\mathrm{SU}(2)\)). Lie algebra:
anti‑Hermitian matrices \(i\mathfrak{su}(2)\) / Hermitian Hamiltonians.
Exponential: \(U(t)=\exp(-iHt)\).
"""))

cells.append(md(r"""
## A2. \(\mathbb{CP}^{d-1}\) and density operators

Pure states of a \(d\)-level system are points of
\(\mathbb{CP}^{d-1}\). Mixed states are density operators
\(\rho=\rho^\dagger\geq 0\), \(\operatorname{Tr}\rho=1\). The set of
density operators is a **convex body** in the real vector space of
Hermitian matrices of trace one; its extreme points are the pure
states (rank‑one projectors). The relative interior of fixed-rank
strata are smooth manifolds; the full set has corners (a manifold
*with corners* / stratified space). We check the geometry numerically
for a qubit (Bloch ball) and a qutrit purity slice.
"""))

cells.append(code(r"""
# Density operators: Bloch ball (qubit) and purity for random qutrit states

def random_density(d, rng, pure=False):
    if pure:
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        v /= norm(v)
        return np.outer(v, np.conj(v))
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)

rng = np.random.default_rng(1)
bloch_pts = []
for _ in range(400):
    rho = random_density(2, rng)
    # r_i = Tr(rho sigma_i)
    r = np.array(
        [np.real(np.trace(rho @ PAULI[s])) for s in ("X", "Y", "Z")]
    )
    bloch_pts.append(r)
bloch_pts = np.array(bloch_pts)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    bloch_pts[:, 0], bloch_pts[:, 1], bloch_pts[:, 2],
    s=8, alpha=0.5, c=norm(bloch_pts, axis=1), cmap="viridis",
)
ax1.set_title("Random qubit states (Bloch ball)")
ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)

pur_mixed = []
pur_pure = []
for _ in range(300):
    rm = random_density(3, rng, pure=False)
    rp = random_density(3, rng, pure=True)
    pur_mixed.append(np.real(np.trace(rm @ rm)))
    pur_pure.append(np.real(np.trace(rp @ rp)))

ax2 = fig.add_subplot(122)
ax2.hist(pur_mixed, bins=30, alpha=0.7, label="mixed (Wishart)")
ax2.axvline(1.0, color="C1", ls="--", lw=2, label="pure (= 1)")
ax2.set_xlabel(r"purity $\mathrm{Tr}(\rho^2)$")
ax2.set_title("Qutrit purity")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.show()

assert np.all(norm(bloch_pts, axis=1) <= 1 + 1e-10)
assert np.allclose(pur_pure, 1.0)
print("max |r| for mixed qubits:", np.max(norm(bloch_pts, axis=1)))
print("pure qutrit purity min/max:", np.min(pur_pure), np.max(pur_pure))
print("mixed qutrit purity range:", np.min(pur_mixed), np.max(pur_mixed))
"""))

cells.append(md(r"""
**Takeaway box — A2.**  
Pure states: manifold \(\mathbb{CP}^{d-1}\). Mixed states: convex set
of density operators (stratified by rank). Geometry of distinguishability
(Bures, quantum Fisher) lives on the interior of full-rank states —
see Bengtsson–Życzkowski for the Riemannian picture.
"""))

cells.append(md(r"""
## A3. Unitary gates \(\mathrm{U}(n)\) and Hermitian generators

\(\mathrm{U}(n)=\{U:U^\dagger U=I\}\) is a compact Lie group. Its Lie
algebra consists of **anti‑Hermitian** matrices \(X^\dagger=-X\). Writing
\(X=-iH\) with \(H=H^\dagger\), every one-parameter subgroup is
\(U(t)=\exp(-i t H)\). Quantum computation's gate set sits here
(\(n=2^k\) for \(k\) qubits).
"""))

cells.append(code(r"""
# U(n): exp(-iH) is unitary when H is Hermitian; dimension check

def random_hermitian(n, rng):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / 2

rng = np.random.default_rng(2)
H = random_hermitian(4, rng)
U = expm(-1j * H)
print("||U^† U - I|| =", norm(U.conj().T @ U - np.eye(4)))
print("|det U| =", abs(det(U)))

# Real dimension of U(n) is n^2; of su(n) (traceless anti-Hermitian) is n^2-1
n = 4
# sample: n^2 independent real parameters for Hermitian H
print(f"real dim of u({n}) / Hermitian generators: {n * n}")
print(f"real dim of su({n}) (traceless): {n * n - 1}")
"""))

cells.append(md(r"""
**Takeaway box — A3.**  
Lie group: \(\mathrm{U}(n)\) (quantum gates). Lie algebra: \(i\) times
Hermitian matrices. Exponential: time evolution / gate synthesis.
"""))

cells.append(md(r"""
## A4. The symplectic group \(\mathrm{Sp}(2n,\mathbb{R})\)

Phase space \(\mathbb{R}^{2n}\) carries the standard symplectic matrix
\(J\) (\(J^\top=-J\), \(J^2=-I\)). A real matrix \(S\) is **symplectic**
when \(S^\top J S=J\). These transformations are the linear canonical
transformations of classical mechanics and the Gaussian (linear optics)
transformations of continuous-variable quantum information.

> **Further reading.** M. de Gosson, *Symplectic Geometry and Quantum
> Mechanics*, Birkhäuser, 2006, Ch. 2 (the symplectic group).
"""))

cells.append(code(r"""
# Sp(2,R) ≅ SL(2,R): check S.T @ J @ S = J, and act on a covariance ellipse

def J_matrix(n):
    J = np.zeros((2 * n, 2 * n))
    for i in range(n):
        J[2 * i, 2 * i + 1] = 1
        J[2 * i + 1, 2 * i] = -1
    return J

J = J_matrix(1)
# A one-parameter subgroup: squeezing / rotation generators in sp(2,R)
# K_rot = [[0,-1],[1,0]] generates rotations; K_sq = [[0,1],[1,0]] squeeze

def sp_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def sp_squeeze(r):
    return np.array([[np.exp(r), 0.0], [0.0, np.exp(-r)]])

for name, S in [("rot", sp_rot(0.4)), ("sq", sp_squeeze(0.5)),
                ("prod", sp_rot(0.3) @ sp_squeeze(0.2))]:
    err = norm(S.T @ J @ S - J)
    print(f"{name}: ||S^T J S - J|| = {err:.2e}, det={det(S):.6f}")

# Covariance ellipse of a Gaussian: Σ = S S^T starting from identity
S = sp_rot(0.5) @ sp_squeeze(0.7)
Sigma = S @ S.T
# Williamson / uncertainty: Σ + i (ħ/2) J ≥ 0 with ħ=1 → Σ + i J/2 ≥ 0
W = Sigma + 1j * J / 2
evals = eigvalsh(W)
print("eigenvalues of Σ + i J/2 (should be >= 0):", np.real(evals))

fig, ax = plt.subplots()
ts = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(ts), np.sin(ts)], axis=0)
ellipse = S @ circle
ax.plot(circle[0], circle[1], "k--", lw=1, label="vacuum")
ax.plot(ellipse[0], ellipse[1], "C0-", lw=2, label="S · vacuum")
ax.set_aspect("equal")
ax.set_title(r"Symplectic image of the unit circle in $\mathbb{R}^2$")
ax.legend(fontsize=9)
plt.show()
"""))

cells.append(md(r"""
**Takeaway box — A4.**  
Lie group: \(\mathrm{Sp}(2n,\mathbb{R})\) (linear canonical /
Gaussian CV gates). Lie algebra: \(\mathfrak{sp}(2n,\mathbb{R})\)
(Hamiltonian matrices for quadratic forms). Exponential: Gaussian
unitaries' classical shadow.
"""))

cells.append(md(r"""
## A5. Statistical manifolds and the Fisher metric

A smooth family of probability densities \(\{p_\theta\}\) parametrised
by \(\theta\in\Theta\subset\mathbb{R}^m\) is a **statistical manifold**.
The **Fisher information matrix**

\[
g_{ij}(\theta)
  =\mathbb{E}_\theta
   \bigl[\partial_i\log p_\theta\;\partial_j\log p_\theta\bigr]
\]

is a Riemannian metric on \(\Theta\) (Rao, 1945). Geodesics and
distances derived from \(g\) organise asymptotic statistics and, in
quantum form, metrology.

> **Further reading.** C. R. Rao, “Information and the accuracy
> attainable in the estimation of statistical parameters,” *Bull.
> Calcutta Math. Soc.* **37** (1945) 81–91. S. Amari,
> *Differential-Geometrical Methods in Statistics*, Springer, 1985.
> S. Amari and H. Nagaoka, *Methods of Information Geometry*, AMS,
> 2000 (Ch. 2–3).
"""))

cells.append(code(r"""
# Bernoulli(p) as a 1-D statistical manifold; Fisher metric g(p)=1/(p(1-p))

def fisher_bernoulli(p):
    return 1.0 / (p * (1.0 - p))

ps = np.linspace(0.05, 0.95, 200)
g = fisher_bernoulli(ps)

# Fisher arc length from 0.2 to 0.8: integral sqrt(g) dp = Delta arcosh-like
# For Bernoulli, the Fisher distance is 2|arcsin(sqrt(p1))-arcsin(sqrt(p0))|
def fisher_dist_bernoulli(p0, p1):
    return 2 * np.abs(np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))

p0, p1 = 0.2, 0.8
# Numerical integrate sqrt(g)
integ = np.trapezoid(np.sqrt(fisher_bernoulli(np.linspace(p0, p1, 2000))),
                     np.linspace(p0, p1, 2000))
analytic = fisher_dist_bernoulli(p0, p1)
print("Fisher distance 0.2→0.8 numeric:", integ)
print("Fisher distance analytic:      ", analytic)

fig, ax = plt.subplots()
ax.plot(ps, g, "C0-", lw=2)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$g(p)=1/(p(1-p))$")
ax.set_title("Fisher metric on the Bernoulli family")
plt.show()
"""))

cells.append(md(r"""
**Takeaway box — A5.**  
Manifold: parameter space of a smooth statistical model. Metric:
Fisher–Rao. Quantum thermodynamics and quantum estimation use the
quantum Fisher information on density-operator manifolds (Amari–Nagaoka,
Ch. on quantum systems).
"""))

# ---------------------------------------------------------------------------
# Part II-B
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II‑B — Light labs

## B1. Minkowski spacetime as a smooth manifold

\(\mathbb{R}^{1,3}\) with metric \(\eta=\mathrm{diag}(-1,1,1,1)\) (or
the opposite sign convention) is a smooth manifold diffeomorphic to
ordinary \(\mathbb{R}^4\), but the **metric** distinguishes timelike,
null, and spacelike vectors. Lorentz transformations are the Lie group
\(\mathrm{O}(1,3)\) preserving \(\eta\).

> **Further reading.** S. Carroll, *Spacetime and Geometry*, Addison
> Wesley / Cambridge reprint, Ch. 1–2. R. Wald, *General Relativity*,
> Chicago, 1984, Ch. 1–2 (for the manifold + metric viewpoint).
"""))

cells.append(code(r"""
# Light cone in 1+1 Minkowski space

eta = np.diag([-1.0, 1.0])
xs = np.linspace(-2, 2, 400)
fig, ax = plt.subplots()
ax.fill_between(xs, xs, -xs, where=(xs >= 0), alpha=0.08, color="C0")
ax.fill_between(xs, xs, -xs, where=(xs <= 0), alpha=0.08, color="C0")
ax.plot(xs, xs, "k-", lw=1, label=r"null $t=\pm x$")
ax.plot(xs, -xs, "k-", lw=1)
ax.scatter([0], [0], c="C3", zorder=5)
ax.set_aspect("equal")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.set_title(r"Minkowski plane $\mathbb{R}^{1,1}$: light cone")
ax.legend(fontsize=9)
plt.show()

# Classify a few vectors
def kind(v):
    s = v @ eta @ v
    if abs(s) < 1e-12:
        return "null"
    return "timelike" if s < 0 else "spacelike"

for v in [(1, 0), (0, 1), (1, 1), (1, 2)]:
    print(v, kind(np.array(v, dtype=float)))
"""))

cells.append(md(r"""
**Takeaway box — B1.**  
Manifold: spacetime (here flat Minkowski). Extra structure: Lorentzian
metric. Isometry Lie group: Lorentz / Poincaré.
"""))

cells.append(md(r"""
## B2. \(U(1)\) bundles and Berry phase (sketch with code)

A **principal \(U(1)\) bundle** over a base \(M\) assigns a circle's
worth of phases to each point of \(M\), glued consistently. In quantum
mechanics, a normalised eigenstate \(|n(\mathbf{R})\rangle\) depending
on slow parameters \(\mathbf{R}\) defines a line bundle; transporting
the state around a closed loop in \(\mathbf{R}\)-space accumulates a
geometric **Berry phase**.

For a spin‑1/2 in a magnetic field \(\mathbf{B}=B\,\hat{\mathbf{n}}\),
a loop of \(\hat{\mathbf{n}}\) on the Bloch sphere enclosing solid
angle \(\Omega\) gives Berry phase \(-\Omega/2\) for the ground state.

> **Further reading.** M. Nakahara, *Geometry, Topology and Physics*,
> 2nd ed., Taylor & Francis, 2003, Ch. 9–10 (bundles, connections,
> Berry phase §10.6). M. V. Berry, “Quantal phase factors accompanying
> adiabatic changes,” *Proc. R. Soc. Lond. A* **392** (1984) 45–57.
"""))

cells.append(code(r"""
# Berry phase for spin-1/2: loop at fixed latitude theta on the Bloch sphere
# Analytic: gamma = -pi (1 - cos(theta))  [= -Omega/2, Omega = 2pi(1-cos theta)]

def ground_state(theta, phi):
    # H ~ n · sigma; ground state for spin along -n if we take energy -|B|
    # Standard: |n-; lower> = (sin(theta/2) e^{-i phi}, -cos(theta/2)) up to phase
    return np.array(
        [np.sin(theta / 2) * np.exp(-1j * phi), -np.cos(theta / 2)],
        dtype=complex,
    )

def berry_phase_latitude(theta, n_steps=500):
    phis = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    states = np.stack([ground_state(theta, phi) for phi in phis], axis=0)
    # Remove discontinuous chart jumps by fixing local phase continuity
    for k in range(1, n_steps):
        overlap = np.vdot(states[k - 1], states[k])
        if abs(overlap) > 0:
            states[k] *= np.exp(-1j * np.angle(overlap))
    # Close the loop
    total = 1.0 + 0j
    for k in range(n_steps):
        total *= np.vdot(states[k], states[(k + 1) % n_steps])
    # After phase alignment, open-path product ≈ 1; geometric phase from closure:
    closure = np.vdot(states[-1], states[0])
    # Standard discrete Berry: arg of product of overlaps around the loop
    prod = 1.0 + 0j
    for k in range(n_steps):
        prod *= np.vdot(states[k], states[(k + 1) % n_steps])
    return np.angle(prod)

theta = np.pi / 3
gamma_num = berry_phase_latitude(theta)
gamma_an = -np.pi * (1 - np.cos(theta))
print("Berry phase numeric:", gamma_num)
print("Berry phase analytic -π(1-cos θ):", gamma_an)
print("abs error:", abs(((gamma_num - gamma_an + np.pi) % (2 * np.pi)) - np.pi))
"""))

cells.append(md(r"""
**Takeaway box — B2.**  
Base manifold: parameter space (here \(S^2\) of field directions).
Bundle: \(U(1)\) phases of eigenstates. Connection: Berry connection;
curvature → solid angle / monopole.
"""))

# ---------------------------------------------------------------------------
# Part II-C
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II‑C — Map, schematics, and reading paths

These topics are too rich for a full lab here. Each subsection states
**what the manifold is**, why physicists or cryptographers care, and
where to read. Dedicated follow‑on notebooks can graduate from these
entries.
"""))

cells.append(md(r"""
## C1. Calabi–Yau manifolds

A **Calabi–Yau** \(m\)-fold is (in the physicist's working sense) a
compact Kähler manifold with vanishing first Chern class, equivalently
one admitting a Ricci‑flat Kähler metric with holonomy in
\(\mathrm{SU}(m)\) (Yau's solution of the Calabi conjecture). In
string theory, the extra spatial dimensions are often modelled as a
Calabi–Yau threefold; the topology controls the low‑energy particle
spectrum.

We do **not** attempt to construct Ricci‑flat metrics in code here.

> **Further reading.** D. Joyce, *Compact Manifolds with Special
> Holonomy*, OUP, 2000, Ch. on Calabi–Yau. S.‑T. Yau, “On the Ricci
> curvature of a compact Kähler manifold and the complex Monge–Ampère
> equation, I,” *Comm. Pure Appl. Math.* **31** (1978) 339–411.
> M. Gross, D. Huybrechts, D. Joyce, *Calabi–Yau Manifolds and Related
> Geometries*, Springer, 2003 (survey lectures).

**Off‑ramp:** a future “string compactifications / CY” notebook.
"""))

cells.append(md(r"""
## C2. \(G_2\) manifolds

\(G_2\) is an exceptional Lie group; a Riemannian 7‑manifold with
holonomy \(G_2\) is Ricci‑flat and carries a parallel 3‑form. Such
spaces appear in M‑theory compactifications. Compact examples were
constructed by Joyce in the 1990s.

> **Further reading.** Joyce, *Compact Manifolds with Special
> Holonomy*, OUP, 2000, Part II (\(G_2\) and \(\mathrm{Spin}(7)\)).

**Off‑ramp:** graduate with the CY notebook or a dedicated exceptional
holonomy note.
"""))

cells.append(md(r"""
## C3. Twistor space \(\mathbb{CP}^{3}\)

**Twistor theory** (Penrose) encodes points and null structure of
Minkowski spacetime in complex projective geometry. A basic dictionary:
points of \(\mathbb{CP}^{3}\) correspond to null lines in (complexified)
Minkowski space; spacetime points correspond to lines
(\(\mathbb{CP}^{1}\)s) in twistor space.

> **Further reading.** R. Penrose, “Twistor algebra,” *J. Math. Phys.*
> **8** (1967) 345–366. R. Penrose and W. Rindler, *Spinors and
> Space‑Time*, Vol. 2, Cambridge, 1986, Ch. 6.

**Off‑ramp:** a dedicated twistor notebook (geometry first, then field
equations).
"""))

cells.append(md(r"""
## C4. Elliptic curves, lattices, and PQC (off‑ramp)

An **elliptic curve** over a field (e.g. \(\mathbb{F}_p\)) is a smooth
genus‑one curve with a chosen point; its points form an abelian group
used in elliptic‑curve cryptography. Modularally, isomorphism classes
of elliptic curves over \(\mathbb{C}\) are parametrised by the quotient
of the upper half‑plane by \(\mathrm{SL}_2(\mathbb{Z})\) — a
foundational orbifold / stacky manifold picture.

**Lattices** \(\Lambda\subset\mathbb{R}^n\) are discrete subgroups; the
quotient tori \(\mathbb{R}^n/\Lambda\) are smooth manifolds, and hard
problems on lattices underpin much of **post‑quantum cryptography
(PQC)**.

This foundations notebook stops at the geometric invitation. A
**dedicated PQC / cryptology notebook** should use:

- NIST PQC standards (ML‑KEM, ML‑DSA, SLH‑DSA, …);
- **RFCs** and Internet‑Drafts for protocol integration (e.g. hybrid
  key exchange in TLS 1.3 — see RFC 9954 and related documents);
- the **Open Quantum Safe** stack:
  [`liboqs`](https://github.com/open-quantum-safe/liboqs) and
  [`oqs-provider`](https://github.com/open-quantum-safe/oqs-provider).

> **Further reading (geometry / ECC).** J. H. Silverman, *The
> Arithmetic of Elliptic Curves*, 2nd ed., Springer, 2009. J. H.
> Silverman, *An Introduction to Lattices, Lattice Reduction, and
> Lattice‑Based Cryptography*, PCMI lecture notes.
"""))

cells.append(code(r"""
# Tiny schematic: group law sketch on y^2 = x^3 + a x + b over the reals
# (illustrative only — not a crypto implementation)

a, b = -1.0, 1.0
xs = np.linspace(-2, 2, 400)
ysq = xs ** 3 + a * xs + b
mask = ysq >= 0
fig, ax = plt.subplots()
ax.plot(xs[mask], np.sqrt(ysq[mask]), "C0-", lw=2)
ax.plot(xs[mask], -np.sqrt(ysq[mask]), "C0-", lw=2)
# chord-and-tangent cartoon: three colinear points
ax.set_title(r"Real locus of $y^2=x^3-x+1$ (schematic)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")
plt.show()
print("Crypto/PQC implementations: use RFCs + OQS, not this plot.")
"""))

cells.append(md(r"""
**Takeaway box — C4.**  
Manifolds / quotients: elliptic curves (as groups/manifolds), modular
curves, lattice tori. Engineering depth: **separate** PQC notebook with
OQS + RFCs.
"""))

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part III — Bridge to product formulas (module 05)

Hamiltonian simulation writes an evolution operator as a **product of
simpler exponentials**. That only makes sense once $\exp(t\mathbf{X})$
is defined on a matrix Lie algebra and lands in a matrix Lie group —
Part I.

If $[\mathbf{A},\mathbf{B}]=0$, then
$\exp(t(\mathbf{A}+\mathbf{B}))=\exp(t\mathbf{A})\exp(t\mathbf{B})$.
If not, the **Lie product formula** still recovers the sum in the limit

$$
\exp\bigl(t(\mathbf{A}+\mathbf{B})\bigr)
  =\lim_{r\to\infty}
   \Bigl(\exp\frac{t\mathbf{A}}{r}\,\exp\frac{t\mathbf{B}}{r}\Bigr)^{r}.
$$

Finite $r$ gives a Trotter step of order $O(t^2/r)$ (and higher-order
Suzuki formulae improve the power of $1/r$). Rigorous remainder bounds
in terms of nested commutators are in Childs et al., *Phys. Rev. X*
**11**, 011020 (2021).

In the coupled-oscillator notebooks, §5.1 of the $N=2$ tutorial keeps a
*thin* reminder of the exponential map and then moves to Pauli
decomposition and Trotter. This foundations notebook is the place for
the geometry; module 05 is the place for the simulation pipeline.

> **Further reading.** Hall (BCH chapter). M. Suzuki, “General theory
> of fractal path integrals…,” *J. Math. Phys.* **32** (1991) 400–407.
> A. M. Childs et al., “Theory of Trotter error with commutator
> scaling,” *Phys. Rev. X* **11** (2021) 011020,
> [arXiv:1912.08854](https://arxiv.org/abs/1912.08854).
"""))

cells.append(code(r"""
# Lie–Trotter sanity check on 2x2 matrices: ||exp(t(A+B)) - (exp(tA/r) exp(tB/r))^r||

rng = np.random.default_rng(3)
A = random_hermitian(2, rng)
B = random_hermitian(2, rng)
t = 1.0
exact = expm(-1j * t * (A + B))
errs = []
for r in [1, 2, 4, 8, 16, 32]:
    step = expm(-1j * (t / r) * A) @ expm(-1j * (t / r) * B)
    approx = np.linalg.matrix_power(step, r)
    errs.append(norm(exact - approx))
print("r: error")
for r, e in zip([1, 2, 4, 8, 16, 32], errs):
    print(f"{r:2d}: {e:.4e}")

plt.figure()
plt.loglog([1, 2, 4, 8, 16, 32], errs, "o-", lw=2)
plt.xlabel("r (Trotter steps)")
plt.ylabel(r"$\|U - U_r\|$")
plt.title("Lie product formula error (2×2 Hermitian example)")
plt.grid(True, which="both", alpha=0.3)
plt.show()
"""))

cells.append(md(r"""
## Summary

- Smooth manifolds let us do calculus where global addition fails;
  charts + \(C^\infty\) transitions make derivatives chart‑independent
  (Lee).
- Lie groups package manifolds of transformations; Lie algebras are
  their infinitesimal generators; the exponential map builds
  one‑parameter subgroups (Hall).
- Quantum information, Gaussian CV, and statistics supply Tier‑A
  manifolds you can plot and test in this notebook
  (Bengtsson–Życzkowski, de Gosson, Amari).
- Relativity and bundles appear at Tier‑B (Carroll/Wald, Nakahara,
  Berry).
- CY, \(G_2\), twistors, and crypto appear at Tier‑C as maps with
  citations; **PQC belongs in a dedicated RFC + OQS notebook**.

## Global further reading (bookshelf)

1. Lee, *Introduction to Smooth Manifolds*, 2nd ed., Springer, 2013.
2. Hall, *Lie Groups, Lie Algebras, and Representations*, 2nd ed.,
   Springer, 2015 (arXiv:math‑ph/0005032).
3. Bengtsson & Życzkowski, *Geometry of Quantum States*, 2nd ed., CUP,
   2017.
4. de Gosson, *Symplectic Geometry and Quantum Mechanics*, Birkhäuser,
   2006.
5. Amari & Nagaoka, *Methods of Information Geometry*, AMS, 2000.
6. Nakahara, *Geometry, Topology and Physics*, 2nd ed., 2003.
7. Joyce, *Compact Manifolds with Special Holonomy*, OUP, 2000.
8. Penrose & Rindler, *Spinors and Space‑Time*, Vol. 2, CUP, 1986.
9. Silverman, *The Arithmetic of Elliptic Curves*, 2nd ed., Springer,
   2009.
10. Open Quantum Safe: https://openquantumsafe.org/ and the `liboqs`
    / `oqs-provider` repositories.
"""))

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
)
OUT.write_text(nbf.writes(nb))
print(f"Wrote {OUT} with {len(cells)} cells")
