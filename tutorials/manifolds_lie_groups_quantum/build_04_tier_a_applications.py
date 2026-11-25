#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 04 (source of truth): Tier A application labs.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_04_tier_a_applications.py

The generated notebook is ``04_tier_a_applications.ipynb``
beside this file. Markdown and the five lab checks are Part II-A
from ``build_notebook.py`` (Bloch sphere, density operators,
unitary gates, the symplectic group, and the Fisher metric).
Examples (M1)–(M3), charts, and Lie groups are notebooks 01–03.
Tiers B and C are notebooks 05 and 06. This builder does not
rewrite that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part II-A uses level-2 and level-3
headings only, so those titles stay plain heading cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("04_tier_a_applications.ipynb")


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


def md(source: str, *, collapsed: bool | None = None) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(fix_math(source.strip() + "\n"))
    if collapsed is not None:
        cell.metadata["jp-MarkdownHeadingCollapsed"] = collapsed
    return cell


def code(source: str, *, hide_input: bool = False) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(source.strip() + "\n")
    if hide_input:
        # JupyterLab collapses the source; nbconvert hide-input tag as backup.
        cell.metadata["jupyter"] = {"source_hidden": True, "outputs_hidden": False}
        tags = list(cell.metadata.get("tags", []))
        if "hide-input" not in tags:
            tags.append("hide-input")
        cell.metadata["tags"] = tags
    return cell


# Heading banners (N=2 orange family for the deepest level). Level-4
# example titles such as (M1) must outrank their (M1.1)–style children
# visually — slate parent strip, orange child strip — otherwise the
# hierarchy in JupyterLab reads upside-down (plain #### under loud #####).
_SUB_STYLE = (
    "border-left:5px solid #455a64;"
    "background:#eceff1;"
    "padding:0.55em 0.9em;"
    "margin:0.65em 0 0.2em 0;"
    "font-size:1.18em;"
    "font-weight:700;"
    "line-height:1.35;"
)
_SUBSUB_STYLE = (
    "border-left:4px solid #e67e22;"
    "background:#fff4e5;"
    "padding:0.5em 0.85em;"
    "margin:0.55em 0 0.15em 0;"
    "font-size:1.08em;"
    "font-weight:600;"
    "line-height:1.35;"
)


def style_subsection_heading(title: str) -> str:
    """HTML <h4> banner for example-level titles such as (M1), (P1)."""
    return f'<h4 style="{_SUB_STYLE}">{title}</h4>'


def style_subsubsection_heading(title: str) -> str:
    """HTML <h5> banner for a subsubsection title (math OK inside)."""
    return f'<h5 style="{_SUBSUB_STYLE}">{title}</h5>'


_HEADING_LINE = re.compile(r"^(#{1,6})\s+\S")


def explode_heading_cells(
    cells_in: list[nbf.NotebookNode],
    *,
    default_collapsed: bool = False,
) -> list[nbf.NotebookNode]:
    """Match the N=2 coupled-oscillators notebook: each ATX heading in its
    own markdown cell with ``jp-MarkdownHeadingCollapsed``, body until the
    next heading in the following cell. JupyterLab then expands/collapses
    all cells under that heading.

    Level-4 headings (``####``, e.g. (M1)) become slate ``<h4>`` banners;
    level-5+ (``#####``, e.g. M1.1–M1.5) become orange ``<h5>`` banners.
    """
    out: list[nbf.NotebookNode] = []
    saw_doc_title = False

    def flush_body(buf: list[str]) -> None:
        text = "".join(buf).strip()
        if text:
            out.append(md(text))

    for cell in cells_in:
        if cell.cell_type != "markdown":
            out.append(cell)
            continue

        # Honour an explicit collapsed flag already set on a heading-only cell.
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        lines = src.splitlines(keepends=True)
        body: list[str] = []
        for line in lines:
            if _HEADING_LINE.match(line):
                flush_body(body)
                body = []
                level = len(line) - len(line.lstrip("#"))
                heading_only = line.rstrip("\n").rstrip()
                title = re.sub(r"^#{1,6}\s+", "", heading_only)
                # Document title (# only, first time): no collapse meta (N=2).
                if level == 1 and not saw_doc_title and not heading_only.startswith(
                    "# Part"
                ):
                    saw_doc_title = True
                    out.append(md(heading_only))
                else:
                    if level == 1:
                        saw_doc_title = True
                    explicit = cell.metadata.get("jp-MarkdownHeadingCollapsed")
                    flag = (
                        default_collapsed
                        if explicit is None
                        else bool(explicit)
                    )
                    if level >= 5:
                        out.append(
                            md(style_subsubsection_heading(title), collapsed=flag)
                        )
                    elif level == 4:
                        out.append(
                            md(style_subsection_heading(title), collapsed=flag)
                        )
                    else:
                        out.append(md(heading_only, collapsed=flag))
            else:
                body.append(line)
        flush_body(body)

    return out

LOCATOR = r'''
# Tier A application labs

This notebook is Part II‑A from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
the Bloch sphere ($\mathbb{CP}^{1}$), density operators on
$\mathbb{CP}^{d-1}$, unitary gates $\mathrm{U}(n)$, the symplectic
group $\mathrm{Sp}(2n,\mathbb{R})$, and the Fisher metric on a
statistical manifold. The wording below is that text. Examples
(M1)–(M3), charts, and Lie groups are notebooks 01–03. Tiers B
and C are notebooks 05 and 06. The map is `00_series_guide.ipynb`.

The code cells check those five labs. Lab A2 uses the Pauli
matrices defined in the A1 cell, as when these labs shared one notebook. There
is no sibling demo.
'''

SETUP = r'''
# Setup — foundations notebook (numpy / scipy / matplotlib only)
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, det, eigvalsh
from scipy.linalg import expm, logm

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (5.5, 4.0), "font.size": 11})

print("numpy", np.__version__)
'''

PART_IIA = r'''
# Part II‑A — Application labs (full code)

Each lab below has the same shape: **why the object shows up**, the
**equations** that make it precise, a **coded check**, and a short
**takeaway** tying manifold / Lie group / exponential together.

> **Further reading (QI geometry).** I. Bengtsson and K. Życzkowski,
> *Geometry of Quantum States: An Introduction to Quantum
> Entanglement*, 2nd ed., Cambridge University Press, 2017. Chapter 4
> treats complex projective spaces; the Bloch sphere is $\mathbb{CP}^1$.
'''

A1 = r'''
## A1. The Bloch sphere ($\mathbb{CP}^{1}$)

### Background

A **qubit** is the simplest quantum system with two levels (spin‑1/2,
ground/excited states of a superconducting circuit, polarisation of a
photon, …). Its pure states are unit vectors
$|\psi\rangle\in\mathbb{C}^2$, but physics does not care about an
overall complex factor: $|\psi\rangle$ and $\lambda|\psi\rangle$
($\lambda\neq 0$) describe the same ray. The space of such rays is the
complex projective line $\mathbb{CP}^{1}$. As a real smooth manifold
that space is ordinary $S^2$ — the **Bloch sphere**. That is why every
textbook picture of a qubit is a point on a sphere, not a vector in
$\mathbb{C}^2$.

### Geometry

$$
\mathbb{CP}^{1}\;\cong\; S^2.
$$

In the computational basis one may write

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
speed $2$ (in units $\hbar=1$). So the **Lie algebra** (Hermitian
generators / anti‑Hermitian $X=-iH$) drives motion on the **manifold**
of states via the **exponential** in the **Lie group** $\mathrm{U}(2)$.

### Where it shows up

- **Single‑qubit gates** are rotations of $\mathbf{r}$ (or of the
  sphere itself under $\mathrm{SU}(2)$).
- **State tomography** recovers $\mathbf{r}$ from Pauli expectation
  values $\langle\sigma_i\rangle$.
- **NMR / ESR / superconducting control** design microwave pulses that
  steer $\mathbf{r}$ along chosen paths.
- **Decoherence** (not modelled in the pure‑state code below) pulls
  $\mathbf{r}$ inside the ball — which is the mixed‑state story of A2.
'''

A1_CODE = r'''
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
'''

A1_TAKE = r'''
**Takeaway box — A1.**  
Manifold: \(\mathbb{CP}^{1}\cong S^2\) (pure qubit states). Lie group
acting: \(\mathrm{U}(2)\) (or \(\mathrm{SU}(2)\)). Lie algebra:
anti‑Hermitian matrices \(i\mathfrak{su}(2)\) / Hermitian Hamiltonians.
Exponential: \(U(t)=\exp(-iHt)\).
'''

A2 = r'''
## A2. $\mathbb{CP}^{d-1}$ and density operators

### Background

Not every quantum system is a qubit, and not every state is pure.
A $d$-level system (qutrit, qudit, truncated oscillator, …) has pure
states that are rays in $\mathbb{C}^d$ — points of the complex
projective space $\mathbb{CP}^{d-1}$. When the system is entangled with
an environment, or we simply lack full knowledge, the correct object is
a **density operator** $\rho$: a Hermitian, positive‑semidefinite
matrix with $\operatorname{Tr}\rho=1$. Pure states sit at the extreme
points ($\rho=|\psi\rangle\langle\psi|$); everything else is a
statistical mixture.

### Geometry

Pure states of a $d$-level system are points of $\mathbb{CP}^{d-1}$.
The set of all density operators is a **convex body** in the real
vector space of Hermitian matrices of trace one. Its extreme points
are the pure states (rank‑one projectors). Fixed‑rank subsets are
smooth manifolds in the relative interior; the full set has corners
(a stratified space / manifold with corners). For $d=2$ that body is
the **Bloch ball** $|\mathbf{r}|\le 1$; the sphere $|\mathbf{r}|=1$ is
exactly A1.

A useful scalar invariant is the **purity**

$$
\mathcal{P}(\rho)=\operatorname{Tr}(\rho^2),
\qquad
\frac{1}{d}\le\mathcal{P}(\rho)\le 1,
$$

with $\mathcal{P}=1$ if and only if $\rho$ is pure.

### Where it shows up

- **Open quantum systems** and noise channels act as maps on $\rho$,
  typically shrinking purity.
- **Quantum information geometry** (Bures metric, quantum Fisher
  information) lives on full‑rank density operators — distances that
  quantify how distinguishable two states are.
- **Tomography and certification** reconstruct $\rho$ or check that a
  prepared state is close to a target pure state.
- **Entanglement** for bipartite systems is diagnosed on the joint
  $\rho_{AB}$; reduced states $\rho_A=\operatorname{Tr}_B\rho_{AB}$
  are generically mixed even when $\rho_{AB}$ is pure.

We check the geometry numerically for a qubit (Bloch ball) and a
qutrit purity slice.
'''

A2_CODE = r'''
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
'''

A2_TAKE = r'''
**Takeaway box — A2.**  
Pure states: manifold \(\mathbb{CP}^{d-1}\). Mixed states: convex set
of density operators (stratified by rank). Geometry of distinguishability
(Bures, quantum Fisher) lives on the interior of full-rank states —
see Bengtsson–Życzkowski for the Riemannian picture.
'''

A3 = r'''
## A3. Unitary gates $\mathrm{U}(n)$ and Hermitian generators

### Background

Closed quantum evolution preserves inner products: if
$|\psi(0)\rangle$ is normalised, so is $|\psi(t)\rangle$, and relative
phases between components are the entire dynamical content. The maps
that do that are **unitary** matrices. Every gate in a digital quantum
circuit, and every continuous drive under a time‑dependent
Hamiltonian, is (ideally) an element of $\mathrm{U}(n)$ — or of the
special unitary group $\mathrm{SU}(n)$ once an irrelevant global phase
is fixed. For $k$ qubits one takes $n=2^k$.

### Geometry

$$
\mathrm{U}(n)=\{U\in\mathrm{M}_n(\mathbb{C}):U^\dagger U=I\}
$$

is a compact Lie group of real dimension $n^2$. Its Lie algebra
$\mathfrak{u}(n)$ consists of **anti‑Hermitian** matrices
$X^\dagger=-X$. Writing $X=-iH$ with $H=H^\dagger$, every one-parameter
subgroup is

$$
U(t)=\exp(-i t H).
$$

The traceless subalgebra $\mathfrak{su}(n)$ (dimension $n^2-1$)
generates $\mathrm{SU}(n)$. Controllability questions in quantum
control ask which subgroup you can reach by switching among a finite
list of Hamiltonians — a Lie‑algebra generation problem.

### Where it shows up

- **Gate synthesis / compilation:** approximate a target $U_*$ by a
  product of native exponentials $\exp(-i t_j H_j)$.
- **Hamiltonian simulation** (module 05): write
  $\exp(-i t(A+B+\cdots))$ as a product of simpler exponentials
  (Trotter–Suzuki) — only after the exponential map of Part I is in
  place.
- **Pulse‑level control:** shaped microwave / laser drives implement
  time‑ordered exponentials of $H(t)$.
'''

A3_CODE = r'''
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
'''

A3_TAKE = r'''
**Takeaway box — A3.**  
Lie group: \(\mathrm{U}(n)\) (quantum gates). Lie algebra: \(i\) times
Hermitian matrices. Exponential: time evolution / gate synthesis.
'''

A4 = r'''
## A4. The symplectic group $\mathrm{Sp}(2n,\mathbb{R})$

### Background

Position and momentum $(q_1,p_1,\ldots,q_n,p_n)$ live in a $2n$‑dimensional
**phase space**. Classical Hamiltonian mechanics cares about more than
ordinary volumes: it cares about the **symplectic** area form that
makes Hamilton's equations covariant. Linear maps that preserve that
form are the **symplectic** matrices. The same group appears in
quantum optics and continuous‑variable (CV) quantum information: beam
splitters, phase shifters, and squeezers act as Gaussian unitaries
whose classical shadow on means and covariances is exactly a
symplectic transformation.

### Geometry

Phase space $\mathbb{R}^{2n}$ carries the standard symplectic matrix
$J$ ($J^\top=-J$, $J^2=-I$). A real matrix $S$ is **symplectic** when

$$
S^\top J S = J.
$$

The set of all such $S$ is the Lie group $\mathrm{Sp}(2n,\mathbb{R})$.
For $n=1$ one has $\mathrm{Sp}(2,\mathbb{R})\cong\mathrm{SL}(2,\mathbb{R})$.
Quadratic Hamiltonians generate one‑parameter subgroups by the matrix
exponential in the Lie algebra $\mathfrak{sp}(2n,\mathbb{R})$.

A Gaussian state's covariance matrix $\Sigma$ transforms as
$\Sigma\mapsto S\Sigma S^\top$. Physical states obey an uncertainty
relation that can be written, with $\hbar=1$, as
$\Sigma+\mathrm{i}J/2\ge 0$.

### Where it shows up

- **Classical mechanics:** linear canonical transformations; small
  oscillations; stability of fixed points.
- **Quantum optics / CV QI:** squeezing, two‑mode entanglement
  (EPR‑type states), Gaussian boson sampling backends.
- **Metrology:** squeezed vacuum improves phase sensitivity beyond
  shot noise within the Gaussian sector.

> **Further reading.** M. de Gosson, *Symplectic Geometry and Quantum
> Mechanics*, Birkhäuser, 2006, Ch. 2 (the symplectic group).
'''

A4_CODE = r'''
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
'''

A4_TAKE = r'''
**Takeaway box — A4.**  
Lie group: \(\mathrm{Sp}(2n,\mathbb{R})\) (linear canonical /
Gaussian CV gates). Lie algebra: \(\mathfrak{sp}(2n,\mathbb{R})\)
(Hamiltonian matrices for quadratic forms). Exponential: Gaussian
unitaries' classical shadow.
'''

A5 = r'''
## A5. Statistical manifolds and the Fisher metric

### Background

Estimation problems ask: given samples from a probability distribution
that depends on an unknown parameter $\theta$, how accurately can
$\theta$ be recovered? Differential geometry enters because the family
of distributions itself is a curved space, and there is a natural
notion of distance that matches statistical distinguishability.

### Geometry

A smooth family of probability densities $\{p_\theta\}$ parametrised
by $\theta\in\Theta\subset\mathbb{R}^m$ is a **statistical manifold**.
The **Fisher information matrix**

$$
g_{ij}(\theta)
  =\mathbb{E}_\theta
   \bigl[\partial_i\log p_\theta\;\partial_j\log p_\theta\bigr]
$$

is a Riemannian metric on $\Theta$ (Rao, 1945). The Cramér–Rao bound
says that the covariance of an unbiased estimator is at least
$g(\theta)^{-1}$: more Fisher information means tighter possible
error bars. Geodesics and distances derived from $g$ organise
asymptotic statistics.

For a Bernoulli random variable with success probability $p\in(0,1)$,

$$
g(p)=\frac{1}{p(1-p)},
$$

which blows up at the endpoints — extreme probabilities are easier to
tell apart from their neighbours than $p\approx 1/2$ is.

### Where it shows up

- **Classical statistics:** asymptotic efficiency, exponential families,
  natural gradients in machine learning (Amari).
- **Quantum metrology:** the **quantum Fisher information** on density
  operators bounds how well a phase or field can be estimated; squeezed
  and entangled probes aim to raise that information per resource.
- **Thermodynamics / inference:** geometries on spaces of Gibbs states
  connect to response and fluctuation formulae.

> **Further reading.** C. R. Rao, “Information and the accuracy
> attainable in the estimation of statistical parameters,” *Bull.
> Calcutta Math. Soc.* **37** (1945) 81–91. S. Amari,
> *Differential-Geometrical Methods in Statistics*, Springer, 1985.
> S. Amari and H. Nagaoka, *Methods of Information Geometry*, AMS,
> 2000 (Ch. 2–3).
'''

A5_CODE = r'''
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
'''

A5_TAKE = r'''
**Takeaway box — A5.**  
Manifold: parameter space of a smooth statistical model. Metric:
Fisher–Rao. Quantum thermodynamics and quantum estimation use the
quantum Fisher information on density-operator manifolds (Amari–Nagaoka,
Ch. on quantum systems).
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(PART_IIA))
    cells.append(md(A1))
    cells.append(code(A1_CODE))
    cells.append(md(A1_TAKE))
    cells.append(md(A2))
    cells.append(code(A2_CODE))
    cells.append(md(A2_TAKE))
    cells.append(md(A3))
    cells.append(code(A3_CODE))
    cells.append(md(A3_TAKE))
    cells.append(md(A4))
    cells.append(code(A4_CODE))
    cells.append(md(A4_TAKE))
    cells.append(md(A5))
    cells.append(code(A5_CODE))
    cells.append(md(A5_TAKE))
    return cells


def main() -> None:
    cells = explode_heading_cells(build_cells(), default_collapsed=False)
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
    )
    nbf.write(nb, OUT)
    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    n_code = sum(1 for c in cells if c.cell_type == "code")
    print(f"Wrote {OUT} with {len(cells)} cells ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    main()
