#!/usr/bin/env python3
"""Build the twistor-theory notebook (run; commit the .ipynb)."""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name(
    "twistor_theory_geometry_and_penrose_transform.ipynb"
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
# Twistor Theory — Geometry and the Penrose Transform

The manifolds foundations notebook (Tier‑C §C3) already wrote the
**incidence dictionary** between twistor space
$\mathbb{PT}=\mathbb{CP}^{3}$ and null geometry in complexified
Minkowski space $\mathbb{CM}$. This notebook starts from that
dictionary and adds three layers that §C3 deferred:

1. **2‑spinor coordinates** for twistors and events,
2. **reality conditions** that select real Minkowski space
   $M\subset\mathbb{CM}$,
3. a **Penrose‑transform sketch** for massless helicity fields.

It stays geometry‑first. Modern amplitude methods and integrable‑system
constructions appear only as maps with citations.

> **Upstream.** Manifolds notebook §B1 (Minkowski as a smooth manifold)
> and §C3 (incidence $\omega^{A}=ix^{AA'}\pi_{A'}$). Open those before
> this file if the ambient spaces are unfamiliar.
"""))

cells.append(md(r"""
## Prerequisites

- Manifolds foundations §B1 and §C3 recommended.
- Comfort with complex projective space $\mathbb{CP}^{n}$ and with
  the Minkowski metric $\eta=\mathrm{diag}(-1,1,1,1)$.
- No prior spinor course assumed: Part I introduces the $2\times 2$
  isomorphism used here.

## Learning objectives

1. Write a twistor as $(\omega^{A},\pi_{A'})$ and recover a null
   direction from $\pi_{A'}$.
2. State the incidence relation and compute $\omega$ from $(x,\pi)$
   in coordinates.
3. Explain what a reality condition selects ($M$ inside $\mathbb{CM}$)
   without claiming a full $\mathrm{SU}(2,2)$ course.
4. Sketch the Penrose‑transform idea (contour integral of a
   homogeneous function of $Z$ yields a massless field on $M$).
5. Point to Huggett–Tod / Penrose–Rindler for the next depth; know
   that amplitudes and integrable systems are off‑ramps, not labs here.
"""))

cells.append(md(r"""
## How this notebook is organised

| Part | Content |
|------|---------|
| I | Spinors, incidence in coordinates, null rays from $\pi$ |
| II | Reality conditions (real slice) |
| III | Penrose transform sketch |
| IV | Maps: amplitudes, integrable systems; bookshelf |

**Citation rule.** Every section names textbooks or articles a reader
can open. We do not invent references.
"""))

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part I — Spinors and incidence in coordinates

## Working setup (declared once)

The same three spaces as manifolds §C3:

1. **Real Minkowski** $M=\mathbb{R}^{1,3}$ with
   $\eta=\mathrm{diag}(-1,1,1,1)$.
2. **Complexified Minkowski** $\mathbb{CM}=\mathbb{C}^{1,3}$.
3. **Twistor space**
   $\mathbb{PT}=\mathbb{CP}^{3}=(\mathbb{C}^{4}\setminus\{\mathbf{0}\})/\mathbb{C}^{\times}$.

Identify $\mathbb{C}^{1,3}$ with $2\times 2$ complex matrices via
$$
x^{AA'}
  =\frac{1}{\sqrt{2}}
  \begin{pmatrix}
  t+z & x-iy \\
  x+iy & t-z
  \end{pmatrix},
  \qquad
  (t,x,y,z)\in\mathbb{CM}.
$$
Then $\det(x^{AA'})=\tfrac12\eta_{\mu\nu}x^{\mu}x^{\nu}$ (up to the
overall factor convention for the $\sqrt{2}$ normalisation). Null
vectors are exactly the rank‑$\le 1$ matrices.

A **twistor** is a nonzero $Z^{\alpha}=(\omega^{A},\pi_{A'})\in\mathbb{C}^{4}$,
defined up to scale. The **incidence relation** with an event $x$ is
$$
\omega^{A}=ix^{AA'}\pi_{A'}.
$$
"""))

cells.append(code(r"""
import numpy as np
import matplotlib.pyplot as plt

# Minkowski → 2×2 Hermitian (real events) / general complex (CM)
def four_to_spinor(t, x, y, z):
    return (1 / np.sqrt(2)) * np.array(
        [[t + z, x - 1j * y],
         [x + 1j * y, t - z]],
        dtype=complex,
    )

def spinor_to_four(X):
    # Inverse of four_to_spinor (same √2 convention)
    t = (X[0, 0] + X[1, 1]) / np.sqrt(2)
    z = (X[0, 0] - X[1, 1]) / np.sqrt(2)
    x = (X[0, 1] + X[1, 0]) / np.sqrt(2)
    y = 1j * (X[0, 1] - X[1, 0]) / np.sqrt(2)
    return np.array([t, x, y, z])

eta = np.diag([-1.0, 1.0, 1.0, 1.0])

def msq(v):
    return float(np.real(v @ eta @ v))

# Null check on a real lightlike vector
v_null = np.array([1.0, 1.0, 0.0, 0.0])
X_null = four_to_spinor(*v_null)
print("η(v,v) for (1,1,0,0):", msq(v_null))
print("det(X) for that vector:", complex(np.linalg.det(X_null)))
print("rank hint (singular values):", np.linalg.svd(X_null, compute_uv=False))
"""))

cells.append(md(r"""
### Incidence: $\omega$ from $(x,\pi)$

Fix an event $x$ and a nonzero spinor $\pi_{A'}=(\pi_{0'},\pi_{1'})$.
Then $\omega^{A}=ix^{AA'}\pi_{A'}$ is a linear map
$\mathbb{C}^{2}\to\mathbb{C}^{2}$. Scaling
$(\omega,\pi)\mapsto\lambda(\omega,\pi)$ leaves the projective twistor
$[Z]\in\mathbb{PT}$ unchanged and does not change the null line.
"""))

cells.append(code(r"""
def omega_from_incidence(X, pi):
    # ω^A = i X^{AA'} π_{A'}  (π as length-2 complex vector)
    pi = np.asarray(pi, dtype=complex).reshape(2)
    return 1j * (X @ pi)

# Event at the origin: every π gives ω = 0 → twistors (0, π)
X0 = four_to_spinor(0, 0, 0, 0)
pi = np.array([1.0, 0.0], dtype=complex)
om = omega_from_incidence(X0, pi)
print("At x=0, π=(1,0) → ω =", om)

# A nontrivial real event
X1 = four_to_spinor(1.0, 0.2, -0.1, 0.3)
om1 = omega_from_incidence(X1, pi)
Z = np.concatenate([om1, pi])
print("Twistor Z=(ω|π) at x=(1,0.2,-0.1,0.3), π=(1,0):\n ", Z)

# Scale invariance of the projective point [Z] ∈ CP³
lam = 2.5 + 0.5j
Z2 = lam * Z
print("Z and λZ proportional?", np.allclose(Z2, lam * Z))
"""))
cells.append(md(r"""
### Null direction from $\pi$ alone

When $\pi_{A'}\neq 0$, the matrix $\pi_{A'}\bar\pi_{A}$ (for real
null momenta; here we use the complex bilinear form
$p^{AA'}=\pi^{A}\pi^{A'}$ in the complexified setting with a choice of
index raising) determines a null direction. In the real future‑null
case a standard flag is
$p^{AA'}=\pi^{A}\bar\pi^{A'}$. The lab below builds a real null
4‑vector from a $\pi\in\mathbb{C}^{2}\setminus\{0\}$ that way and
checks $\eta(p,p)=0$.
"""))

cells.append(code(r"""
def null_momentum_from_pi(pi):
    # Real future-null p^μ from π via p^{AA'} = π^A conj(π)^{A'}
    pi = np.asarray(pi, dtype=complex).reshape(2)
    # Hermitian p^{AA'} = π ⊗ π̄^T
    P = np.outer(pi, np.conjugate(pi))
    return spinor_to_four(P)

for pi in [np.array([1.0, 0.0]), np.array([1.0, 1.0]),
           np.array([1.0, 1j])]:
    p = null_momentum_from_pi(pi)
    # p from Hermitian P is real
    p = np.real(p)
    print("π =", pi, "→ p =", np.round(p, 6), " η(p,p) =", msq(p))
"""))

cells.append(md(r"""
**Takeaway box — Part I.**  
Twistor $Z=(\omega,\pi)$ up to scale; incidence
$\omega^{A}=ix^{AA'}\pi_{A'}$; $\pi$ alone fixes a null direction.
This is the coordinate form of manifolds §C3.
"""))

# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II — Reality conditions

Complexified Minkowski $\mathbb{CM}$ is a convenience: the physical
spacetime of special relativity is the **real** slice $M$. Twistor
space carries a natural pseudo‑Hermitian form of signature $(2,2)$
(the $\mathrm{SU}(2,2)$ structure). Reality conditions select those
twistors / lines that correspond to **real** null geodesics and **real**
events.

### What we need here (not a full $\mathrm{SU}(2,2)$ course)

For this notebook it is enough to remember:

- An event $x\in\mathbb{CM}$ is **real** when $x^{AA'}$ is Hermitian
  (equivalently $(t,x,y,z)\in\mathbb{R}^{1,3}$ in our chart).
- A twistor line $\mathbb{CP}^{1}\subset\mathbb{PT}$ corresponds to a
  real event iff it is compatible with that Hermitian condition under
  incidence.
- Null rays of real $M$ are the real loci of the complex null lines
  attached to points of $\mathbb{PT}$.

> **Further reading.** Huggett–Tod, Ch. 2–3; Penrose–Rindler Vol. 2,
> Ch. 6 (reality and the Klein quadric picture).

We check the Hermitian / real‑slice statement in coordinates and stop.
"""))

cells.append(code(r"""
def is_hermitian(X, tol=1e-12):
    return np.allclose(X, X.conjugate().T, atol=tol)

# Real event → Hermitian spinor matrix
Xr = four_to_spinor(1.0, 0.3, -0.2, 0.5)
print("Real event Hermitian?", is_hermitian(Xr))

# Complex event → not Hermitian
Xc = four_to_spinor(1.0, 0.3j, -0.2, 0.5)
print("Complex event Hermitian?", is_hermitian(Xc))

# Round-trip real four-vector
v = np.array([1.0, 0.3, -0.2, 0.5])
v_back = np.real(spinor_to_four(four_to_spinor(*v)))
print("Round-trip error:", np.max(np.abs(v - v_back)))
"""))

cells.append(md(r"""
**Takeaway box — Part II.**  
Real Minkowski = Hermitian $x^{AA'}$ in this chart. Full twistor
reality (which lines in $\mathbb{PT}$ are “real”) is in Huggett–Tod;
we only need the event‑side statement to keep $M$ inside $\mathbb{CM}$
honest.
"""))

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part III — Penrose transform (sketch)

The **Penrose transform** produces solutions of massless field
equations on $M$ from suitable cohomology classes (or, in the classical
contour‑integral picture, from homogeneous functions) on regions of
$\mathbb{PT}$.

### Geometry → field (one formula)

For helicity $n/2$ a classical contour‑integral schematic is
$$
\phi_{A_1\cdots A_n}(x)
  =\int_{\Gamma_x}
   \pi_{A_1}\cdots\pi_{A_n}\,
   f(Z)\,
   \pi_{C'}d\pi^{C'},
$$
where $Z=(\omega,\pi)$ is constrained by incidence at the event $x$,
$f$ is homogeneous of the appropriate degree, and $\Gamma_x$ is a
contour in the $\mathbb{CP}^{1}$ of twistors through $x$. The output
$\phi$ satisfies the massless helicity equations on (regions of) $M$.

We do **not** evaluate cohomology groups here. The lab below only
illustrates the **incidence fibre** $\mathbb{CP}^{1}$ at a fixed real
event — the contour's geometric support — and plots a sample
homogeneous function restricted to that fibre in an affine chart.
"""))

cells.append(code(r"""
# Fibre CP^1 through a fixed real event x: π varies, ω = i X π.
X = four_to_spinor(0.0, 0.0, 0.0, 0.0)  # origin for a clean chart
# Affine chart on CP^1: π = (1, ζ), ζ ∈ C ≅ R^2
n = 200
zs = np.linspace(-2, 2, n)
ZRe, ZIm = np.meshgrid(zs, zs)
zeta = ZRe + 1j * ZIm
# Homogeneous f(Z) caricature on the fibre: f ~ 1/(1+ζ^2) (schematic)
f = 1.0 / (1.0 + zeta**2)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
pcm = ax.pcolormesh(ZRe, ZIm, np.real(f), shading="auto", cmap="coolwarm")
ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
ax.set_title(r"Schematic $f$ on the $\mathbb{CP}^{1}$ fibre at $x=0$")
ax.set_aspect("equal")
fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
print("Contour Γ_x would be a closed path in this ζ-plane (not drawn).")
print("Incidence at x=0: ω = 0 for all π, so Z = (0,0,π0',π1').")
"""))

cells.append(md(r"""
> **Further reading.** Penrose, “Twistor algebra” (1967); Huggett–Tod,
> Ch. 4–5 (Penrose transform); Penrose–Rindler Vol. 2, Ch. 6.

**Takeaway box — Part III.**  
Massless fields on $M$ ← suitable data on $\mathbb{PT}$, integrated
over the incident $\mathbb{CP}^{1}$. This notebook stops at the
geometry of that fibre and the shape of the integral.
"""))

# ---------------------------------------------------------------------------
# Part IV
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part IV — Maps and further reading

## Where the geometry goes next (not labs here)

- **Scattering amplitudes.** Twistor and momentum‑twistor variables
  reorganise perturbative gauge‑theory amplitudes (modern descendants
  of the Penrose programme).
- **Integrable systems.** Several soliton / self‑dual equations admit
  twistor constructions (Ward, Hitchin, …).
- **Curved twistor spaces.** Go beyond flat $\mathbb{CP}^{3}$ for
  self‑dual gravity / nonlinear graviton (Penrose 1976).

## Bridges in this lab

| Topic | Where |
|------|--------|
| Minkowski manifold + light cone | Manifolds §B1 |
| Incidence dictionary (stub → formula) | Manifolds §C3 |
| Spinors, reality, Penrose sketch | **This notebook** |
| PQC / lattices / ECC | Manifolds §C4 → PQC module |

## Global further reading (bookshelf)

1. R. Penrose, “Twistor algebra,” *J. Math. Phys.* **8** (1967)
   345–366.
2. R. Penrose and W. Rindler, *Spinors and Space‑Time*, Vol. 2, CUP,
   1986.
3. S. A. Huggett and K. P. Tod, *An Introduction to Twistor Theory*,
   2nd ed., CUP, 1994.
4. Manifolds foundations notebook in this repository (§B1, §C3).
"""))

cells.append(md(r"""
## Summary

- Twistors are points of $\mathbb{PT}=\mathbb{CP}^{3}$; incidence
  $\omega^{A}=ix^{AA'}\pi_{A'}$ ties them to null geometry in
  $\mathbb{CM}$.
- Real events are Hermitian $x^{AA'}$; that is the reality cut used
  here.
- The Penrose transform builds massless fields from data on
  $\mathbb{PT}$ by contour integration over the incident
  $\mathbb{CP}^{1}$.
- Amplitudes and integrable systems remain citation maps; deepen via
  Huggett–Tod / Penrose–Rindler.
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
