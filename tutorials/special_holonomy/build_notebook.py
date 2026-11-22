#!/usr/bin/env python3
"""Build the special-holonomy notebook (run; commit the .ipynb)."""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name(
    "special_holonomy_calabi_yau_and_g2.ipynb"
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
# Special Holonomy — Calabi–Yau and $G_2$

The manifolds foundations notebook (Tier‑C §C1–C2) already gave working
definitions: compact Kähler $X$ with $c_1(X)=0$ (Calabi–Yau), and a
7‑manifold with parallel stable 3‑form $\varphi$ (holonomy in $G_2$).
This notebook adds the **differential-geometry tool** those definitions
sit in — **holonomy** of a metric connection — then deepens both
packages with equations, schematics, and reading paths.

We do **not** construct Ricci‑flat metrics in code, nor run a string
compactification pipeline. Twistors and PQC stay in their own modules.

> **Upstream.** Manifolds §C1–C2 (working setups + cartoons). Open those
> first if the ambient language is unfamiliar.
"""))

cells.append(md(r"""
## Prerequisites

- Manifolds foundations §C1–C2 recommended; Part I DG toolset helpful
  (embedded constraint, $T_pM$, charts).
- Comfort with Riemannian metrics and linear algebra of orthogonal /
  unitary groups.

## Learning objectives

1. State what the holonomy group of a metric connection is, and why
   “special holonomy” means a proper subgroup of $\mathrm{O}(n)$ (or
   $\mathrm{U}(m)$) forced by parallel tensors.
2. Recall the Calabi–Yau package: $c_1=0$, Yau’s Ricci‑flat metric,
   holonomy in $\mathrm{SU}(m)$, holomorphic volume form.
3. Write the model associative 3‑form $\varphi_0$ on $\mathbb{R}^{7}$
   and explain $G_2=\mathrm{Aut}(\varphi_0)$ as a stabilizer.
4. Point to Joyce for compact exact‑holonomy $G_2$ examples and to Yau /
   Gross–Huybrechts–Joyce for CY depth; know $\mathrm{Spin}(7)$ as the
   8‑dimensional sibling (map only here).
"""))

cells.append(md(r"""
## How this notebook is organised

| Part | Content |
|------|---------|
| I | Holonomy as a DG tool; Berger list as a map |
| II | Calabi–Yau deepen |
| III | $G_2$ deepen |
| IV | $\mathrm{Spin}(7)$ map; bridges; bookshelf |

**Citation rule.** Primary sources are named; we do not invent theorems.
"""))

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part I — Holonomy as a differential-geometry tool

## Working setup

Let $(M,g)$ be a connected Riemannian $n$‑manifold with Levi‑Civita
connection $\nabla$. Fix a basepoint $p\in M$ and an orthonormal frame
of $T_pM$. Parallel transport of that frame around loops based at $p$
yields orthogonal transformations of $T_pM$. The **holonomy group**
$\mathrm{Hol}_p(\nabla)\subset\mathrm{O}(T_pM)\cong\mathrm{O}(n)$ is
the group of all such transformations (restricted holonomy uses
null‑homotopic loops). Up to conjugacy it is independent of $p$ when
$M$ is connected; we write $\mathrm{Hol}(g)$.

**Special holonomy** means $\mathrm{Hol}(g)$ sits in a proper subgroup
$H\subset\mathrm{O}(n)$ (or $\mathrm{U}(m)\subset\mathrm{O}(2m)$ in the
Kähler setting) that preserves extra parallel tensors — e.g. a complex
structure, a holomorphic volume form, or a stable 3‑form.

Berger’s list (roughly): for irreducible nonsymmetric Riemannian
manifolds the possible restricted holonomy groups are tightly
constrained; Calabi–Yau and $G_2$ appear on that list as
$\mathrm{SU}(m)$ and $G_2$. We use the list as a **map**, not a proof.
"""))

cells.append(code(r"""
# Dimension bookkeeping for the groups that appear below (schematic).
# Not a holonomy computation — just sizes of the ambient orthogonal /
# unitary groups vs the special subgroups.

rows = [
    ("O(2m) for m=3 (real dim 6)", "O(6)", 15),
    ("U(3) < O(6)", "U(3)", 9),
    ("SU(3) (CY3 holonomy candidate)", "SU(3)", 8),
    ("O(7)", "O(7)", 21),
    ("G2 < O(7)", "G2", 14),
    ("O(8)", "O(8)", 28),
    ("Spin(7) < O(8)", "Spin(7)", 21),
]
print(f"{'role':40s}  {'group':8s}  dim")
for role, name, dim in rows:
    print(f"{role:40s}  {name:8s}  {dim:3d}")
"""))

cells.append(md(r"""
**Takeaway box — Part I.**  
Holonomy = parallel-transport image in $\mathrm{O}(n)$. Special
holonomy = parallel tensors forcing a smaller $H$. CY and $G_2$ are
two entries on Berger’s list with rich compact examples.
"""))

# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II — Calabi–Yau (deepen)

## Working setup (from manifolds §C1, sharpened)

A compact Kähler manifold $(X,g,J,\omega)$ of complex dimension $m$ is
**Calabi–Yau** in the physicist’s working sense when any of the
following equivalent packages holds (under standard topological
hypotheses; see Yau, Joyce):

1. $c_1(X)=0$ in $H^{2}(X;\mathbb{Z})$;
2. the canonical bundle $K_X$ is holomorphically trivial — a nowhere
   vanishing holomorphic volume form $\Omega\in H^{0}(X,K_X)$;
3. there is a Ricci‑flat Kähler metric in the given Kähler class
   (Yau’s solution of the Calabi conjecture), with holonomy contained
   in $\mathrm{SU}(m)$.

Threefolds ($m=3$) are the standard internal spaces for many string /
M‑theory reductions to four dimensions. Hodge numbers
$h^{p,q}=\dim H^{p,q}(X)$ organise deformations and, in physics
language, count moduli and generations.

**DG link.** Parallelism of $\Omega$ (or of the Ricci‑flat condition)
cuts holonomy from $\mathrm{U}(m)$ down toward $\mathrm{SU}(m)$.
"""))

cells.append(code(r"""
# Cartoon: Weierstrass discriminant for an elliptic fibre
# y^2 = x^3 + f x + g,  Δ = -16(4f^3+27g^2).
# Smooth fibres where Δ≠0 — building block language for elliptic CY.
# Not a Ricci-flat metric.

import numpy as np
import matplotlib.pyplot as plt

f = -1.0
gs = np.linspace(-1.5, 1.5, 400)
disc = -16.0 * (4.0 * f**3 + 27.0 * gs**2)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(gs, disc, "C0-", lw=2)
ax.axhline(0, color="0.5", lw=1)
ax.fill_between(gs, disc, 0, where=(disc > 0), alpha=0.12, color="C2",
                label=r"$\Delta>0$ (smooth fibres, schematic)")
ax.fill_between(gs, disc, 0, where=(disc < 0), alpha=0.12, color="C3")
ax.set_xlabel(r"$g$")
ax.set_ylabel(r"$\Delta(f,g)$")
ax.set_title(r"Weierstrass $\Delta$ at $f=-1$ (elliptic-fibre cartoon)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("CY claim: c1=0 + Kähler => Ricci-flat metric exists (Yau).")
print("Code plots Δ only; it does not build that metric.")
"""))

cells.append(md(r"""
### Where it shows up

- String / heterotic / type‑II compactifications on CY threefolds.
- Mirror symmetry (exchanged Hodge data).
- Pure geometry: moduli of Ricci‑flat metrics.

> **Further reading.** Yau (1978); Joyce, *Compact Manifolds with
> Special Holonomy*; Gross–Huybrechts–Joyce survey lectures.

**Takeaway box — Part II.**  
CY = compact Kähler + $c_1=0$ ⇒ Ricci‑flat metric, holonomy in
$\mathrm{SU}(m)$, holomorphic volume form. Code: discriminant cartoon
only.
"""))

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part III — $G_2$ (deepen)

## Working setup (from manifolds §C2, sharpened)

1. On $\mathbb{R}^{7}$ with coordinates $(x^{1},\ldots,x^{7})$, the
   model **associative** 3‑form is
   $$
   \varphi_0
     =dx^{123}+dx^{145}+dx^{167}+dx^{246}
      -dx^{257}-dx^{347}-dx^{356}.
   $$
2. $G_2\subset\mathrm{GL}(7,\mathbb{R})$ is the stabilizer of
   $\varphi_0$ (equivalently $\mathrm{Aut}(\mathbb{O})$ of the
   octonions). $\dim G_2=14$. The metric and orientation are recovered
   algebraically from $\varphi_0$.
3. A smooth oriented 7‑manifold $Y$ with a stable 3‑form $\varphi$ has
   **holonomy in $G_2$** when $\nabla\varphi=0$ for the Levi‑Civita
   connection of the metric determined by $\varphi$. Then $Y$ is
   Ricci‑flat. Compact examples with holonomy **exactly** $G_2$ were
   constructed by Joyce.

**DG link.** Parallel $\varphi$ is the tensor that cuts
$\mathrm{Hol}$ from $\mathrm{O}(7)$ down to $G_2$ (or a subgroup).
"""))

cells.append(code(r"""
# Inventory of φ0 triples + dimension bar chart (schematic).

terms = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 5, 6),
]
signs = [+1, +1, +1, +1, -1, -1, -1]
assert len(set(terms)) == 7

n_three_forms = 35  # C(7,3)
dim_GL7, dim_G2 = 49, 14
print("C(7,3) =", n_three_forms, "  dim GL(7) =", dim_GL7, "  dim G2 =", dim_G2)
print("Model φ0 signed triples:")
for s, t in zip(signs, terms):
    print(f"  {s:+d} dx^{t[0]}{t[1]}{t[2]}")

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(["3-forms", "GL(7)", "G2"], [n_three_forms, dim_GL7, dim_G2],
       color=["C0", "C1", "C2"])
ax.set_ylabel("dimension")
ax.set_title(r"$G_2\subset\mathrm{GL}(7)$ sizes (schematic)")
plt.tight_layout()
plt.show()
"""))

cells.append(md(r"""
### Where it shows up

- M‑theory compactifications on $G_2$‑holonomy 7‑folds
  ($\mathcal{N}=1$ in four dimensions).
- Exceptional holonomy geometry as pure math (with $\mathrm{Spin}(7)$).

> **Further reading.** Joyce, *Compact Manifolds with Special Holonomy*,
> Part II.

**Takeaway box — Part III.**  
$G_2$‑manifold = Ricci‑flat 7‑fold with parallel stable 3‑form.
Model $\varphi_0$ is an explicit tensor on $\mathbb{R}^{7}$; compact
exact‑holonomy examples exist (Joyce).
"""))

# ---------------------------------------------------------------------------
# Part IV
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part IV — Maps, bridges, and further reading

## $\mathrm{Spin}(7)$ (map only)

In eight dimensions the exceptional holonomy sibling is
$\mathrm{Spin}(7)\subset\mathrm{O}(8)$, characterised by a parallel
stable 4‑form (Cayley form). Compact examples again exist (Joyce). We
stop at the naming; deepen in Joyce Part II.

## Bridges in this lab

| Topic | Where |
|------|--------|
| CY / $G_2$ invitations | Manifolds §C1–C2 |
| Holonomy tool + deepen | **This notebook** |
| Twistor incidence / Penrose | [`../twistor_theory`](../twistor_theory/) |
| Lattices / ECC / PQC | Manifolds §C4 → PQC module |

## Global further reading (bookshelf)

1. D. Joyce, *Compact Manifolds with Special Holonomy*, OUP, 2000.
2. S.‑T. Yau, *Comm. Pure Appl. Math.* **31** (1978) 339–411.
3. M. Gross, D. Huybrechts, D. Joyce, *Calabi–Yau Manifolds and Related
   Geometries*, Springer, 2003.
4. Manifolds foundations notebook in this repository (§C1–C2).
"""))

cells.append(md(r"""
## Summary

- Holonomy records parallel transport; special holonomy means parallel
  tensors force a smaller structure group.
- Calabi–Yau: $c_1=0$ / trivial $K_X$ / Ricci‑flat Kähler /
  $\mathrm{SU}(m)$.
- $G_2$: parallel stable 3‑form on a 7‑manifold; model $\varphi_0$ on
  $\mathbb{R}^{7}$; compact examples by Joyce.
- $\mathrm{Spin}(7)$ is the 8‑dimensional map entry; string /
  M‑theory compactification depth stays in the cited monographs.
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
