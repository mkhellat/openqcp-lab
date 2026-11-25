#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 06 (source of truth): Tier C offramps.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_06_tier_c_offramps.py

The generated notebook is ``06_tier_c_offramps.ipynb``
beside this file. Markdown and the schematic checks are
Part II‑C from ``build_notebook.py`` (Calabi-Yau, G2,
twistor incidence, and elliptic curves / lattices). Examples
(M1)–(M3), charts and atlases, Lie groups, and Tiers A and B
stay in the mega notebook. This builder does not rewrite
that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part II‑C uses level-2 and level-3
headings only, so those titles stay plain heading cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("06_tier_c_offramps.ipynb")


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
# Tier C offramps

This notebook is Part II‑C from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
Calabi–Yau manifolds, $G_2$ holonomy, twistor incidence, and
elliptic curves and lattices. The wording below is that text.
Examples (M1)–(M3), charts and atlases, Lie groups and the
exponential map, and Tiers A and B stay in the mega notebook.

The code cells draw the Weierstrass discriminant, inventory the
model $G_2$ 3-form, sketch twistor incidence, and plot a lattice
torus. There is no sibling demo.
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

PART_IIC = r"""
# Part II‑C — Geometry‑depth maps and reading paths

Tier‑C topics stay **invitation‑depth** relative to Tier A (no full
string / holonomy / crypto course), but each of **C1–C4** is now written
formula‑first: a declared working setup, the geometric object with at
least one explicit equation or dictionary, a schematic where it helps,
and a takeaway. Graduate off‑ramps:

- Twistors → [`../twistor_theory`](../twistor_theory/)
- CY / $G_2$ (holonomy deepen) →
  [`../special_holonomy`](../special_holonomy/)
- PQC / lattices / ECC engineering →
  [`../post_quantum_cryptography`](../post_quantum_cryptography/)
- Combinatorial manifolds ($k=1,\ldots,5$) →
  [`../simplicial_manifolds`](../simplicial_manifolds/)
  (mother notebook `00_series_guide.ipynb`)

**Off‑ramp:** [`../simplicial_manifolds`](../simplicial_manifolds/)
(mother notebook `00_series_guide.ipynb`) — when a finite simplicial
complex is a combinatorial manifold, for $k=1,\ldots,5$. Smooth
charts, Lie groups, and the exponential map stay in Part I of this
notebook.
"""

C1 = r"""
## C1. Calabi–Yau manifolds

### Background

String theory needs more spatial dimensions than the three we see.
Those extra directions are usually taken to be small and curled into a
compact space whose shape affects which particles and couplings appear
at low energy. The spaces that preserve enough supersymmetry to be
phenomenologically useful are **Calabi–Yau** manifolds.

### Working setup (declared once)

Throughout (C1) we use the physicist's working package (equivalent
conditions under standard compactness / simply‑connected hypotheses;
see Yau, Joyce):

1. A **complex manifold** $X$ of complex dimension $m$ (real dimension
   $2m$), with holomorphic tangent bundle $T^{1,0}X$.
2. A **Kähler** structure $(g,J,\omega)$: a Riemannian metric $g$
   compatible with a complex structure $J$, and Kähler form
   $\omega(u,v)=g(Ju,v)$.
3. The **first Chern class** $c_1(X)=c_1(T^{1,0}X)\in H^{2}(X;\mathbb{Z})$.

A compact Kähler $X$ is **Calabi–Yau** (in this working sense) when
$c_1(X)=0$. Yau's theorem then supplies a unique Ricci‑flat Kähler
metric in each Kähler class: the holonomy reduces to a subgroup of
$\mathrm{SU}(m)$. Equivalently, $X$ admits a nowhere‑vanishing
holomorphic volume form $\Omega\in H^{0}(X,K_X)$ (trivial canonical
bundle $K_X=\det T^{*1,0}X$).

Threefolds ($m=3$) are the standard internal spaces for heterotic /
type‑II compactifications to four dimensions. Hodge numbers
$h^{p,q}=\dim H^{p,q}(X)$ and cycles control generations and moduli.

We do **not** construct Ricci‑flat metrics in code. The lab below only
checks a **toy algebraic condition** that appears in the Weierstrass
model of an elliptic K3 / CY fibre — the discriminant of
$y^{2}=x^{3}+fx+g$ — as a reminder that “CY geometry” starts from
equations on ambient projective space, not from slogans.

### Where it shows up

- **String compactifications:** topology of the internal space →
  four‑dimensional effective field theory.
- **Mirror symmetry:** pairs of Calabi–Yau spaces with exchanged Hodge
  data.
- **Existence / moduli of Ricci‑flat metrics** as pure geometry (Yau).

> **Further reading.** D. Joyce, *Compact Manifolds with Special
> Holonomy*, OUP, 2000, Ch. on Calabi–Yau. S.‑T. Yau, “On the Ricci
> curvature of a compact Kähler manifold and the complex Monge–Ampère
> equation, I,” *Comm. Pure Appl. Math.* **31** (1978) 339–411.
> M. Gross, D. Huybrechts, D. Joyce, *Calabi–Yau Manifolds and Related
> Geometries*, Springer, 2003 (survey lectures).

**Off‑ramp:** [`../special_holonomy`](../special_holonomy/) (holonomy
tool + CY deepen). This section stops at the working definition and
the discriminant cartoon.
"""

C1_CODE = r"""
# Cartoon only: Weierstrass discriminant Δ = -16(4f³+27g²) for y²=x³+f x+g.
# Singular fibres where Δ=0; smooth elliptic fibres where Δ≠0.
# Not a CY metric — just the algebraic locus that appears in elliptic
# fibrations used as CY building blocks.

f = -1.0
gs = np.linspace(-1.5, 1.5, 400)
disc = -16.0 * (4.0 * f**3 + 27.0 * gs**2)

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.plot(gs, disc, "C0-", lw=2)
ax.axhline(0, color="0.5", lw=1)
ax.fill_between(gs, disc, 0, where=(disc > 0), alpha=0.12, color="C2",
                label=r"$\Delta>0$ (smooth fibres, schematic)")
ax.fill_between(gs, disc, 0, where=(disc < 0), alpha=0.12, color="C3",
                label=r"$\Delta<0$")
ax.set_xlabel(r"$g$")
ax.set_ylabel(r"$\Delta(f,g)$")
ax.set_title(r"Weierstrass discriminant at fixed $f=-1$ (cartoon)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
print("CY claim in this notebook: c1=0 + Kähler ⇒ Ricci-flat metric exists (Yau).")
print("Code here only plots Δ; it does not build a Ricci-flat metric.")
"""

C1_TAKE = r"""
**Takeaway box — C1.**  
Working definition: compact Kähler $X$ with $c_1(X)=0$ (holomorphic
volume form; holonomy in $\mathrm{SU}(m)$ after Yau). Code: discriminant
cartoon only. Next depth:
[`../special_holonomy`](../special_holonomy/).
"""

C2 = r"""
## C2. $G_2$ manifolds

### Background

M‑theory is formulated in eleven dimensions. Compactifying on a
7‑dimensional internal space while keeping some supersymmetry selects
manifolds whose holonomy is the exceptional Lie group $G_2$, not
$\mathrm{SU}(3)$ as in Calabi–Yau threefolds (C1).

### Working setup (declared once)

1. **$G_2$** is the exceptional simple Lie group of dimension $14$,
   realised as the automorphism group of the normed algebra of
   octonions — equivalently, as the stabilizer in $\mathrm{GL}(7,\mathbb{R})$
   of a generic 3‑form on $\mathbb{R}^{7}$.
2. On $\mathbb{R}^{7}$ with coordinates $(x^{1},\ldots,x^{7})$, a
   standard associative 3‑form is
   $$
   \varphi_0
     =dx^{123}+dx^{145}+dx^{167}+dx^{246}
      -dx^{257}-dx^{347}-dx^{356},
   $$
   where $dx^{ijk}=dx^{i}\wedge dx^{j}\wedge dx^{k}$. The metric and
   orientation are recovered from $\varphi_0$ by a pointwise algebraic
   formula (Joyce).
3. A smooth oriented 7‑manifold $Y$ with a stable 3‑form $\varphi$ has
   **holonomy contained in $G_2$** when $\nabla\varphi=0$ for the
   Levi‑Civita connection of the metric determined by $\varphi$. Then
   $Y$ is Ricci‑flat. Compact examples with holonomy **exactly** $G_2$
   were constructed by Joyce in the 1990s.

We do **not** build $G_2$‑holonomy metrics here. The lab only verifies
that the model 3‑form $\varphi_0$ is alternating of rank $\binom{7}{3}$
data reduced to the seven terms above — a coordinate check that the
object being preserved is a concrete tensor on $\mathbb{R}^{7}$.

### Where it shows up

- **M‑theory compactifications** to four dimensions with
  $\mathcal{N}=1$ supersymmetry.
- **Exceptional holonomy geometry** alongside $\mathrm{Spin}(7)$ in
  eight dimensions.

> **Further reading.** Joyce, *Compact Manifolds with Special
> Holonomy*, OUP, 2000, Part II ($G_2$ and $\mathrm{Spin}(7)$).

**Off‑ramp:** [`../special_holonomy`](../special_holonomy/) ($G_2$
deepen + $\mathrm{Spin}(7)$ map).
"""

C2_CODE = r"""
# Coordinate check: list the seven monomials in φ0 and confirm they are
# distinct oriented triples in {1..7}. No metric construction.

terms = [
    (1, 2, 3),
    (1, 4, 5),
    (1, 6, 7),
    (2, 4, 6),
    (2, 5, 7),  # appears with a minus in φ0
    (3, 4, 7),  # minus
    (3, 5, 6),  # minus
]
signs = [+1, +1, +1, +1, -1, -1, -1]
assert len(terms) == 7 and len(set(terms)) == 7
assert all(len(set(t)) == 3 and all(1 <= i <= 7 for i in t) for t in terms)

# Count of independent 3-form coefficients on R^7 vs G2 stabilizer dimension
n_three_forms = 35  # C(7,3)
dim_GL7 = 49
dim_G2 = 14
print("C(7,3) =", n_three_forms, "  dim GL(7) =", dim_GL7, "  dim G2 =", dim_G2)
print("Model φ0 uses", len(terms), "signed triples:")
for s, t in zip(signs, terms):
    print(f"  {s:+d} dx^{t[0]}{t[1]}{t[2]}")

# Tiny bar chart: dimensions in the stabilizer story
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(["3-forms", "GL(7)", "G2"], [n_three_forms, dim_GL7, dim_G2],
       color=["C0", "C1", "C2"])
ax.set_ylabel("dimension")
ax.set_title(r"Sizes in the $G_2\subset\mathrm{GL}(7)$ picture (schematic)")
plt.tight_layout()
plt.show()
"""

C2_TAKE = r"""
**Takeaway box — C2.**  
A $G_2$‑manifold is a Ricci‑flat 7‑fold with parallel stable 3‑form
$\varphi$ (holonomy in $G_2$). Model $\varphi_0$ on $\mathbb{R}^{7}$ is
an explicit tensor; compact exact‑holonomy examples exist (Joyce). Code:
coordinate inventory only. Next depth:
[`../special_holonomy`](../special_holonomy/).
"""

C3 = r"""
## C3. Twistor space $\mathbb{CP}^{3}$

### Background

Part II‑B treated flat spacetime $\mathbb{R}^{1,3}$ with its Lorentzian
metric and null cone. Penrose's twistor programme asks whether that
**null** (light‑ray) structure is better organised in a complex
projective space than in $\mathbb{R}^{1,3}$ itself. Massless fields and
some integrable systems become cleaner once incidence in projective
geometry replaces metric algebra on the light cone.

### Working setup (extrinsic, declared once)

Throughout (C3) we work with three declared spaces:

1. **Real Minkowski space** $M=\mathbb{R}^{1,3}$ with metric
   $\eta=\mathrm{diag}(-1,1,1,1)$ as in II‑B (B1).
2. **Complexified Minkowski space**
   $\mathbb{CM}=\mathbb{C}^{1,3}$ — the complexification of $M$ as an
   affine space (same $\eta$, now on complex vectors). Null lines of
   $\mathbb{CM}$ are the complex geodesics with $\eta(v,v)=0$.
3. **Twistor space**
   $$
   \mathbb{PT}
     :=\mathbb{CP}^{3}
     =(\mathbb{C}^{4}\setminus\{\mathbf{0}\})/\mathbb{C}^{\times},
   $$
   the set of complex lines through the origin in $\mathbb{C}^{4}$. A
   point of $\mathbb{PT}$ is a homogeneous coordinate
   $Z=[Z^{0}:Z^{1}:Z^{2}:Z^{3}]$, with $Z\sim\lambda Z$ for
   $\lambda\in\mathbb{C}^{\times}$.

No claim is made that $\mathbb{PT}$ is a Lie group, nor that we build
the full 2‑spinor calculus here. The object is the **incidence
dictionary** between $\mathbb{PT}$ and null geometry in $\mathbb{CM}$.

### Geometry — incidence dictionary

Write a twistor in 2‑spinor blocks
$Z^{\alpha}=(\omega^{A},\pi_{A'})$ with $A,A'\in\{0,1\}$, so
$Z\in\mathbb{C}^{4}$ and $[Z]\in\mathbb{PT}$. A point
$x\in\mathbb{CM}$ has Hermitian spinor coordinates $x^{AA'}$
(the standard isomorphism $\mathbb{C}^{1,3}\cong\mathrm{Herm}(2)$).
The **incidence relation** is the linear equation
$$
\omega^{A}
  =ix^{AA'}\pi_{A'}
  \qquad
  (A=0,1).
$$
For fixed nonzero $\pi_{A'}$, this is one linear constraint on $x$;
its solution set in $\mathbb{CM}$ is a **null line** (complex light
ray). Scaling $(\omega,\pi)\mapsto\lambda(\omega,\pi)$ does not change
the line, so the geometric object attached to a **point** of
$\mathbb{PT}$ is that null line — not a single event of $M$.

Conversely, fix an event $x\in\mathbb{CM}$. The twistors incident with
$x$ are the solutions $(\omega,\pi)$ of the same equation. The free
data is the projective class $[\pi_{A'}]\in\mathbb{CP}^{1}$, and
$\omega^{A}$ is then determined by $x$ and $\pi$. Thus a spacetime
point corresponds to a **line**
$\mathbb{CP}^{1}\subset\mathbb{PT}$ — the set of all twistors through
that event.

In short (one dictionary, both directions):

| in $\mathbb{PT}$ | in $\mathbb{CM}$ |
|---|---|
| point $[Z]\in\mathbb{CP}^{3}$ | one null line |
| line $\mathbb{CP}^{1}\subset\mathbb{CP}^{3}$ | one spacetime point $x$ |

Incidence replaces some metric calculations on the light cone with
projective linear algebra in $\mathbb{CP}^{3}$. Full contour‑integral
formulae for massless fields, and the reality conditions that select
real $M\subset\mathbb{CM}$, are in the dedicated twistor notebook.

### Where it shows up

- **Classical relativity / massless fields:** contour‑integral formulae
  for solutions of helicity field equations (Penrose transform).
- **Scattering amplitudes** (modern descendants): twistor and
  momentum‑twistor variables in gauge‑theory amplitude methods.
- **Integrable systems:** several soliton equations have natural
  twistor constructions.

> **Further reading.** R. Penrose, “Twistor algebra,” *J. Math. Phys.*
> **8** (1967) 345–366. R. Penrose and W. Rindler, *Spinors and
> Space‑Time*, Vol. 2, Cambridge, 1986, Ch. 6. S. A. Huggett and
> K. P. Tod, *An Introduction to Twistor Theory*, 2nd ed., Cambridge,
> 1994, Ch. 1–2 (incidence in coordinates).

**Off‑ramp:** the dedicated twistor notebook
[`../twistor_theory`](../twistor_theory/) — geometry first (spinors,
reality, Penrose transform), then a field‑equation sketch. This
section stops at the incidence dictionary and the schematic below.
"""

C3_CODE = r"""
# Schematic only: incidence cartoon (not a twistor engine).
# Left: one null ray in R^{1,1}  ↔  one point of PT.
# Right: CP^1 of directions at a fixed event  ↔  line in PT.

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# --- left: Minkowski null line ---
ax = axes[0]
xs = np.linspace(-1.5, 1.5, 200)
ax.plot(xs, xs, "C0-", lw=2.5, label=r"null ray ($t=x$)")
ax.plot(xs, -xs, color="0.75", lw=1, ls="--")
ax.scatter([0], [0], c="C3", s=40, zorder=5)
ax.annotate(
    r"one $[Z]\in\mathbb{CP}^{3}$" + "\n" + r"$\leftrightarrow$ this null line",
    xy=(0.9, 0.9), xytext=(0.15, 1.25),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="0.3"),
)
ax.set_aspect("equal")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.set_title(r"Point of $\mathbb{PT}$ $\leftrightarrow$ null line")
ax.legend(fontsize=8, loc="lower right")

# --- right: CP^1 of twistors through one event ---
ax = axes[1]
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), "C0-", lw=2.5)
# three sample π-directions on the circle (S^1 ⊂ CP^1 chart caricature)
for ang, lab in [(0.3, r"$[\pi]_1$"), (2.0, r"$[\pi]_2$"), (4.2, r"$[\pi]_3$")]:
    ax.scatter([np.cos(ang)], [np.sin(ang)], c="C3", s=36, zorder=5)
    ax.annotate(lab, xy=(np.cos(ang), np.sin(ang)),
                xytext=(1.15 * np.cos(ang), 1.15 * np.sin(ang)),
                fontsize=9, ha="center")
ax.scatter([0], [0], c="k", s=28, zorder=5)
ax.annotate(
    r"event $x$" + "\n" + r"$\leftrightarrow$ this $\mathbb{CP}^{1}$",
    xy=(0, 0), xytext=(-0.2, -1.35),
    fontsize=9, ha="center",
)
ax.set_aspect("equal")
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.axis("off")
ax.set_title(r"Spacetime point $\leftrightarrow$ line in $\mathbb{PT}$")

fig.suptitle(
    r"Incidence cartoon (schematic): $\mathbb{PT}=\mathbb{CP}^{3}$ vs null geometry",
    fontsize=11,
)
plt.tight_layout()
plt.show()
print("Incidence: ω^A = i x^{AA'} π_{A'}  (homogeneous; not evaluated numerically here).")
"""

C3_TAKE = r"""
**Takeaway box — C3.**  
Manifold: twistor space $\mathbb{PT}=\mathbb{CP}^{3}$. Dictionary:
point of $\mathbb{PT}$ $\leftrightarrow$ null line in $\mathbb{CM}$;
spacetime point $\leftrightarrow$ $\mathbb{CP}^{1}$ in $\mathbb{PT}$,
via $\omega^{A}=ix^{AA'}\pi_{A'}$. Next depth:
[`../twistor_theory`](../twistor_theory/).
"""

C4 = r"""
## C4. Elliptic curves, lattices, and PQC (off‑ramp)

### Background

Public‑key cryptography needs hard problems with an easy “trapdoor”
direction. For decades that meant factoring and discrete logarithms —
including discrete logs on **elliptic curves**, which give smaller keys
than multiplicative groups of finite fields. Large quantum computers
break those via Shor's algorithm, so **post‑quantum cryptography
(PQC)** shifts to problems believed harder for quantum adversaries,
notably on **lattices**.

### Working setup (declared once)

**Elliptic curves.** Over a field $k$ of $\mathrm{char}\neq 2,3$, a
Weierstrass model
$$
E:\; y^{2}=x^{3}+Ax+B,
\qquad
\Delta:=-16(4A^{3}+27B^{2})\neq 0,
$$
cuts out a smooth genus‑one curve. With a chosen point $O$ (the point
at infinity in the projective model), the chord‑and‑tangent law makes
$E(k)$ an abelian group. Over $\mathbb{C}$, every elliptic curve is a
complex torus $\mathbb{C}/\Lambda$ for a lattice
$\Lambda=\mathbb{Z}+\mathbb{Z}\tau$ with $\mathrm{Im}\,\tau>0$;
isomorphism classes are parametrised by the modular curve
$\mathrm{SL}_{2}(\mathbb{Z})\backslash\mathbb{H}$.

**Lattices.** A **lattice** $\Lambda\subset\mathbb{R}^{n}$ is a discrete
cocompact subgroup (equivalently $\Lambda=B\mathbb{Z}^{n}$ for
invertible $B\in\mathrm{GL}(n,\mathbb{R})$). The quotient
$\mathbb{R}^{n}/\Lambda$ is a smooth $n$‑torus. Cryptographic hardness
(short / close vectors, LWE, …) is stated on lattices; the manifold
picture here is only that the torus $\mathbb{R}^{n}/\Lambda$ is a
compact flat manifold whose geometry depends on $\Lambda$.

This section is the **geometric invitation**. Implementations,
parameter sets, and hybrid TLS live in
[`../post_quantum_cryptography`](../post_quantum_cryptography/).

### Where it shows up

- **Classical ECC** in TLS, SSH, blockchains (still widely deployed).
- **PQC standards:** ML‑KEM, ML‑DSA, SLH‑DSA, … for encryption and
  signatures meant to survive quantum adversaries.
- **Protocol engineering:** hybrid key exchange in TLS; Open Quantum
  Safe ([`liboqs`](https://github.com/open-quantum-safe/liboqs),
  [`oqs-provider`](https://github.com/open-quantum-safe/oqs-provider)).

> **Further reading (geometry / ECC).** J. H. Silverman, *The
> Arithmetic of Elliptic Curves*, 2nd ed., Springer, 2009. J. H.
> Silverman, *An Introduction to Lattices, Lattice Reduction, and
> Lattice‑Based Cryptography*, PCMI lecture notes.
"""

C4_CODE = r"""
# Left: real Weierstrass locus + Δ check.
# Right: 2D lattice Λ = Z v1 + Z v2 and the fundamental parallelogram
# (torus R²/Λ schematic). Not a crypto implementation.

a, b = -1.0, 1.0
disc = -16.0 * (4.0 * a**3 + 27.0 * b**2)
xs = np.linspace(-2, 2, 400)
ysq = xs ** 3 + a * xs + b
mask = ysq >= 0

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

ax = axes[0]
ax.plot(xs[mask], np.sqrt(ysq[mask]), "C0-", lw=2)
ax.plot(xs[mask], -np.sqrt(ysq[mask]), "C0-", lw=2)
ax.set_title(rf"$y^2=x^3{a:+}x{b:+}$" + rf"  ($\Delta={disc:.0f}$)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_aspect("equal")

ax = axes[1]
v1 = np.array([1.0, 0.0])
v2 = np.array([0.4, 0.9])
# draw a few lattice points
pts = [i * v1 + j * v2 for i in range(-2, 3) for j in range(-2, 3)]
P = np.array(pts)
ax.scatter(P[:, 0], P[:, 1], c="C0", s=22, zorder=3)
# fundamental parallelogram
para = np.array([[0, 0], v1, v1 + v2, v2, [0, 0]])
ax.plot(para[:, 0], para[:, 1], "C3-", lw=2)
ax.fill(para[:, 0], para[:, 1], color="C3", alpha=0.15)
ax.set_aspect("equal")
ax.set_title(r"Lattice $\Lambda\subset\mathbb{R}^2$; torus $=\mathbb{R}^2/\Lambda$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

fig.suptitle("C4 schematics: elliptic curve (real) and lattice torus", fontsize=11)
plt.tight_layout()
plt.show()
print(f"Weierstrass Δ = {disc:.6g} (nonsingular over R chart iff Δ≠0).")
print("PQC implementations: use the PQC module + OQS/RFCs, not these plots.")
"""

C4_TAKE = r"""
**Takeaway box — C4.**  
Elliptic curve: smooth Weierstrass model ($\Delta\neq 0$) with group
law; over $\mathbb{C}$, a torus $\mathbb{C}/\Lambda$. Lattice:
discrete $\Lambda\subset\mathbb{R}^{n}$; torus $\mathbb{R}^{n}/\Lambda$.
Engineering depth:
[`../post_quantum_cryptography`](../post_quantum_cryptography/).
"""


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(PART_IIC))
    cells.append(md(C1))
    cells.append(code(C1_CODE))
    cells.append(md(C1_TAKE))
    cells.append(md(C2))
    cells.append(code(C2_CODE))
    cells.append(md(C2_TAKE))
    cells.append(md(C3))
    cells.append(code(C3_CODE))
    cells.append(md(C3_TAKE))
    cells.append(md(C4))
    cells.append(code(C4_CODE))
    cells.append(md(C4_TAKE))
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
