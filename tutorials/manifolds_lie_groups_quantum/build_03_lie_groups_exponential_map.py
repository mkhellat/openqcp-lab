#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 03 (source of truth): Lie groups and exp map.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_03_lie_groups_exponential_map.py

The generated notebook is ``03_lie_groups_exponential_map.ipynb``
beside this file. Markdown and the matrix-exponential check are
Part I §§3–5 from ``build_notebook.py`` (tangent vectors, the
exponential map, Lie groups and Lie algebras, and GL(n, C)).
Examples (M1)–(M3) and the physics examples are notebook 01,
charts are notebook 02, and the tiers are notebooks 04–06. This
builder does not rewrite that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part I §§3–5 have level-2 headings only,
so the section titles stay plain heading cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("03_lie_groups_exponential_map.ipynb")

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
# Lie groups and the exponential map

This notebook is Part I §§3–5 from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
tangent vectors and the exponential map, Lie groups and Lie
algebras, and $\mathrm{GL}(n,\mathbb{C})$ with the matrix
exponential series. The wording below is that text. Examples
(M1)–(M3) and physics examples (P1)–(P3) are notebook 01. Charts
and atlases are notebook 02. The application tiers are notebooks
04–06. Part III is notebook 07. The map is `00_series_guide.ipynb`.

The code cell checks the power series against `scipy.linalg.expm`,
the one-parameter subgroup ODE, and $\exp(X)\exp(-X)=I$. There is
no sibling demo.
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

SECTION_3 = r'''
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
'''

SECTION_4 = r'''
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
'''

SECTION_5 = r'''
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
'''

SERIES = r'''
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
'''

TAKEAWAY = r'''
**Takeaway box — \(\mathrm{GL}\) / \(\mathfrak{gl}\).**  
Manifold / Lie group: invertible matrices. Lie algebra: all matrices.
Exponential: series (1), always defined; solves the one-parameter
subgroup ODE. Hamiltonians sit in the algebra; time evolution sits in
the group when the exponential is unitary (next section) or, more
generally, invertible.
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(SECTION_3))
    cells.append(md(SECTION_4))
    cells.append(md(SECTION_5))
    cells.append(code(SERIES))
    cells.append(md(TAKEAWAY))
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
