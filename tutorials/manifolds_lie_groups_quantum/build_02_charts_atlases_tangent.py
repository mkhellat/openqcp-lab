#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 02 (source of truth): charts, atlases, transitions.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_02_charts_atlases_tangent.py

The generated notebook is ``02_charts_atlases_tangent.ipynb``
beside this file. Markdown and the circle check are Part I §2
from ``build_notebook.py`` (atlas, charts, and transition maps
on the circle). Examples (M1)–(M3) and the physics examples
are notebook 01; Lie groups and the tiers are notebooks 03–06.
This builder does not rewrite that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part I §2 has a level-2 heading only, so
the section title stays a plain heading cell.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("02_charts_atlases_tangent.ipynb")


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
# Charts, atlases, and transition maps

This notebook is Part I §2 from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
a chart, an atlas, and $C^\infty$ transition maps, checked on
two angle charts of the circle. The wording below is that text.
Examples (M1)–(M3) and physics examples (P1)–(P3) are notebook
01. Tangent vectors, Lie groups, and the exponential map are
notebook 03. The application tiers are notebooks 04–06. The map
is `00_series_guide.ipynb`.

The code cell checks the two angle charts on $S^1$ and the
transition on the upper and lower arcs. The figure is drawn in
that cell. There is no sibling demo.
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

SECTION_2 = r'''
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
'''

CIRCLE = r'''
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
'''

TAKEAWAY = r'''
**Takeaway box — what is the manifold here?**  
\(M=S^1\), dimension \(1\). Charts: open arcs with angle coordinates.
Transition maps are \(\theta\mapsto\theta\) or \(\theta\mapsto\theta+2\pi\),
both \(C^\infty\) on their domains. Smoothness of functions on the
circle does not depend on which chart we use.
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(SECTION_2))
    cells.append(code(CIRCLE))
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
