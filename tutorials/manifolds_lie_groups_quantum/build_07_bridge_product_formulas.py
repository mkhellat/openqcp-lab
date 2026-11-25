#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 07 (source of truth): bridge to product formulas.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_07_bridge_product_formulas.py

The generated notebook is ``07_bridge_product_formulas.ipynb``
beside this file. Markdown and the Lie–Trotter check are Part III
from ``build_notebook.py`` (the Lie product formula and the 2×2
Hermitian example). Examples (M1)–(M3), charts, Lie groups, and
Tiers A–C are notebooks 01–06. The series map, the global summary,
and the bookshelf are notebook 00. This builder does not rewrite
that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part III has a level-1 title only, so
that title stays a plain heading cell.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("07_bridge_product_formulas.ipynb")

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
# Bridge to product formulas

This notebook is Part III from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
the Lie product formula and the 2×2 Hermitian Trotter check.
The wording below is that text. Examples (M1)–(M3), charts,
Lie groups, and Tiers A–C are notebooks 01–06. The series map
and the global summary are `00_series_guide.ipynb`.

The code cell checks
$\|\exp(t(A+B))-(\exp(tA/r)\exp(tB/r))^r\|$
for $r=1,2,4,8,16,32$. It calls `random_hermitian`, which the
mega notebook defines in the Tier A $\mathrm{U}(n)$ lab. The
setup cell below defines that same helper so this check runs
alone. There is no sibling demo.
'''

SETUP = r'''
# Setup — foundations notebook (numpy / scipy / matplotlib only)
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, det, eigvalsh
from scipy.linalg import expm, logm

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (5.5, 4.0), "font.size": 11})

# Defined in notebook 04's Tier A U(n) cell. Part III calls it.
def random_hermitian(n, rng):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / 2

print("numpy", np.__version__)
'''

PART_III = r'''
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
'''

TROTTER = r'''
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
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(PART_III))
    cells.append(code(TROTTER))
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
