#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 05 (source of truth): Tier B light labs.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_05_tier_b_light_labs.py

The generated notebook is ``05_tier_b_light_labs.ipynb``
beside this file. Markdown and the two lab checks are Part II-B
from ``build_notebook.py`` (Minkowski spacetime and the Berry
phase on a latitude of the Bloch sphere). Examples (M1)–(M3),
charts and atlases, Lie groups, and Tiers A and C stay in the
mega notebook. This builder does not rewrite that text.

Heading cells are split the same way as the mega notebook:
level-4 titles become slate banners, and level-5 titles become
orange banners. The banner helpers below are the ones in
``build_notebook.py``. Part II-B uses level-2 and level-3
headings only, so those titles stay plain heading cells.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("05_tier_b_light_labs.ipynb")


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
# Tier B light labs

This notebook is Part II‑B from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
Minkowski spacetime as a Lorentzian manifold, and a $U(1)$ Berry
phase on a latitude of the Bloch sphere. The wording below is
that text. Examples (M1)–(M3), charts and atlases, Lie groups
and the exponential map, and Tiers A and C stay in the mega
notebook.

The code cells check the light cone in $\mathbb{R}^{1,1}$ and
the latitude Berry phase. There is no sibling demo.
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

PART_IIB = r'''
# Part II‑B — Light labs

Same three‑part shape as Tier A (background → geometry → applications),
with shorter code.

## B1. Minkowski spacetime as a smooth manifold

### Background

Special relativity is not “Euclidean geometry with an extra
coordinate.” Events form a four‑dimensional set of points, and the
physically meaningful notion of length is the **Minkowski** interval,
which treats time differently from space. Light rays sit on the
boundary between travel that is possible for massive particles and
travel that is not.

### Geometry

$\mathbb{R}^{1,3}$ with metric $\eta=\mathrm{diag}(-1,1,1,1)$ (or the
opposite sign convention) is a smooth manifold diffeomorphic to
ordinary $\mathbb{R}^4$, but the **metric** distinguishes timelike,
null, and spacelike vectors:

$$
v\cdot v = \eta_{\mu\nu}v^\mu v^\nu
  \quad
  \begin{cases}
  <0 & \text{timelike (massive worldlines)},\\
  =0 & \text{null (light rays)},\\
  >0 & \text{spacelike}.
  \end{cases}
$$

Lorentz transformations are the Lie group $\mathrm{O}(1,3)$ preserving
$\eta$; adjoining translations gives the Poincaré group.

### Where it shows up

- **Particle kinematics:** energy–momentum relations, decay thresholds,
  collider frame changes.
- **Field theory:** causal structure of wave equations; light cones
  bound signal propagation.
- **General relativity:** curved spacetime is a Lorentzian manifold;
  Minkowski space is the flat local model (Wald, Carroll).

> **Further reading.** S. Carroll, *Spacetime and Geometry*, Addison
> Wesley / Cambridge reprint, Ch. 1–2. R. Wald, *General Relativity*,
> Chicago, 1984, Ch. 1–2 (for the manifold + metric viewpoint).
'''

B1_CODE = r'''
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
'''

B1_TAKE = r'''
**Takeaway box — B1.**  
Manifold: spacetime (here flat Minkowski). Extra structure: Lorentzian
metric. Isometry Lie group: Lorentz / Poincaré.
'''

B2 = r'''
## B2. $U(1)$ bundles and Berry phase (sketch with code)

### Background

If you slowly cycle the knobs of a Hamiltonian and return them to the
start, the quantum state need not return with only the usual dynamical
phase $e^{-iEt/\hbar}$. An extra **geometric** phase can remain — it
depends on the *path* in parameter space, not on how fast you traverse
it (in the adiabatic idealisation). Berry’s insight was that this phase
is the holonomy of a natural connection on the bundle of eigenstates.

### Geometry

A **principal $U(1)$ bundle** over a base $M$ assigns a circle's worth
of phases to each point of $M$, glued consistently. In quantum
mechanics, a normalised eigenstate $|n(\mathbf{R})\rangle$ depending on
slow parameters $\mathbf{R}$ defines a line bundle over the parameter
manifold; transporting the state around a closed loop accumulates a
geometric **Berry phase**.

For a spin‑1/2 in a magnetic field $\mathbf{B}=B\,\hat{\mathbf{n}}$, a
loop of $\hat{\mathbf{n}}$ on the Bloch sphere enclosing solid angle
$\Omega$ gives Berry phase

$$
\gamma = -\frac{\Omega}{2}
$$

for the ground state (monopole curvature of strength $1/2$ at the
origin of $\mathbf{B}$‑space).

### Where it shows up

- **Molecular / condensed‑matter physics:** electronic states that pick
  up Berry phases when nuclear coordinates cycle; anomalous velocity
  terms in solids.
- **Quantum control:** holonomic gates that encode computation in
  geometric phases, aiming for robustness to timing noise.
- **Topology:** Chern numbers and related invariants are integrals of
  Berry curvature — the same geometry at a global scale.

> **Further reading.** M. Nakahara, *Geometry, Topology and Physics*,
> 2nd ed., Taylor & Francis, 2003, Ch. 9–10 (bundles, connections,
> Berry phase §10.6). M. V. Berry, “Quantal phase factors accompanying
> adiabatic changes,” *Proc. R. Soc. Lond. A* **392** (1984) 45–57.
'''

B2_CODE = r'''
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
'''

B2_TAKE = r'''
**Takeaway box — B2.**  
Base manifold: parameter space (here \(S^2\) of field directions).
Bundle: \(U(1)\) phases of eigenstates. Connection: Berry connection;
curvature → solid angle / monopole.
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(PART_IIB))
    cells.append(code(B1_CODE))
    cells.append(md(B1_TAKE))
    cells.append(md(B2))
    cells.append(code(B2_CODE))
    cells.append(md(B2_TAKE))
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
