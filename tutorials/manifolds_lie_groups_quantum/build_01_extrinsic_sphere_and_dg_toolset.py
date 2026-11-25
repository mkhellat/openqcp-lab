#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 01 (source of truth): extrinsic sphere and DG toolset.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_01_extrinsic_sphere_and_dg_toolset.py

The generated notebook is ``01_extrinsic_sphere_and_dg_toolset.ipynb``
beside this file. Markdown and the check cell are the (M1)–(M3)
blocks from ``build_notebook.py``, including the hidden launch of
``cylinder_group_demo.py``. Physics examples (P1)–(P3) and Part I
§2 onward stay in the mega notebook. This builder does not rewrite
that text.

Heading cells are split the same way as the mega notebook: level-4
titles such as (M1) become slate banners, and (M1.1)–(M1.5) become
orange banners. The banner helpers below are the ones in
``build_notebook.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("01_extrinsic_sphere_and_dg_toolset.ipynb")

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
# Extrinsic sphere and the differential-geometry toolset

This notebook is examples (M1)–(M3) from
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`:
the unit sphere in $\mathbb{R}^3$, directions in the plane, and
rotation matrices, with the cylinder-group figure. The wording
below is that text. Physics examples (P1)–(P3), charts and
atlases (§2), Lie groups, and the application tiers stay in the
mega notebook.

The code cells check the extrinsic calculus from (M1)–(M3). The
cylinder figure is `cylinder_group_demo.py` in this folder; the
notebook only launches it. Example titles such as (M1) use a slate
banner; (M1.1)–(M1.5) use an orange banner.
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

PART_I = r'''
# Part I — Smooth manifolds and Lie groups

## 1. Sets of points that are not vector spaces

In a real vector space $V$ you may pick any two vectors $u,v\in V$ and
any scalar $\alpha\in\mathbb{R}$ and form

$$
u+v\in V,
\qquad
\alpha u\in V.
$$

The examples below are sets for which those operations on *points*
either leave the set, depend on arbitrary choices, or are not the
combination law actually used — so they are not vector spaces of
points. We state that briefly where it is obvious, and spend the space
on the **differential** fact: curves and functions on the set still
admit ordinary calculus once velocities live in tangent spaces and
coordinates are local. §2 packages that pattern as charts on a general
manifold. The examples use only linear algebra and (for physics)
undergraduate quantum mechanics.

### 1.1 Mathematics examples

#### (M1) The unit sphere in $\mathbb{R}^3$

**Working setup (extrinsic, declared once).** Throughout (M1) we
treat $S^2$ as a subset of the Euclidean space $\mathbb{R}^3$. Write
$\cdot$ for the standard bilinear product
$(u,v)\mapsto u_1v_1+u_2v_2+u_3v_3$ on $\mathbb{R}^3$, and
$\|u\|=\sqrt{u\cdot u}$. Define
$$
S^2
  =\bigl\{p\in\mathbb{R}^3:p\cdot p=1\bigr\}.
$$
So both a point of $S^2$ and a vector in $\mathbb{R}^3$ live in **the
same** ambient vector space; the product $a\cdot b$ is only ever used
when $a,b\in\mathbb{R}^3$.

**Curves and derivatives (ordinary calculus).** A $C^1$ curve on the
sphere is a $C^1$ map
$\gamma:(-\varepsilon,\varepsilon)\to\mathbb{R}^3$ whose values lie in
$S^2$. The derivative $\gamma'(t)$ means the ordinary derivative of
this $\mathbb{R}^3$-valued map **at the parameter** $t\in\mathbb{R}$ —
not “at the point $\gamma(t)$.” Both $\gamma(t)$ and $\gamma'(t)$ are
elements of $\mathbb{R}^3$. Differentiating the scalar identity
$\gamma(t)\cdot\gamma(t)=1$ at $t$ gives
$2\,\gamma(t)\cdot\gamma'(t)=0$, hence
$\gamma(t)\cdot\gamma'(t)=0$. That identity is available only because
both factors sit in $\mathbb{R}^3$.

**Tangent plane as a subspace of the same $\mathbb{R}^3$.** For
$p\in S^2$ set
$$
T_p S^2
  :=\bigl\{v\in\mathbb{R}^3:p\cdot v=0\bigr\}.
$$
This is a linear **subspace** of $\mathbb{R}^3$ of dimension $2$ (the
orthogonal complement of $\mathrm{span}\{p\}$). It is not a separate
abstract “$2$-D space” floating beside $\mathbb{R}^3$: every
$v\in T_p S^2$ is still a triple of real numbers, and
$\gamma'(t)\in T_{\gamma(t)}S^2\subset\mathbb{R}^3$.

This set $S^2$ is **not** a vector subspace of $\mathbb{R}^3$ under the
ambient $+/\cdot$: it is not closed under addition (or under scaling).
For any distinct pair of points $p,q\in S^2$, the sum $p+q$ lies on
$S^2$ if and only if $p\cdot q=-1/2$ (then $\|p+q\|=1$); otherwise
$p+q\notin S^2$. So ambient addition is not a closed operation on
$S^2$.

##### (M1.1) Why continuous $+/\cdot$ on $S^2$ cannot work

Ambient $+/\cdot$ failed. Could some **other** pair of operations on
the points of $S^2$ still satisfy the vector-space axioms?

**Why we insist on continuity.** A discontinuous map
$\oplus:S^2\times S^2\to S^2$ is still an operation on the **set** of
points of $S^2$. Nothing in set language forbids it. Scalar fields and
other functions *defined on* $S^2$ may also be discontinuous; analysis
studies those routinely. Continuity of a map $f$ at $p$ means: for
every $\varepsilon>0$ there exists $\delta>0$ such that
$d(x,p)<\delta$ implies $d\bigl(f(x),f(p)\bigr)<\varepsilon$ — for each
target tolerance around $f(p)$, some domain tolerance around $p$ works.
The sizes need not match, and there is no “vice versa” in that
definition.

The issue is different. The sphere already carries a topology (nearness
of points) used for limits, curves, and — later — derivatives. A
**topological** vector space on that sphere is a vector-space structure
whose $\oplus$ and $\odot$ are continuous **with respect to that same
nearness**. If $p_n\to p$ on $S^2$ but $p_n\oplus q\not\to p\oplus q$,
then limits and addition do not mesh: you cannot pass to the limit
inside a sum. That is allowed as raw algebra on a set; it is not a
structure that interacts with the differential geometry this notebook
is building on $S^2$. So the question we ask is not “does any binary
operation on the set exist?” but “can $+/\cdot$ obey the vector-space
axioms **and** be continuous for the sphere’s own topology?” The maps
under discussion are
$$
\oplus:S^2\times S^2\to S^2,
\qquad
\odot:\mathbb{R}\times S^2\to S^2.
$$

**What fails.** Under that requirement, **no** such $\oplus$ and
$\odot$ exist. The reason does not use a special formula: take *any*
continuous pair that satisfies the axioms and look at the shape of
$S^2$.

- **Compactness.** The whole sphere fits in a finite box in
  $\mathbb{R}^3$; every sequence of points on $S^2$ has a subsequence
  converging to a point still on $S^2$. Call that compactness: the
  surface has no unused “room at infinity.”

  Let $\mathbf{0}\in S^2$ be the zero of the assumed structure, and
  suppose $v\neq\mathbf{0}$. Define
  $$
  f:\mathbb{R}\to S^2,
  \qquad
  f(t)=t\odot v.
  $$
  Continuity of $\odot$ makes $f$ continuous. The axioms make $f$
  one-to-one: $t_1\odot v=t_2\odot v$ implies $t_1=t_2$. So as $t$
  runs over all real numbers, $f(t)$ traces a continuous curve on
  $S^2$ that **never revisits** a point. That is asking the compact
  sphere to contain a continuous, endless, non-repeating copy of the
  real line. It cannot. Therefore no nonzero $v$ can exist, and the
  space would have to be $\{\mathbf{0}\}$ alone — but $S^2$ has many
  points. Contradiction. So no continuous vector-space operations on
  $S^2$ exist.

- **Unboundedness (same obstruction, said with $\mathbb{R}^n$).**
  **What.** Ordinary $\mathbb{R}^n$ with its usual $+/\cdot$ is the model
  finite-dimensional real vector space. Along the line
  $t\mapsto t\cdot e_1=(t,0,\dots,0)$ you get a new point for every real
  $t$; as $|t|\to\infty$ those points leave every finite box. So
  $\mathbb{R}^n$ is **not** compact.
  **Why this matters for $S^2$.** If continuous $\oplus,\odot$ made $S^2$
  into a finite-dimensional real vector space, that structure would be
  the same kind of object as $\mathbb{R}^n$: you could pick a basis and
  pass to coordinates by a continuous bijection with continuous inverse
  (a **homeomorphism**) between $S^2$ and some $\mathbb{R}^d$.
  Homeomorphic spaces share compactness: either both are compact or
  neither is.
  **How it fails.** $S^2$ *is* compact (fits in a box; sequential
  compactness as above). $\mathbb{R}^d$ is *not*. They cannot be
  homeomorphic. Therefore $S^2$ cannot carry continuous operations
  making it a finite-dimensional real vector space — the same dead end
  as the $f(t)=t\odot v$ argument, viewed as “$S^2$ would have to look
  like flat $\mathbb{R}^d$, but it does not.”

- **A chart is not a global $\oplus$.**
  **What.** From one pole $N$ (say the north pole), **stereographic
  projection** sends every other point of $S^2$ to a unique point of the
  equatorial plane $\mathbb{R}^2$: draw the line from $N$ through
  $p\in S^2\setminus\{N\}$ and record where it hits that plane. Call the
  inverse map $\sigma:\mathbb{R}^2\to S^2\setminus\{N\}$. It is
  continuous, bijective onto the punctured sphere, and continuously
  invertible there — a chart.
  **Why one might try this.** On $\mathbb{R}^2$ the usual $+/\cdot$
  *do* give a continuous vector-space structure. Transport them to the
  punctured sphere by
  $$
  p\oplus q
    :=\sigma\bigl(\sigma^{-1}(p)+\sigma^{-1}(q)\bigr),
  \qquad
  t\odot p
    :=\sigma\bigl(t\cdot\sigma^{-1}(p)\bigr).
  $$
  On $S^2\setminus\{N\}$ these are continuous and obey the vector-space
  axioms — because $\sigma$ just renames points of the plane.
  **How it fails for the whole $S^2$.** The chart never includes $N$.
  As a point $x\in\mathbb{R}^2$ runs to infinity,
  $\sigma(x)$ approaches $N$ on the sphere. Ordinary scaling
  $t\cdot x$ with $|t|\to\infty$, or sums that leave every compact set
  in the plane, correspond under $\sigma$ to points on $S^2$ that pile
  up at $N$. There is no continuous way to assign $N\oplus p$ or
  $t\odot N$ so that the same axioms hold and the operations stay
  continuous at $N$: the missing pole is exactly where the plane’s
  “infinity” was compressed. So stereographic $+/\cdot$ is a continuous
  vector-space structure on the **open** set $S^2\setminus\{N\}$, not on
  the **closed** sphere. Compactness / unboundedness already said no
  global continuous structure exists; this bullet shows that the most
  natural local copy of $\mathbb{R}^2$ does not sneak past that ban.

(Closure without the axioms is also not enough: e.g.
$(p,q)\mapsto(p+q)/\|p+q\|$ stays on $S^2$ when $p+q\neq\mathbf{0}$,
but it is not a vector-space addition.)

##### (M1.2) No circle-style group on the closed sphere

On the circle $S^1$ each point may be recorded by a real angle
$\theta$, with the understanding that $\theta$ and $\theta+2\pi k$
(any integer $k$) name the **same** point — written
$\theta\in\mathbb{R}/2\pi\mathbb{Z}$ (“reals modulo full turns”). The
rule $(\theta,\alpha)\mapsto\theta+\alpha$ (mod $2\pi$) takes two
points of $S^1$, returns a third still on $S^1$, is smooth, has
identity $\theta=0$, and has inverses $\theta\mapsto -\theta$. That is
a smooth group operation **on the set $S^1$ itself**: the group
elements are exactly the points of the circle. On **all** of $S^2$
there is **no** smooth (even continuous) binary operation that takes
two points of the sphere, returns a third still on the sphere, and
obeys the group axioms everywhere. (Spherical coordinates
$(\phi,\theta)$ are **local coordinates** from a chart: they assign
numbers only on an open patch of $S^2$, and they fail at the poles and
along a cut in $\phi$. Adding those coordinate numbers — even
$\theta\bmod\pi$ — is arithmetic in the coordinate plane, not a group
operation defined on the points of the whole sphere.)

**Punctured** $S^2\setminus\{N,S\}$ is different: it is a cylinder.
Send $\theta\in(0,\pi)$ to $u=\cot\theta\in\mathbb{R}$ and keep
$\phi\in S^1$; then $(u,\phi)$ with componentwise addition is a Lie
group $\mathbb{R}\times S^1$. That group lives on the open cylinder,
not on the closed sphere — it does not extend continuously over the
poles.

**Addition on the punctured sphere (explicit).** Invert the
coordinate: $\sin\theta=1/\sqrt{1+u^2}$ and
$\cos\theta=u/\sqrt{1+u^2}$ for $\theta\in(0,\pi)$, so the point of
$S^2$ with cylinder coordinates $(u,\phi)$ is
$$
p(u,\phi)
  =
  \Bigl(
    \frac{\cos\phi}{\sqrt{1+u^2}},\,
    \frac{\sin\phi}{\sqrt{1+u^2}},\,
    \frac{u}{\sqrt{1+u^2}}
  \Bigr).
$$
The group operation is componentwise on the cylinder,
$$
(u_1,\phi_1)+(u_2,\phi_2)
  =(u_1+u_2,\,\phi_1+\phi_2\bmod 2\pi),
$$
and the sum of two sphere points (in this structure) is the single
sphere point
$p(u_1+u_2,\,\phi_1+\phi_2)$.

Interactive figure: **static** mode takes $(u_i,\phi_i)$ as expressions
or sliders; **animate** mode sets $\theta$ and/or $\phi$ ranges
(multiples of $\pi$), builds frames, then scrubs them with a Play /
frame control. Camera: **elev** / **azim** (or **Snap cam**). The
cylinder sum is $p(u_1+u_2,\phi_1+\phi_2)$, not $p_1+p_2$ in
$\mathbb{R}^3$.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

The interactive UI lives in
<code>cylinder_group_demo.py</code> next to this notebook so the
tutorial cell stays short. JupyterLab also collapses the launch cell
input by default (<code>source_hidden</code>). Open the <code>.py</code>
file in the editor if you want the full implementation.

</details>
'''

CYLINDER = r'''
# Launch only — implementation: cylinder_group_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "cylinder_group_demo.py").exists():
    _here = Path("tutorials/manifolds_lie_groups_quantum")
sys.path.insert(0, str(_here.resolve()))

from cylinder_group_demo import launch_cylinder_group_demo

launch_cylinder_group_demo()
'''

AFTER_FIGURE = r'''
The vector-space hope is already dead from (M1.1). The remaining claim
is stronger: even a circle-style **Lie** group (group operations
smooth as maps on the embedded sphere) cannot live on closed $S^2$.
That is where $S^0$, $S^1$, $S^3$ vs $S^2$ enters. The nowhere-zero
map $p\mapsto X(p)\in T_p S^2\subset\mathbb{R}^3$ below is **not** an
extra demand we invent; it is what such a group would **force**.

**What.** Suppose $S^2$ carried a smooth group operation $\ast$ with
identity $e\in S^2$ (inverses smooth too). Pick a $C^1$ curve of group
elements
$$
g:(-\varepsilon,\varepsilon)\to\mathbb{R}^3,
\qquad
g\bigl((-\varepsilon,\varepsilon)\bigr)\subset S^2,
\qquad
g(0)=e,
$$
such that the ordinary derivative **at the parameter** $0$ satisfies
$g'(0)\neq\mathbf{0}$ in $\mathbb{R}^3$. By the working setup,
$g'(0)\in T_e S^2\subset\mathbb{R}^3$. (Such a curve exists because
$T_e S^2\neq\{\mathbf{0}\}$.) Note the three different objects: the
**map** $g$ lives in a space of paths
$C^1\bigl((-\varepsilon,\varepsilon),\mathbb{R}^3\bigr)$; each
**value** $g(t)$ is a point of $S^2\subset\mathbb{R}^3$; the
**number** $g'(0)$ is one vector in the subspace $T_e S^2$. Now use
the assumed product $\ast$. Write $\mu(a,b)=a\ast b$. For each
**fixed** $p\in S^2$ define
$$
\gamma_p:(-\varepsilon,\varepsilon)\to\mathbb{R}^3,
\qquad
\gamma_p(t):=\mu\bigl(p,\,g(t)\bigr)=p\ast g(t).
$$
Then $\gamma_p(t)\in S^2\subset\mathbb{R}^3$ and $\gamma_p(0)=p\ast e=p$.
Differentiate the composite **at the parameter** $t$ by the chain rule
in the second slot of $\mu$ (first slot held fixed at $p$):
$$
\gamma_p'(t)
  =D_2\mu\bigl(p,\,g(t)\bigr)\,[g'(t)]
  \;\in\; T_{\gamma_p(t)}S^2\subset\mathbb{R}^3.
$$
In particular, at $t=0$,
$$
X(p)
  :=\gamma_p'(0)
  =D_2\mu(p,e)\,[g'(0)]
  \;\in\; T_p S^2\subset\mathbb{R}^3.
$$
So $X(p)$ is the image of the single vector $g'(0)$ under the
linear map $D_2\mu(p,e):T_e S^2\to T_p S^2$.
**Why $X(p)$ cannot vanish.** Left multiplication by $p$,
$L_p(q)=p\ast q$, is a smooth bijection $S^2\to S^2$ with smooth
inverse $L_{p^{-1}}$, hence a diffeomorphism. Therefore its
differential at the identity,
$D_2\mu(p,e)=DL_p|_e:T_e S^2\to T_p S^2$, is a linear isomorphism.
A linear isomorphism sends nonzero vectors to nonzero vectors, so
$$
X(p)=\mathbf{0}
  \quad\Longleftrightarrow\quad
g'(0)=\mathbf{0}.
$$
We chose $g$ with $g'(0)\neq\mathbf{0}$, hence
$X(p)\neq\mathbf{0}$ in $T_p S^2\subset\mathbb{R}^3$ for **every**
$p\in S^2$. (Equivalently: if $\gamma_p'(0)=\mathbf{0}$, then
$g=\lambda_p\circ\gamma_p$ with $\lambda_p=L_{p^{-1}}$ yields
$g'(0)=\mathbf{0}$ by the chain rule — the same identity.)
**How that bites.** Smoothness of $\ast$ makes
$p\mapsto X(p)\in\mathbb{R}^3$ continuous. A smooth group structure on
all of $S^2$ would therefore give a continuous map
$X:S^2\to\mathbb{R}^3$ with $X(p)\in T_p S^2$ and $X(p)\neq\mathbf{0}$
everywhere. Hairy ball forbids that. Therefore no such group structure
exists. (The rotation try that produces a field vanishing at the poles
is (M1.3)–(M1.4).)

##### (M1.3) Explicit derivative of $\gamma_p(t)=p\ast g(t)$ at $t=0$

**DG toolset so far (from M1.1–M1.2).** Extrinsic manifold
$S^2\subset\mathbb{R}^3$; tangent space
$T_p S^2=\{v\in\mathbb{R}^3:p\cdot v=0\}$; differential of a map in the
second slot $D_2\mu$; left translation $L_p$ with
$DL_p|_e$ a linear isomorphism $T_e\to T_p$.

**What.** Evaluate the same identity on two models: the circle (a true
Lie group on its points) and the sphere under an $\mathrm{SO}(3)$
**action** (not a product $\ast$ on pairs of points of $S^2$).

**How — general (embedded Lie group).** Let $\mu(p,q)=p\ast q$ on
$G\subset\mathbb{R}^N$, $\gamma_p(t)=\mu(p,g(t))$. Then
$$
\gamma_p'(t)
  =D_2\mu\bigl(p,\,g(t)\bigr)\,[g'(t)],
\qquad
X(p)
  =\gamma_p'(0)
  =D_2\mu(p,e)\,[g'(0)]
  \;\in\; T_p G\subset\mathbb{R}^N.
$$
Brackets mean “apply the linear map,” not a Euclidean product of
points. A map $p\mapsto X(p)\in T_p G$ of this kind is a **vector
field** on $G$ (tool we will reuse).

**How — circle (group law on the points).** On
$S^1\cong\mathbb{R}/2\pi\mathbb{Z}$, $\mu(\theta,\alpha)=\theta+\alpha$,
$g(t)=t$, so $g'(0)=1$ and
$X(\theta)=\frac{\mathrm{d}}{\mathrm{d}t}|_{0}(\theta+t)=1$ in the
angle chart — nowhere zero, as (M1.2) requires.

**How — sphere under rotation about $z$ (action, not $\ast$).** Let
$\mathrm{SO}(3)$ act on $S^2\subset\mathbb{R}^3$ by matrix–vector
multiplication. Take the **one-parameter subgroup**
$$
R_z(t)=
\begin{pmatrix}
\cos t & -\sin t & 0\\
\sin t & \cos t & 0\\
0 & 0 & 1
\end{pmatrix},
\qquad
\gamma_p(t):=R_z(t)\,p
=
\begin{pmatrix}
x\cos t-y\sin t\\
x\sin t+y\cos t\\
z
\end{pmatrix}
\quad\bigl(p=(x,y,z)^\top\bigr).
$$
Differentiate **at the parameter** $t=0$:
$$
X(p)
  =\gamma_p'(0)
  =
\begin{pmatrix}-y\\ x\\ 0\end{pmatrix}
  =\mathbf{E}(p)
  \;\in\;\mathbb{R}^3.
$$
Equivalently $X(p)=R_z'(0)\,p$ with generator
$R_z'(0)=\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix}$
(an element of the **Lie algebra** $\mathfrak{so}(3)$ of skew matrices —
foreshadowing the exponential map). Ambient Euclid gives
$p\cdot\mathbf{E}(p)=0$, so $\mathbf{E}(p)\in T_p S^2$. In the
spherical chart, $\mathbf{E}=\sin\theta\,\hat{\boldsymbol{\phi}}$
(due east). This $X$ is a continuous vector field on all of $S^2$, but
(M1.4) shows it vanishes at the poles — hence **not** the nowhere-zero
field a smooth group law on $S^2$ would force.

##### (M1.4) Poles: continuous field vs nowhere-zero field

**DG toolset.** A **(continuous) vector field** on $S^2$ is a
continuous map $X:S^2\to\mathbb{R}^3$ with $X(p)\in T_p S^2$ for every
$p$. “Nowhere zero” means $X(p)\neq\mathbf{0}$ everywhere. Hairy ball:
no continuous nowhere-zero vector field on $S^2$.

The field $\mathbf{E}(x,y,z)=(-y,\,x,\,0)$ is continuous on all of
$S^2$, but $\mathbf{E}(N)=\mathbf{E}(S)=\mathbf{0}$ — the rotation
stalls at the poles. Renormalize on the open belt $x^2+y^2>0$:
$$
\hat{\mathbf{E}}(p)
  =\frac{(-y,\,x,\,0)}{\sqrt{x^2+y^2}}.
$$
Then $\hat{\mathbf{E}}(p)\neq\mathbf{0}$ for $0<\theta<\pi$, yet
$\hat{\mathbf{E}}$ has **no** continuous extension to $N$ or $S$: along
longitude $\phi$ the limit is $(-\sin\phi,\,\cos\phi,\,0)$, which
depends on $\phi$. Remove the poles and $\hat{\mathbf{E}}$ is fine
(cylinder $\mathbb{R}\times S^1$ from (M1.2)). On the **closed**
sphere, hairy ball forbids any continuous nowhere-zero $X$. So (M1.2)'s
hypothesis fails: no smooth $\ast$ on all of $S^2$. Among spheres only
$S^0$, $S^1$, $S^3$ carry a smooth group structure on the whole
sphere.

**Path vs field.** One smooth curve
$\gamma:(-\varepsilon,\varepsilon)\to S^2$ through a pole is fine: it
needs only $\gamma'(t_0)\in T_{\gamma(t_0)}S^2$ at each parameter. A
group law forces a **simultaneous** nonzero $X(p)$ at every $p$; that
is what fails.

**Takeaway box — M1.4.**  
Continuous vector field $\mathbf{E}$ exists but vanishes at poles;
unit belt field $\hat{\mathbf{E}}$ cannot extend continuously; hairy
ball ⇒ no smooth circle-style group on closed $S^2$.

##### (M1.5) Local calculus: constraint, charts, chain rule

We stop hunting for global $+/\cdot$ on points of $S^2$. What fits is
linear structure on **velocities**, plus ordinary calculus in
**charts**.

**Constraint ⇒ tangent space (ambient).** If
$\gamma:(-\varepsilon,\varepsilon)\to\mathbb{R}^3$ is $C^1$ with
$\gamma(t)\in S^2$, both $\gamma(t)$ and $\gamma'(t)$ live in
$\mathbb{R}^3$. Differentiate $\gamma\cdot\gamma=1$ at the parameter
$t$:
$$
\gamma(t)\cdot\gamma'(t)=0
  \quad\Longleftrightarrow\quad
\gamma'(t)\in T_{\gamma(t)}S^2.
$$
Differentiating a curve on $S^2$ does **not** return a point of $S^2$;
it returns a vector in the ambient $\mathbb{R}^3$ constrained to that
plane. (Tool: **differentiate the defining equation** to read the
tangent space of an embedded manifold.)

**Charts.** Away from the poles, spherical coordinates
$$
\begin{aligned}
x&=\sin\theta\cos\phi,\\
y&=\sin\theta\sin\phi,\\
z&=\cos\theta,
\end{aligned}
\qquad
\theta\in(0,\pi),\quad
\phi\in(0,2\pi),
$$
give a chart $\varphi:U\subset S^2\to\mathbb{R}^2$,
$\varphi(x,y,z)=(\theta,\phi)$, with $\varphi^{-1}$ smooth into
$\mathbb{R}^3$. A scalar $f:S^2\to\mathbb{R}$ is smooth on $U$ iff
$f\circ\varphi^{-1}$ is smooth in $(\theta,\phi)$. Along
$t\mapsto\bigl(\theta(t),\phi(t)\bigr)$,
$$
\frac{d}{dt}(f\circ\gamma)(t)
  =\partial_\theta(f\circ\varphi^{-1})\,\theta'(t)
  +\partial_\phi(f\circ\varphi^{-1})\,\phi'(t):
$$
derivatives of scalars on $S^2$ are ordinary $\mathbb{R}^2$ derivatives
in the chart ($\theta',\phi'$ at the **parameter** $t$). Charts
covering the poles and $C^\infty$ **transition maps** on overlaps are
the atlas story of §2 — the intrinsic upgrade of this extrinsic
picture.

**Punchline for (M1).** $S^2\subset\mathbb{R}^3$ is not closed under
ambient $+/\cdot$, admits no continuous vector-space structure and no
smooth circle-style group on the whole closed sphere; curves have
$\mathbb{R}^3$-derivatives in $T_{\gamma(t)}S^2$; scalars are
differentiated by charts + chain rule. Toolset in hand: embedded
constraint, $T_pM$, differential $DF$, vector field, chart,
one-parameter subgroup / Lie-algebra generator (preview).

#### (M2) Directions in the plane (projective tool)

**Working setup.** A **direction** in $\mathbb{R}^2$ is a line through
the origin. Declare
$$
v\sim w
\quad\text{iff}\quad
\exists\,\lambda\in\mathbb{R}\setminus\{0\}
\text{ with }w=\lambda v.
$$
The space of directions is the quotient manifold (set for now)
$$
\mathbb{RP}^{1}
  =\bigl(\mathbb{R}^2\setminus\{0\}\bigr)\big/\sim.
$$
Then $v$ and $2v$ are different vectors but the **same** point of
$\mathbb{RP}^{1}$.

**Why $+$ fails on classes.** If $v,w$ are not parallel, the class of
$v+w$ depends on representative lengths: $v+w$ and $v+2w$ are not
proportional in general, so addition of directions is not well-defined.
Scaling a class by $2$ does nothing new ($v\sim 2v$).

**DG tool.** Quotients by free scaling (projective spaces) are standard
manifolds; the same pattern over $\mathbb{C}$ gives
$\mathbb{CP}^{1}$ (pure qubit states in (P1) / Bloch sphere).

#### (M3) Rotation matrices (matrix Lie group)

**Working setup.** A matrix $R\in\mathrm{M}_3(\mathbb{R})$ is a
**rotation** when
$$
R^\top R = I,
\qquad
\det R = 1.
$$
Write $\mathrm{SO}(3)$ for that set. It is an **embedded** subset of the
vector space $\mathrm{M}_3(\mathbb{R})\cong\mathbb{R}^{9}$, cut out by
smooth equations — same extrinsic pattern as $S^2\subset\mathbb{R}^3$.
It is **not** a linear subspace: if $R,S\in\mathrm{SO}(3)$, then $R+S$
typically fails $(R+S)^\top(R+S)=I$.

**Group law.** The structure that closes is **composition**
$(R,S)\mapsto RS$:
$$
(RS)^\top(RS)=S^\top R^\top R S=I,
\qquad
\det(RS)=1.
$$
So $\mathrm{SO}(3)$ is a **matrix Lie group**: manifold + smooth group
operations, realised inside matrices. A smooth curve $R(t)$ in
$\mathrm{SO}(3)$ has velocity $R'(t)$ in the ambient
$\mathrm{M}_3(\mathbb{R})$; at $t=0$ with $R(0)=I$, the constraint
$R^\top R=I$ forces $R'(0)^\top+R'(0)=0$ — i.e. $R'(0)$ lies in the
Lie algebra $\mathfrak{so}(3)$ of skew-symmetric matrices (compare
$R_z'(0)$ in (M1.3)). The matrix exponential will reconnect algebra →
group later.

**Link.** The eastward field $\mathbf{E}$ of (M1.3) was exactly the
velocity of the $\mathrm{SO}(3)$-orbit $t\mapsto R_z(t)p$.
'''

SECTION_13 = r'''
### 1.3 What we need next

Each example is a set of points that is not a vector space under
addition. From (M1)–(M3) we already carry a **differential-geometry
toolset in embryo**: embedded constraint → $T_pM$; differential
$DF$ / $D_2\mu$; vector field; chart + chain rule; projective
quotient; matrix Lie group with Lie algebra of velocities at the
identity. Section 2 turns these into definitions (atlas, smooth maps,
tangent bundle language) on a worked circle example, then feeds the
Lie-group / exponential story later.

> **Further reading.** J. M. Lee, *Introduction to Smooth Manifolds*,
> 2nd ed., Springer, 2013, Ch. 1 (smooth manifolds and smooth maps).
> For a picture-first companion, V. Guillemin and A. Pollack,
> *Differential Topology*, AMS Chelsea, 2010 (reprint), Ch. 1.
'''

CHECKS = r'''
# Concrete checks for §1: S^2 calculus, ray equivalence, SO(3) sum vs product

# (M1) Differentiating the constraint: great-circle velocity is tangent
# gamma(t) = (cos t, sin t, 0) lies on S^2; gamma'(t) = (-sin t, cos t, 0)
# satisfies gamma(t) · gamma'(t) = 0. Chain rule in spherical coords for f=z.

def gamma(t):
    return np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)

def gamma_dot(t):
    return np.stack([-np.sin(t), np.cos(t), np.zeros_like(t)], axis=-1)

ts = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
g, gd = gamma(ts), gamma_dot(ts)
assert np.allclose(np.sum(g * g, axis=-1), 1.0)
assert np.allclose(np.sum(g * gd, axis=-1), 0.0)

# Spherical chart on this equator curve: theta=pi/2, phi=t
# f = z = cos(theta) => along the curve, d(f∘gamma)/dt = 0 = -sin(theta) theta'
theta, phi = np.pi / 2, ts
f_chart = np.cos(theta)  # constant along this curve
df_dt_chain = -np.sin(theta) * 0.0 + 0.0  # theta'=0, and f independent of phi
df_dt_ambient = gd[:, 2]  # d(z)/dt along gamma
assert np.allclose(df_dt_ambient, 0.0)
assert np.allclose(df_dt_chain, df_dt_ambient)

print("(M1) max |gamma·gamma'| =", np.max(np.abs(np.sum(g * gd, axis=-1))))
print("(M1) chain rule for f=z on equator: OK")

# (M1.3)–(M1.4) E field from Rz action: E(p)=(-y,x,0), tangent, zeros at poles
def E_field(p):
    x, y, z = p
    return np.array([-y, x, 0.0])

for p in [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
          np.array([0.0, 0.0, -1.0])]:
    Ep = E_field(p)
    print("(M1.3) p =", p, " E =", Ep, " p·E =", np.dot(p, Ep))
    assert np.isclose(np.dot(p, Ep), 0.0)
assert np.allclose(E_field(np.array([0.0, 0.0, 1.0])), 0.0)  # pole stall

# (M2) same direction vs ill-defined sum of classes
v = np.array([1.0, 2.0])
w = 2.0 * v
print("(M2) v, 2v =", v, w, " parallel?", np.isclose(v[0] * w[1], v[1] * w[0]))
a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
s1, s2 = a + b, a + 2.0 * b
# s1 and s2 are not parallel ⇒ class of a+b depends on representative of [b]
print("(M2) a+b =", s1, " a+2b =", s2,
      " parallel?", np.isclose(s1[0] * s2[1], s1[1] * s2[0]))

# (M3) Two rotations: sum leaves SO(3), product stays; Rz'(0) skew
def is_SO3(R, tol=1e-10):
    return (
        norm(R.T @ R - np.eye(3)) < tol
        and abs(det(R) - 1.0) < tol
    )

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

R1, R2 = Rz(0.4), Rz(0.9)
print("(M3) R1 in SO(3)?", is_SO3(R1), " R2 in SO(3)?", is_SO3(R2))
print("(M3) R1+R2 in SO(3)?", is_SO3(R1 + R2),
      " R1@R2 in SO(3)?", is_SO3(R1 @ R2))
assert is_SO3(R1) and is_SO3(R2) and is_SO3(R1 @ R2)
assert not is_SO3(R1 + R2)
# generator at identity: d/dt Rz(t)|_0 is skew-symmetric (so(3))
h = 1e-6
Rz_dot0 = (Rz(h) - Rz(0.0)) / h
print("(M3) Rz'(0) ≈\n", np.round(Rz_dot0, 6))
print("(M3) skew check ||A+A.T|| =", norm(Rz_dot0 + Rz_dot0.T))
'''


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(LOCATOR))
    cells.append(code(SETUP))
    cells.append(md(PART_I))
    cells.append(code(CYLINDER, hide_input=True))
    # (M1.2) continues after the figure. (P1)–(P3) stay in the mega
    # notebook; section 1.3 is the toolset summary that closes (M1)–(M3).
    cells.append(md(AFTER_FIGURE.rstrip() + "\n\n" + SECTION_13.lstrip("\n")))
    cells.append(code(CHECKS))
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
