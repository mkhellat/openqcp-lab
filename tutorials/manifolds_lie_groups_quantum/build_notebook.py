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


cells: list[nbf.NotebookNode] = []

# ---------------------------------------------------------------------------
# Title / objectives
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Smooth Manifolds, Lie Groups, and the Exponential Map

Many sets that appear in mathematics and theoretical physics are not
vector spaces: if $p$ and $q$ are points of the set, $p+q$ or
$\alpha p$ need not lie in the set, and need not be meaningful.
Concrete examples are in §1.

Such sets arise as domains of points on which geometry and physical
law are local and differential: they concern how structure and
quantities vary among neighbouring points. What the points *are*
depends on the subject — for example:

- in **mathematics**: the domains that forced the invention of
  manifolds — curved surfaces and higher-dimensional geometries that
  admit calculus only through local coordinates like open sets in
  $\mathbb{R}^n$ (not a single global vector-space chart), directions
  up to scaling, and continuous groups of transformations — so that
  derivatives, curvature, and differential equations can be stated
  intrinsically on those sets;
- in **special and general relativity**: **events**; the domain is
  spacetime itself, and the independent variables need not single out
  time as a privileged parameter of the set;
- in **quantum mechanics**, **quantum information**, and **quantum
  computing**: pure states up to phase, mixed states, and unitary
  gates (finite transformations), as well as control parameters of
  Hamiltonians;
- in **statistical physics** and **quantum statistical physics**:
  macroscopic states of ensembles, and parameters of families of
  probability distributions or Gibbs states;
- in **twistor theory**: points of the complex projective set
  $\mathbb{PT}=\mathbb{CP}^{3}$ that reorganise null structure in
  spacetime (incidence dictionary in Part II‑C §C3);
- in **post-quantum cryptography and cryptology**: geometric carriers
  such as elliptic curves and lattice quotients that underwrite hard
  problems (treated here only as a map; depth belongs in a dedicated
  notebook).

The language of that local variation is multivariable calculus: smooth
real-valued functions, smooth curves, and derivatives, written in local
coordinates. Manifolds were introduced in mathematics precisely to
carry that language on a set of points that is only locally like an
open set in $\mathbb{R}^n$, when a single global vector-space domain is
the wrong description of the points.

A **smooth manifold** is a set equipped with local coordinates in open
subsets of $\mathbb{R}^d$ so that those calculus notions are defined
via ordinary calculus in $\mathbb{R}^d$ (details in §2). Differentiating
a curve through a point $p$ yields an element of a vector space
$T_pM$, the tangent space at $p$ — not an operation that adds points
of the set itself.

If a smooth manifold $G$ is also a group and the group operations are
smooth, $G$ is a **Lie group**. Its **Lie algebra** is the tangent
space at the identity, $\mathfrak{g}=T_eG$, which is a vector space.
For the matrix groups used in this notebook, the exponential series
$\exp(X)=\sum_{k\ge 0}X^k/k!$ maps $\mathfrak{g}$ into $G$, and
$t\mapsto\exp(tX)$ is the smooth curve in $G$ with derivative $X$ at
$t=0$.

This notebook develops that language with pictures and code, then
applications at depths A / B / C. It is a foundations module for the
geometry of states, transformations, and parameters used across the
tutorials and tools here. Part III bridges in particular to product
formulas and the coupled-oscillator Hamiltonian-simulation notebooks
(module 05); it does not replace them.
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
   background, equations, executable checks, and application notes.
5. Read Tier‑B/C vignettes with enough background *and* geometry to know
   *what* the manifold is, *why it matters*, and where to go next
   (twistor incidence in II‑C §C3; PQC/crypto off‑ramp in §C4).
"""))

cells.append(md(r"""
## How this notebook is organised

| Part | Content |
|------|---------|
| I | Charts, tangent vectors, Lie groups, \(\mathrm{GL}/\mathfrak{gl}\), exponential map |
| II‑A | Full labs: Bloch / \(\mathbb{CP}^{1}\); \(\mathbb{CP}^{d-1}\) & density operators; \(\mathrm{U}(2^n)\); \(\mathrm{Sp}(2n,\mathbb{R})\); statistical manifolds |
| II‑B | Light labs: Lorentzian Minkowski; \(U(1)\) / Berry phase |
| II‑C | Tier‑C geometry‑depth: Calabi–Yau; \(G_2\); twistors; elliptic curves & lattices (PQC off‑ramp) |
| III | Bridge to product formulas / module 05 |

**Collapsible sections.** Same pattern as the N=2 coupled-oscillators
notebook: each Markdown heading lives in its own cell with JupyterLab’s
`jp-MarkdownHeadingCollapsed` flag. In JupyterLab (or the notebook ToC),
click the triangle beside a heading to collapse or expand that section,
subsection, or subsubsection — including the code cells under it.
Example subsections (`####`, e.g. (M1)) use a slate banner; their
subsubsections (`#####`, e.g. M1.1–M1.5) use an orange-accent banner
(same family as the N=2 callout boxes) so the hierarchy stays readable.


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
"""))

cells.append(code(r"""
# Launch only — implementation: cylinder_group_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "cylinder_group_demo.py").exists():
    _here = Path("tutorials/manifolds_lie_groups_quantum")
sys.path.insert(0, str(_here.resolve()))

from cylinder_group_demo import launch_cylinder_group_demo

launch_cylinder_group_demo()
""", hide_input=True))

cells.append(md(r"""
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

### 1.2 Physics examples

#### (P1) Pure states of a qubit

A pure state of a two-level quantum system is a unit vector
$|\psi\rangle\in\mathbb{C}^2$,

$$
|\psi\rangle
  =\begin{pmatrix}\alpha\\\beta\end{pmatrix},
\qquad
|\alpha|^2+|\beta|^2=1,
$$

but physics does not distinguish an overall complex factor: for every
$\lambda\in\mathbb{C}\setminus\{0\}$,

$$
|\psi\rangle
\quad\text{and}\quad
\lambda|\psi\rangle
$$

describe the **same** preparation (all expectation values
$\langle\psi|A|\psi\rangle/\langle\psi|\psi\rangle$ agree). Equivalently
one may keep $\|\psi\|=1$ and identify

$$
|\psi\rangle\sim e^{i\phi}|\psi\rangle,
\qquad\phi\in\mathbb{R}.
$$

The space of pure qubit states is therefore **not** the vector space
$\mathbb{C}^2$, and not even the unit sphere $S^3\subset\mathbb{C}^2\cong\mathbb{R}^4$
once global phases are removed. A standard set of representatives is

$$
|\psi(\theta,\phi)\rangle
  =\cos\frac{\theta}{2}\,|0\rangle
  +e^{i\phi}\sin\frac{\theta}{2}\,|1\rangle,
\qquad
\theta\in[0,\pi],\quad\phi\in[0,2\pi),
$$

which labels points of an ordinary sphere $S^2$ (the Bloch sphere) by

$$
\mathbf{r}
  =(\sin\theta\cos\phi,\;\sin\theta\sin\phi,\;\cos\theta).
$$

You need calculus on that sphere (paths of states, derivatives of
expectation values); you do **not** add two Bloch vectors as if they
were the states themselves and expect to stay on pure states — the
sum of two pure-state projectors is generally mixed.

#### (P2) Orientations of a rigid body

The orientation of a rigid body in three-dimensional space is exactly
a rotation of a body frame into a space frame: an element of
$\mathrm{SO}(3)$ from (M3). A continuous motion is a smooth curve
$R(t)\in\mathrm{SO}(3)$. Angular velocity lives in the linear space of
antisymmetric $3\times 3$ matrices (skew-symmetric generators), while
the orientation itself lives in the nonlinear set $\mathrm{SO}(3)$.
Adding two orientation matrices is again the wrong operation;
composing them (or following the curve $R(t)$) is the right one.

#### (P3) Direction of a magnetic field (or spin axis)

In many control and NMR-style settings the laboratory field direction
is a unit vector $\hat{\mathbf{n}}\in S^2$. The Hamiltonian of a
spin‑1/2 aligned with that field has the form

$$
H = \tfrac12 B\, \hat{\mathbf{n}}\cdot\boldsymbol{\sigma}
$$

(with Pauli matrices $\boldsymbol{\sigma}$ and strength $B$). The
**parameter** that labels the family of Hamiltonians is the point
$\hat{\mathbf{n}}$ on $S^2$, not a vector you add freely in
$\mathbb{R}^3$: $\hat{\mathbf{n}}$ and $2\hat{\mathbf{n}}$ are not two
different directions. Smoothly changing the field direction means
moving on $S^2$, which returns us to (M1) with a physical reading.

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
"""))

cells.append(code(r"""
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

# (P1) Global phase does not change Pauli expectations
def bloch(psi):
    psi = psi / norm(psi)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.array([
        np.real(np.vdot(psi, X @ psi)),
        np.real(np.vdot(psi, Y @ psi)),
        np.real(np.vdot(psi, Z @ psi)),
    ])

psi = np.array([np.cos(0.3), np.exp(1j * 0.5) * np.sin(0.3)], dtype=complex)
psi_phase = np.exp(1j * 1.7) * psi
print("(P1) Bloch(ψ)      =", bloch(psi))
print("(P1) Bloch(e^{iφ}ψ)=", bloch(psi_phase))
assert np.allclose(bloch(psi), bloch(psi_phase))
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

Each lab below has the same shape: **why the object shows up**, the
**equations** that make it precise, a **coded check**, and a short
**takeaway** tying manifold / Lie group / exponential together.

> **Further reading (QI geometry).** I. Bengtsson and K. Życzkowski,
> *Geometry of Quantum States: An Introduction to Quantum
> Entanglement*, 2nd ed., Cambridge University Press, 2017. Chapter 4
> treats complex projective spaces; the Bloch sphere is $\mathbb{CP}^1$.
"""))

cells.append(md(r"""
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
# Part II‑C — Geometry‑depth maps and reading paths

Tier‑C topics stay **invitation‑depth** relative to Tier A (no full
string / holonomy / crypto course), but each of **C1–C4** is now written
formula‑first: a declared working setup, the geometric object with at
least one explicit equation or dictionary, a schematic where it helps,
and a takeaway. Graduate off‑ramps:

- Twistors → [`../twistor_theory`](../twistor_theory/)
- PQC / lattices / ECC engineering →
  [`../post_quantum_cryptography`](../post_quantum_cryptography/)
- CY / $G_2$ compactifications → future dedicated notes (cited in C1–C2)
"""))

cells.append(md(r"""
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

**Off‑ramp:** a future “string compactifications / CY” notebook; this
section stops at the working definition and the discriminant cartoon.
"""))

cells.append(code(r"""
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
"""))

cells.append(md(r"""
**Takeaway box — C1.**  
Working definition: compact Kähler $X$ with $c_1(X)=0$ (holomorphic
volume form; holonomy in $\mathrm{SU}(m)$ after Yau). Code: discriminant
cartoon only. Next depth: dedicated CY / compactification notebook.
"""))

cells.append(md(r"""
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

**Off‑ramp:** dedicated exceptional‑holonomy note (or shared with a
future CY compactification notebook).
"""))

cells.append(code(r"""
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
"""))

cells.append(md(r"""
**Takeaway box — C2.**  
A $G_2$‑manifold is a Ricci‑flat 7‑fold with parallel stable 3‑form
$\varphi$ (holonomy in $G_2$). Model $\varphi_0$ on $\mathbb{R}^{7}$ is
an explicit tensor; compact exact‑holonomy examples exist (Joyce). Code:
coordinate inventory only.
"""))

cells.append(md(r"""
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
"""))

cells.append(code(r"""
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
"""))

cells.append(md(r"""
**Takeaway box — C3.**  
Manifold: twistor space $\mathbb{PT}=\mathbb{CP}^{3}$. Dictionary:
point of $\mathbb{PT}$ $\leftrightarrow$ null line in $\mathbb{CM}$;
spacetime point $\leftrightarrow$ $\mathbb{CP}^{1}$ in $\mathbb{PT}$,
via $\omega^{A}=ix^{AA'}\pi_{A'}$. Next depth:
[`../twistor_theory`](../twistor_theory/).
"""))

cells.append(md(r"""
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
"""))

cells.append(code(r"""
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
"""))

cells.append(md(r"""
**Takeaway box — C4.**  
Elliptic curve: smooth Weierstrass model ($\Delta\neq 0$) with group
law; over $\mathbb{C}$, a torus $\mathbb{C}/\Lambda$. Lattice:
discrete $\Lambda\subset\mathbb{R}^{n}$; torus $\mathbb{R}^{n}/\Lambda$.
Engineering depth:
[`../post_quantum_cryptography`](../post_quantum_cryptography/).
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
- Tier‑C is geometry‑depth throughout: Calabi–Yau ($c_1=0$), $G_2$
  (parallel $3$‑form), twistors (incidence
  $\omega^{A}=ix^{AA'}\pi_{A'}$), elliptic curves / lattice tori.
  Graduate off‑ramps: [`twistor_theory`](../twistor_theory/);
  PQC module for standards and OQS.

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

cells = explode_heading_cells(cells, default_collapsed=False)

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
