#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 06 (source of truth): applications.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_06_applications.py

The generated notebook is ``06_applications.ipynb`` beside this file.
Three banners share the file at v1. A later edition may split them
into notebooks 06 and 07. The banners are scientific computing
(manifold meshes, FEM/DEC pointer, TDA caveat, data-structure
reprise), QFT renormalizability (power counting through the
superficial degree, building on the notebook 03 appetizer), and
cryptology-adjacent hardness (the notebooks 04--05 table read as a
resource, with explicit non-claims).

Interactive drawing lives in ``mesh_manifold_demo.py``. The notebook
cell that shows it is launch-only. A visible cell prints the three
mesh verdicts. A second visible cell evaluates scalar power-counting
arithmetic. No cell builds a Feynman diagram, a counterterm, or a
cryptosystem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("06_applications.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Applications

Three banners share this notebook. They may split later into a
scientific-computing notebook and a following notebook (06 and 07),
or into one file per banner. The cut is the heading of each banner.
Cross-references name the notebook that already holds the theorem,
so the text on each side of a cut still points at 03, 04, and 05.

The applications path in the series guide is 00 → 01 → 02 → 06.
The QFT banner uses the appetizer in notebook 03. The cryptology
banner uses the landscape in notebooks 04 and 05. Neither banner
replaces those notebooks.

No cell searches for a homeomorphism, builds a counterterm, or
defines a key. The mesh figure compares one combinatorial surface
with two complexes that fail the link condition. The power-counting
cell evaluates the degree formulas on small integers.

## Learning objectives

1. Apply the closed-surface link condition to a triangle mesh, and
   choose a facet list, an incidence matrix, a half-edge, or a
   Quad-Edge according to the question.
2. Compute the mass dimension of a polynomial coupling and the
   superficial degree of a connected scalar diagram, and read the
   sign of that dimension as the power-counting class.
3. Place $\chi$ and the genus in the $2$-dimensional
   Einstein–Hilbert action, in the string and ribbon weights, and in
   the loop count of a graph, and keep those integers out of the
   sign test for a coupling in dimension $4$.
4. Match each row of the notebooks 04–05 table to four predicates
   (sample, plant, hide, verify), and state the non-claims that
   follow.
"""))

    cells.append(md(r"""
## Scientific computing

Manifold meshes, a pointer to finite elements and to discrete
exterior calculus, the persistent-homology caveat, and the
data-structure table from notebook 03. Half-edge and Quad-Edge stay
different operators.
""", collapsed=False))

    cells.append(md(r"""
### When a mesh is a closed combinatorial surface

Notebook 02 fixes the test this banner uses. A pure $2$-complex $K$
is a closed combinatorial surface when every vertex link is a single
cycle,
$$
\operatorname{Lk}(v, K) \cong S^{1}.
$$
`closed_surface_link_condition` is that sentence. On a complex that
passes, every edge $e$ lies in exactly two triangles,
$$
\#\{\tau \in K : e \subset \tau,\; \dim\tau = 2\} = 2.
$$
The count is a reading of the facet list. It is not a vertex degree.
`vertex_degrees` reads the adjacency list of the $1$-skeleton: the
number of neighbouring vertices. An edge used by three triangles can
still meet a vertex of degree $2$.

A file from a modeller can fail earlier than this test, and it can
fail later. The three stages are different predicates.

**Stage 1.** The data are not a simplicial complex. Two triangles
whose intersection is a proper subsegment of an edge (a T-junction,
a hanging node) do not meet in a common face. Notebook 01 stores a
facet list and its downward closure. A partial overlap is not an
element of that closure. Duplicate indices for one geometric point
are a different vertex set from the one the picture suggests.

**Stage 2.** The data are a simplicial complex, and the link formula
fails. Two triangles that meet only at a vertex are pure of
dimension $2$. The shared vertex has two disjoint edges as its link.
Three triangles that meet at one edge are also a complex. That edge
has incidence $3$, so the count above is not $2$, and the vertex
links are not cycles. These are the two dirty presets in the figure.

**Stage 3.** The link formula holds, and the surface is the wrong
one for the boundary-value problem. The tetrahedron boundary passes
and has $\chi = 2$. A torus passes and has $\chi = 0$. Both are
manifold meshes. Which one was required is a question about the
domain, answered by the classification sketch in notebook 03, not by
the link test alone.

A conforming finite-element mesh, in the sense of Ciarlet, asks that
the intersection of two closed elements be a common face. Stage 1 is
that demand at the level of the complex. The lowest-order Lagrange
space on a triangle mesh uses it so that traces agree on shared
edges. This notebook does not assemble a stiffness matrix.

Discrete exterior calculus places cochains on a mesh that has already
passed a manifold test, and takes the coboundary as the discrete
exterior derivative. On a primal complex that coboundary is the dual
of the boundary operator $\partial_k$ from `lib/boundary.py`. The
Hodge star is a second operator: it pairs a primal cochain with a
cochain on a dual cell. Building the dual is a different storage
question from storing $\partial_k$. The table below sends those
questions to different structures.

A half-edge stores one directed side of one oriented triangle, with
`next` along the face and `twin` across the unique opposite face.
Incidence $3$ means there is no unique opposite face, so `twin` is
not a function on that edge. A Quad-Edge stores one undirected edge
together with the operator $\mathrm{Rot}$ whose order-$4$ cycle
visits the primal arrow and the dual arrow (Guibas and Stolfi). The
same missing opposite face means $\mathrm{Rot}$ is not defined there
either. The two algebras agree that a branched edge is outside their
domain. They are not two names for one encoding. Notebook 03 builds
a teaching subset on complexes that pass the link condition, and
refuses the classical twin on $\mathbb{R}\mathrm{P}^{2}$ for an
orientation reason. The branched preset here is the incidence reason.

**What.** The link formula, the edge-incidence count, and the three
stages.

**Why.** A mesh library that asks only for vertex degrees will accept
the branched preset's outer vertices. The facet list records the
third triangle. The adjacency list does not.

**How.** The visible cell prints both numbers for three complexes.
The hidden cell draws them. Orange marks the pinched vertex or the
branched edge.

**When.** The object is a triangle mesh offered to a surface code.
A chain-level identity $\partial^{2} = 0$ is still `boundary_matrix`,
on whatever complex one has, manifold or not. Notebook 01 already
separates that identity from the manifold test.
"""))

    cells.append(md(r"""
### Persistent homology does not evaluate the link

Persistent homology takes a filtration of complexes and returns the
ranks of homology that persist across scales (Edelsbrunner and Harer,
*Computational Topology*; Carlsson, Bull. Amer. Math. Soc. 46 (2009),
255–308). A long bar in degree $1$ is evidence about $H_1$. It is a
statement about cycles modulo boundaries in that filtration.

The link formula is a statement about $\operatorname{Lk}(v, K)$ for
each vertex of one fixed complex. A barcode does not output those
links. The two computations can be run on the same point cloud and
can disagree, because they answer different questions.

The low-dimensional illustration is the pinched preset. Two triangles
glued at a vertex have the homology of a point: both are disks, and
the wedge of two disks is contractible. There is no persistent $H_1$
feature to discover. The link of the shared vertex is nevertheless
disconnected, and `closed_surface_link_condition` returns false.
"No holes" is not a manifold certificate.

The high-dimensional illustration is already in notebook 05. Edwards
(arXiv:math/0610573) proves that the double suspension of a homology
$3$-sphere is homeomorphic to $S^{5}$. The links of the suspension
vertices fail the link formula, so the complex is not a combinatorial
manifold. Homology, and in that case homeomorphism type, can say
"sphere" while stage 2 fails. A pipeline that trusts a Betti number
as a combinatorial check has substituted one predicate for the other.

The converse direction is the same gap. A combinatorial torus has
$\chi = 0$ and $b_1 = 2$, and so does any complex with those ranks.
The ranks do not certify circular vertex links.

**What.** Homology ranks of a filtration, versus the link formula.

**Why.** "Looks like a manifold" in a topological-data-analysis
report is a claim about homology. Stage 2 is a claim about links.

**How.** Read the pinched complex as the case where homology is
trivial and the link test fails. Read notebook 05 for the case where
the realization is a sphere and the link test fails.

**When.** The input is a point cloud or a filtration. After a complex
has been chosen, the manifold question is the link formula again.
"""))

    cells.append(md(r"""
### Which structure, for which mesh question

Notebook 03 compared these encodings on surfaces. The column that
changes here is the mesh question they answer. Half-edge and
Quad-Edge remain different rows.

| Question | Structure | Why this one |
|----------|-----------|----------------|
| the triangles that were given | facet list | the constructor input; stage 1 fails before this list exists |
| $\chi$, stars, links, the link formula | downward closure, `link` | every face, not only the top |
| vertex degree, a $k=1$ walk | `adjacency_list` | unsigned neighbours; silent on how many triangles contain an edge |
| edge incidence $1$, $2$, or $\ge 3$ | facet list, counted per edge | the integer that admits or refuses `twin` |
| incidence signs, $\partial^{2}=0$, the DEC coboundary | `boundary_matrix` | the linear map $C_k \to C_{k-1}$; the coboundary is its dual |
| circulate one face, name a primal flip | half-edge, `next` and `twin` | the walk is primal; `twin` needs one opposite face |
| the same edge read in the dual, a Hodge star | Quad-Edge, $\mathrm{Rot}$ and $\mathrm{Onext}$ | one bundle holds the primal arrow and the dual arrow |
| barcode ranks of a filtration | a filtered complex, outside this table | homology ranks are not link spheres |
| a dense matrix of the same $\partial_1$ | contrast, then discard | the tetrahedron boundary is already $4 \times 6$ |

`lib/` stores the facet list, the adjacency list, and the sparse
boundary. Half-edge and Quad-Edge stay in `surface_tds_demo.py`, and
only on complexes that can support them. The branched preset is drawn
here so the incidence that blocks both operators is visible. Neither
operator is constructed for that preset.

Guibas and Stolfi, ACM Transactions on Graphics 4 (1985), is the
Quad-Edge reference, as in notebook 03. A half-edge is the surface encoding whose
opposite pointer is `twin`. Treating the two names as synonyms drops
the dual from the Hodge star, or invents a dual the half-edge does
not store.
"""))

    cells.append(md(r"""
### Three complexes

The dropdown is the tetrahedron boundary, the pinched pair, and the
branched triple. The title line is the link condition and $\chi$,
computed from the complex.

- The manifold mesh is the tetrahedron boundary in
  $\mathbb{R}^{3}$. Six edges, each in two triangles. Every vertex
  has degree $3$. The camera sliders apply only there.
- The pinch is two triangles that meet at vertex $0$. All six edges
  have incidence $1$. Vertex $0$ has degree $4$. Degree $4$ also
  occurs on honest triangulations, so the degree does not name the
  pinch. The link does.
- The branch is three triangles on the edge $\{0,1\}$. That edge has
  incidence $3$. Vertices $2$, $3$, and $4$ have degree $2$.

`mesh_report` prints the three verdicts. The figure lives in
`mesh_manifold_demo.py`. The last cell of this banner only checks
that the file is beside the notebook and calls
`launch_mesh_manifold_demo`.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

Coordinates, incidence highlighting, and the widgets live in
<code>mesh_manifold_demo.py</code>. The visible cell is
<code>example_facts</code> and <code>mesh_report</code>. JupyterLab
collapses the launch input (<code>source_hidden</code>).

</details>
"""))

    cells.append(code(r"""
# Verdicts only. The drawing is the hidden cell below.
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "mesh_manifold_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "mesh_manifold_demo.py").exists():
    raise FileNotFoundError(
        "mesh_manifold_demo.py missing next to this notebook"
    )
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from mesh_manifold_demo import example_facts, mesh_report

facts = example_facts()
assert [fact.manifold for fact in facts] == [True, False, False]
assert [fact.max_edge_incidence for fact in facts] == [2, 1, 3]
assert [fact.chi for fact in facts] == [2, 1, 1]
branched = facts[2]
assert dict(branched.degrees)[2] == 2
assert branched.branched_edges == ((0, 1),)
print(mesh_report())
"""))

    cells.append(code(r"""
# Launch only — implementation: mesh_manifold_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "mesh_manifold_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "mesh_manifold_demo.py").exists():
    raise FileNotFoundError(
        "mesh_manifold_demo.py missing next to this notebook"
    )
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from mesh_manifold_demo import launch_mesh_manifold_demo

launch_mesh_manifold_demo()
""", hide_input=True))

    cells.append(md(r"""
## QFT renormalizability

Power counting, the superficial degree, and the sign that sorts
theories by spacetime dimension. Notebook 03 supplies $\chi$ of a
surface. This banner says where that integer enters and which tests
do not read it.
""", collapsed=False))

    cells.append(md(r"""
### Two integers from notebook 03

On a closed Riemannian surface, Gauss–Bonnet gives
$\int_{\Sigma} R\,\mathrm{d}A = 4\pi\,\chi(\Sigma)$, and the
Einstein–Hilbert action with the normalisation from notebook 03 is
$$
S_{\mathrm{EH}}[g]
=
\frac{1}{16\pi G}
\int_{\Sigma} R\,\mathrm{d}A
=
\frac{\chi(\Sigma)}{4G}.
$$
For a connected closed oriented surface the same integer is the
genus,
$$
\chi = 2 - 2g.
$$
The closed-string perturbation series weights the genus-$g$ amplitude
by that Euler characteristic,
$$
\mathcal{A}
=
\sum_{g\ge 0} g_s^{\,2g-2}\,\mathcal{A}_g
=
\sum_{g\ge 0} g_s^{\,-\chi}\,\mathcal{A}_g.
$$
Both identities are topological. The first says that the
$2$-dimensional Einstein–Hilbert integrand integrates to a constant
on each component. The second says that the string coupling is raised
to $-\chi$ of the worldsheet. Neither identity mentions the mass
dimension of a coupling in a $4$-dimensional Lagrangian.

The rest of this banner is that mass dimension, and the degree of a
Feynman integrand. The appetizer numbers are used again only in the
section that places $\chi$.
"""))

    cells.append(md(r"""
### Mass dimension

Work in units $\hbar = c = 1$. The action is dimensionless, so a
Lagrangian density in $d$ spacetime dimensions has
$$
[S] = 0,
\qquad
S = \int \mathrm{d}^{d}x\,\mathcal{L},
\qquad
[\mathcal{L}] = d.
$$
A real scalar with canonical kinetic term
$\frac12 \partial_{\mu}\phi\,\partial^{\mu}\phi$ then has
$$
[\phi] = \frac{d-2}{2},
$$
because $[\partial] = 1$ and $2[\phi] + 2 = d$.

The interaction $\lambda_n \phi^{n} / n!$ is a term in $\mathcal{L}$, so
$$
[\lambda_n] + n[\phi] = d,
$$
which is
$$
[\lambda_n]
=
d - \frac{n}{2}(d-2).
$$
Two cases used below are
$$
[\lambda_4] = 4 - d,
\qquad
[\lambda_3] = \frac{6-d}{2}.
$$
The same counting on the Yang–Mills coupling, normalised so that
$F = \partial A + g A^{2}$, gives
$$
[g_{\mathrm{YM}}] = \frac{4-d}{2}.
$$
Newton's constant in the Einstein–Hilbert action
$\frac{1}{16\pi G}\int \sqrt{|g|}\, R$ has $[R] = 2$ when the metric is
dimensionless, so
$$
[G] = 2 - d.
$$
A four-fermion coupling with $[\psi] = (d-1)/2$ has
$$
[G_F] = 2 - d
$$
as well. The operator multiplied by $1/G$ is the curvature scalar.
The operator multiplied by $G_F$ is a product of four fermion fields.
The shared dimension does not identify the theories. Both numbers are
recorded so the sign test has a non-scalar example beside $\phi^{4}$.

**What.** $[\phi]$, $[\lambda_n]$, $[g_{\mathrm{YM}}]$, $[G]$, and
$[G_F]$, as integers or rational numbers read off $\mathcal{L}$.

**Why.** The superficial degree below is this list together with the
number of vertices. A theory is sorted by the sign of its couplings'
dimensions before any integral is estimated.

**How.** The code cell evaluates $[\lambda_4]$ at $d \in \{2,3,4,6\}$,
$[\lambda_3]$ at $d = 6$, $[\lambda_6]$ at $d = 3$,
$[g_{\mathrm{YM}}]$ at $d = 4$, and $[G]$ at $d = 2$ and $d = 4$.

**When.** The Lagrangian is a polynomial in fields and derivatives of
non-negative order, and the kinetic term fixes the propagator. The
scalar formula does not become a graviton integral by substitution.
$[G]$ is used only as a sign.
"""))

    cells.append(md(r"""
### Superficial degree

Take a connected diagram of a single scalar interaction $\phi^{n}$,
with $V$ vertices, $E$ external legs, and $I$ internal edges. Each
vertex has $n$ legs, and each internal edge uses two of them, so
$$
n V = E + 2 I.
$$
The loop number of a connected diagram is
$$
L = I - V + 1.
$$
With a propagator of degree $-2$ in momentum, the integrand of the
loop integrals has degree
$$
D = d\, L - 2 I
$$
at large loop momentum. Eliminating $I$ and $L$ with the two counts
and with $[\phi] = (d-2)/2$ gives the same integer as
$$
D
=
d - [\phi]\, E - [\lambda_n]\, V.
$$
The coefficient of $V$ is $-[\lambda_n]$. Raising the vertex count by
one raises $D$ by $-[\lambda_n]$.

$D$ is superficial. It is the degree of the integrand in the loop
momenta of the whole diagram. A negative $D$ does not speak for a
subgraph: a subgraph can have its own non-negative degree. If $L = 0$
there is no loop integral, and $D$ is only the engineering dimension
of that monomial.

The graph that supplies $L$ has an Euler characteristic of its own.
Regard the diagram as a connected $1$-complex with $V$ vertices and
$I$ internal edges. Then
$$
\chi_{\Gamma} = V - I,
\qquad
L = 1 - \chi_{\Gamma}.
$$
This $\chi_{\Gamma}$ counts loops. It is the integer notebook 03
warned against confusing with Gauss–Bonnet. The code cell checks
$L = 1 - \chi_{\Gamma}$ on the same rows it uses for $D$.

**What.** $I$, $L$, $D$, and $\chi_{\Gamma}$ for one connected scalar
diagram.

**Why.** Power counting is the sign of $[\lambda_n]$ inside $D$, once
the leg count is fixed. The sign is visible only after $D$ is written.

**How.** The cell fixes $E = 4$ and $n = 4$, varies $V$, and checks
that the integrand form $d L - 2 I$ equals the coupling form
$d - [\phi] E - [\lambda] V$.

**When.** The diagram is connected, the interaction is $\phi^{n}$
without extra derivatives at the vertex, and the propagator is
$1/k^{2}$. Gravity, gauge theory, and the Fermi interaction are not
fed through this scalar identity. Their coupling dimensions stand in
the previous section as signs.
"""))

    cells.append(md(r"""
### The sign, by dimension

Fix one interaction whose coupling has dimension $\delta$.

- If $\delta > 0$, then $D$ falls as $V$ grows. Only finitely many
  pairs $(V, E)$ can have $D \ge 0$. The power-counting class is
  super-renormalizable.
- If $\delta = 0$, then $D$ is independent of $V$. It depends on the
  external legs. Finitely many leg structures have $D \ge 0$, and
  they can appear at every loop order. The class is renormalizable.
- If $\delta < 0$, then $D$ grows with $V$. New leg structures keep
  acquiring $D \ge 0$ as vertices are added. The class is
  non-renormalizable.

"Renormalizable" here means that the list of counterterm monomials
required by this degree count is finite. It does not mean that the
diagrams are finite. $\phi^{4}$ in $d = 4$ has $[\lambda_4] = 0$ and,
for $E = 4$, superficial degree $D = 0$ at every $V \ge 2$ in the
connected family below. Those diagrams are logarithmically divergent
at the level of $D$, and the theory is renormalizable.
Super-renormalizable theories still have divergent diagrams at small
$V$. The sign limits how the list grows. It does not empty the list.

A Lagrangian with several couplings is sorted by the lowest
dimension among them. One irrelevant operator is enough to put the
theory in the third class.

| Term | Dimension of the coupling | Vanishes at | Class in that dimension |
|------|---------------------------|-------------|-------------------------|
| $\phi^{4}$ | $4 - d$ | $d = 4$ | renormalizable in $4$; super below; non above |
| $\phi^{3}$ | $(6-d)/2$ | $d = 6$ | renormalizable in $6$ |
| $\phi^{6}$ | $6 - 2d$ | $d = 3$ | renormalizable in $3$ |
| Yang–Mills | $(4-d)/2$ | $d = 4$ | renormalizable in $4$ |
| Einstein–Hilbert, $[G]$ | $2 - d$ | $d = 2$ | see the next section; in $d = 4$, $\delta = -2$ |
| Fermi, $[G_F]$ | $2 - d$ | $d = 2$ | in $d = 4$, $\delta = -2$ |

The $d = 2$ entry for $[G]$ is the appetizer, not a second copy of the
scalar sign test. In $d = 2$ the integral that would have been the
bulk action collapses to $\chi /(4G)$. Power counting sees a
dimensionless $G$ and stops. Gauss–Bonnet is further information: the
would-be marginal bulk integrand is topological. The scalar formula
for $D$ does not see that collapse.

In $d = 4$, $[G] = -2$. Each additional graviton vertex raises the
superficial degree. Pure gravity is power-counting non-renormalizable
in four dimensions. That sentence is the sign test. It is not yet a
statement that a particular diagram diverges. 't Hooft and Veltman
(Ann. Inst. H. Poincaré A 20 (1974), 69–94) showed that pure gravity
is one-loop finite on shell. Goroff and Sagnotti (Nucl. Phys. B266
(1986), 709–736) showed a two-loop divergence in pure gravity. The
first loop can cancel while the sign of $[G]$ is already negative.
The notebook records those citations and does not recompute either
integral.

A coupling with $\delta < 0$ still organises an effective theory
below a cutoff, with operators ordered by dimension. Donoghue
(Phys. Rev. D 50 (1994), 3874) treats general relativity that way.
The class name "non-renormalizable" is the statement that the
operator tower is infinite. It is the reason an effective-field-theory
cutoff is part of the definition of the expansion.

The cell prints the $\phi^{4}$ rows and checks the identities. It does
not construct a diagram, a counterterm Lagrangian, or a graviton
amplitude.
"""))

    cells.append(code(r"""
# Power counting for a canonical scalar. No diagram is built.
from fractions import Fraction


def field_dimension(d):
    # Mass dimension (d - 2) / 2 of a canonical scalar.
    return Fraction(d - 2, 2)


def coupling_dimension(n, d):
    # Mass dimension of the coefficient of phi^n / n!.
    return d - n * field_dimension(d)


def diagram_counts(n, V, E):
    # Internal edges and loops of one connected diagram.
    legs = n * V - E
    if legs < 0 or legs % 2:
        raise ValueError("external legs do not match the vertices")
    internal = legs // 2
    loops = internal - V + 1
    if loops < 0:
        raise ValueError("not a connected diagram")
    return internal, loops


def superficial_degree(n, d, V, E):
    # Integrand degree d*L - 2*I. Propagator degree is -2.
    internal, loops = diagram_counts(n, V, E)
    return d * loops - 2 * internal


def by_sign(dimension):
    if dimension > 0:
        return "super-renormalizable"
    if dimension == 0:
        return "renormalizable"
    return "non-renormalizable"


print("phi^4 coupling")
for d in (2, 3, 4, 6):
    dim = coupling_dimension(4, d)
    print(f"  d={d}  [lambda]={dim}  {by_sign(dim)}")

print("connected phi^4 diagrams with E=4")
for d, vertices in ((4, 2), (4, 3), (4, 4), (6, 2), (6, 3), (6, 4)):
    internal, loops = diagram_counts(4, vertices, 4)
    degree = superficial_degree(4, d, vertices, 4)
    closed = (
        d
        - field_dimension(d) * 4
        - coupling_dimension(4, d) * vertices
    )
    assert degree == closed
    chi_graph = vertices - internal
    assert loops == 1 - chi_graph
    print(
        f"  d={d} V={vertices} I={internal} L={loops} "
        f"chi_graph={chi_graph} D={degree}"
    )

assert coupling_dimension(3, 6) == 0
assert coupling_dimension(6, 3) == 0
assert all(superficial_degree(4, 4, vertices, 4) == 0 for vertices in (2, 3, 4))
assert (
    superficial_degree(4, 6, 3, 4) - superficial_degree(4, 6, 2, 4)
    == -coupling_dimension(4, 6)
)


def yang_mills_coupling(d):
    # Mass dimension (4 - d) / 2.
    return Fraction(4 - d, 2)


def einstein_hilbert_G(d):
    # Mass dimension 2 - d of Newton's constant.
    return 2 - d


print(f"Yang-Mills [g] at d=4: {yang_mills_coupling(4)}")
print(
    "Einstein-Hilbert [G]: "
    f"d=2 -> {einstein_hilbert_G(2)}, "
    f"d=4 -> {einstein_hilbert_G(4)}"
)
assert yang_mills_coupling(4) == 0
assert einstein_hilbert_G(2) == 0
assert einstein_hilbert_G(4) == -2
"""))

    cells.append(md(r"""
### Where $\chi$ and the genus enter

Four places use an Euler characteristic. They do not share a meaning.

**The surface in the appetizer.** $\chi(\Sigma)$ is the alternating
face count of a closed surface, or $\int R\,\mathrm{d}A / 4\pi$. In
two dimensions it is the value of the Einstein–Hilbert action, up to
the constant $1/(4G)$. For the oriented connected case it is $2-2g$.

**The string weight.** The closed-string sum is organised by $g$, or
equivalently by $-\chi$ of the worldsheet. The exponent
$2g - 2 = -\chi$ is that topological weight. It multiplies
$\mathcal{A}_g$. It is not a coupling dimension in a
$4$-dimensional bulk Lagrangian.

**The ribbon-graph weight.** With the 't Hooft coupling
$\lambda = g_{\mathrm{YM}}^{2} N$ held fixed, a connected vacuum
diagram of a $U(N)$ gauge theory is weighted by $N^{\chi}$ of its
ribbon graph ('t Hooft, Nucl. Phys. B72 (1974), 461–473). A closed
orientable ribbon graph has $\chi = 2 - 2g$. The power of $N$ sorts
diagrams by genus. $[g_{\mathrm{YM}}] = (4-d)/2$ is a different
number, read off the Lagrangian, and it is zero in $d = 4$ whether
or not the $1/N$ expansion is written.

**The loop count.** $\chi_{\Gamma} = V - I$ and $L = 1 - \chi_{\Gamma}$
convert a connected diagram into an integer $L$ that then appears in
$D = d L - 2 I$. This is bookkeeping inside the superficial-degree
formula. It becomes Gauss–Bonnet only if the diagram has been
identified with a surface, which is the string or ribbon reading
above, not the default for a $\phi^{4}$ graph.

In dimension $4$ there is a further topological density, distinct from
the Einstein–Hilbert integrand. The Chern–Gauss–Bonnet theorem on a
closed oriented Riemannian $4$-manifold reads
$$
\chi(M)
=
\frac{1}{32\pi^{2}}
\int_{M}
\left(
R_{\mu\nu\rho\sigma} R^{\mu\nu\rho\sigma}
-
4 R_{\mu\nu} R^{\mu\nu}
+
R^{2}
\right)
\mathrm{dvol}.
$$
The integrand is quadratic in curvature. The Einstein–Hilbert
integrand is linear in curvature. The identity fixes the integral of
that quadratic density. It leaves $[G] = 2 - d$ unchanged, and in
$d = 4$ that dimension is $-2$.

**What.** Four roles for an Euler number, plus the $4$-dimensional
Euler density.

**Why.** The appetizer is easy to over-read as a renormalizability
test. Writing the roles side by side keeps the string weight, the
loop count, and the sign of $[G]$ from collapsing into one sentence.

**How.** Match each formula in this section to one row of the sign
table. The string row uses $\chi(\Sigma)$. The $\phi^{4}$ row uses
$4 - d$. The Einstein–Hilbert row uses $2 - d$, and uses $\chi$ only
through the $d = 2$ collapse and through the separate $4$-dimensional
density.

**When.** The question is which integer a given formula actually
contains. A triangulated $4$-manifold's face-count $\chi$, as in
notebook 05, is a filter on homeomorphism type. It is not an input
to $[\lambda_4]$.
"""))

    cells.append(md(r"""
### Tests that do not read $\chi$

The sign of $[\lambda_4] = 4 - d$ is fixed by $d$ and by the
field dimension. $\chi(S^{2}) = 2$ and $\chi(S^{4}) = 2$ do not
appear in the formula. Both spheres have Euler characteristic $2$
because the dimension is even, by the formula
$\chi(S^{n}) = 1 + (-1)^{n}$ from notebook 05. That common value
does not put $\phi^{4}$ and Einstein–Hilbert in the same
power-counting class. In $d = 4$ one coupling has dimension $0$ and
the other has dimension $-2$.

$S_{\mathrm{EH}} = \chi /(4G)$ is an identity in two dimensions. The
derivation uses Gauss–Bonnet, which has the dimension of the surface
built into $\int R\,\mathrm{d}A = 4\pi\chi$. Carrying the identity
into $d = 4$ would replace the Einstein–Hilbert integrand by the
quadratic Euler density above. Those are different actions. The
quadratic density being topological does not cancel the negative
dimension of $G$ in front of $\int R$.

The genus expansion weights amplitudes by $g_s^{-\chi}$. It assumes a
sum over worldsheets. It does not compute $D$ for a bulk diagram, and
a bulk theory's power-counting class is not settled by naming the
weight. The ribbon factor $N^{\chi}$ likewise sorts a gauge-theory
expansion by genus after $[g_{\mathrm{YM}}]$ has already been read
off as $(4-d)/2$.

$\chi_{\Gamma} = 1 - L$ is an input to $D$ only as the loop number.
Two diagrams with the same $L$ and different coupling dimensions have
different $D$. The graph Euler characteristic does not override
$[\lambda]$.

**What.** The formulas of the sign test, written so that $\chi$ of a
manifold is absent from them.

**Why.** Notebook 03 deferred this separation. The deferral is the
reason this banner exists.

**How.** Compare $[\lambda_4] = 4 - d$ with $S_{\mathrm{EH}} = \chi/(4G)$.
The first depends on the spacetime dimension. The second, in the
setting where it holds, depends on the topology of a surface.

**When.** The claim under discussion is renormalizability in the
power-counting sense in dimension $d \ge 2$. A computation of
$V - E + F$ for a mesh stays in the first banner.
"""))

    cells.append(md(r"""
## Cryptology and hardness

Hardness as a resource, read off the notebooks 04–05 table.
Survey names of neighbouring ideas stay names. The non-claims are
part of the banner.
""", collapsed=False))

    cells.append(md(r"""
### Four predicates

A decision problem supplies a cryptographic resource, in the sense
used here, only when four predicates hold for a security parameter
$n$.

**Sample.** A probabilistic polynomial-time algorithm outputs an
instance whose size is polynomial in $n$.

**Plant.** The same algorithm outputs a witness together with the
instance: a trapdoor, or a planted solution the holder can use.

**Hide.** Without that witness, the average instance drawn by the
sampler is hard. The quantifier is the sampler's distribution.
A theorem that some infinite family of instances has no algorithm is
a different quantifier.

**Verify.** The honest operation — checking the witness, or applying
the trapdoor — runs in probabilistic polynomial time.

The four are a checklist, not a protocol. No distribution is defined
in this notebook. The checklist is the shape a construction would
have to match. The next section evaluates the checklist on theorems
that already appear in notebooks 04 and 05. Those theorems are
worst-case statements about a finite complex presented as input.
None of them mentions a security parameter.
"""))

    cells.append(md(r"""
### The landscape as a resource check

Notebook 05 is the source of the rows. The hypotheses (which Novikov,
which target, what $\in\mathrm{NP}$ means) stay there. This table
adds only the four predicates.

| Row | Sample | Plant | Hide | Verify |
|-----|--------|-------|------|--------|
| $k=1,2$. Degrees, cycles, surface classification | the check is the algorithm | none | no hidden instance | the check itself |
| $k=3$. $S^{3}$ recognition $\in\mathrm{NP}$ (Schleimer) | no sampler in the theorem | no trapdoor | hardness is open | yes-instances have short certificates |
| $k=3$. Oriented homeomorphism decidable (Kuperberg) | — | — | a decision procedure exists | the procedure is the honest algorithm |
| Knot genus $\mathrm{NP}$-complete (Agol–Hass–Thurston), named in notebook 05 | a worst-case reduction | no trapdoor in the theorem | worst-case, not an average over a planted distribution | an $\mathrm{NP}$ certificate |
| $k=4$. Homeomorphism undecidable (Markov); target $\#_{k}(S^{2}\times S^{2})$ | arbitrary triangulations; no polynomial sampler of hard pairs | no | worst-case; every algorithm fails on some infinite family | the honest party has no algorithm on that family either |
| $k=4$. $S^{4}$ recognition open | no theorem | no theorem | no theorem | no theorem |
| $k\ge 5$. $S^{d}$ recognition undecidable (S. P. Novikov) | same shape as the Markov row, with target $S^{d}$ | no | worst-case undecidable | fails on the worst case, for the honest party as well |

Schleimer's theorem (arXiv:math/0407047, Theorem 16.1 in the
preprint; Proc. Sympos. Pure Math. 82 (2011), Theorem 15.1) is
membership of $3$-sphere recognition in $\mathrm{NP}$. Membership
means yes-instances have certificates that a deterministic verifier
accepts in polynomial time. That is the Verify column, for yes
instances of one recognition problem. The 2004 preprint asks whether
the problem is $\mathrm{NP}$-hard and leaves the question open.
Notebook 05 records the direction of the bound: an upper bound on
$S^{3}$ recognition is not a lower bound on a problem in dimension
$4$. The same direction applies here. An upper bound is not the Hide
column.

Kuperberg's theorem (Pacific J. Math. 301 (2019), 189–241) makes
oriented $3$-manifold homeomorphism decidable. A decidable problem
fails Hide. It is the wrong shape for a resource whose secrecy is
the absence of an algorithm.

Agol, Hass, and Thurston (STOC 2002; arXiv:math/0205057) prove that
$3$-manifold knot genus is $\mathrm{NP}$-complete. Notebook 05 cites
that theorem as the hardness result Schleimer points to, still in
dimension $3$, still about knots. $\mathrm{NP}$-completeness is a
worst-case reduction from every language in $\mathrm{NP}$. The
reduction does not plant a witness into a distribution, and it does
not name a security parameter. The Verify column holds because the
problem is in $\mathrm{NP}$. Hide, in the average-case sense above,
is not part of the statement.

Markov's theorem (Dokl. Akad. Nauk SSSR 121 (1958), 218–220) says
that for every $n \ge 4$ no algorithm decides homeomorphism of closed
triangulated $n$-manifolds. In dimension $4$ the unrecognizable
target is $\#_{k}(S^{2}\times S^{2})$, not $S^{4}$. Notebook 05
separates those targets by $\chi = 2 + 2k$. Undecidability fails
Verify on the worst case: the honest party does not have an
algorithm that the theorem says does not exist. A resource that the
legitimate user cannot evaluate is not a scheme. The open status of
$S^{4}$ recognition is a third problem. An open problem has no entry
in any of the four columns.

S. P. Novikov's theorem, as stated in Volodin, Kuznetsov, and
Fomenko (Russian Math. Surveys 29 (1974), no. 5, 169–171), says that
recognizing $S^{d}$ is undecidable for every $d \ge 5$. The shape
matches the Markov row with a different target. P. S. Novikov's
unsolvability of the word problem is the group-theory input named in
notebook 05. It is not a sampler of manifolds.

**What.** The four predicates on each row of the landscape.

**Why.** "Hard" and "undecidable" and "in $\mathrm{NP}$" are three
results in notebooks 04 and 05. Only a row that filled all four
columns would be a resource in the sense of this banner. None does.

**How.** Read one row of notebook 05, then the same row here. Keep
the target of the theorem (the sphere, a connected sum, a knot
genus) fixed while reading the columns.

**When.** The question is whether a complexity theorem is already a
cryptographic construction. The answer on this table is that the
predicates the construction would need are not the theorems.
"""))

    cells.append(md(r"""
### Neighbouring names

A few ideas sit next to this table in the literature. They are named
so the table is not asked to be them. This notebook does not state
their public data, element lengths, or operations.

Combinatorial-group proposals take a search problem in a finitely
presented group as a hypothesis. Anshel, Anshel, and Goldfeld (Math.
Res. Lett. 6 (1999), 287–291) and Ko, Lee, Cheon, Han, Kang, and Park
(CRYPTO 2000, LNCS 1880, 166–183) name the conjugacy search problem
in the braid group. The braid group is a particular group. Its word
problem is solvable; that is classical for Artin's braid group, and
it is why the braid group is not an instance of P. S. Novikov's
theorem on the general finitely presented word problem. A hypothesis
that a search problem is hard in practice, for parameter sizes the
proposal would have to choose, is not an undecidability theorem and
not Schleimer's theorem. Those sizes are not chosen here.

The face-count $\chi$ is sometimes suggested as a fingerprint of a
mesh. On a closed combinatorial surface, $\chi$ is $V - E + F$, which
notebook 03 computes from the facet list in linear time in the number
of faces. Every torus returns $0$. A function that is constant on an
infinite family collides on that family. The speed of the computation
is the reason it is a poor fingerprint, not a reason it is a hidden
resource. Sample is easy, Plant is empty, Hide fails because the
value is the face count, and Verify is the same addition.

Knot genus, cited above, is the hardness theorem already on the
table. Naming it again in this paragraph would not add a
distribution. The citation stays in the previous section.

**What.** Two names that are easy to attach to the landscape, and
the predicate each one actually fills.

**Why.** A solvable word problem in one group is not the undecidable
word problem Markov uses. A linear-time invariant is not a hidden
instance.

**How.** Match the braid sentence to P. S. Novikov's theorem as
notebook 05 separates the two Novikovs. Match $\chi$ as a fingerprint
to the face-count formula in notebook 03.

**When.** The claim is that a published idea "uses topology, so it
uses this series." The series' theorems are the rows above. A
construction that does not fill the four predicates is not licensed
by a row that fails them.
"""))

    cells.append(md(r"""
### Non-claims

The following bounds are part of the result of this banner.

This notebook does not define a cryptosystem, a key schedule, a
challenge distribution, or a security parameter.

$S^{3}$ recognition is not a cipher, a signature, a hash, or a key
exchange in this file. Schleimer's theorem is not used as one.
Membership in $\mathrm{NP}$ is not read as $\mathrm{NP}$-hardness,
and $\mathrm{NP}$-hardness is not read as an average-case one-way
function.

Homeomorphism of $4$-manifolds and recognition of $S^{d}$ for
$d \ge 5$ are not ciphers. Markov's theorem and S. P. Novikov's
theorem are not key generation. The open status of $S^{4}$
recognition is not a temporary parameter set.

The braid-group papers are not instantiated. No public element, no
private element, and no operation on braids is specified.

$\chi$ of a complex is not proposed as a hash. Knot genus is not
proposed as a one-way function.

No sentence in this banner is an instruction to deploy recognition of
$S^{3}$, or recognition of any other manifold, as cryptography.
"""))

    cells.append(md(r"""
## Closing

The three banners are one file at v1 and may split at their headings
into notebooks 06 and 07.

A triangle mesh reaches a surface code only after it is a simplicial
complex and the link formula holds. Vertex degree does not record
edge incidence. The branched example has vertices of degree $2$ and
an edge in three triangles, which is why `twin` and $\mathrm{Rot}$
are both undefined there and still not the same operator. Persistent
homology reports ranks. The pinched complex is contractible and fails
the link test. Edwards' double suspension, in notebook 05, is
homeomorphic to $S^{5}$ and fails the link test.

$[\lambda_4] = 4 - d$ and $D = d - [\phi] E - [\lambda] V$ sort
$\phi^{4}$ by dimension: renormalizable at $d = 4$, where $D$ for
$E = 4$ is independent of $V$. $[G] = 2 - d$ is zero at $d = 2$,
where the action equals $\chi/(4G)$, and $-2$ at $d = 4$. The string
weight $g_s^{-\chi}$, the ribbon weight $N^{\chi}$, and the graph
identity $L = 1 - \chi_{\Gamma}$ use Euler characteristics for three
further jobs. None of them is the sign of $[\lambda_4]$. The
$4$-dimensional Euler density integrates to $\chi$ and is not the
Einstein–Hilbert integrand. The cell checks the scalar identities
and does not produce a counterterm. One-loop on-shell finiteness of
pure gravity and the two-loop divergence are citations, not
calculations of this notebook.

The notebooks 04–05 landscape does not fill sample, plant, hide, and
verify on any row. $S^{3}$ recognition supplies certificates, which
is an upper bound. Knot genus is $\mathrm{NP}$-complete as a
worst-case statement about knots. Undecidability in dimensions
$\ge 4$ removes the honest algorithm on the worst case. This file
defines no scheme and deploys no recognition problem as one.
"""))

    return cells


def main() -> None:
    cells = build_cells()
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
