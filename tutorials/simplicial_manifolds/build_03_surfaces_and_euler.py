#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 03 (source of truth): surfaces and Euler characteristic.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_03_surfaces_and_euler.py

The generated notebook is ``03_surfaces_and_euler.ipynb`` beside this
file. It specialises the link criterion of notebook 02 at dimension
$k=2$, computes $\\chi = V-E+F$ on a sphere, a torus, and
$\\mathbb{R}\\mathrm{P}^{2}$, and compares half-edge walks with the
Quad-Edge operator $\\mathrm{Rot}$. Interactive figures live in
``euler_surface_demo.py`` and ``surface_tds_demo.py``. A short QFT
appetizer records that the 2-dimensional Einstein–Hilbert action is
proportional to $\\chi$, and points to notebook 06 for power counting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("03_surfaces_and_euler.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Surfaces and the Euler characteristic

Notebook 02 fixes the local test for a closed combinatorial surface:
the complex is pure of dimension $2$, and the link of every vertex is
one cycle. `closed_surface_link_condition` is that test. This notebook
does not rebuild it. On a complex that passes, every edge lies in
exactly two triangles, and the global integer
$$
\chi = V - E + F
$$
is defined. The gallery computes it for a sphere, a torus, and
$\mathbb{R}\mathrm{P}^{2}$.

The same complexes are where a mesh encoding has to be chosen.
Counting faces answers $\chi$. A half-edge answers an oriented walk
around a triangle. A Quad-Edge answers that walk and the same walk on
the dual. Those are different operators. The last section is a short
QFT appetizer: in two dimensions the Einstein–Hilbert action is
proportional to $\chi$. Notebook 06 is the power-counting argument
this identity does not replace.

## Learning objectives

1. Compute $\chi = V - E + F$ from the downward closure, and match it
   to $\sum_d (-1)^d f_d$.
2. Separate the sphere, the torus, and $\mathbb{R}\mathrm{P}^{2}$ by
   the pair (orientable, $\chi$).
3. Check the degree-defect form $\sum_v(6-\deg v) = 6\chi$ on each
   gallery surface.
4. Choose face counts, $\partial_k$, a half-edge, or a Quad-Edge by
   the question the algorithm asks.
5. State $S_{\mathrm{EH}}\propto\chi$ in two dimensions, and where
   notebook 06 takes up renormalizability.
"""))

    cells.append(md(r"""
## Closed triangulations

A complex in this notebook is a closed triangulation when it passes
`closed_surface_link_condition`. Notebook 02 records why that forces
every edge into two triangles and every vertex link into a single
cycle of length $\ge 3$. The count of those triangles is $F$, the
count of edges is $E$, and the count of vertices is $V$. Each triangle
contributes three edge ends and each edge is shared by two triangles,
so
$$
3F = 2E.
$$

The Euler characteristic of a finite complex is the alternating face
count, omitting the empty simplex:
$$
\chi = \sum_{d\ge 0}(-1)^d f_d.
$$
On a pure $2$-complex the sum stops at the triangles,
$$
\chi = f_0 - f_1 + f_2 = V - E + F.
$$
`euler_characteristic` evaluates the sum from `faces_by_dim()`. The
facet list alone is the wrong store: the tetrahedron boundary has
four facets and $\chi = 2$.

**What.** $f_d$ is the number of nonempty $d$-simplices in the
downward closure.

**Why.** $\chi$ is defined as that alternating sum. The sum needs no
signs and no dual.

**How.** Build the closure once, then take the cardinality of each
dimension. The code cell prints $V$, $E$, $F$ and the same integer
from `euler_characteristic`.

**When.** The question is $\chi$, the $f$-vector, or the degree defect
below. Incidence signs and $\partial^2 = 0$ use `boundary_matrix`.
A walk around one triangle uses a half-edge. A walk that also runs on
the dual mesh uses a Quad-Edge. Notebook 02's adjacency list remains
the tool for a degree or a $k=1$ cycle; an adjacency list does not
store triangles, so it cannot see $F$.
"""))

    cells.append(md(r"""
## Three surfaces

The claims, checked in the next cell:

| Surface | Complex | $V$ | $E$ | $F$ | $\chi$ | orientable |
|---------|---------|----:|----:|----:|-------:|:----------:|
| sphere $S^{2}$ | boundary of a tetrahedron | 4 | 6 | 4 | 2 | yes |
| torus $T^{2}$ | $3\times 3$ grid, opposite sides identified | 9 | 27 | 18 | 0 | yes |
| torus $T^{2}$ | $7$-vertex cyclic triangulation | 7 | 21 | 14 | 0 | yes |
| $\mathbb{R}\mathrm{P}^{2}$ | $6$-vertex triangulation | 6 | 15 | 10 | 1 | no |

$$
\chi(S^{2}) = 2,
\qquad
\chi(T^{2}) = 0,
\qquad
\chi(\mathbb{R}\mathrm{P}^{2}) = 1.
$$

The two tori are there so $\chi$ is not mistaken for a vertex count.
Both are orientable and both have $\chi = 0$. The gallery draws the
grid, because the square shows the identifications. The $7$-vertex
complex is the smallest triangulation of the torus; the cell counts
it and does not draw it.

Orientability is a second computation. Order each triangle so that
every shared edge is used once in each direction. On a closed
combinatorial surface that assignment exists exactly when the surface
is orientable. The sphere and both tori receive one. On
$\mathbb{R}\mathrm{P}^{2}$ the propagation meets an edge whose two
triangles already demand opposite signs; the cell prints that edge.
The half-edge sketch later refuses to invent a twin there. $\chi$ does
not need the twin.

A disjoint union of two tetrahedron boundaries still passes the link
condition from notebook 02, one component at a time. Its Euler
characteristic is $2+2 = 4$. The classification formulas in the next
section are for a single component.
"""))

    cells.append(code(r"""
# Face-count χ for the gallery, the 7-vertex torus, and a disjoint union.
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "lib").is_dir():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "lib").is_dir():
    raise FileNotFoundError("lib/ missing next to this notebook")
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from euler_surface_demo import (
    classification_label,
    coherently_orientable,
    degree_defects,
    disjoint_union,
    face_counts,
    orientation_attempt,
    real_projective_plane,
    surface_report,
    tetrahedron_boundary,
    torus_grid,
    torus_seven,
)
from lib import closed_surface_link_condition, euler_characteristic


def show(name, K):
    print(surface_report(name, K))
    print()


sphere = tetrahedron_boundary()
torus = torus_grid()
small_torus = torus_seven()
plane = real_projective_plane()

show("sphere", sphere)
show("torus, 3×3 grid", torus)
show("torus, 7 vertices", small_torus)
show("real projective plane", plane)

pair = disjoint_union(sphere, sphere)
show("two spheres, disjoint", pair)

for name, K, chi, orientable in (
    ("sphere", sphere, 2, True),
    ("grid torus", torus, 0, True),
    ("7-vertex torus", small_torus, 0, True),
    ("RP2", plane, 1, False),
):
    vertices, edges, faces = face_counts(K)
    assert vertices - edges + faces == chi
    assert euler_characteristic(K) == chi
    assert 3 * faces == 2 * edges
    assert closed_surface_link_condition(K) is True
    assert coherently_orientable(K) is orientable
    assert sum(degree_defects(K).values()) == 6 * chi

assert euler_characteristic(pair) == 4
assert closed_surface_link_condition(pair) is True
assert classification_label(pair) == "2 components, χ=4"
conflict = orientation_attempt(plane).conflict_edge
print("RP2 orientation conflict on edge", conflict)
assert conflict is not None
"""))

    cells.append(md(r"""
## Classification sketch

For a connected closed surface, the homeomorphism type is fixed by
orientability together with $\chi$.

An orientable surface has a genus $g\ge 0$ and
$$
\chi = 2 - 2g.
$$
The sphere is $g = 0$. The torus is $g = 1$. The labels printed above
are this formula solved for $g$.

A non-orientable surface has $k\ge 1$ crosscaps and
$$
\chi = 2 - k.
$$
$\mathbb{R}\mathrm{P}^{2}$ is $k = 1$. The Klein bottle is $k = 2$,
so it also has $\chi = 0$. The torus and the Klein bottle are not
homeomorphic. $\chi$ alone does not separate them; the orientation bit
does. The gallery does not triangulate the Klein bottle. The formula
is the point of naming it.

The labels require one component. Two spheres have $\chi = 4$, which
is the sum of the characteristics, and the printed label is the
component count rather than a genus. Notebook 02 already allowed that
disjoint union through the link condition. Connectedness is extra data
for the classification line, read here by a walk on the adjacency list
of the $1$-skeleton.

This is a sketch of the classification, checked on four complexes. It
is not a census of triangulations, and it is not a recognition
procedure in dimension $3$ or higher. Those questions belong to
notebooks 04 and 05. In dimension $2$ the pair (orientable, $\chi$)
is the global invariant once the link condition holds.
"""))

    cells.append(md(r"""
## A taste of discrete curvature

On a closed triangulation the handshakes $3F = 2E$ and
$\sum_v\deg(v) = 2E$ rearrange into
$$
\sum_v (6 - \deg v) = 6V - 2E = 6\chi.
$$
Each report in the computation cell prints that sum next to $6\chi$. The
sphere has four vertices of degree $3$, defect $3$, and the sum is
$12$. The grid torus is $6$-regular, every defect is $0$, and the sum
is $0$. $\mathbb{R}\mathrm{P}^{2}$ has six vertices of degree $5$,
defect $1$, and the sum is $6$.

The integer $6 - \deg(v)$ is the combinatorial stand-in for an angle
defect $2\pi - \sum \theta$. Six equilateral triangles around a vertex
fill the plane and give defect $0$. Gauss–Bonnet for a closed
Riemannian surface is $\int K\,\mathrm{d}A = 2\pi\chi$. The degree sum
has the same shape, an alternating count of local defects equal to a
multiple of $\chi$, and it stores no metric. The gallery uses it as a
check that the face counts and the degrees describe one complex.
"""))

    cells.append(md(r"""
## The same $\chi$, read off $\partial_k$

`boundary_matrix(K, k)` is the sparse map $\partial_k\colon C_k\to
C_{k-1}$ from notebook 01. Its number of columns is $f_k$, because the
columns are the oriented $k$-simplices. In this unaugmented complex
$\partial_0$ has no rows. For a pure $2$-complex
$$
\chi = \mathrm{ncols}(\partial_0) - \mathrm{ncols}(\partial_1)
+ \mathrm{ncols}(\partial_2).
$$
That is the face-count formula written in the shapes of the chain
maps. Use it when a boundary matrix is already the object in hand:
incidence signs, the identity $\partial_{k-1}\partial_k = 0$, or a
later rank computation. Homology ranks satisfy
$\chi = b_0 - b_1 + b_2$ as well, and those ranks need
$\mathrm{rank}(\partial)$, which is more work than the cardinality of
a face bucket. This notebook does not compute Betti numbers.

A dense matrix is a contrast, not a store. `toarray()` on $\partial_1$
of the tetrahedron boundary is a $4\times 6$ integer matrix, small
enough to read, and full of the zeros a sparse format declines to
keep. The adjacency list of the same complex stores twelve neighbour
slots ($2E$). A dense $V\times V$ block stores sixteen entries here
and $V^{2}$ entries on a larger mesh. $\chi$ is not a row sum of
either matrix.

**What.** Column counts of the sparse boundary matrices.

**Why.** They are $f_k$ again, so they reproduce $\chi$ without a
second census of the facet list.

**How.** Read `.shape[1]` for $k = 0, 1, 2$ and alternate the signs.
The dense array is printed only for the tetrahedron.

**When.** The surrounding computation is already about chains. When
the question is only $\chi$, `euler_characteristic` is the direct
call. When the question is a walk, neither matrix is the walk.
"""))

    cells.append(code(r"""
# Column counts of ∂_k reproduce χ. toarray() is a contrast on 4 vertices.
from lib import boundary_matrix


def chi_from_columns(K):
    counts = [boundary_matrix(K, k).shape[1] for k in range(3)]
    return counts[0] - counts[1] + counts[2], counts


sphere = tetrahedron_boundary()
vertices, edges, faces = face_counts(sphere)
chi, counts = chi_from_columns(sphere)
print("face_counts", (vertices, edges, faces))
print("column counts f_0, f_1, f_2", counts)
print("χ from columns", chi)
print("χ from euler_characteristic", euler_characteristic(sphere))
assert chi == euler_characteristic(sphere) == vertices - edges + faces
assert chi_from_columns(torus_grid())[0] == 0
assert chi_from_columns(real_projective_plane())[0] == 1

partial = boundary_matrix(sphere, 1)
print("∂_1 type", type(partial).__name__, "shape", partial.shape)
print(partial.toarray())
slots = sum(len(neighbours) for neighbours in sphere.adjacency_list().values())
print("adjacency-list neighbour slots", slots)
print("dense V×V entries", vertices * vertices)
assert slots == 2 * edges
"""))

    cells.append(md(r"""
## Half-edge, then Quad-Edge

The gallery never builds either structure. $\chi$ does not ask for an
edge pointer. The structures earn their place when the algorithm
walks.

A half-edge is one directed side of one oriented triangle. The
teaching sketch stores
$$
h \mapsto \bigl(\mathrm{origin}(h),\,\mathrm{twin}(h),\,\mathrm{next}(h),\,\mathrm{face}(h)\bigr).
$$
On a coherently oriented closed triangulation
$$
\mathrm{twin}(\mathrm{twin}(h)) = h,
\qquad
\mathrm{next}^{3}(h) = h
$$
for each triangular face, and the vertex circulator
$$
h \mapsto \mathrm{next}(\mathrm{twin}(h))
$$
returns to $h$ after $\deg(\mathrm{origin}(h))$ steps. Each step of
that circulator is an outgoing primal edge. The ring is the star of
the vertex in the surface, not a tour of the dual.

**What.** Constant-time pointers for an oriented primal walk.

**Why.** The next edge of a face, and the other side of the same
undirected edge, are the two moves a triangle mesh makes constantly.
The facet list can find them by scanning the triangles on an edge.
The half-edge stores the answer on the edge.

**How.** `build_halfedge` consumes the coherent orientation already
used for the classification bit, then writes `next` around each
triangle and `twin` across each undirected edge. The trace prints one
face ring and one vertex ring on the tetrahedron boundary. Both have
length $3$.

**When.** Walk the boundary of a face, walk the star of a vertex, or
name the diagonal a flip would rewrite. Graphics meshes and most
triangle-mesh surgery use this encoding. It has no pointer onto a
dual edge. Use it when the algorithm stays on the primal surface.

`twin` exists for every edge only when a coherent orientation exists.
On $\mathbb{R}\mathrm{P}^{2}$ the orientation conflicts, and
`build_halfedge` raises rather than store a one-way edge. That
refusal is the orientability witness again. Face counts still return
$\chi = 1$. A single triangle is refused for a different reason: its
three edges have no second face, and this sketch does not add the
boundary sentinel a production DCEL would use. The gallery is closed.

A flip, described and not performed: two triangles that share an edge
$\{u,v\}$ have opposite vertices $a$ and $b$. Replacing $\{u,v\}$ by
$\{a,b\}$ rewrites `next` on those two faces and the twins of the new
diagonal. On the tetrahedron boundary $\{a,b\}$ is already an edge, so
the trace says a flip would not create a simplex. On the grid torus
the trace names a missing diagonal and stops. An industrial kernel
would edit the pointers. `surface_tds_demo.py` reports the pair.
"""))

    cells.append(md(r"""
## Quad-Edge

Guibas and Stolfi, *Primitives for the manipulation of general
subdivisions and the computation of Voronoi diagrams*, ACM Transactions
on Graphics 4 (1985), package one undirected edge as four directed
records in a single cycle of an operator $\mathrm{Rot}$:
$$
\mathrm{Rot}^{4}(e) = e,
\qquad
\mathrm{Sym}(e) = \mathrm{Rot}^{2}(e).
$$
If $e$ is a directed primal edge, $\mathrm{Rot}(e)$ is a directed dual
edge, from the left face of $e$ to the right face of $e$.
$\mathrm{Sym}(e)$ is the reversed primal edge. $\mathrm{Rot}^{3}(e)$
is the reversed dual edge.

$\mathrm{Onext}(e)$ is the next record counterclockwise about the
origin of $e$. On a primal record that origin is a mesh vertex, and
the cycle length is $\deg(v)$. On a dual record that origin is a primal
face, and on a triangulation the cycle length is $3$. The same
permutation implements both rings. The face walk that the half-edge
calls $\mathrm{next}$ is recovered by
$$
\mathrm{Lnext}(e) = \mathrm{Rot}^{-1}\bigl(\mathrm{Onext}(\mathrm{Rot}(e))\bigr).
$$
The sketch sets the dual $\mathrm{Onext}$ from
$\mathrm{Onext}(\mathrm{Rot}(e)) = \mathrm{Rot}(\mathrm{next}(e))$, so
the identity holds by construction, and the trace checks that
$\mathrm{Lnext}$ reproduces $\mathrm{next}$ on every half-edge of the
tetrahedron.

$\mathrm{Sym}$ reverses a primal arrow, which is what $\mathrm{twin}$
does. $\mathrm{Sym}$ is $\mathrm{Rot}$ applied twice, so the cycle
that reverses the arrow also contains the dual. $\mathrm{twin}$ has no
dual slot. $\mathrm{Rot}$ is not $\mathrm{twin}$, and it is not
$\mathrm{next}$.

**What.** One edge algebra for a subdivision and its dual.

**Why.** An algorithm that alternates between a mesh and its dual
otherwise keeps two complexes and a handwritten correspondence.
$\mathrm{Rot}$ is that correspondence. Guibas–Stolfi used it so that a
Delaunay edge and the Voronoi edge dual to it are one record bundle.

**How.** Four records per undirected edge of an oriented closed
triangulation. Primal $\mathrm{Onext}$ copies the half-edge vertex
circulator. Dual $\mathrm{Onext}$ copies the face ring through
$\mathrm{Rot}$. The trace prints the $4$-cycle of the arrow $0\to 1$
on the tetrahedron, then the $\mathrm{Onext}$ ring about one dual
origin, then the cycle lengths on the grid torus: primal length $6$,
dual length $3$.

**When.** The algorithm talks to both a mesh and its dual: Delaunay
and Voronoi together, a discrete exterior calculus that passes between
primal edges and dual edges, a flip whose dual edge must move with the
primal diagonal. The trace's torus sentence names that splice and does
not perform it. Prefer a half-edge when the dual is not an object in
the algorithm. Prefer face counts when the only integer wanted is
$\chi$. The classical $\mathrm{Rot}$ cycle wants a left face and a
right face, so the sketch builds it on the sphere and the torus and
refuses $\mathbb{R}\mathrm{P}^{2}$ with the same orientation conflict.

The figure steps the three operators on the tetrahedron boundary.
Orange is a half-edge, offset into its face. Blue is $\mathrm{Rot}$
of that edge once it has left the primal: the segment joining the
centroids of the two faces that share the edge. The right-hand
$4$-cycle is one undirected edge. $\mathrm{Sym}$ sits opposite $e$.
$\mathrm{Rot}(e)$ sits on a dual corner.
"""))

    cells.append(md(r"""
## Which structure, for which question

| Question | Structure | Why this one |
|----------|-----------|----------------|
| $\chi$, the $f$-vector, $\sum(6-\deg v)$ | face counts, `euler_characteristic` | the alternating sum is the definition |
| incidence signs, $\partial^{2}=0$, column counts | `boundary_matrix` | the map is linear on oriented simplices |
| degree, a $k=1$ cycle (notebook 02) | `adjacency_list` | unsigned neighbour sets, no faces above dimension $1$ |
| walk a triangle, walk a vertex star, name a primal flip | half-edge | `next` and `twin` are the walk |
| the same walk on the dual, a paired subdivision | Quad-Edge, $\mathrm{Rot}$ and $\mathrm{Onext}$ | one bundle holds the primal arrow and the dual arrow |
| show what the zeros would have cost | a dense array, then leave it | the tetrahedron's $\partial_1$ is already $4\times 6$ |

Half-edge and Quad-Edge stay in `surface_tds_demo.py`. `lib/` keeps
the facet list, the adjacency list, and the sparse boundary. Notebook
06 can reprise this table for meshes in computation; the operators
above are the ones that reprise would call.
"""))

    cells.append(code(r"""
# Operator trace. Drawings of the same walks are the hidden cell below.
from surface_tds_demo import trace_report

print(trace_report())
"""))

    cells.append(md(r"""
## Gallery

The dropdown is the sphere, the grid torus, and
$\mathbb{R}\mathrm{P}^{2}$. The title line is $V - E + F$ computed
from the complex, not typed into the picture.

- The sphere is the tetrahedron boundary in $\mathbb{R}^{3}$. The
  camera sliders apply only there.
- The torus is the $3\times 3$ grid drawn as a square. Hollow dots on
  the far sides are the same vertices as the opposite side. They are
  not a tenth vertex. Matching arrows mark the two identifications.
  $\chi = 0$ with nine vertices; the $7$-vertex count in the cell
  above is the same $\chi$ with fewer vertices.
- $\mathbb{R}\mathrm{P}^{2}$ is the $6$-vertex triangulation drawn in
  a disk. Antipodal points of the circle are one point. Crossings
  inside the disk belong to the drawing. $\chi = 1$ is the face count.

Drawing and widgets live in `euler_surface_demo.py`. The next cell
only checks that the file is beside the notebook and calls
`launch_euler_surface_demo`.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

The gallery lives in <code>euler_surface_demo.py</code> next to this
notebook, so this cell stays a launch stub. JupyterLab collapses the
launch input (<code>source_hidden</code>). Open the <code>.py</code>
to change coordinates or the fundamental-domain arrows.

</details>
"""))

    cells.append(code(r"""
# Launch only — implementation: euler_surface_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "euler_surface_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "euler_surface_demo.py").exists():
    raise FileNotFoundError("euler_surface_demo.py missing next to this notebook")
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from euler_surface_demo import launch_euler_surface_demo

launch_euler_surface_demo()
""", hide_input=True))

    cells.append(md(r"""
## Stepping the operators

The second figure is the tetrahedron boundary again, with a choice of
operator.

- **next** walks one face. Three steps close the triangle. The arrow
  stays primal.
- **next(twin)** walks the outgoing edges at one vertex. The length is
  the degree. The ring does not visit a face-centroid.
- **Rot** walks the $4$-cycle of one undirected edge. Two of the four
  stops are primal. The other two are the dual edge, drawn in blue
  between face centroids, and its reverse.

`surface_tds_demo.py` holds the pointers and the figure. The cell is
a launch stub.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

Half-edge records, Quad-Edge records, and the stepper live in
<code>surface_tds_demo.py</code>. The printed trace in the earlier
cell is <code>trace_report()</code> in that file. This cell only
launches the figure.

</details>
"""))

    cells.append(code(r"""
# Launch only — implementation: surface_tds_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "surface_tds_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "surface_tds_demo.py").exists():
    raise FileNotFoundError("surface_tds_demo.py missing next to this notebook")
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from surface_tds_demo import launch_surface_tds_demo

launch_surface_tds_demo()
""", hide_input=True))

    cells.append(md(r"""
## A QFT appetizer

Gauss–Bonnet on a closed Riemannian surface reads
$$
\int_{\Sigma} K\,\mathrm{d}A = 2\pi\,\chi(\Sigma).
$$
In two dimensions the scalar curvature is twice the Gaussian
curvature, $R = 2K$, so
$$
\int_{\Sigma} R\,\mathrm{d}A = 4\pi\,\chi(\Sigma).
$$
The Einstein–Hilbert action with the usual normalisation is
$$
S_{\mathrm{EH}}[g]
=
\frac{1}{16\pi G}\int_{\Sigma} R\,\mathrm{d}A
=
\frac{\chi(\Sigma)}{4G}.
$$
The $4\pi$ in Gauss–Bonnet and the $16\pi G$ in the action produce
$1/(4G)$. On a connected surface the integral depends on the metric
only through $\chi$. Two metrics on the torus give the same
Einstein–Hilbert value, because both have $\chi = 0$. The sphere and
$\mathbb{R}\mathrm{P}^{2}$ do not.

Closed oriented worldsheets are classified by the genus, and
$$
\chi = 2 - 2g.
$$
The closed-string perturbation series weights the genus-$g$ amplitude
by the Euler characteristic of that worldsheet,
$$
\mathcal{A}
=
\sum_{g\ge 0} g_s^{\,2g-2}\,\mathcal{A}_g
=
\sum_{g\ge 0} g_s^{\,-\chi}\,\mathcal{A}_g,
$$
since $2g - 2 = -\chi$. The power of the string coupling is a
topological weight. It is the same $\chi$ the gallery computed.

Power-counting renormalizability in four-dimensional spacetime is a
different test. It asks how the superficial degree of divergence of a
$1$PI diagram grows with the number of loops and external legs, given
the mass dimension of the couplings and the dimension of spacetime.
Finitely many counterterm types is the renormalizable case. That
arithmetic uses the Feynman rules of a $d=4$ theory. It does not read
$\chi$ of a triangulated surface, and $\chi(S^{2}) = 2$ does not
decide whether a given four-dimensional theory is renormalizable. The
identity $S_{\mathrm{EH}}\propto\chi$ says that two-dimensional pure
gravity, with this action, has no local bulk propagating mode. It does
not say that four-dimensional gravity is renormalizable, and it does
not say that it is not.

A Feynman diagram also has a graph Euler characteristic, built from
its vertices, edges, and connected components, and that integer counts
loops. The graph count is not Gauss–Bonnet, and it is not the
renormalizability test either. Notebook 06 walks the power-counting
argument: superficial degree, which couplings are renormalizable in
which dimension, and where a genus weight does and does not enter.
The number to carry forward from this notebook is $\chi$, together
with the fact that the $2$-dimensional Einstein–Hilbert action is that
number.
"""))

    cells.append(md(r"""
## Summary

On a closed triangulation, $\chi = V - E + F$, and
$3F = 2E$. The tetrahedron boundary has $\chi = 2$. The $3\times 3$
grid torus and the $7$-vertex torus both have $\chi = 0$. The
$6$-vertex triangulation of $\mathbb{R}\mathrm{P}^{2}$ has $\chi = 1$.
`euler_characteristic` and the column counts of $\partial_k$ agree
with that arithmetic. A dense matrix was printed for the tetrahedron
only so the zeros were visible.

For one connected component the classification sketch is
$\chi = 2 - 2g$ when the surface is orientable and $\chi = 2 - k$
when it is not. The torus and the Klein bottle share $\chi = 0$ and
differ by the orientation bit. Two disjoint spheres have $\chi = 4$.
The degree defects sum to $6\chi$, which is the combinatorial form of
the same global constraint, without a metric.

Face counts answer $\chi$. Half-edge `next` and `twin` answer an
oriented primal walk, and the sketch builds them only after a coherent
orientation exists. Quad-Edge $\mathrm{Rot}$ answers the same edge
together with its dual; $\mathrm{Sym} = \mathrm{Rot}^{2}$ is the
reversal, and $\mathrm{Lnext}$ reproduces `next`.
$\mathbb{R}\mathrm{P}^{2}$ keeps its $\chi$ and refuses the classical
twin. Notebook 02's adjacency list is still the tool for a $k=1$
degree walk.

In two dimensions $S_{\mathrm{EH}} = \chi/(4G)$, and the closed-string
weight is $g_s^{-\chi}$. Notebook 06 is where power counting in $d=4$
is spelled out. Notebook 04 leaves surfaces: vertex links become
$2$-spheres, and $\chi$ no longer classifies the manifold.
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
