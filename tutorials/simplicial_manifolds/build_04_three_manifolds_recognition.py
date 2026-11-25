#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 04 (source of truth): three-manifolds and recognition.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_04_three_manifolds_recognition.py

The generated notebook is ``04_three_manifolds_recognition.ipynb`` beside
this file. It specialises the link criterion of notebook 02 at dimension
$k=3$, using ``vertex_links_are_spheres_s2``, and separates that local
check from homeomorphism type. The complexity statement is Schleimer's
theorem that 3-sphere recognition lies in NP (arXiv:math/0407047, 2004;
Proc. Sympos. Pure Math. 82, 2011). The notebook does not implement a
certificate, a normal surface, or a crushing step.

No sibling demo. The vertex link of the model $\\partial\\Delta^{4}$ is the
tetrahedron boundary already drawn in notebooks 02 and 03.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("04_three_manifolds_recognition.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Three-manifolds and recognition

Notebook 02 fixes the closed combinatorial manifold criterion: the
complex is pure of dimension $k$, and the link of every nonempty
simplex is a piecewise-linear sphere of the complementary dimension.
Notebook 03 specialises that test at $k=2$ and computes
$$
\chi = V - E + F.
$$
On a connected closed surface the pair (orientable, $\chi$) determines
the homeomorphism type. This notebook specialises the same criterion
at $k=3$. The local question becomes whether every vertex link is a
combinatorial $2$-sphere. The global question is which $3$-manifold
the geometric realization is, up to homeomorphism.

Those questions have different answers. The boundary of the $4$-simplex
is the model PL $3$-sphere, and a disjoint union of two copies has the
same vertex links and the same Euler characteristic. Schleimer's
theorem (2004; published 2011) places recognition of the $3$-sphere in
NP: a yes-instance carries a polynomial-size certificate. The theorem
is that membership. It is not the statement "$3$-sphere recognition is
NP-hard." No cell builds the certificate. Notebook 05 is the
dimension $\ge 4$ landscape, where the proved statements are
undecidability theorems.

## Learning objectives

1. Read a triangulation from a facet list of tetrahedra, and compute
   $\chi = V - E + F - T$ from the downward closure.
2. State the $k=3$ link criterion, and run
   `vertex_links_are_spheres_s2` with the limits in its docstring.
3. Separate that local answer from homeomorphism type on
   $\partial\Delta^{4}$ and on a disjoint union of two copies.
4. State Schleimer's theorem as $3$-sphere recognition $\in\mathrm{NP}$,
   name the certificate, and leave the certificate uncomputed.
5. Point the $k\ge 4$ undecidability statements at notebook 05.
"""))

    cells.append(md(r"""
## A triangulation, stored as a facet list

A complex in this notebook is given by its tetrahedra. The canonical
store is the facet list: each facet is a $4$-element set of vertices,
and a simplex is determined by its vertices. The downward closure
fills in the triangles, edges, and vertices. Write
$$
(f_0,f_1,f_2,f_3) = (V, E, F, T)
$$
for those counts, and
$$
\chi
=
\sum_{d=0}^{3}(-1)^d f_d
=
V - E + F - T.
$$
`euler_characteristic` evaluates the same alternating sum from
`faces_by_dim()`. The facet list alone is the wrong store for $\chi$:
five tetrahedra have $T = 5$ and, on the model below, $\chi = 0$.

The model PL $3$-sphere is the boundary of the $4$-simplex, the case
$m = 3$ of the model sphere in notebook 02:
$$
\partial\Delta^{4}
=
\{\rho\subsetneq\{0,1,2,3,4\}\mid \rho\neq\emptyset\}.
$$
The library omits the empty face. The facets are the five tetrahedra
obtained by deleting one vertex of $\{0,1,2,3,4\}$. This complex is the
model, so its realization is $S^{3}$ for the same reason the
tetrahedron boundary is the model $S^{2}$ in notebooks 02 and 03. That
identification is the definition of the model. It is not a recognition
procedure.

Each tetrahedron has four triangular faces. When every triangle lies
in exactly two tetrahedra, those faces satisfy
$$
4T = 2F.
$$
The model meets the count: $T = 5$ and $F = 10$. The same count holds
for every closed combinatorial $3$-manifold. It also holds for
complexes that fail the vertex-link test, so the count is not the
manifold criterion. The next section is that criterion. The suspension
section is a complex on which $4T = 2F$ holds and a vertex link is a
torus.

**What.** The facet list of tetrahedra, and the $f$-vector of its
downward closure.

**Why.** $\chi$ is the alternating face count. The five facet names do
not display $V$, $E$, or $F$.

**How.** Build `SimplicialComplex` from the five tetrahedra. Read
`faces_by_dim()`. Compare $V - E + F - T$ with `euler_characteristic`.

**When.** The question is the triangulation, the $f$-vector, or
$\chi$. The link of a vertex is a further complex, built in the next
section. A path-component count uses the adjacency list, further
down. Half-edge and Quad-Edge walks stay in notebook 03: this notebook
does not circulate around a face.
"""))

    cells.append(code(r"""
# Facet list of the model PL 3-sphere, then the f-vector.
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

from lib import (
    SimplicialComplex,
    euler_characteristic,
    vertex_links_are_spheres_s2,
)


def from_facets(facets):
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen) if frozen else frozenset()
    return SimplicialComplex(vertices, frozen)


def simplex_boundary(dim):
    # Boundary of the standard dim-simplex on vertices 0 .. dim.
    vertices = list(range(dim + 1))
    facets = [tuple(v for v in vertices if v != omit) for omit in vertices]
    return from_facets(facets)


def face_vector(K):
    by_dim = K.faces_by_dim()
    return tuple(len(by_dim.get(d, ())) for d in range(K.dimension() + 1))


def triangle_degrees(K):
    # How many tetrahedra contain each triangle. Sums to 4T.
    triangles = K.faces_by_dim().get(2, ())
    return [sum(1 for tet in K.facets if tri <= tet) for tri in triangles]


def component_count(K):
    # Path-components of the realization, via the 1-skeleton.
    adj = K.adjacency_list()
    unseen = set(K.vertices)
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            vertex = stack.pop()
            for neighbour in adj[vertex]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    return count


model = simplex_boundary(4)
vertices, edges, faces, tetrahedra = face_vector(model)
print("f-vector (V, E, F, T)", (vertices, edges, faces, tetrahedra))
print("chi", euler_characteristic(model))
print("components", component_count(model))
print("triangle degrees", sorted(set(triangle_degrees(model))))

assert model.dimension() == 3 and model.is_pure(3)
assert (vertices, edges, faces, tetrahedra) == (5, 10, 10, 5)
assert vertices - edges + faces - tetrahedra == 0
assert euler_characteristic(model) == 0
assert 4 * tetrahedra == 2 * faces
assert set(triangle_degrees(model)) == {2}
assert sum(triangle_degrees(model)) == 4 * tetrahedra
assert component_count(model) == 1
"""))

    cells.append(md(r"""
## Vertex links are $2$-spheres

Specialise the criterion of notebook 02 at $k = 3$. A complex $K$ is a
closed combinatorial $3$-manifold when $K$ is pure of dimension $3$ and
every nonempty simplex $\sigma$ has PL-sphere link
$$
\operatorname{Lk}(\sigma, K)
\cong_{\mathrm{PL}}
\partial\Delta^{3-\dim\sigma}.
$$

| $\dim\sigma$ | Link required | Model, on $\partial\Delta^{4}$ |
|-------------:|---------------|--------------------------------|
| $0$ | $\partial\Delta^{3}$, a combinatorial $2$-sphere | tetrahedron boundary on the opposite four vertices |
| $1$ | a cycle of length $\ge 3$ | the opposite triangle's boundary, a $3$-cycle |
| $2$ | two points and no edge | the two vertices that complete a tetrahedron |
| $3$ | the empty complex | a facet has no coface |

`vertex_links_are_spheres_s2` iterates the vertices only. It returns
true when $K$ is pure of dimension $3$, nonempty, and every vertex link
is connected, passes `closed_surface_link_condition`, and has Euler
characteristic $2$. Notebook 03's classification makes that package a
combinatorial $2$-sphere: one component, circular vertex links, and
$\chi = 2$.

On a simplicial complex the rest of the table follows from the vertex
row. The link of an edge $\{u,v\}$ is the link of the vertex $v$ inside
$\operatorname{Lk}(\{u\}, K)$. Notebook 02 already records that a
combinatorial $2$-sphere has circular vertex links and two-point edge
links. The function does not take a second pass over edges or
triangles. The cell below checks one edge and one triangle of the
model directly, so the implication is visible on this complex.

The predicate answers the local question. Several complexes share the
answer true.

- The model $\partial\Delta^{4}$ is the PL $3$-sphere by the definition
  in notebook 02. The predicate also returns true. The definition is
  the reason the realization is $S^{3}$.
- A disjoint union of closed combinatorial $3$-manifolds still has
  spherical vertex links. Homeomorphism type is not a connectedness
  hypothesis of the link formula. The disjoint-union section is the
  example, and $\chi$ does not separate it from the model.
- A lens space $L(p,q)$ with $p\ge 2$ has fundamental group
  $\mathbb{Z}/p\mathbb{Z}$, so it is not homeomorphic to $S^{3}$. Any
  combinatorial triangulation of it has vertex links $S^{2}$ and, by
  the vanishing of $\chi$ below, Euler characteristic $0$. No such
  triangulation is entered in this notebook. The same sentence applies
  to $S^{1}\times S^{2}$, whose fundamental group is $\mathbb{Z}$.

The predicate answers false for a different list of reasons.

- Wrong dimension. The tetrahedron boundary is the model $S^{2}$ from
  notebook 03. It is pure of dimension $2$.
- Boundary. A single tetrahedron is a PL $3$-ball. The link of a
  vertex is the opposite triangle, a combinatorial disk, with
  $\chi = 1$. Every triangle of the ball lies in one tetrahedron, so
  the triangle link is a single point, a PL $0$-ball. Notebook 02
  already separated that ball from a sphere.
- A disconnected link whose Euler characteristic happens to be $2$.
  Two tetrahedra that meet only at a vertex,
  $$
  \mathcal{F}(K)
  =
  \bigl\{\{0,1,2,3\},\,\{0,4,5,6\}\bigr\},
  $$
  give the shared vertex a link of two disjoint triangles. That link
  has $\chi = 2$ and two path-components. A combinatorial $2$-sphere is
  one component. The predicate refuses the pinch. This is the
  three-dimensional form of the pinched triangles in notebook 02.

A true return is the local check on a simplicial complex. It is not
Schleimer's certificate. A face-pairing triangulation that is not a
simplicial complex is not an input the function can see. Those limits
are the recognition section.

**What.** The link of a simplex, as a complex on its own facet list.

**Why.** The manifold criterion is a statement about those links. The
parent facet list does not display them.

**How.** `K.link(σ)` deletes $\sigma$ from every simplex that properly
contains it. On a vertex, pass the resulting complex to the surface
checks from notebooks 02 and 03.

**When.** The question is the local combinatorial type. Homeomorphism
to $S^{3}$ is the recognition section. An adjacency list still cannot
see a triangle, so it cannot state the link.
"""))

    cells.append(code(r"""
# Links on the model, then three complexes the predicate refuses.
from lib import closed_surface_link_condition, links_are_circles

model = simplex_boundary(4)
vertex_link = model.link(frozenset({0}))
opposite = frozenset({1, 2, 3, 4})
print("vertex link f-vector", face_vector(vertex_link), "chi", euler_characteristic(vertex_link))
assert vertex_link.vertices == opposite
assert vertex_link.facets == frozenset(frozenset(opposite - {v}) for v in opposite)
assert closed_surface_link_condition(vertex_link)
assert euler_characteristic(vertex_link) == 2
assert vertex_links_are_spheres_s2(model) is True

edge_link = model.link(frozenset({0, 1}))
print("edge link facets", sorted(tuple(sorted(edge)) for edge in edge_link.facets))
assert edge_link.facets == frozenset(
    {frozenset({2, 3}), frozenset({2, 4}), frozenset({3, 4})}
)
assert links_are_circles(edge_link) is True

triangle_link = model.link(frozenset({0, 1, 2}))
print("triangle link vertices", sorted(triangle_link.vertices))
assert triangle_link.dimension() == 0
assert triangle_link.vertices == frozenset({3, 4})
assert triangle_link.facets == frozenset({frozenset({3}), frozenset({4})})

# Symmetry of the model: every vertex, edge, and triangle link.
vertices = model.faces_by_dim()[0]
edges = model.faces_by_dim()[1]
triangles = model.faces_by_dim()[2]
assert len(vertices) == 5 and len(edges) == 10 and len(triangles) == 10
for vertex in vertices:
    link = model.link(vertex)
    assert closed_surface_link_condition(link)
    assert euler_characteristic(link) == 2
    assert len(link.vertices) == 4
for edge in edges:
    assert links_are_circles(model.link(edge))
    assert len(model.link(edge).vertices) == 3
for triangle in triangles:
    link = model.link(triangle)
    assert link.dimension() == 0 and len(link.vertices) == 2

surface = simplex_boundary(3)
print("tetrahedron boundary: dim", surface.dimension(), "chi", euler_characteristic(surface))
assert surface.dimension() == 2
assert euler_characteristic(surface) == 2
assert vertex_links_are_spheres_s2(surface) is False

ball = from_facets([(0, 1, 2, 3)])
ball_link = ball.link(frozenset({0}))
print("ball f-vector", face_vector(ball), "chi", euler_characteristic(ball))
print("ball vertex link chi", euler_characteristic(ball_link))
assert face_vector(ball) == (4, 6, 4, 1)
assert euler_characteristic(ball) == 1
assert euler_characteristic(ball_link) == 1
assert set(triangle_degrees(ball)) == {1}
assert vertex_links_are_spheres_s2(ball) is False

pinch = from_facets([(0, 1, 2, 3), (0, 4, 5, 6)])
shared = pinch.link(frozenset({0}))
print(
    "pinch link of 0: f-vector",
    face_vector(shared),
    "chi",
    euler_characteristic(shared),
    "components",
    component_count(shared),
)
assert face_vector(pinch) == (7, 12, 8, 2)
assert euler_characteristic(pinch) == 1
assert component_count(pinch) == 1
assert euler_characteristic(shared) == 2
assert component_count(shared) == 2
assert vertex_links_are_spheres_s2(pinch) is False
"""))

    cells.append(md(r"""
## Every triangle in two tetrahedra

The relation $4T = 2F$ says that every triangle link is a $0$-sphere.
It does not say that every vertex link is a $2$-sphere. The suspension
of a closed surface is a complex that separates the two sentences.

Let $X$ be the $7$-vertex triangulation of the torus from notebook 03,
the cyclic complex generated by $(0,1,3)$ and $(0,2,3)$ modulo $7$.
That complex has $\chi(X) = 0$. Form the suspension by coning every
triangle of $X$ to a new apex $N = 7$ and to a new apex $S = 8$:
$$
\mathcal{F}(\Sigma X)
=
\bigl\{\tau\cup\{N\}\mid \tau\in\mathcal{F}(X)\bigr\}
\cup
\bigl\{\tau\cup\{S\}\mid \tau\in\mathcal{F}(X)\bigr\}.
$$
Each triangle of $X$ then lies in exactly two tetrahedra. Each new
triangle, spanned by an edge of $X$ and one apex, lies in two
tetrahedra as well, because every edge of the torus lies in two
triangles. So $4T = 2F$ holds on $\Sigma X$.

The link of the apex $N$ is $X$ itself. A combinatorial $2$-sphere has
Euler characteristic $2$. The torus has Euler characteristic $0$, so
the apex fails the vertex test and
`vertex_links_are_spheres_s2` returns false.

The face-count Euler characteristic of a suspension is
$$
\chi(\Sigma X) = 2 - \chi(X),
$$
the inclusion-exclusion count for two cones glued along $X$. Each cone
is contractible, so each contributes $1$, and the intersection
contributes $\chi(X)$. Here $\chi(\Sigma X) = 2$. A closed
$3$-manifold has Euler characteristic $0$, by the duality count in the
next section. The suspension is therefore not a closed $3$-manifold,
and the link of the apex is the witness the face count is summarising.

The $3\times 3$ grid torus of notebook 03 has the same $\chi$ and would
suspend the same way. The $7$-vertex complex is the shorter facet list.
"""))

    cells.append(code(r"""
# Suspension of the 7-vertex torus: triangle count holds, apex link does not.
def torus_seven():
    # Same cyclic triangulation as notebook 03.
    facets = []
    for i in range(7):
        facets.append((i, (i + 1) % 7, (i + 3) % 7))
        facets.append((i, (i + 2) % 7, (i + 3) % 7))
    return from_facets(facets)


def suspension(surface, north, south):
    facets = []
    for triangle in surface.facets:
        facets.append(tuple(triangle) + (north,))
        facets.append(tuple(triangle) + (south,))
    return from_facets(facets)


torus = torus_seven()
suspended = suspension(torus, 7, 8)
apex = suspended.link(frozenset({7}))
print("torus f-vector", face_vector(torus), "chi", euler_characteristic(torus))
print("suspension f-vector", face_vector(suspended), "chi", euler_characteristic(suspended))
print("apex link equals the torus", apex.facets == torus.facets)

assert face_vector(torus) == (7, 21, 14)
assert euler_characteristic(torus) == 0
assert closed_surface_link_condition(torus)
assert face_vector(suspended) == (9, 35, 56, 28)
assert euler_characteristic(suspended) == 2 - euler_characteristic(torus)
assert set(triangle_degrees(suspended)) == {2}
assert component_count(suspended) == 1
assert apex.facets == torus.facets
assert euler_characteristic(apex) == 0
assert vertex_links_are_spheres_s2(suspended) is False
"""))

    cells.append(md(r"""
## Homeomorphism type is a second question

Every closed $3$-manifold $M$ has Euler characteristic $0$. Poincaré
duality pairs Betti numbers by $b_k = b_{3-k}$: over any field when
$M$ is orientable, and over $\mathbb{F}_{2}$ with no orientation
hypothesis. The alternating sum collapses:
$$
\chi(M)
=
b_0 - b_1 + b_2 - b_3
=
0.
$$
The same vanishing holds for every closed manifold of odd dimension.
Notebook 03 could separate the sphere, the torus, and
$\mathbb{R}\mathrm{P}^{2}$ by $\chi$ because those surfaces are
even-dimensional and the values $2$, $0$, and $1$ all occur. On closed
$3$-manifolds the value is identically $0$.

This notebook does not compute Betti numbers. The face-count $\chi$ on
the examples is the check. Schleimer's polynomial-time test that a
triangulated $3$-manifold is a homology $3$-sphere, via the Smith
normal form of the boundary matrices, is a further algorithm and is
not run here.

The model and a disjoint union of two copies therefore look alike to
both of the numerical tests this notebook does run:

| Complex | $V$ | $E$ | $F$ | $T$ | $\chi$ | vertex links $S^{2}$ | components |
|---------|----:|----:|----:|----:|-------:|:---------------------:|-----------:|
| $\partial\Delta^{4}$ | 5 | 10 | 10 | 5 | 0 | yes | 1 |
| two copies of $\partial\Delta^{4}$ | 10 | 20 | 20 | 10 | 0 | yes | 2 |

The realization of a simplicial complex is path-connected exactly when
its $1$-skeleton is. The component count is a walk on
`adjacency_list()`, the same graph walk notebook 02 uses for cycles.
The model has one component. The union has two. The $3$-sphere is
connected, so the union is not homeomorphic to $S^{3}$. The predicate
returns true on both complexes. It is doing the local job. Connectedness
of $|K|$ is extra data, read off the adjacency list, and it still does
not name a lens space: a lens space triangulation is connected, has
$\chi = 0$, and has vertex links $S^{2}$.

Connected sum with $S^{3}$ does not change homeomorphism type:
$M\# S^{3}$ is homeomorphic to $M$. Schleimer records this as a
consequence of Alexander's theorem. The connected sum of two lens
spaces with nontrivial fundamental group is a closed $3$-manifold
whose combinatorial triangulations pass the vertex test and are not
homeomorphic to $S^{3}$. No connected sum is triangulated in this
notebook.

**What.** Path-components of the $1$-skeleton.

**Why.** The link predicate and $\chi$ both agree on the model and on
the disjoint union. The component count is the datum that separates
those two complexes.

**How.** One adjacency list, then a walk that deletes vertices as it
visits them. The same walk, restricted to a vertex link, is what
connectedness of that link means inside the sphere test.

**When.** The question is how many pieces $|K|$ has. The walk does not
read triangles, so it does not decide whether a link is $S^{2}$, and
it does not decide whether a connected complex is $S^{3}$.
"""))

    cells.append(code(r"""
# Disjoint union: the local predicate passes, and S^3 is connected.
def disjoint_union(left, right):
    shift = (max(left.vertices) + 1) if left.vertices else 0
    facets = set(left.facets)
    for face in right.facets:
        facets.add(frozenset(vertex + shift for vertex in face))
    vertices = frozenset().union(*facets) if facets else frozenset()
    return SimplicialComplex(vertices, frozenset(facets))


model = simplex_boundary(4)
pair = disjoint_union(model, model)
print("union f-vector", face_vector(pair), "chi", euler_characteristic(pair))
print("components", component_count(pair))
print("vertex links S^2", vertex_links_are_spheres_s2(pair))

assert face_vector(pair) == (10, 20, 20, 10)
assert euler_characteristic(pair) == 0
assert euler_characteristic(pair) == euler_characteristic(model)
assert set(triangle_degrees(pair)) == {2}
assert vertex_links_are_spheres_s2(pair) is True
assert component_count(model) == 1
assert component_count(pair) == 2
"""))

    cells.append(md(r"""
## The same $\chi$, read off $\partial_k$

`boundary_matrix(K, k)` is the sparse map $\partial_k\colon C_k\to C_{k-1}$
from notebook 01. Its number of columns is $f_k$. For a pure
$3$-complex
$$
\chi
=
\mathrm{ncols}(\partial_0)
-
\mathrm{ncols}(\partial_1)
+
\mathrm{ncols}(\partial_2)
-
\mathrm{ncols}(\partial_3).
$$
That is the face-count formula written in the shapes of the chain
maps. On the model and on the disjoint union the alternating column
count is $0$. The chain maps reproduce $\chi$. They do not acquire a
homeomorphism type by being written down.

Use the column count when a boundary matrix is already the object in
hand: incidence signs, or the identity $\partial_{k-1}\partial_k = 0$.
Homology ranks satisfy $\chi = b_0 - b_1 + b_2 - b_3$ as well, and
those ranks need $\mathrm{rank}(\partial)$, which is more work than the
cardinality of a face bucket and which this notebook does not compute.
When the question is only $\chi$, `euler_characteristic` is the direct
call.

**What.** Column counts of the sparse boundary matrices.

**Why.** They are $f_k$ again, so they reproduce $\chi$ without a
second census of the facet list.

**How.** Read `.shape[1]` for $k = 0, 1, 2, 3$ and alternate the signs.

**When.** The surrounding computation is already about chains. A
normal-surface coordinate vector, the object Schleimer's certificate
carries, is not one of these four matrices.
"""))

    cells.append(code(r"""
# Column counts of ∂_k reproduce χ. They stay 0 on both complexes.
from lib import boundary_matrix


def chi_from_columns(K):
    counts = [boundary_matrix(K, k).shape[1] for k in range(4)]
    chi = counts[0] - counts[1] + counts[2] - counts[3]
    return chi, counts


model = simplex_boundary(4)
pair = disjoint_union(model, model)
for name, K in (("model", model), ("union", pair)):
    chi, counts = chi_from_columns(K)
    print(name, "column counts", counts, "chi", chi)
    assert chi == euler_characteristic(K) == 0
"""))

    cells.append(md(r"""
## Which structure, for which question

| Question | Structure | Why this one |
|----------|-----------|----------------|
| the tetrahedra, $\chi$, the relation $4T = 2F$ | facet list, then `faces_by_dim` | $\chi$ is the alternating face count; $T$ is the number of facets |
| link of a vertex, an edge, or a triangle | `link`, a complex built from the closure | the criterion is a statement about that complex |
| circular links inside a vertex link | `closed_surface_link_condition`, `links_are_circles` | notebooks 02 and 03, applied to the link |
| path-components of $|K|$ | `adjacency_list` | the realization is connected exactly when the $1$-skeleton is |
| incidence signs, column counts, $\partial^{2} = 0$ | `boundary_matrix` | the map is linear on oriented simplices |
| an oriented walk on a surface, or a walk on its dual | half-edge, Quad-Edge | notebook 03; no cell here circulates a face |
| a certificate that $|K|\cong S^{3}$ | a normal or almost-normal surface vector | Schleimer's witness, described next and not stored |

The library's canonical store remains the facet list. A face-pairing
triangulation in Schleimer's sense, several model tetrahedra plus
permutations identifying their faces, can fail to be a simplicial
complex: one tetrahedron with paired faces is a triangulation of
$S^{3}$ in that encoding, and this library cannot store it, because a
simplex here is a set of vertices. The examples above are the
simplicial-complex case.
"""))

    cells.append(md(r"""
## $3$-sphere recognition lies in NP

The recognition problem takes a triangulation $T$ and asks whether the
underlying space $|T|$ is homeomorphic to $S^{3}$. Schleimer's input is
a face-pairing triangulation: $n$ labeled model tetrahedra, together
with isometries pairing their faces. He writes $\mathrm{size}(T) = n$.
The simplicial complexes above are legal inputs of that problem. The
problem also accepts triangulations this notebook cannot store.

Three results sit in a line. Each is an upper bound on the cost of
deciding the problem, as Schleimer records them in the 2011 article.

- Rubinstein (Haifa lectures, 1992; ICM, Zürich 1994, published 1995)
  gave an algorithm, using almost normal $2$-spheres. Thompson (Math.
  Res. Lett. 1 (1994), 613–630) proved a form of it with thin position.
  Schleimer's Theorem 1.1 places that algorithm in EXPTIME.
- Casson (MSRI lecture, 1997) crushes a triangulation along a normal
  $2$-sphere and reduces the space bound. Schleimer's Theorem 14.1
  places Casson's version in PSPACE.
- Schleimer's Theorem 15.1 places the problem in NP. The article was
  posted as arXiv:math/0407047 in 2004. In that preprint the same
  statement is Theorem 16.1. Ivanov (Geom. Dedicata 131 (2008), 1–26)
  gave another proof of membership in NP.

A decision problem lies in NP when there is a polynomial $q$ such that
every yes-instance $T$ has a certificate of length at most
$q(\mathrm{size}(T))$, and a deterministic verifier accepts the pair
(instance, certificate) in time bounded by a polynomial in
$\mathrm{size}(T)$. Membership is an upper bound: yes-instances have
short proofs. NP-hard is a different property. A problem is NP-hard
when every decision problem in NP polynomial-time reduces to it.
NP-complete means both. Schleimer's theorem is the membership
$$
\text{$3$-sphere recognition}\in\mathrm{NP}.
$$
It is not the statement "$3$-sphere recognition is NP-hard." The 2004
preprint asks whether the problem is NP-hard and leaves the question
unanswered. The 2011 article does not turn Theorem 15.1 into a
hardness theorem. The result Schleimer cites as NP-complete is a
different problem: $3$-manifold knot genus (Agol, Hass, and Thurston,
STOC 2002, 761–766; arXiv:math/0205057). Unknot recognition, by
contrast, is a membership result of the same shape as Theorem 15.1:
Hass, Lagarias, and Pippenger (J. ACM 46 (1999), 185–211) place it in
NP.

The certificate in the proof of Theorem 15.1 is a sequence of pairs
$(T_i, v(S_i))$, $i = 0, \ldots, n$.

1. $T_0$ is the input triangulation.
2. $S_i$ is a normal or almost-normal $2$-sphere in $|T_i|$, written as
   a vector $v(S_i)$ of integer coordinates, with weight at most
   $\exp(\mathrm{size}(T_i))$. The weight bound keeps the bit length of
   the coordinates polynomial in $\mathrm{size}(T_i)$.
3. If $S_i$ is normal and not vertex-linking, $T_{i+1}$ is obtained by
   crushing $T_i$ along $S_i$: cut along the sphere, cone the resulting
   boundary spheres to points, and collapse the non-tetrahedral cells.
4. If $S_i$ is almost normal, it normalizes to vertex-linking
   $2$-spheres in both directions, and $T_{i+1}$ is $T_i$ with the
   component containing $S_i$ deleted.
5. The last triangulation $T_n$ is empty.

A normal surface meets each tetrahedron in normal disks, triangles and
quadrilaterals whose boundaries cross distinct edges of the
tetrahedron. An almost-normal surface is normal except in one
tetrahedron, which contains one exceptional piece. The verifier checks
that $|T|$ is a $3$-manifold and a homology $3$-sphere, that each
$S_i$ is a connected $2$-sphere, and that each crushing or deletion
step matches the next triangulation. On a homology $3$-sphere, crushing
along a normal $2$-sphere preserves connected sum, up to $3$-sphere
summands, and the empty connected sum is $S^{3}$ by the convention in
the paper. That is why the sequence is allowed to end empty.

This notebook stops at that description. `lib/` has no normal
coordinate, no almost-normal piece, and no crushing move.
`vertex_links_are_spheres_s2` is not a verifier for the sequence.
Schleimer's Theorem 3.4 is a polynomial-time check that $|T|$ is a
$3$-manifold, separate from the certificate. Vertex links must be
spheres or disks. The link of an edge midpoint is a sphere, a disk, or
$\mathbb{R}\mathrm{P}^{2}$, and $\mathbb{R}\mathrm{P}^{2}$ fails the
manifold test. A simplicial complex, each simplex a set of vertices,
does not encode that last gluing. The function in `lib/checks.py` is
the tutorial-scale vertex half of the check, on complexes small enough
to list by hand. Theorem 3.4 is not Theorem 15.1.

Pairwise homeomorphism of closed $3$-manifolds is a broader decision
problem than recognition of the single target $S^{3}$. This notebook
does not assign the broader problem a complexity class. The
undecidability theorems start at dimension $4$. Notebook 05 records
them: homeomorphism of $4$-manifolds is undecidable (Markov-type), and
sphere recognition in dimensions $\ge 5$ is undecidable
(Novikov-type). An undecidable problem admits no algorithm. A
polynomial verifier for $S^{3}$ does not place an undecidable problem
in NP, and it does not make dimension $4$ NP-hard.

Notebook 02 names a gap that is neither Theorem 15.1 nor those
undecidability results: the double suspension of a homology sphere can
be homeomorphic to a sphere while the suspension points have
non-spherical links, so the complex fails the combinatorial criterion.
That complex is not built here. High-dimensional sphere recognition is
notebook 05.
"""))

    cells.append(md(r"""
## Summary

A triangulation here is a facet list of tetrahedra. The model
$\partial\Delta^{4}$ has $(V,E,F,T) = (5,10,10,5)$ and $\chi = 0$. It
is the PL $3$-sphere because it is the model sphere of notebook 02.
Every vertex link is the tetrahedron boundary, every edge link is a
$3$-cycle, and every triangle link is two points.
`vertex_links_are_spheres_s2` returns true. The tetrahedron boundary
itself, a single tetrahedron, and two tetrahedra meeting at one vertex
return false: wrong dimension, a disk link, and a two-component link
whose Euler characteristic is nevertheless $2$.

The suspension of the $7$-vertex torus satisfies $4T = 2F$ and has
$\chi = 2$. The link of an apex is the torus, so the vertex test fails.
Closed $3$-manifolds have $\chi = 0$ by Poincaré duality. That integer
classified surfaces in notebook 03 and does not classify $3$-manifolds.
Column counts of $\partial_k$ reproduce the same integer.

A disjoint union of two copies of the model also has $\chi = 0$ and
spherical vertex links, with two path-components of the adjacency
list. The $3$-sphere is connected. Lens spaces and $S^{1}\times S^{2}$
are connected complexes of the same local type and are not entered
here. The predicate is the local check. Homeomorphism type is the
other question.

Schleimer's theorem (arXiv:math/0407047, 2004; Proc. Sympos. Pure
Math. 82 (2011), Theorem 15.1) says that $3$-sphere recognition lies
in NP. The certificate is a sequence of triangulations and normal or
almost-normal $2$-sphere vectors, ending at the empty triangulation.
The theorem does not say that the problem is NP-hard; the 2004
preprint leaves hardness open. Knot genus is the NP-complete problem
Schleimer points to instead. No cell builds a surface vector or crushes
a triangulation. Notebook 05 takes up dimension $\ge 4$, where the
statements are undecidability of homeomorphism and of high-dimensional
sphere recognition.
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
