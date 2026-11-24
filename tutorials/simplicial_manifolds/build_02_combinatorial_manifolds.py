#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 02 (source of truth): combinatorial manifolds.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_02_combinatorial_manifolds.py

The generated notebook is ``02_combinatorial_manifolds.ipynb`` beside
this file. It states the link of a simplex and the closed
combinatorial-manifold criterion, separates that condition from the
PL and topological names, and specialises at $k=1$ to a walk on the
adjacency list. The interactive figure is ``link_demo.py``; the
notebook only launches it.

Notebook 01 owns the facet list, the star, purity, and the first
boundary matrix. This notebook uses those objects and does not rebuild
their sections. Surfaces and $V-E+F$ wait for notebook 03.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("02_combinatorial_manifolds.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Combinatorial manifolds

Notebook 01 builds a complex from a facet list, closes it downward,
and indexes the 1-skeleton by an adjacency list. The star of a
simplex $\sigma$ is the set of cofaces of $\sigma$. This notebook
deletes $\sigma$ from each proper coface. The result is the link.
A closed combinatorial manifold is a pure complex whose links are
piecewise-linear spheres of the complementary dimension.

At dimension $1$ that sentence becomes a graph fact, read from the
adjacency list: every degree is $2$, and each component is a cycle.
The figure at the end scrubs vertices and edges on a tetrahedron
boundary, on two triangles pinched at a vertex, and on a $5$-cycle.

## Learning objectives

1. Write $\operatorname{Lk}(\sigma, K)$ and the dimension formula
   $k - \dim\sigma - 1$.
2. State the closed combinatorial $k$-manifold criterion, and give
   separate meanings to combinatorial, PL, and topological.
3. Compare three vertex links in dimension $2$: a $3$-cycle, a single
   edge, and two disjoint edges.
4. At $k=1$, turn the criterion into degrees and a component walk on
   the adjacency list.
5. Launch the link scrubber and read one link off the figure.
"""))

    cells.append(md(r"""
## The link of a simplex

Let $\sigma$ be a simplex of a finite simplicial complex $K$. The
link of $\sigma$ is the set of simplices obtained by deleting $\sigma$
from every simplex that properly contains it:
$$
\operatorname{Lk}(\sigma, K)
=
\{\tau\setminus\sigma \mid \tau\in K,\ \sigma\subsetneq\tau\}.
$$
The same collection is the set of simplices disjoint from $\sigma$
whose union with $\sigma$ still lies in $K$:
$$
\operatorname{Lk}(\sigma, K)
=
\{\rho\in K \mid \rho\cap\sigma = \emptyset,\ \rho\cup\sigma\in K\}.
$$
Either formula includes the faces of those pieces, so the link is
itself a simplicial complex. `SimplicialComplex.link` stores it as a
fresh facet list. The parent complex is unchanged.

If $K$ is pure of dimension $k$ and $\sigma\in K$ has dimension $i$,
every piece $\tau\setminus\sigma$ coming from a facet has dimension
$k - i - 1$, and
$$
\dim\operatorname{Lk}(\sigma, K) = k - i - 1.
$$
The link of a facet is empty: nothing properly contains a facet. The
empty complex has dimension $-1$, and the formula returns
$k - k - 1 = -1$. If $\sigma$ is not a face of $K$, the link is empty
as well.

The code cell uses the boundary of a tetrahedron, the complex whose
facets are the four triangles on vertices $\{0,1,2,3\}$. Notebook 01
drew this complex; here only the links are printed. The link of the
vertex $\{0\}$ is the opposite $3$-cycle. The link of the edge
$\{0,1\}$ is two points.
"""))

    cells.append(code(r"""
# Setup, then two links on the tetrahedron boundary.
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
    closed_surface_link_condition,
    is_disjoint_union_of_cycles,
    links_are_circles,
    vertex_degrees,
)


def from_facets(facets):
    # Facet list in, complex out. Same constructor wrapper as notebook 01.
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen)
    return SimplicialComplex(vertices, frozen)


def simplex_boundary(dim):
    # Boundary of the standard dim-simplex on vertices 0 .. dim.
    vertices = list(range(dim + 1))
    facets = [[v for v in vertices if v != omit] for omit in vertices]
    return from_facets(facets)


def show_link(name, K, sigma):
    link = K.link(sigma)
    dim_sigma = len(sigma) - 1
    expected = K.dimension() - dim_sigma - 1
    facets = sorted(tuple(sorted(facet)) for facet in link.facets)
    print(f"--- {name}")
    print("sigma", tuple(sorted(sigma)), "dim", dim_sigma)
    print("link dimension", link.dimension(), "expected", expected)
    print("link facets", facets)


sphere = simplex_boundary(3)
show_link("tetrahedron boundary, vertex", sphere, frozenset({0}))
show_link("tetrahedron boundary, edge", sphere, frozenset({0, 1}))
"""))

    cells.append(md(r"""
## PL spheres of low dimension

A piecewise-linear homeomorphism is a homeomorphism that is linear on
the simplices of some subdivision of its domain and codomain. Write
$\cong_{\mathrm{PL}}$ for that relation. The boundary of the standard
simplex $\Delta^{m+1}$ is the model sphere of dimension $m$:
$$
\partial\Delta^{m+1}
=
\{\rho\subsetneq\{0,1,\ldots,m+1\}\mid \rho\neq\emptyset\}
$$
as an abstract complex (the library omits the empty face, and the
geometric boundary is the matching union of convex hulls). A complex
$L$ is a PL $m$-sphere when
$$
L \cong_{\mathrm{PL}} \partial\Delta^{m+1}.
$$
The dimension of $\partial\Delta^{m+1}$ is $m$.

Three dimensions are visible by listing the complexes.

| $m$ | Model | Complexes that satisfy the relation |
|-----|--------|--------------------------------------|
| $-1$ | $\partial\Delta^{0}$, the empty complex | the empty complex |
| $0$ | $\partial\Delta^{1}$, two points | two vertices and no edge |
| $1$ | $\partial\Delta^{2}$, a $3$-cycle | any cycle of length $\ge 3$ |

A simplicial complex has no loop and no digon, so a cycle begins at
length $3$. Any two cycles of length $\ge 3$ are PL-homeomorphic: a
subdivision inserts vertices until the polygons match. The $5$-cycle
below is a PL $1$-sphere for that reason. It is a different abstract
complex from $\partial\Delta^{2}$.

A PL $m$-ball is a complex PL-homeomorphic to the simplex $\Delta^{m}$.
A single point is a PL $0$-ball. A single edge is a PL $1$-ball. A
path of edges is a PL $1$-ball as well.

For $m\ge 2$ the relation $\cong_{\mathrm{PL}}\partial\Delta^{m+1}$ is
still the definition. Notebook 03 takes up combinatorial surfaces, and
notebooks 04 and 05 take up recognition in dimensions $3$ and higher.
"""))

    cells.append(md(r"""
## The criterion, and three names

$K$ is a closed combinatorial $k$-manifold when $K$ is pure of
dimension $k$ and every nonempty simplex has PL-sphere link of the
complementary dimension. In one formula, using
$\dim\partial\Delta^{k-\dim\sigma} = k - \dim\sigma - 1$,
$$
K\text{ is pure of dimension }k
\quad\text{and}\quad
\forall\,\sigma\in K,\ 
\operatorname{Lk}(\sigma, K)\cong_{\mathrm{PL}}\partial\Delta^{k-\dim\sigma}.
$$
Purity is the condition from notebook 01: every facet has dimension
$k$. The displayed formula then forces the link dimension
$k - \dim\sigma - 1$.

| $\dim\sigma$ | Dimension of the link | PL sphere, as used here |
|--------------|------------------------|-------------------------|
| $k$ | $-1$ | empty complex |
| $k-1$ | $0$ | two points, no edge |
| $k-2$ | $1$ | one cycle of length $\ge 3$ |
| $\le k-3$ | $\ge 2$ | $\cong_{\mathrm{PL}}\partial\Delta^{k-\dim\sigma}$; later notebooks |

**Combinatorial.** The formula is a property of the abstract complex
$K$. It is checked from the facet list, by building each link. At
$k=1$ the same check is a walk on the adjacency list, in the section
below.

**PL.** The geometric realization $|K|$ from notebook 01 inherits a
piecewise-linear atlas when the formula holds: on each geometric
simplex the chart is linear. A closed PL $k$-manifold is a topological
manifold equipped with such an atlas. Satisfying the link formula
makes $|K|$ a closed PL $k$-manifold, with that triangulation as a
PL atlas.

**Topological.** A topological $k$-manifold without boundary is a
Hausdorff, second-countable space in which every point has an open
neighbourhood homeomorphic to $\mathbb{R}^{k}$. The realization of a
combinatorial manifold is a topological manifold. Homeomorphism type
of $|K|$ is a coarser relation than $\cong_{\mathrm{PL}}$ of the
links. A triangulation of a topological sphere can fail the link
formula; the double suspension of a homology sphere is the classical
witness. Notebooks 04 and 05 own that distinction and the recognition
theorems. This notebook checks links on complexes small enough to
draw.

A combinatorial $k$-manifold with boundary uses the same dimension
count, and each link is either a PL sphere or a PL ball, with at
least one ball. The manifold boundary is the set of simplices whose
links are balls. The chain boundary $\partial_{k}$ of notebook 01 is
a different operator. One comparison is enough to see the balls: the
filled triangle is a combinatorial disk. The link of a vertex is the
opposite edge, a PL $1$-ball, and the link of an edge is a single
point, a PL $0$-ball.

At $k=2$ the vertex links carry the whole closed criterion.
`closed_surface_link_condition` requires purity of dimension $2$ and
a single cycle as the link of every vertex. A cycle has all degrees
$2$. In the link of a vertex $v$, the degree of a neighbour $w$ equals
the number of triangles that contain the edge $\{v,w\}$. Degree $2$
in every vertex link forces every edge into exactly two triangles, so
each edge link is a PL $0$-sphere. Facet links are empty by purity.
The tetrahedron boundary passes. The filled triangle fails, because
its vertex links are edges.
"""))

    cells.append(code(r"""
# Closed combinatorial surface, and a combinatorial disk.
sphere = simplex_boundary(3)
disk = from_facets([[0, 1, 2]])

show_link("sphere vertex", sphere, frozenset({0}))
show_link("sphere edge", sphere, frozenset({0, 1}))
print(
    "closed_surface_link_condition",
    closed_surface_link_condition(sphere),
)

show_link("disk vertex", disk, frozenset({0}))
show_link("disk edge", disk, frozenset({0, 1}))
print(
    "closed_surface_link_condition",
    closed_surface_link_condition(disk),
)
"""))

    cells.append(md(r"""
## Pinched complexes

Both examples in this section are pure of dimension $2$. Both fail
the link formula. They fail it for different links.

Two triangles that meet only at a vertex,
$$
\mathcal{F}(K)=\bigl\{\{0,1,2\},\,\{0,3,4\}\bigr\},
$$
give the link
$$
\operatorname{Lk}(\{0\}, K)
=
\bigl\{\{1\},\{2\},\{1,2\},\{3\},\{4\},\{3,4\}\bigr\}.
$$
The facets of that link are the two edges $\{1,2\}$ and $\{3,4\}$. A
PL $1$-sphere is one cycle. Two components are two edges. The
realization is two disks whose intersection is a point, so the image
of vertex $0$ has no neighbourhood homeomorphic to $\mathbb{R}^{2}$.
This complex fails the topological condition and the combinatorial
condition together. It is the picture in the scrubber.

The link of an outer vertex, for example $\{1\}$, is the single edge
$\{0,2\}$. That is the PL $1$-ball already seen on the disk. The
pinched vertex is the one whose link is disconnected.

Three triangles on one edge,
$$
\mathcal{F}(K)
=
\bigl\{\{0,1,2\},\,\{0,1,3\},\,\{0,1,4\}\bigr\},
$$
fail at the shared edge first. The link of $\{0,1\}$ is three points,
one from each triangle. A PL $0$-sphere has two points. The link of
the endpoint $\{0\}$ is a tripod: the three edges $\{1,2\}$,
$\{1,3\}$, $\{1,4\}$, joined at vertex $1$. That graph is connected,
and vertex $1$ has degree $3$, so it is not a cycle. Connectedness of
the link is a weaker condition than the link being a sphere.

Parent degree does not sort these examples. In the pinch,
$\deg(0)=4$. A vertex of a $4$-cycle link also has degree $4$, and
that vertex can be a manifold point. The edges of the link decide.
Those edges are
$$
E\bigl(\operatorname{Lk}(\{v\},K)\bigr)
=
\bigl\{\{a,b\}\in K^{(1)} \mid \{v,a,b\}\in K\bigr\},
$$
which the parent's adjacency row does not list. The row
$\operatorname{Adj}(v)$ lists the vertices of the link. The cycle
test, when it runs, walks the adjacency list of the link.
"""))

    cells.append(code(r"""
# Two pure 2-complexes whose links are not spheres.
pinch = from_facets([[0, 1, 2], [0, 3, 4]])
branch = from_facets([[0, 1, 2], [0, 1, 3], [0, 1, 4]])

show_link("pinch vertex 0", pinch, frozenset({0}))
show_link("pinch vertex 1", pinch, frozenset({1}))
print("pinch Adj(0)", sorted(pinch.adjacency_list()[0]))
print(
    "closed_surface_link_condition",
    closed_surface_link_condition(pinch),
)

show_link("three triangles, shared edge", branch, frozenset({0, 1}))
show_link("three triangles, vertex 0", branch, frozenset({0}))
print(
    "closed_surface_link_condition",
    closed_surface_link_condition(branch),
)
"""))

    cells.append(md(r"""
## Dimension $1$: cycles, via the adjacency list

Restrict the criterion to $k=1$. Purity means every facet is an edge.
The only cofaces of a vertex $\{v\}$ are the edges that contain $v$,
and deleting $v$ from the edge $\{v,w\}$ leaves the point $\{w\}$.
There is no triangle, so the link contains no edge:
$$
\operatorname{Lk}(\{v\}, K)
=
\bigl\{\{w\} \mid w\in\operatorname{Adj}(v)\bigr\},
\qquad
\deg(v) = \#\operatorname{Adj}(v).
$$
The link is a PL $0$-sphere exactly when that row has two entries:
$$
\operatorname{Lk}(\{v\}, K)\cong_{\mathrm{PL}}\partial\Delta^{1}
\iff
\deg(v) = 2.
$$
The link of an edge is empty, because a pure $1$-complex has no
$2$-simplex. Empty is $\partial\Delta^{0}$, the PL sphere of dimension
$-1$. Purity has already made every edge link a sphere. The vertex
rows are the remaining test.

A finite undirected graph in which every degree equals $2$ is a
disjoint union of cycles. A simplicial cycle has length at least $3$.
Therefore
$$
\begin{aligned}
&K\text{ is a closed combinatorial }1\text{-manifold}\\
&\iff
K\text{ is pure of dimension }1\text{ and }\deg(v)=2\text{ for every }v\\
&\iff
K\text{ is a disjoint union of cycles of length }\ge 3.
\end{aligned}
$$
Connectedness is not required. Two disjoint cycles are a closed
$1$-manifold with two components. The link of each vertex is still one
$0$-sphere. The criterion constrains links, and it allows $K$ itself
to be disconnected. The same reading holds for surfaces:
`closed_surface_link_condition` allows a disjoint union, because each
vertex link is one circle.

**What.** `adjacency_list()` is the dictionary
$v\mapsto\operatorname{Adj}(v)$ built once from $K^{(1)}$.
`vertex_degrees` is the dictionary of row sizes.
`is_disjoint_union_of_cycles` is the criterion in this graph form:
pure of dimension $1$, every row of size $2$, then one walk per
component, rejecting a component shorter than $3$.
`links_are_circles` is the same criterion in link form: that graph
test, plus the explicit check that each vertex link is two points and
no edge. On a pure $1$-complex the two predicates agree, because a
link edge would be a triangle.

**Why an adjacency list.** Degree is the cardinality of one row. The
walk's next vertex is the other member of the current row, the
neighbour that differs from the vertex just left. One build answers
both questions, in time proportional to the number of vertices plus
the number of edges. Notebook 01 counted those neighbour slots,
$2\,\#K^{(1)}$, against the $V^{2}$ entries of a dense matrix. The
signed matrix $\partial_{1}$ from that notebook records an orientation
on each end of an edge. This walk reads an unordered neighbour.

**How.** The code cell follows the library walk. Build `Adj`. If any
row has size other than $2$, stop. Otherwise start at the least
unused vertex, step to `min(Adj(curr) \ {prev})`, and count vertices
until the walk returns. Each successful return contributes one cycle
length. `is_disjoint_union_of_cycles` is that procedure.

**When.** Use the parent's adjacency list for this $k=1$ test, and
for the equality between $\operatorname{Adj}(v)$ and the vertex set
of $\operatorname{Lk}(\{v\},K)$ in every dimension. Use the link's
own facet list, and then the link's own adjacency list, when the link
has edges. That is the pinched vertex and the tripod above, and it is
the $3$-cycle link of a tetrahedron vertex: the parent's row lists
$\{1,2,3\}$, and the link's rows list the opposite cycle. A degree in
the parent complex does not see those opposite edges.
"""))

    cells.append(code(r"""
# k=1: degrees and a walk on Adj. The predicates match on these four.
# The tetrahedron's vertex link is where the walk runs on the link.


def cycle_lengths(K):
    # One adjacency list, then a walk. None means "not a degree-2 pure 1-complex".
    if K.dimension() != 1 or not K.is_pure(1):
        return None
    adj = K.adjacency_list()
    if not adj or any(len(neighbours) != 2 for neighbours in adj.values()):
        return None
    unseen = set(adj)
    lengths = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        prev = start
        curr = min(adj[start])
        walked = 1
        while curr != start:
            if curr not in unseen:
                return None
            unseen.remove(curr)
            walked += 1
            onward = adj[curr] - {prev}
            if len(onward) != 1:
                return None
            prev, curr = curr, min(onward)
        if walked < 3 or prev not in adj[start]:
            return None
        lengths.append(walked)
    return lengths


def report_one_complex(name, K):
    adj = K.adjacency_list()
    print(f"--- {name}")
    print("dimension", K.dimension(), "pure of dim 1", K.is_pure(1))
    print("degrees", vertex_degrees(K))
    print("Adj", {v: sorted(neighbours) for v, neighbours in adj.items()})
    print("cycle_lengths", cycle_lengths(K))
    print("is_disjoint_union_of_cycles", is_disjoint_union_of_cycles(K))
    print("links_are_circles", links_are_circles(K))


cycle = from_facets([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
path = from_facets([[0, 1], [1, 2]])
union = from_facets(
    [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [6, 7], [7, 5]]
)
filled = from_facets([[0, 1, 2]])

report_one_complex("C5", cycle)
link0 = cycle.link(frozenset({0}))
print("C5 link of 0", sorted(link0.vertices), "dim", link0.dimension())
assert set(link0.vertices) == set(cycle.adjacency_list()[0])

report_one_complex("path of length 2", path)
print(
    "path link of endpoint 0",
    sorted(path.link(frozenset({0})).vertices),
)
report_one_complex("C5 disjoint from C3", union)
report_one_complex("filled triangle", filled)

assert cycle_lengths(cycle) == [5]
assert cycle_lengths(union) == [5, 3]
assert cycle_lengths(path) is None
assert cycle_lengths(filled) is None
assert links_are_circles(cycle) and links_are_circles(union)
assert not links_are_circles(path) and not links_are_circles(filled)

sphere = simplex_boundary(3)
vertex_link = sphere.link(frozenset({0}))
print("--- tetrahedron, link of vertex 0, walked on the link")
print("parent Adj(0)", sorted(sphere.adjacency_list()[0]))
print(
    "link Adj",
    {v: sorted(n) for v, n in vertex_link.adjacency_list().items()},
)
print("cycle_lengths", cycle_lengths(vertex_link))
assert set(vertex_link.vertices) == set(sphere.adjacency_list()[0])
assert cycle_lengths(vertex_link) == [3]

pinch = from_facets([[0, 1, 2], [0, 3, 4]])
pinch_link = pinch.link(frozenset({0}))
print("--- pinch, link of vertex 0, degrees from the link's Adj")
print("parent Adj(0)", sorted(pinch.adjacency_list()[0]))
print("link degrees", vertex_degrees(pinch_link))
print("cycle_lengths", cycle_lengths(pinch_link))
assert cycle_lengths(pinch_link) is None
"""))

    cells.append(md(r"""
## A scrubber for the link

The figure shows one geometric realization of three complexes from
the cells above.

- The tetrahedron boundary starts on vertex $\{0\}$. The orange
  $3$-cycle is the opposite face's boundary, and it is the link. The
  opposite triangle is a facet of $K$; its interior is not a simplex
  of the link. Scrub to the edge $\{0,1\}$ and the link is the two
  points $\{2\}$ and $\{3\}$.
- The pinched complex starts on the shared vertex. The orange set is
  two disjoint edges. Scrub to an outer vertex and the link is a
  single edge, a PL $1$-ball.
- The $5$-cycle starts on a vertex. The link is two points and no
  edge, so the orange marks are two vertices of the pentagon, and the
  two cycle edges that touch the chosen vertex stay gray. Those edges
  are cofaces. Scrub to an edge and the link panel is empty: the PL
  sphere of dimension $-1$.

The slider walks vertices, then edges. Elev and azim apply to the
tetrahedron only. Drawing and widgets live in `link_demo.py`, in this
folder. Read and edit that file. The next cell only checks that the
file is beside the notebook, imports `launch_link_demo`, and calls it.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

The interactive UI lives in
<code>link_demo.py</code> next to this notebook, so this cell stays a
launch stub. JupyterLab collapses the launch input
(<code>source_hidden</code>). Open the <code>.py</code> in the editor
to change coordinates, colours, or the scrubber.

</details>
"""))

    cells.append(code(r"""
# Launch only — implementation: link_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "link_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "link_demo.py").exists():
    raise FileNotFoundError("link_demo.py missing next to this notebook")
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from link_demo import launch_link_demo

launch_link_demo()
""", hide_input=True))

    cells.append(md(r"""
## Summary

The link of $\sigma$ is obtained by deleting $\sigma$ from every
simplex that properly contains it. When $K$ is pure of dimension $k$,
$$
\dim\operatorname{Lk}(\sigma, K) = k - \dim\sigma - 1.
$$
$K$ is a closed combinatorial $k$-manifold when it is pure of
dimension $k$ and
$$
\operatorname{Lk}(\sigma, K)\cong_{\mathrm{PL}}\partial\Delta^{k-\dim\sigma}
$$
for every nonempty simplex $\sigma$. That condition puts a PL atlas on
the realization $|K|$, which is then a topological manifold.
Homeomorphism type of a space can hold for a triangulation that fails
the link formula; notebooks 04 and 05 take up recognition.

On the tetrahedron boundary, vertex links are $3$-cycles and edge
links are two points, so `closed_surface_link_condition` is true. A
filled triangle has vertex links that are single edges: PL balls, the
manifold-boundary pattern. Two triangles meeting at a vertex have a
disconnected link there. Three triangles on one edge have a three-point
link on that edge.

At $k=1$ the link of a vertex is the adjacency-list row, with no edges,
and the link of every edge is empty. The criterion becomes: pure of
dimension $1$, and every degree equal to $2$. The walk that checks the
cycles reads those rows. `is_disjoint_union_of_cycles` and
`links_are_circles` agree on that class. When a link has edges, as it
does for a vertex in a surface, the walk uses the adjacency list of
the link, built after the facet list has produced the link.

Notebook 03 specialises the same criterion at $k=2$, writes
$V-E+F$, and compares the sphere, the torus, and
$\mathbb{R}\mathrm{P}^{2}$. It also compares surface mesh encodings
— **half-edge / DCEL** versus **Quad-Edge** (Guibas–Stolfi) — and
picks by the question (oriented walk versus primal+dual edge algebra).
Those structures are the wrong tool for the $k=1$ degree walk above;
adjacency lists remain the fit for that job.
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
