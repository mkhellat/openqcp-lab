#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 01 (source of truth): simplicial complexes.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_01_simplicial_complexes.py

The generated notebook is ``01_simplicial_complexes.ipynb`` beside this
file. It introduces the facet list, the adjacency list, and a first
look at the sparse boundary, and it stops before the
combinatorial-manifold criterion (notebook 02). The interactive figure
is ``complex_draw_demo.py``; the notebook only launches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("01_simplicial_complexes.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Simplicial complexes

A finite simplicial complex is a finite vertex set together with a
collection of simplices closed under taking faces. This notebook
builds that collection from a **facet list**, reads the **full set of
faces**, indexes the 1-skeleton as an **adjacency list**, and writes
signed face incidence as a sparse boundary $\partial_k$. It then
computes stars, tests purity, and draws one geometric realization.

Notebook 02 asks when the realization is a combinatorial manifold.
This notebook stops at the complex.

The code cells run from the top. The first one puts `lib/` on the
path and defines `from_facets`.

## Learning objectives

1. Separate an abstract simplex from a geometric simplex, and locate
   the centroid by barycentric coordinates.
2. Recover the face set from a facet list, and read the $f$-vector.
3. Choose the facet list, the closure, the adjacency list, or
   $\partial_k$ by the question being asked, including the sparsity
   count $2\,\#K^{(1)}$ against $V^2$.
4. On the filled triangle, read the matrix of $\partial_1$ and check
   $\partial_1\circ\partial_2=0$.
5. Compute a star, and test whether every facet has the same dimension.
"""))

    cells.append(md(r"""
## Abstract simplices and geometric simplices

An abstract simplex of dimension $n$ is a finite set $\sigma$ with
$$
\#\sigma = n+1, \qquad \dim\sigma = \#\sigma - 1.
$$
A face of $\sigma$ is any subset $\tau\subseteq\sigma$. Counting the
empty face, $\sigma$ has $2^{\#\sigma}$ faces. The library omits the
empty face, so the stored nonempty faces of one simplex number
$2^{\#\sigma}-1$.

A geometric $n$-simplex on affinely independent points
$v_0,\ldots,v_n\in\mathbb{R}^d$ is the convex hull
$$
\operatorname{conv}\{v_0,\ldots,v_n\}
=
\Bigl\{
  \sum_{i=0}^{n} t_i v_i
  \;\Big|\;
  t_i\ge 0,\quad
  \sum_{i=0}^{n} t_i = 1
\Bigr\}.
$$
The numbers $t_i$ are the barycentric coordinates of the point. The
centroid is the point with $t_i = 1/(n+1)$ for every $i$.

`SimplicialComplex` stores the abstract set. Coordinates, when a
picture needs them, live in the drawing demo rather than in `lib/`.
The symbol $|K|$ is reserved below for the realization of a whole
complex. A single geometric simplex is written as a convex hull, so
the set $\sigma$ and the point set in $\mathbb{R}^d$ stay distinct.
"""))

    cells.append(code(r"""
# Setup. Abstract 2-simplex from a one-element facet list, then one
# point of a geometric realization (the centroid).
from pathlib import Path
import sys

import numpy as np

_here = Path.cwd()
if not (_here / "lib").is_dir():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "lib").is_dir():
    raise FileNotFoundError("lib/ missing next to this notebook")
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from lib import SimplicialComplex


def from_facets(facets):
    # Facet list in, complex out. Vertices are the union of the facets.
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen)
    return SimplicialComplex(vertices, frozen)


triangle = from_facets([[0, 1, 2]])
print("abstract vertices", sorted(triangle.vertices))
print("dimension", triangle.dimension())

positions = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
    ]
)
barycentric = np.full(3, 1.0 / 3.0)
centroid = barycentric @ positions
print("barycentric", barycentric)
print("sum", float(barycentric.sum()), "min", float(barycentric.min()))
print("centroid", np.round(centroid, 6))
"""))

    cells.append(md(r"""
## Faces and dimension

A finite collection $K$ of nonempty finite sets is an abstract
simplicial complex when it is downward closed: $\sigma\in K$ and
$\emptyset\neq\tau\subseteq\sigma$ imply $\tau\in K$. The vertices are
the one-element members of $K$.

The simplices of dimension $k$, and the $f$-vector, are
$$
K^{(k)} = \{\sigma\in K\mid \dim\sigma = k\},
\qquad
f_k = \#K^{(k)},
\qquad
f(K) = (f_0,\ldots,f_{\dim K}).
$$
The dimension of the complex is the largest simplex dimension,
$$
\dim K = \max_{\sigma\in K}\dim\sigma,
$$
or $-1$ when $K$ has no simplices. For a nonempty complex the same
maximum is attained on a facet.

A filled triangle has $f(K)=(3,3,1)$. The one nonempty 2-simplex
contributes $2^{3}-1=7$ nonempty faces, which is $f_0+f_1+f_2$.
"""))

    cells.append(code(r"""
# Full simplex set of the filled triangle, grouped by dimension.
triangle = from_facets([[0, 1, 2]])
by_dim = triangle.faces_by_dim()
f_vector = tuple(len(by_dim[d]) for d in range(triangle.dimension() + 1))
print("f-vector", f_vector)
print("nonempty faces", len(triangle.all_simplices()))
for dim in range(triangle.dimension() + 1):
    faces = sorted(tuple(sorted(face)) for face in by_dim[dim])
    print(f"dim {dim}", faces)
"""))

    cells.append(md(r"""
## Facet list

A facet is an inclusion-maximal simplex. The facet list is
$$
\mathcal{F}(K)
=
\{\sigma\in K\mid \tau\in K\text{ and }\sigma\subseteq\tau \implies \tau=\sigma\}.
$$
Downward closure rebuilds every nonempty face from that list:
$$
K
=
\{\tau\neq\emptyset\mid \tau\subseteq\sigma\text{ for some }\sigma\in\mathcal{F}(K)\}.
$$

**What.** $\mathcal{F}(K)$ is the canonical store.
`SimplicialComplex` keeps that frozenset, and a vertex set equal to
the union of the facets. An isolated vertex is a singleton facet, so
the union still sees it.

**Why.** A triangulation is given by its top-dimensional simplices.
One triangle is a one-element facet list; the closure formula produces
the three edges and the three vertices. Storing those seven sets up
front repeats information the formula already fixes.

**How.** Pass the facet list to the constructor, as `from_facets`
does. `all_simplices` and `faces_by_dim` compute the closure on first
use and cache it on the instance. The cache stays off to the side:
`facets` remains the maximal simplices.

**When.** Use the facet list to build $K$ and to read the maximal
simplices. Purity, below, is a statement about $\mathcal{F}(K)$. Use
the closure for an $f$-vector or a star. Neighbour queries on the
1-skeleton use the adjacency list in the next section.
"""))

    cells.append(code(r"""
# Facet list versus the cached closure. The store is one triangle.
triangle = from_facets([[0, 1, 2]])
facets = sorted(tuple(sorted(facet)) for facet in triangle.facets)
closure = sorted(tuple(sorted(face)) for face in triangle.all_simplices())
print("facet list", facets)
print("closure   ", closure)
print("stored facets", len(facets), "nonempty faces", len(closure))
"""))

    cells.append(md(r"""
## Adjacency list of the 1-skeleton

The 1-skeleton is the undirected graph $\bigl(V,\, K^{(1)}\bigr)$.
Its adjacency list and the degree are
$$
\operatorname{Adj}(v) = \{w\in V\mid \{v,w\}\in K^{(1)}\},
\qquad
\deg(v) = \#\operatorname{Adj}(v).
$$
A dense adjacency matrix is the $V\times V$ array with a one in
column $w$ of row $v$ precisely when $w\in\operatorname{Adj}(v)$.
That array holds $V^2$ entries. The list holds
$$
\sum_{v\in V}\deg(v) = 2\,\#K^{(1)}
$$
neighbour slots (each edge contributes to two rows).

**What.** `adjacency_list()` returns a `dict` from each vertex to the
frozenset of its neighbours. An isolated vertex is present, mapped to
the empty frozenset, because it is a singleton facet.

**Why.** Edge incidence on these complexes is sparse: a mesh degree
stays a small integer while $V$ grows, so most matrix entries would
record an absent edge. The list stores the edges that exist. The book
of two triangles is only mildly sparse ($10$ slots against $4^2=16$).
The cycle $C_{12}$ is the count that shows the gap: $24$ slots against
$12^2=144$.

**How.** Close the facet list once, read $K^{(1)}$, and for each edge
$\{a,b\}$ insert $b$ into $\operatorname{Adj}(a)$ and $a$ into
$\operatorname{Adj}(b)$. After that pass a degree is a set cardinality.
`adjacency_list` does this walk; the code cell repeats it and checks
that the two dictionaries agree. Building the list once is the
indexing step. A later neighbour query reads the stored row.

**When.** Use the adjacency list for the neighbours of one vertex, for
$\deg(v)$, and for a walk along edges. Prefer it to the dense matrix
for those graph-local questions. The rows are unsigned sets. A sign
on an edge, or a face of dimension greater than $1$, is the boundary
in the next section.

Use the facet list when the question is which maximal simplices exist.
On a pure 1-complex the facets already are the edges, and the
adjacency list is that same edge set indexed by vertex. On a complex
with triangles the two stores diverge: $\mathcal{F}(K)$ records
triangles, and $\operatorname{Adj}$ records only the graph. Triangles
are read from the facet list or from `all_simplices`.

| Question | Store |
|----------|--------|
| Build $K$; list the maximal simplices; test purity | Facet list $\mathcal{F}(K)$ |
| Count $f_k$; compute a star | Full simplex set, the closure of $\mathcal{F}(K)$ |
| Neighbours, degree, a walk on edges | Adjacency list of $K^{(1)}$ |
"""))

    cells.append(code(r"""
# Book: facet list holds triangles. Cycle: facet list holds edges.
# Both print Adj and the slot count against V*V. The one-pass helper
# is the indexing step inside adjacency_list.


def adjacency_by_one_pass(K):
    neighbours = {v: set() for v in K.vertices}
    for edge in K.faces_by_dim().get(1, ()):
        a, b = tuple(edge)
        neighbours[a].add(b)
        neighbours[b].add(a)
    return {v: frozenset(neighbours[v]) for v in sorted(neighbours)}


def store_line(name, K):
    adj = K.adjacency_list()
    n = len(K.vertices)
    slots = sum(len(nbrs) for nbrs in adj.values())
    facets = sorted(tuple(sorted(facet)) for facet in K.facets)
    f_vector = tuple(
        len(K.faces_by_dim()[d]) for d in range(K.dimension() + 1)
    )
    print(f"--- {name}")
    print("facet list", facets)
    print("f-vector", f_vector)
    print("Adj", {v: sorted(nbrs) for v, nbrs in adj.items()})
    print(f"neighbour slots {slots}; dense entries {n * n}")


book = from_facets([[0, 1, 2], [0, 1, 3]])
cycle = from_facets([[i, (i + 1) % 12] for i in range(12)])
store_line("book", book)
store_line("cycle C_12", cycle)
assert adjacency_by_one_pass(book) == book.adjacency_list()
assert adjacency_by_one_pass(cycle) == cycle.adjacency_list()
print("one pass matches adjacency_list()")
"""))

    cells.append(md(r"""
## Incidence and the boundary $\partial_k$

A $k$-chain is an integer linear combination of oriented
$k$-simplices. The chain group $C_k$ is the free abelian group on
that basis,
$$
C_k = \bigoplus_{\sigma\in K^{(k)}} \mathbb{Z}\,\sigma.
$$
The boundary is the homomorphism determined on one positively
oriented simplex by deleting one vertex at a time. Write
$(v_0,\ldots,v_k)$ for the increasing order $v_0<\cdots<v_k$, and
write a hat for the omitted vertex. Then
$$
\partial_k\colon C_k\to C_{k-1},
\qquad
\partial_k(v_0,\ldots,v_k)
=
\sum_{i=0}^{k}(-1)^i\,(v_0,\ldots,\hat v_i,\ldots,v_k).
$$
The sign of the face that remains is $(-1)^i$, with $i$ the position
of the deleted vertex in that increasing tuple.

On an edge the sum is two terms,
$$
\partial_1(a,b)=(b)-(a), \qquad a<b.
$$
On the filled triangle the only $2$-simplex is $(0,1,2)$, and
$$
\partial_2(0,1,2)=(1,2)-(0,2)+(0,1).
$$
Apply $\partial_1$ to that chain. Vertex $0$ is reached by $-(0)$ from
the edge $(0,1)$ and by $+(0)$ from $-(0,2)$; vertices $1$ and $2$
cancel the same way. Each vertex appears once with $+1$ and once with
$-1$. The identity
$$
\partial_{k-1}\circ\partial_k=0
$$
is that cancellation, written for every $k$-simplex at once.

The empty simplex is not a cell of $K$. Bases therefore start at
dimension $0$, and the first map this section computes is $\partial_1$.

**What.** `oriented_simplices(K, dim)` lists the increasing vertex-id
tuples of dimension `dim`, in lexicographic order. That list is the
ordered basis of $C_{\mathrm{dim}}$. `boundary_matrix(K, k)` is the
matrix of $\partial_k$ on these bases: one column per $k$-simplex,
one row per $(k-1)$-simplex, entries in $\{-1,0,+1\}$. The stored
operator is sparse: a SciPy CSR matrix when SciPy imports, otherwise
a dictionary of the nonzero keys. `toarray` expands a copy for a
complex small enough to read by hand; the stored map stays sparse.
`boundary_square_is_zero(K, k)` reports whether
$\partial_{k-1}\partial_k$ is the zero matrix.

**Why.** The adjacency list answers which vertices share an edge. Its
rows are unordered sets, so $\{0,1\}$ is the same object from either
end, and a triangle never appears. A chain needs the opposite: an
order, a sign, and a face in every dimension that the alternating
sum names. The two ways around $(0,1,2)$ cancel in
$\partial_1\partial_2$. A neighbour set cannot record that
cancellation. Each column of $\partial_k$ has $k+1$ nonzeros, one per
codimension-one face, so absent incidences are not stored. The
product of two such maps is a sparse multiply. That product is the
calculation $\partial^2=0$.

**How.** Rebuild the filled triangle from the one-element facet list.
`oriented_simplices(K, 1)` prints the edges $(0,1)$, $(0,2)$, $(1,2)$.
`boundary_matrix(K, 1)` has rows $(0,)$, $(1,)$, $(2,)$ and those
edges as columns,
$$
\partial_1
=
\begin{pmatrix}
-1 & -1 & 0 \\
1 & 0 & -1 \\
0 & 1 & 1
\end{pmatrix}.
$$
Column $(0,1)$ is $(1)-(0)$. `boundary_square_is_zero(K, 2)`
multiplies this matrix by the column $\partial_2(0,1,2)=(1,-1,1)$
in the same edge order and checks that every entry is $0$.

The basis tuples stay in increasing vertex-id order. The cell does
not reorder a simplex so that its faces match one chosen orientation
of the whole complex. The relative signs live only in the matrix
entries.

**When.** Use $\partial_k$ when the question is linear algebra on
chains: the sign of a face in a boundary, the identity
$\partial_{k-1}\circ\partial_k=0$, and later a matrix rank set beside
an $f$-vector. Use the adjacency list when the question is local on
the graph and needs no signs: $\deg(v)$, a walk, the neighbours of
one vertex. Use the facet list to name the maximal simplices. The
closure is what both the adjacency list and $\partial_k$ read their
faces from. A walk along edges should not scan columns of
$\partial_1$. A signed boundary should not be read out of
$\operatorname{Adj}$.

| Question | Store |
|----------|--------|
| Build $K$; list the maximal simplices; test purity | Facet list $\mathcal{F}(K)$ |
| Count $f_k$; compute a star | Full simplex set, the closure of $\mathcal{F}(K)$ |
| Neighbours, degree, a walk on edges | Adjacency list of $K^{(1)}$ (no signs) |
| Incidence signs; $\partial_{k-1}\circ\partial_k=0$ | Sparse boundary $\partial_k$ |
"""))

    cells.append(code(r"""
# Filled triangle: oriented edges, boundary_matrix(K, 1) as an array,
# and boundary_square_is_zero(K, 2).
from lib import boundary_matrix, boundary_square_is_zero, oriented_simplices

triangle = from_facets([[0, 1, 2]])
print("oriented vertices", oriented_simplices(triangle, 0))
print("oriented edges   ", oriented_simplices(triangle, 1))
incidence = boundary_matrix(triangle, 1)
print("boundary_matrix(K, 1) shape", incidence.shape)
print(incidence.toarray())
square_vanishes = boundary_square_is_zero(triangle, 2)
print("boundary_square_is_zero(K, 2)", square_vanishes)
assert square_vanishes
"""))

    cells.append(md(r"""
## Stars

The star of a simplex $\sigma\in K$ is the set of its cofaces,
$$
\operatorname{St}(\sigma, K) = \{\tau\in K\mid \sigma\subseteq\tau\}.
$$
Membership is containment of $\sigma$, so $\sigma$ itself belongs to
the star. A proper face $\rho\subsetneq\sigma$ fails
$\sigma\subseteq\rho$, and therefore sits outside the star. A face of
some coface that misses $\sigma$ sits outside as well. The star is
that set of simplices. It is the collection `star` returns, which is
a larger set than the subcomplex generated by closing the cofaces
downward.

On the book $K$ with facets $\{0,1,2\}$ and $\{0,1,3\}$,
$$
\operatorname{St}(\{0,1\}, K)
=
\bigl\{\{0,1\},\,\{0,1,2\},\,\{0,1,3\}\bigr\}.
$$
The four outer edges are faces of those triangles and do not contain
$\{0,1\}$, so they are outside this star. The star of the vertex is
$$
\operatorname{St}(\{0\}, K)
=
\bigl\{
  \{0\},\,\{0,1\},\,\{0,2\},\,\{0,3\},\,\{0,1,2\},\,\{0,1,3\}
\bigr\}.
$$
The edge $\{1,2\}$ meets triangle $\{0,1,2\}$ and still fails
$\{0\}\subseteq\{1,2\}$, so it is outside $\operatorname{St}(\{0\}, K)$.

The star reads the full simplex set. The facet list alone does not
name the edge $\{0,1\}$ until the closure has produced it.
"""))

    cells.append(code(r"""
# Star = cofaces, from the closure of the book's facet list.
book = from_facets([[0, 1, 2], [0, 1, 3]])


def show_star(sigma):
    star = book.star(sigma)
    ordered = sorted(
        (tuple(sorted(face)) for face in star),
        key=lambda face: (len(face), face),
    )
    print("sigma", tuple(sorted(sigma)), "dim", len(sigma) - 1)
    print("star ", ordered)


show_star(frozenset({0, 1}))
show_star(frozenset({0}))
"""))

    cells.append(md(r"""
## Pure complexes

$K$ is pure of dimension $k$ when every facet has that dimension:
$$
\dim\sigma = k \qquad\text{for every }\sigma\in\mathcal{F}(K).
$$
$K$ is pure when one such $k$ exists. Equivalently, every facet has
the same number of vertices. The dimension of $K$ is still the
maximum, including when the facets do not all attain it:
$$
\dim K = \max_{\sigma\in\mathcal{F}(K)}\dim\sigma
$$
for nonempty $K$.

`is_pure(k)` tests the displayed formula. `is_pure()` with no
argument tests that the facet dimensions agree, whatever the common
value is.

The book is pure of dimension $2$: both facets are triangles. The
complex with facets $\{0,1,2\}$ and $\{2,3\}$ has facet dimensions
$2$ and $1$, so it is not pure, and its dimension is $2$.
"""))

    cells.append(code(r"""
# Purity reads facet dimensions. The pendant edge is a 1-dimensional facet.
book = from_facets([[0, 1, 2], [0, 1, 3]])
pendant = from_facets([[0, 1, 2], [2, 3]])
print("book dimension", book.dimension(), "pure of dim 2", book.is_pure(2))
print(
    "pendant dimension",
    pendant.dimension(),
    "pure",
    pendant.is_pure(),
    "facet dims",
    sorted({len(facet) - 1 for facet in pendant.facets}),
)
"""))

    cells.append(md(r"""
## Geometric realization

Choose a point $p_v\in\mathbb{R}^d$ for each vertex $v$ so that, on
every simplex, those points are affinely independent, and so that
intersections follow the abstract faces:
$$
\operatorname{conv}\{p_v\mid v\in\sigma\cap\tau\}
=
\operatorname{conv}\{p_v\mid v\in\sigma\}
\cap
\operatorname{conv}\{p_v\mid v\in\tau\}.
$$
The geometric realization is the union of the convex hulls,
$$
|K|
=
\bigcup_{\sigma\in K}
\operatorname{conv}\{p_v\mid v\in\sigma\}.
$$
The centroid computed above is one point of
$\operatorname{conv}\{p_0,p_1,p_2\}$ for a single triangle. $|K|$ is
the union of such hulls over every face. A face inclusion
$\tau\subseteq\sigma$ becomes an inclusion of convex hulls.

The figure below is one such choice of the $p_v$ for four small
complexes: a filled triangle, the book, the triangle with a pendant
edge, and the boundary of a tetrahedron. The plane figures have no
camera sliders. The tetrahedron adds elev and azim. Blue marks the
chosen simplex, orange marks every other simplex in its star, and
gray marks the rest of $K$. On the book, the shared edge $\{0,1\}$
is blue and both triangles are orange; the outer edges stay gray,
matching $\operatorname{St}(\{0,1\}, K)$ above. Endpoints of a chosen
edge are vertices of that edge, and those 0-simplices sit outside
the star, so their markers stay gray.

The drawing and the widgets are `complex_draw_demo.py`, in this
folder. Read and edit that file. The next cell only checks that the
file is beside the notebook, imports `launch_complex_draw_demo`, and
calls it.

<details>
<summary><b>Where is the plot code?</b> (click to expand)</summary>

The interactive UI lives in
<code>complex_draw_demo.py</code> next to this notebook, so this
cell stays a launch stub. JupyterLab collapses the launch input
(<code>source_hidden</code>). Open the <code>.py</code> in the editor
to change coordinates, colours, or the widget layout.

</details>
"""))

    cells.append(code(r"""
# Launch only — implementation: complex_draw_demo.py (same folder).
from pathlib import Path
import sys

_here = Path.cwd()
if not (_here / "complex_draw_demo.py").exists():
    _here = Path("tutorials/simplicial_manifolds")
if not (_here / "complex_draw_demo.py").exists():
    raise FileNotFoundError(
        "complex_draw_demo.py missing next to this notebook"
    )
_resolved = str(_here.resolve())
if _resolved not in sys.path:
    sys.path.insert(0, _resolved)

from complex_draw_demo import launch_complex_draw_demo

launch_complex_draw_demo()
""", hide_input=True))

    cells.append(md(r"""
## Summary

An abstract simplex is a finite set; a geometric simplex is the
convex hull of affinely independent points, with barycentric
coordinates $t_i$. A complex $K$ is a downward-closed collection of
nonempty simplices. Its $f$-vector counts those simplices by
dimension, and $\dim K$ is the largest dimension in the collection.

The constructor stores the facet list $\mathcal{F}(K)$. The closure
of that list is the full simplex set, which stars and face counts
read. The adjacency list indexes $K^{(1)}$ by vertex and stores
$2\,\#K^{(1)}$ neighbour slots. That is the structure for degrees and
walks; a dense $V\times V$ matrix is the wrong size once the
1-skeleton is sparse. The list has no signs and no face of dimension
greater than $1$. The boundary $\partial_k\colon C_k\to C_{k-1}$ is
the sparse signed incidence of oriented faces. On the filled triangle
the columns of $\partial_1$ are the three edges, and
$\partial_1\circ\partial_2=0$. Triangles as sets stay with the facet
list and the closure; triangles as chains are read from $\partial_2$.

The star of $\sigma$ is the set of cofaces of $\sigma$. A complex is
pure of dimension $k$ when every facet has dimension $k$. The
realization $|K|$ is the union of the geometric simplices. Notebook
02 takes up the question of when $|K|$ is a combinatorial manifold.
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
