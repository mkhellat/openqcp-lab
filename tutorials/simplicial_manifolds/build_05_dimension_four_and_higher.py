#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build notebook 05 (source of truth): dimension four and higher.

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_05_dimension_four_and_higher.py

The generated notebook is ``05_dimension_four_and_higher.ipynb`` beside
this file. It is the computation-path landscape for $k=4$ and $k\\ge 5$.
The local column is the link criterion of notebook 02. The global column
records Markov's undecidability theorem for homeomorphism of closed
$n$-manifolds, $n\\ge 4$, and S. P. Novikov's undecidability theorem for
recognition of $S^{d}$, $d\\ge 5$. Recognition of $S^{4}$ is left open.
Schleimer's theorem stays the $k=3$ statement from notebook 04:
$3$-sphere recognition lies in NP. That membership does not make
dimension $4$ NP-hard.

One code cell counts the model spheres $\\partial\\Delta^{5}$ and
$\\partial\\Delta^{6}$ and the links of the model $4$-sphere. It does not
decide homeomorphism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("05_dimension_four_and_higher.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Dimension four and higher

Notebook 02 fixes the closed combinatorial manifold criterion. Notebook
03 evaluates it at $k=2$, where orientability and
$$
\chi = V - E + F
$$
determine the homeomorphism type of a connected closed surface.
Notebook 04 evaluates it at $k=3$. There the local test asks for
vertex links homeomorphic to $S^{2}$, and the global test of "is this
$S^{3}$?" lies in NP by Schleimer's theorem. This notebook is the
landscape for $k=4$ and for $k\ge 5$. The same two questions stay
separate. Their answers change.

The local question is still the link formula. At $k=4$ every link has
dimension at most $3$, so the formula is a finite check whose hardest
step is $3$-sphere recognition, already discussed in notebook 04. At
$k=5$ a vertex link is a $4$-sphere, and that recognition problem is
open. At $k\ge 6$ a vertex link is a sphere of dimension at least $5$,
and recognizing those spheres admits no algorithm.

The global question is homeomorphism type. For every $n\ge 4$,
A. A. Markov (Dokl. Akad. Nauk SSSR 121 (1958), 218–220) proved that
homeomorphism of closed triangulated $n$-manifolds is undecidable. In
dimension $4$ the manifold his argument fails to recognize is a
connected sum of copies of $S^{2}\times S^{2}$, which is not $S^{4}$.
For every $d\ge 5$, S. P. Novikov proved that recognizing $S^{d}$ is
itself undecidable (the statement included in I. A. Volodin,
V. E. Kuznetsov, and A. T. Fomenko, Russian Math. Surveys 29 (1974),
no. 5, 169–171). P. S. Novikov's unsolvability of the word problem
(Trudy Mat. Inst. Steklov. 44, 1955) is an earlier, different theorem,
used as group-theory input. The two Novikovs are father and son.

No cell searches for a homeomorphism, crushes a triangulation, or
builds a manifold from a group presentation. The only complex the code
builds is a model sphere it is handed by definition, together with a
disjoint union of two copies.

## Learning objectives

1. Keep the link formula and the homeomorphism question as two
   predicates, and read a row of the landscape for each
   $k\in\{1,2,3,4,5\}$ and for $k\ge 6$.
2. Compute the $f$-vector and $\chi$ of the model spheres
   $\partial\Delta^{5}\cong_{\mathrm{PL}} S^{4}$ and
   $\partial\Delta^{6}\cong_{\mathrm{PL}} S^{5}$, and the links of the
   model $4$-sphere, from the facet list.
3. State Markov's theorem for $n\ge 4$, name the $4$-dimensional
   target $\#_{k}(S^{2}\times S^{2})$, and separate that target from
   $S^{4}$ by the formula $\chi = 2 + 2k$.
4. State S. P. Novikov's theorem for $d\ge 5$, and name Smale's
   theorem as the step that starts at dimension $5$.
5. List the $S^{4}$ questions that stay open, including recognition
   and the smooth/PL $4$-dimensional Poincaré conjecture, beside
   Freedman's topological classification.
6. Keep Schleimer's theorem as $3$-sphere recognition
   $\in\mathrm{NP}$. Record why that membership is not an
   NP-hardness theorem in dimension $4$.
"""))

    cells.append(md(r"""
## Two decision problems

Fix a dimension $k\ge 1$ and a finite simplicial complex $K$. Write
$|K|$ for its geometric realization, as in notebook 01. The series
asks two questions, in this order.

**Local.** Is $K$ a closed combinatorial $k$-manifold? Notebook 02
writes the criterion as one formula. $K$ is pure of dimension $k$, and
every nonempty simplex $\sigma$ has PL-sphere link
$$
\operatorname{Lk}(\sigma, K)
\cong_{\mathrm{PL}}
\partial\Delta^{k-\dim\sigma}.
$$
The model PL $m$-sphere is $\partial\Delta^{m+1}$. The link on the
right has dimension $k-\dim\sigma-1$.

**Global.** When the local formula holds, which closed $k$-manifold is
$|K|$, up to homeomorphism? Two special cases have names.

*Homeomorphism of pairs.* Given two finite complexes that are closed
combinatorial $n$-manifolds, decide whether their realizations are
homeomorphic.

*Recognition of a target.* Fix one closed $n$-manifold $X$. Given one
finite complex, decide whether its realization is homeomorphic to $X$.
Sphere recognition is the case $X = S^{n}$.

Recognition of one fixed $X$ can be undecidable while recognition of a
different target in the same dimension stays open. An algorithm for
arbitrary pairs would recognize every target, by pinning one side of
the pair. The converse fails: each target could have its own procedure
without there being one procedure that compares every pair. Dimension
$4$ is where that gap is visible. Markov's theorem makes pairs
undecidable, and it makes $\#_{k}(S^{2}\times S^{2})$ unrecognizable
for a positive number of summands. It leaves $X = S^{4}$ unsettled.

The Euler characteristic is the alternating face count
$$
\chi(K)
=
\sum_{d\ge 0}(-1)^{d} f_d,
$$
with $f_d$ the number of $d$-simplices in the downward closure. For
the sphere,
$$
\chi(S^{n}) = 1 + (-1)^{n}.
$$
Thus $\chi(S^{n})$ is $2$ when $n$ is even and $0$ when $n$ is odd.
For closed $n$-manifolds the connected sum obeys
$$
\chi(M\# N) = \chi(M) + \chi(N) - \chi(S^{n}).
$$
Both formulas are used below as filters. A complex with the wrong
$\chi$ is not the target. A complex with the right $\chi$ is not
thereby recognized.

**What.** The link formula, the homeomorphism relation, and $\chi$.

**Why.** The tables in the next section are these predicates evaluated
by dimension. A slogan that collapses the two columns will mis-name
Schleimer's theorem.

**How.** Read the formula, then the row. The code cell only evaluates
$\chi$ and link face-counts on $\partial\Delta^{5}$ and
$\partial\Delta^{6}$.

**When.** The question is which decision problem a theorem is about.
A walk on a surface mesh still belongs to notebook 03.
"""))

    cells.append(md(r"""
## Landscape

Notebook 00 indexes the series in one table. This notebook is where
the $k=4$ and $k\ge 5$ cells get their hypotheses. Rows $k=1,2,3$
restate notebooks 02–04 so the jump at dimension $4$ sits in the same
grid. The symbol $\in\mathrm{NP}$ is membership, an upper bound.
Undecidable means that no algorithm exists. Those are different
predicates, and the table does not trade one for the other.

| $k$ | Local: the link formula | Global: homeomorphism / recognition |
|----:|-------------------------|--------------------------------------|
| $1$ | Degrees on the adjacency list: every vertex has degree $2$ | The same check. Closed $1$-manifolds are disjoint unions of cycles |
| $2$ | Vertex links are cycles; $\chi$ and orientability | A connected closed surface is classified by the pair (orientable, $\chi$), read off the facet list as in notebook 03 |
| $3$ | Vertex links $\cong S^{2}$ (notebook 04) | $S^{3}$ recognition $\in\mathrm{NP}$ (Schleimer). Closed oriented homeomorphism is decidable (Kuperberg) |
| $4$ | Decidable. The hardest link is a PL $3$-sphere | Homeomorphism undecidable (Markov). $S^{4}$ recognition is open |
| $5$ | Open. A vertex link is a PL $4$-sphere | Homeomorphism undecidable (Markov). $S^{5}$ recognition undecidable (S. P. Novikov) |
| $\ge 6$ | Undecidable. A vertex link is a sphere of dimension $\ge 5$ | $S^{k}$ recognition undecidable (S. P. Novikov). Homeomorphism undecidable (Markov) |

Notebook 00 writes one row "$\ge 5$" whose global cell is "sphere
recognition undecidable (Novikov-type)." Rows $5$ and $\ge 6$ are that
cell. They are split here because the local column changes. At $k=5$
the vertex link is a $4$-sphere, and nobody has proved that
recognizing $S^{4}$ is decidable or undecidable. At $k\ge 6$ the
vertex link falls under S. P. Novikov's theorem, so the link formula
itself has no algorithm. "Local combinatorial checks" in the
cheat-sheet names the criterion. It does not claim a decision
procedure past dimension $4$.

The $k=3$ global cell is two theorems, with different targets.

- Schleimer, arXiv:math/0407047 (2004), Theorem 16.1 in the preprint,
  and Proc. Sympos. Pure Math. 82 (2011), Theorem 15.1: $3$-sphere
  recognition lies in NP. Notebook 04 records the certificate and the
  open hardness question. This notebook uses only the membership.
- Kuperberg, Pacific J. Math. 301 (2019), 189–241, Theorem 1.1:
  homeomorphism of closed oriented triangulated $3$-manifolds is
  decidable, as a corollary of Perelman's geometrization theorem.
  Theorem 1.2 gives an elementary recursive upper bound, a bounded
  tower of exponentials. The paper leaves the non-orientable case
  unwritten. Notebook 04 does not put pairwise homeomorphism in NP.
  Neither does this one. Decidable is the contrast with Markov's
  theorem in the next dimension.

The $k=4$ local cell is a composition of checks already settled in
lower dimensions, on a finite facet list.

| $\dim\sigma$ | Link required | Status on a finite complex |
|-------------:|---------------|----------------------------|
| $4$ | empty complex | compare with the empty facet list |
| $3$ | two points, no edge | a finite count |
| $2$ | one cycle of length $\ge 3$ | the $k=1$ test, on the link |
| $1$ | a PL $2$-sphere | the surface classification of notebook 03 |
| $0$ | a PL $3$-sphere | decidable (Rubinstein–Thompson); $\in\mathrm{NP}$ (Schleimer) |

Rubinstein (ICM, Zürich, 1994; published 1995) and Thompson (Math. Res.
Lett. 1 (1994), 613–630) give an algorithm for $S^{3}$. The
conjunction of the five rows is therefore an algorithm for the link
formula at $k=4$. The algorithm is not written in this notebook.
`vertex_links_are_spheres_s2` is the tutorial-scale vertex half of the
$k=3$ test, and notebook 04 lists what it does not certify. Running it
on a $3$-dimensional link is not $4$-sphere recognition.

At $k=5$ the vertex row would ask for a PL $4$-sphere. That status is
the open cell in the next sections, so the conjunction has no proved
algorithm and no proved obstruction. At $k\ge 6$ the vertex row is
S. P. Novikov's problem. The cone on a complex $L$ has apex link $L$,
so a procedure for the link formula in dimension $\ge 6$ would
recognize spheres of dimension $\ge 5$. The cone is not built here.
The reduction is the reason the local column inherits the theorem.
"""))

    cells.append(md(r"""
## The model spheres are a count

The boundary of the simplex on $n+1$ vertices is the model PL sphere
of dimension $n-1$,
$$
\partial\Delta^{n}
=
\{\rho\subsetneq\{0,1,\ldots,n\}\mid \rho\neq\emptyset\}.
$$
The library omits the empty face. A $k$-face is a subset of size
$k+1$, and every such subset of an $(n+1)$-set is a face of the
boundary when $k\le n-1$, so
$$
f_k(\partial\Delta^{n}) = \binom{n+1}{k+1},
\qquad
k = 0, \ldots, n-1.
$$
The alternating sum equals the sphere formula
$$
\chi(\partial\Delta^{n})
=
\chi(S^{n-1})
=
1 + (-1)^{n-1}.
$$
The case $n = 5$ is the model $S^{4}$: the $f$-vector is
$(6, 15, 20, 15, 6)$ and $\chi = 2$. The case $n = 6$ is the model
$S^{5}$: the $f$-vector is $(7, 21, 35, 35, 21, 7)$ and $\chi = 0$.
The case $n = 4$ is the model $S^{3}$ of notebook 04, with
$(5, 10, 10, 5)$ and $\chi = 0$.

The link of a simplex in $\partial\Delta^{n}$ is the boundary of the
simplex on the complementary vertices. On the model $S^{4}$ the links
are therefore model spheres of lower dimension:

| Simplex in $\partial\Delta^{5}$ | Link | Model |
|----------------------------------|------|--------|
| a vertex | $\partial\Delta^{4}$ | $S^{3}$, $\chi = 0$ |
| an edge | $\partial\Delta^{3}$ | $S^{2}$, $\chi = 2$ |
| a triangle | $\partial\Delta^{2}$ | a $3$-cycle, $\chi = 0$ |
| a tetrahedron | $\partial\Delta^{1}$ | two points, $\chi = 2$ |
| a $4$-simplex | empty | PL sphere of dimension $-1$ |

That table is the link formula on one complex that satisfies it by
construction. The identification
$\partial\Delta^{5}\cong_{\mathrm{PL}} S^{4}$ is the definition of the
model in notebook 02. It is not a test that accepts an arbitrary
facet list.

A disjoint union of two copies has $\chi = 4$ and two
path-components of the $1$-skeleton. Each component is the model, so
the link formula holds and $|K|$ is a disconnected combinatorial
$4$-manifold. Connectedness of $|K|$ is connectedness of the
adjacency list. The sphere $S^{4}$ is connected and has $\chi = 2$.
The union meets neither numerical filter. Filters are one direction.
The next section is about complexes that no alternating sum separates
by an algorithm.

**What.** Binomial face counts of $\partial\Delta^{n}$, the link table
of the model $S^{4}$, and the component count of a disjoint union.

**Why.** $\chi(S^{4}) = 2$ and the link dimensions are the filters
used against Markov's target and against the local column. Both are
visible on a complex with six vertices.

**How.** Build `SimplicialComplex` from the facets of
$\partial\Delta^{5}$ and $\partial\Delta^{6}$. Read
`faces_by_dim()` and `link`. Count path-components with
`adjacency_list()`.

**When.** The question is the model, its $\chi$, or the dimension of
a link. Homeomorphism of a complex we did not build is the Markov
section. Half-edge and Quad-Edge walks stay in notebook 03.
"""))

    cells.append(code(r"""
# Model spheres only. Face counts, links, and one disjoint union.
import math
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
    # Dimension of the complex is dim - 1. This is the model S^{dim-1}.
    vertices = list(range(dim + 1))
    facets = [tuple(v for v in vertices if v != omit) for omit in vertices]
    return from_facets(facets)


def face_vector(K):
    dim = K.dimension()
    if dim < 0:
        return ()
    by_dim = K.faces_by_dim()
    return tuple(len(by_dim.get(d, ())) for d in range(dim + 1))


def binomial_boundary_vector(simplex_dim):
    # f_k(∂Δ^n) = C(n+1, k+1) for k = 0, ..., n-1.
    return tuple(
        math.comb(simplex_dim + 1, k + 1) for k in range(simplex_dim)
    )


def euler_from_vector(fv):
    return sum(((-1) ** d) * count for d, count in enumerate(fv))


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


def disjoint_union(left, right):
    shift = (max(left.vertices) + 1) if left.vertices else 0
    facets = set(left.facets)
    for face in right.facets:
        facets.add(frozenset(vertex + shift for vertex in face))
    vertices = frozenset().union(*facets) if facets else frozenset()
    return SimplicialComplex(vertices, frozenset(facets))


for simplex_dim in (4, 5, 6):
    K = simplex_boundary(simplex_dim)
    fv = face_vector(K)
    sphere_dim = simplex_dim - 1
    chi = euler_characteristic(K)
    print(
        f"dDelta^{simplex_dim}",
        "sphere dim",
        sphere_dim,
        "f",
        fv,
        "chi",
        chi,
        "components",
        component_count(K),
    )
    assert fv == binomial_boundary_vector(simplex_dim)
    assert chi == euler_from_vector(fv) == 1 + (-1) ** sphere_dim
    assert K.dimension() == sphere_dim and K.is_pure(sphere_dim)
    assert component_count(K) == 1

s4 = simplex_boundary(5)
link_expect = {
    (0,): 4,
    (0, 1): 3,
    (0, 1, 2): 2,
    (0, 1, 2, 3): 1,
}
for sigma, boundary_dim in link_expect.items():
    link = s4.link(frozenset(sigma))
    fv = face_vector(link)
    print("link", sigma, "f", fv, "chi", euler_characteristic(link))
    assert fv == binomial_boundary_vector(boundary_dim)
    assert euler_characteristic(link) == 1 + (-1) ** (boundary_dim - 1)

apex_facet = frozenset(range(5))
empty_link = s4.link(apex_facet)
print("link of a 4-simplex", "dim", empty_link.dimension(), "f", face_vector(empty_link))
assert empty_link.dimension() == -1
assert face_vector(empty_link) == ()

# Notebook 04's predicate on the model vertex link. True on every
# closed combinatorial 3-manifold, so the print is not S^3 recognition.
# The link is the model S^3 because its face vector is ∂Δ^4.
vertex_link = s4.link(frozenset((0,)))
print("vertex link passes S^2-link predicate", vertex_links_are_spheres_s2(vertex_link))
assert vertex_links_are_spheres_s2(vertex_link) is True

pair = disjoint_union(s4, s4)
print(
    "union f",
    face_vector(pair),
    "chi",
    euler_characteristic(pair),
    "components",
    component_count(pair),
)
assert euler_characteristic(pair) == 4
assert component_count(pair) == 2
assert face_vector(pair) == tuple(2 * count for count in face_vector(s4))
for vertex in pair.vertices:
    link = pair.link(frozenset((vertex,)))
    assert face_vector(link) == binomial_boundary_vector(4)
    assert vertex_links_are_spheres_s2(link) is True

# χ(#_k(S^2×S^2)) = 2 + 2k. Arithmetic filter only: no triangulation.
def chi_markov_target(k):
    return 2 + 2 * k

print("chi of #_k(S^2 x S^2) for k=1 and k=12", chi_markov_target(1), chi_markov_target(12))
assert chi_markov_target(0) == 2
assert chi_markov_target(12) == 26
assert all(chi_markov_target(k) != chi_markov_target(0) for k in range(1, 13))
print("counts and one arithmetic filter; no homeomorphism algorithm was run")
"""))

    cells.append(md(r"""
## Homeomorphism in dimension $\ge 4$ is undecidable

The group-theory input is older than the topology. P. S. Novikov
(Trudy Mat. Inst. Steklov. 44, 1955) proved that the word problem for
finitely presented groups is undecidable: there is a finite
presentation for which no algorithm decides whether a word represents
the identity. W. W. Boone (Ann. of Math. 70 (1959), 207–265) proved
this independently. From the word problem, S. I. Adian and M. O. Rabin
showed that many properties of the presented group are undecidable as
well. The property used here is triviality. M. O. Rabin (Ann. of
Math. 67 (1958), 172–194) is the English account: there is no
algorithm that takes a finite presentation and decides whether the
group is the one-element group.

H. Seifert and W. Threlfall (*Lehrbuch der Topologie*, 1934) construct,
from a finite presentation $P$, a closed triangulated $4$-manifold
$M(P)$ whose fundamental group is the group presented by $P$. The
construction is effective: a machine can write down the triangulation
from the generators and relators. This notebook does not carry it
out. Markov feeds that manifold to the triviality problem.

| Step | Object | What is known |
|------|--------|----------------|
| 1 | a finite presentation $P$ | triviality of the group is undecidable |
| 2 | $M(P)$, a closed triangulated $4$-manifold | $\pi_1(M(P))$ is the group of $P$ |
| 3 | a fixed target $\#_{k}(S^{2}\times S^{2})$, some $k\ge 1$ | $M(P)$ is homeomorphic to the target if and only if the group is trivial |
| 4 | pairs of closed triangulated $n$-manifolds, any fixed $n\ge 4$ | an algorithm for pairs would recognize the target in the previous row |

Step 3 is Markov's theorem in dimension $4$. Step 4 is the corollary
he records for every $n>3$: the homeomorphism problem for closed
$n$-manifolds is undecidable, because an algorithm that compared
arbitrary pairs would, with one side held fixed, recognize $M^{n}$.
The same paper states the fixed-target form first. For every natural
number $n>3$ there exists a closed $n$-manifold $M^{n}$ such that no
algorithm decides whether a given closed triangulated $n$-manifold is
homeomorphic to $M^{n}$.

The $4$-dimensional target in the expositions of this argument is
$\#_{k}(S^{2}\times S^{2})$ for some $k\ge 1$. C. McA. Gordon (New
Zealand J. Math. 52 (2022), 821–826) proves a concrete case: there is
no algorithm that decides whether a given closed $4$-manifold is
homeomorphic to $\#_{12}(S^{2}\times S^{2})$. Later work has lowered
the number of summands. The existence of some unrecognizable target
is Markov's theorem and does not depend on the smallest such number.
This notebook uses $k=12$ when it wants a published integer, and
$k\ge 1$ when the Euler characteristic is the point.

Apply the connected-sum formula. $\chi(S^{2}\times S^{2}) = 4$ and
$\chi(S^{4}) = 2$, so
$$
\chi(\#_{k}(S^{2}\times S^{2})) = 2 + 2k.
$$
For every $k\ge 1$ the value is an even integer at least $4$. The code
cell checks the arithmetic at $k=1$ ($\chi = 4$) and at Gordon's
$k=12$ ($\chi = 26$). Neither equals $\chi(S^{4}) = 2$. The manifold
Markov's argument makes unrecognizable is therefore not the
$4$-sphere. Undecidability of homeomorphism, and undecidability of
recognition for this particular target, leave recognition of $S^{4}$
on a different line.

The same construction shows a second undecidable problem, because
$\pi_1(M(P))$ is the group of $P$. There is no algorithm that decides
whether a closed triangulated $4$-manifold is simply connected. That
fact still does not settle $S^{4}$. The target
$\#_{1}(S^{2}\times S^{2})$ is simply connected, has $\chi = 4$, and
is not homeomorphic to $S^{4}$. Simple-connectivity is a filter in
one direction, in the same sense as $\chi$.

The input in Markov's paper is a polyhedron given by a triangulation.
A face-pairing triangulation need not be a simplicial complex in the
sense of this library: notebook 04 already records that one
tetrahedron with paired faces can triangulate $S^{3}$ and cannot be
stored as a set of vertices. Barycentric subdivision is an effective
operation and does not change the homeomorphism type. The subdivision
is a simplicial complex, and this library stores it as a facet list.
The presentations that reduce from the triviality problem therefore
yield facet lists of combinatorial manifolds. No procedure decides
homeomorphism for every such list.

**What.** Markov's fixed-target theorem and the corollary for pairs,
for every dimension $n\ge 4$, with the $4$-dimensional target
$\#_{k}(S^{2}\times S^{2})$.

**Why.** This is the global cell of the $k=4$ row. The proof is a
reduction from triviality of finitely presented groups, through the
Seifert–Threlfall manifold, not a reduction from Schleimer's
certificate.

**How.** Read the four-row schematic. The code cell only evaluates
$2+2k$. It stores no presentation and no connected sum.

**When.** The question is homeomorphism of $4$-manifolds, or
recognition of $\#_{k}(S^{2}\times S^{2})$. Recognition of $S^{4}$ is
the next section but one. Sphere recognition in dimensions $\ge 5$
is S. P. Novikov's theorem, which needs an extra ingredient.
"""))

    cells.append(md(r"""
## Why dimension $4$ is the special row

Three comparisons meet at $k=4$, and they are easy to fold into one
sentence. They are three different facts.

**It is the first dimension in which homeomorphism is undecidable.**
For $k\le 2$ the classification is a face count plus a small amount
of combinatorial data (degrees, orientability, $\chi$). For closed
oriented $3$-manifolds, Kuperberg's theorem supplies an algorithm.
Markov's theorem starts at $n>3$. Dimension $4$ is the first row of
the landscape whose global column is an undecidability theorem.

**The local column is still an algorithm, and the global column is
not.** The link table in the landscape section reduces a
$4$-dimensional link formula to spheres of dimension at most $3$.
Those recognition problems are settled: surfaces by the
classification, $S^{3}$ by Rubinstein–Thompson, with Schleimer's
short certificate for the yes-instances. The output of that
composition is one bit, "combinatorial $4$-manifold, yes or no."
Markov's theorem says that the next bit, "homeomorphic to this
particular $4$-manifold," has no algorithm. Local success is the
hypothesis under which the global problem is even posed. It is not
progress toward a homeomorphism test.

**The three manifold categories diverge here, and they diverge
differently again past dimension $4$.** In dimensions $\le 3$ a
topological manifold has a piecewise-linear structure and a smooth
structure, and they are unique in the senses used by the
classification; Moise's theorem is the $3$-dimensional statement.
In dimension $4$ a PL manifold admits a compatible smooth structure:
the obstructions live in homotopy groups of $\mathrm{PL}/O$ that
vanish in this range (M. W. Hirsch and B. Mazur, *Smoothings of
Piecewise Linear Manifolds*, Ann. of Math. Studies 80, Princeton,
1974). The topological category is larger. Rokhlin's theorem (Dokl.
Akad. Nauk SSSR 84 (1952), 221–224) says that the signature of a
closed smooth spin $4$-manifold is divisible by $16$. Freedman's
$E_{8}$ manifold is a closed simply-connected topological
$4$-manifold of signature $8$ (M. H. Freedman, J. Differential Geom.
17 (1982), 357–453), so it has no smooth structure and no PL
structure. S. K. Donaldson (J. Differential Geom. 18 (1983),
279–315) restricts intersection forms of smooth definite
$4$-manifolds further. Homeomorphism and diffeomorphism are different
relations on $4$-manifolds. Markov's theorem is about homeomorphism.

Past dimension $4$ the piecewise-linear and smooth categories diverge
as well. Milnor's exotic $7$-spheres (Ann. of Math. 64 (1956),
399–405) are homeomorphic to $S^{7}$ and not diffeomorphic to it.
The Kirby–Siebenmann class in $H^{4}(M;\mathbb{Z}/2)$ obstructs a PL
structure on a topological manifold of high dimension (R. C. Kirby
and L. C. Siebenmann, *Foundational Essays on Topological Manifolds,
Smoothings, and Triangulations*, Ann. of Math. Studies 88, Princeton,
1977). S. P. Novikov's theorem, in the next section, is again a
statement about homeomorphism to the sphere. It is not a statement
about diffeomorphism.

| Dimension | Categories | Algorithmic row |
|----------:|------------|-----------------|
| $\le 3$ | TOP, PL, and DIFF agree | classification, or Schleimer / Kuperberg at $k=3$ |
| $4$ | PL manifolds smooth; TOP is strictly larger | local link formula decidable; homeomorphism undecidable; $S^{4}$ open |
| $\ge 5$ | TOP, PL, and DIFF can differ | homeomorphism undecidable; sphere recognition undecidable |

The algorithmic row is the subject of this notebook. The category row
is why a careful statement says homeomorphic, PL-homeomorphic, or
diffeomorphic, and then sticks to the word it chose. The theorems
cited for undecidability are homeomorphism theorems about triangulated
input.
"""))

    cells.append(md(r"""
## What stays open for $S^{4}$

Three decision problems in dimension $4$ have three different
statuses. Writing them side by side is the whole of the subtlety.

| Problem | Input | Status |
|---------|--------|--------|
| Homeomorphism of pairs | two closed triangulated $4$-manifolds | undecidable (Markov, 1958) |
| Recognition of $\#_{12}(S^{2}\times S^{2})$ | one closed triangulated $4$-manifold | undecidable (Gordon, 2022; some $k$ already in Markov) |
| Recognition of $S^{4}$ | one finite complex, or one closed PL $4$-manifold | open |

The third row is open both as a topological question and as a PL
question on simplicial input. Nothing in the first two rows closes
it. The Euler characteristic blocks the obvious slide: every manifold
$\#_{k}(S^{2}\times S^{2})$ with $k\ge 1$ has $\chi = 2+2k \neq 2$, so
the unrecognizable target is a different manifold. The fundamental
group blocks the next slide. Triviality of $\pi_1$ is undecidable for
closed triangulated $4$-manifolds, and $S^{4}$ is simply connected,
but so is $\#_{1}(S^{2}\times S^{2})$. An oracle for simple-connectivity
would not be an oracle for the sphere.

Freedman's theorem is a classification statement in the topological
category. A closed topological $4$-manifold that is a homotopy sphere
is homeomorphic to $S^{4}$. Among closed simply-connected topological
$4$-manifolds, the homeomorphism type is determined by the
intersection form and the Kirby–Siebenmann invariant. $S^{4}$ is the
manifold with trivial form. Two gaps sit between that classification
and a procedure that reads a facet list.

- The input to Freedman's theorem is already a topological manifold
  with the homotopy type of the sphere. A simplicial complex arrives
  as a finite list of subsets. Homology of that complex is computable
  from the boundary matrices of notebook 01. Triviality of its
  fundamental group is the group-theory problem above, and the
  presentations that arise from homology spheres are a special
  family. Undecidability for arbitrary presentations does not by
  itself decide that special family. This is why the third row of the
  table stays open rather than moving to the undecidable column.
- Freedman's proof is not a search through triangulations. A
  classification theorem and a decision procedure are different
  mathematical objects. Dimension $3$ needed Perelman before
  Kuperberg could extract an algorithm, long after the geometrization
  conjecture had a precise statement.

The smooth $4$-dimensional Poincaré conjecture is open, and because
PL $4$-manifolds admit compatible smooth structures it is the same
open question as the PL conjecture: whether every smooth homotopy
$4$-sphere is diffeomorphic to $S^{4}$. A homotopy $4$-sphere is a
closed manifold homotopy equivalent to $S^{4}$. An affirmative answer
would say that the geometric objects are standard. It would not write
the algorithm that decides whether a given facet list is one of those
objects. A negative answer would exhibit an exotic homotopy sphere.
It would still leave a recognition problem, now with more than one
diffeomorphism type in the homotopy class.

The local $k=4$ algorithm does not touch this question. Its vertex
links are $3$-spheres. Certifying $\partial\Delta^{5}$ by deleting a
vertex, as the code cell does, uses the definition of the model. The
same deletion on a complex that is not the boundary of a simplex is
not supplied by any function in `lib/`.

**What.** The open status of $S^{4}$ recognition, Freedman's
topological theorem, and the open smooth/PL Poincaré conjecture in
dimension $4$.

**Why.** The cheat-sheet's phrase "$S^{4}$ subtleties" is this table.
Markov's target, the sphere, and a homotopy sphere are three spaces.

**How.** Compare $\chi$ and $\pi_1$ as one-way filters. Stop when the
filter no longer separates $S^{4}$ from the remaining inputs.

**When.** The target is $S^{4}$. For $S^{d}$ with $d\ge 5$ the
corresponding cell is a theorem, proved in the next section.
"""))

    cells.append(md(r"""
## Sphere recognition is undecidable for $d\ge 5$

S. P. Novikov's theorem answers the sphere row in every dimension
$d\ge 5$. There is no algorithm that takes a finite simplicial complex
of dimension $d$ and decides whether its realization is homeomorphic
to $S^{d}$. The proof was published as a section of I. A. Volodin,
V. E. Kuznetsov, and A. T. Fomenko, Russian Math. Surveys 29 (1974),
no. 5, 169–171. The group-theory engine inside the proof is
P. S. Novikov's word problem. The son uses the father's theorem. The
topological statement is the son's.

The reason the argument cannot start at $d=4$ is S. Smale's theorem
(Ann. of Math. 74 (1961), 391–406): a simply connected homology
$d$-sphere is homeomorphic to $S^{d}$ when $d\ge 5$. A homology
sphere has the homology of $S^{d}$. With trivial fundamental group it
is a homotopy sphere, and Smale's h-cobordism theorem promotes that
to a homeomorphism with the standard sphere. The promotion uses the
Whitney trick, which needs dimension at least $5$. In dimension $4$
the corresponding smooth/PL promotion is the open Poincaré
conjecture, and Freedman's topological promotion is not the
effective step this reduction needs.

| Step | Object | What is known |
|------|--------|----------------|
| 1 | a word $w$ in a finitely presented group with unsolvable word problem | no algorithm decides whether $w$ represents $1$ |
| 2 | a triangulated homology $d$-sphere $X_w$, $d\ge 5$ | $\pi_1(X_w)$ is trivial if and only if $w$ represents $1$; the complex is produced effectively from $w$ |
| 3 | Smale's theorem | if $\pi_1(X_w)$ is trivial, then $X_w$ is homeomorphic to $S^{d}$ |
| 4 | the recognition question | $X_w$ is homeomorphic to $S^{d}$ if and only if $w$ represents $1$ |

Step 4 says that a sphere-recognition algorithm would solve the word
problem. Hence no such algorithm exists. The complexes $X_w$ built in
the proof are manifolds, so the problem stays undecidable when the
input is restricted to combinatorial $d$-manifolds. It stays
undecidable without that promise, because an algorithm for arbitrary
complexes would restrict to one. A non-manifold is not homeomorphic
to $S^{d}$, and the reduction never needs those inputs.

A connected sum extends the theorem from the sphere to an arbitrary
target. Let $M$ be any closed $d$-manifold, $d\ge 5$, and form
$M\# X_w$. The fundamental group of a connected sum is the free
product
$$
\pi_1(M\# X_w) \cong \pi_1(M) * \pi_1(X_w).
$$
For finitely generated groups, $G * H \cong G$ if and only if $H$ is
the trivial group. Thus $\pi_1(M\# X_w) \cong \pi_1(M)$ if and only if
$w$ represents $1$. In that case Smale's theorem supplies a
homeomorphism from $X_w$ to $S^{d}$, and
$M\# X_w$ is homeomorphic to $M$. If $w$ does not represent $1$, the
fundamental groups differ, so $M\# X_w$ is not homeomorphic to $M$.
This notebook does not build the sum. The conclusion is the one used
in the landscape: for $d\ge 5$, no closed $d$-manifold has a
recognition algorithm. The sphere is the case $M = S^{d}$. Dimension
$4$ has no Smale step, and the connected-sum extension does not fire
there either. That is another face of the $S^{4}$ gap: unrecognizable
$4$-manifolds exist, and the proof that every target is unrecognizable
begins at dimension $5$.

The same dimension threshold explains the local column for $k\ge 6$.
The link of a vertex is a finite simplicial complex of dimension
$k-1\ge 5$. S. P. Novikov's theorem says there is no algorithm that
decides whether that link is homeomorphic to a sphere. The link
formula demands that it be a PL sphere. The local predicate in
dimension $\ge 6$ is therefore undecidable. At $k=5$ the vertex link
is $4$-dimensional, Smale's hypothesis is one short, and the local
predicate is open for the same reason $S^{4}$ recognition is open.

**What.** S. P. Novikov's theorem, Smale's theorem as the dimension
threshold, and the passage from the sphere to an arbitrary target.

**Why.** This is the global cell of every row $k\ge 5$, and the local
cell of every row $k\ge 6$.

**How.** Read the four-row schematic. Identify which Novikov is the
word problem and which is the sphere.

**When.** The dimension is at least $5$. In dimension $4$, use
Markov's theorem for pairs and for $\#_{k}(S^{2}\times S^{2})$, and
keep $S^{4}$ in the open row.
"""))

    cells.append(md(r"""
## A complex homeomorphic to a sphere can fail the link formula

The local and global columns are different predicates even when both
are undecidable. The classical witness is the double suspension
theorem. R. D. Edwards ("Suspensions of homology spheres",
arXiv:math/0610573) proves that the double suspension of a homology
$n$-sphere is homeomorphic to $S^{n+2}$. For a homology $3$-sphere the
realization is homeomorphic to $S^{5}$.

The suspension points spoil the link formula. Suspend a homology
$3$-sphere once: each of the two new vertices has that homology
sphere as its link, and the link is not $S^{3}$. Suspend again.
Edwards' theorem says the realization is homeomorphic to $S^{5}$.
The links of the outer suspension vertices are suspensions of the
homology sphere, and those links are not PL $4$-spheres. The complex
is homeomorphic to $S^{5}$ and it is not a combinatorial manifold.
Notebook 02 named this witness and left the recognition theorems to
notebooks 04 and 05. The witness has dimension $5$, inside the range
of S. P. Novikov's theorem and one dimension above the open $S^{4}$
row.

The example separates the two readings of "is this a sphere?"

- **Homeomorphism.** $|K|\cong S^{5}$. S. P. Novikov's theorem says
  that no algorithm answers this for arbitrary input of dimension
  $\ge 5$. On this particular complex the answer is yes, by Edwards'
  theorem, and the complex is not produced by the code cell.
- **Link formula.** $K$ is not a combinatorial $5$-manifold. The
  failure is local, at the suspension vertices, and it is the reason
  notebook 02 refuses to treat "homeomorphic to a sphere" as a
  synonym for the PL link criterion.

High-dimensional sphere recognition is the homeomorphism reading.
The combinatorial-manifold check is the link reading. They disagree
on the double suspension. Undecidability of sphere recognition is
already in force at dimension $5$. Undecidability of the link formula
begins at dimension $6$, where a vertex link has dimension at least
$5$. No cell builds a homology sphere or a suspension.
"""))

    cells.append(md(r"""
## Schleimer's theorem stays in dimension $3$

The false step this notebook exists to block is the sentence
"Schleimer put $3$-sphere recognition in NP, so dimension $4$ is
NP-hard." The sentence chains three substitutions, and each one
changes the claim.

**The problem changes.** Schleimer's Theorem 15.1 (2011; Theorem 16.1
in arXiv:math/0407047, 2004) is recognition of one $3$-manifold, the
sphere $S^{3}$. Markov's theorem is homeomorphism of $4$-manifolds,
and recognition of $\#_{k}(S^{2}\times S^{2})$. Recognition of $S^{4}$
is a third problem, and it is open. A theorem about the first problem
is not a theorem about the second or the third. Notebook 04 describes
the certificate: a polynomial-length sequence of triangulations and
normal or almost-normal $2$-sphere vectors. That sequence is a
witness for $S^{3}$. It is not a witness for a $4$-dimensional
homeomorphism.

**The direction of the bound changes.** A language lies in NP when
there is a polynomial $q$ such that every yes-instance has a
certificate of length at most $q$ of the input length, and a
deterministic verifier accepts the pair (instance, certificate) in
polynomial time. Membership is an upper bound: yes-instances have
short proofs. A language is NP-hard when every language in NP
polynomial-time many-one reduces to it. NP-complete means both.
Schleimer proves membership for $S^{3}$ recognition. The 2004
preprint asks whether that problem is NP-hard and leaves the question
unanswered. The 2011 article does not turn Theorem 15.1 into a
hardness theorem. The NP-complete problem Schleimer points to instead
is $3$-manifold knot genus (Agol, Hass, and Thurston, STOC 2002;
arXiv:math/0205057), which is still a dimension-$3$ problem about
knots. An upper bound on problem A does not produce a lower bound on
problem B.

**The dimension and the date change.** Markov's paper is 1958.
Schleimer's preprint is 2004. The reduction Markov writes starts from
Adian–Rabin triviality and the Seifert–Threlfall $4$-manifold. It
does not start from a normal surface in a $3$-dimensional
triangulation. A result about dimension $4$ that predates the
dimension-$3$ certificate is not a corollary of that certificate.

One further distinction keeps the correction from inventing a new
slogan. Undecidability of $4$-manifold homeomorphism implies that
this problem lies outside NP. Every language in NP is decidable: the
certificate length is at most polynomial, the verifier is
deterministic, and the search over that finite set of certificates
halts. Markov's theorem says the search for a homeomorphism algorithm
does not halt with a correct procedure, because none exists. Outside
NP is an upper-bound statement of a negative kind. NP-hard is a
lower-bound statement. Some undecidable languages are NP-hard under
polynomial-time reductions; the halting problem is the standard
example, since a bounded search for a satisfying assignment can be
posed as a question about halting. This notebook therefore does not
say "undecidable, hence not NP-hard." It says something narrower,
which is all the citations support.

- Markov's theorem is undecidability of homeomorphism. The paper does
  not state an NP-hardness theorem, and the reduction it gives is
  from an undecidable group-theory problem.
- Schleimer's theorem is membership of $S^{3}$ recognition in NP.
  It does not mention dimension $4$.
- The sentence that derives NP-hardness of the $k=4$ row from
  Schleimer's membership substitutes the problem, the direction of
  the bound, and the dimension.

The corrected $k=4$ row is the landscape row already stated. The
link formula is decidable. Homeomorphism is undecidable. $S^{4}$
recognition is open. None of those three clauses is the clause
"$k=4$ is NP-hard."

**What.** The three substitutions in the false step, and the
corrected row.

**Why.** Notebook 00 defers this correction to notebooks 04 and 05.
Notebook 04 owns the membership direction inside dimension $3$. This
notebook owns the refusal to carry that membership into dimension $4$.

**How.** Match each clause of the false sentence to the theorem it
would have to cite. Stop when the citation is about a different
problem.

**When.** The claim under discussion is complexity of dimension $4$.
A certificate for $S^{3}$ is notebook 04, and it is not recomputed
here.
"""))

    cells.append(md(r"""
## Which structure, for which question

| Question | Structure | Why this one |
|----------|-----------|----------------|
| the model $\partial\Delta^{n}$, its $f$-vector, $\chi$ | facet list, then `faces_by_dim` | $\chi$ is the alternating face count; the binomial formula counts subsets of the vertex set |
| link of a simplex on the model | `link`, a complex on the complementary vertices | the link formula is a statement about that complex |
| path-components of a disjoint union | `adjacency_list` | $|K|$ is connected exactly when the $1$-skeleton is |
| the $S^{2}$-link predicate on a vertex link of the model | `vertex_links_are_spheres_s2` | notebook 04's local test, applied to a $3$-complex we already know is $\partial\Delta^{4}$ |
| incidence signs, $\partial^{2} = 0$ | `boundary_matrix` | not used; column counts would recover $\chi$ and $\chi$ does not recognize $S^{4}$ |
| an oriented surface walk, or a walk on the dual | half-edge, Quad-Edge | notebook 03; no cell here circulates a face |
| homeomorphism, or recognition of $S^{d}$ or of $\#_{k}(S^{2}\times S^{2})$ | no object stored in `lib/` | the theorems are reductions from group presentations; the notebook records the reduction and does not run it |

The canonical store is still the facet list. The code builds three
complexes from facet lists: $\partial\Delta^{5}$, $\partial\Delta^{6}$, and
a disjoint union of two copies of the first. The arithmetic
$2+2k$ builds nothing.
"""))

    cells.append(md(r"""
## Summary

The link formula and the homeomorphism question are different
predicates. On the model $\partial\Delta^{5}$ the formula holds by
construction: the $f$-vector is $(6, 15, 20, 15, 6)$, $\chi = 2$, and
the links are $\partial\Delta^{4}$, $\partial\Delta^{3}$,
$\partial\Delta^{2}$, two points, and the empty complex, in that order
as the simplex grows. $\partial\Delta^{6}$ has $\chi = 0$. A disjoint
union of two copies of the model $4$-sphere has $\chi = 4$ and two
path-components. Each vertex link still passes the
dimension-$3$ predicate of notebook 04. That predicate does not
recognize $S^{3}$ on its own, and the union is not $S^{4}$. The
counts are filters.

Markov (Dokl. Akad. Nauk SSSR 121 (1958), 218–220) proved that for
every $n\ge 4$ the homeomorphism problem for closed triangulated
$n$-manifolds is undecidable, by reduction from the triviality
problem for finite presentations. In dimension $4$ the unrecognizable
target is $\#_{k}(S^{2}\times S^{2})$ for some $k\ge 1$. Gordon (New
Zealand J. Math. 52 (2022), 821–826) takes $k = 12$. The Euler
characteristic $2+2k$ separates every such target from $S^{4}$.
Recognition of $S^{4}$ is open. Freedman's theorem classifies
topological homotopy $4$-spheres and is not a facet-list algorithm.
The smooth and PL Poincaré conjectures in dimension $4$ are open.
The local link formula in dimension $4$ is nevertheless decidable,
because every link has dimension at most $3$.

S. P. Novikov (Russian Math. Surveys 29 (1974), no. 5, 169–171)
proved that recognizing $S^{d}$ is undecidable for every $d\ge 5$.
The proof uses P. S. Novikov's word problem and Smale's theorem that
a simply connected homology $d$-sphere is homeomorphic to $S^{d}$
when $d\ge 5$. A connected-sum argument extends unrecognizability
from the sphere to every closed target in those dimensions. The link
formula in dimension $\ge 6$ is undecidable because a vertex link is
such a sphere. The double suspension of a homology $3$-sphere is
homeomorphic to $S^{5}$ and fails the link formula, so the two
predicates can disagree.

Schleimer's theorem remains $3$-sphere recognition $\in\mathrm{NP}$.
It is a membership result in dimension $3$. It is not NP-hardness,
it does not mention dimension $4$, and it is not the source of
Markov's theorem. The $k=4$ row is: local checks decidable,
homeomorphism undecidable, $S^{4}$ recognition open.
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
