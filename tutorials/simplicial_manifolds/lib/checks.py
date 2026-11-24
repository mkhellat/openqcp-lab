# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Local combinatorial-manifold checks, by dimension.

Dimension 1 goes through the adjacency-list cycle test. Dimension 2
asks each vertex link to be a single circle. The 3-dimensional check
asks each vertex link to be a combinatorial 2-sphere and documents
why that is not a recognition procedure for $S^3$.
"""

from __future__ import annotations

from .complex import SimplicialComplex
from .invariants import euler_characteristic, is_disjoint_union_of_cycles


def links_are_circles(K: SimplicialComplex) -> bool:
    """Closed combinatorial 1-manifold check on a pure 1-complex.

    A closed 1-manifold is a disjoint union of circles. In a simplicial
    complex that is the same statement as: pure of dimension 1, and
    every vertex link is a 0-sphere (two points, no edge).

    The degree form of "link is two points" is read from the adjacency
    list inside `is_disjoint_union_of_cycles` (one list, then a walk).
    This function then checks the link complex of each vertex is that
    0-sphere, so the manifold wording and the graph wording have to
    agree.
    """
    if not is_disjoint_union_of_cycles(K):
        return False
    for vertex in K.vertices:
        link = K.link(frozenset((vertex,)))
        if link.dimension() != 0 or len(link.vertices) != 2:
            return False
        if link.facets != frozenset(frozenset((v,)) for v in link.vertices):
            return False
    return True


def closed_surface_link_condition(K: SimplicialComplex) -> bool:
    """Whether every vertex link is a single circle.

    This is the combinatorial closed-surface condition: `K` is pure of
    dimension 2, and the link of each vertex is one cycle (a
    combinatorial 1-sphere), not a path and not several cycles.

    A disjoint union of closed surfaces passes: connectedness is not
    required. The boundary of a tetrahedron passes. Two triangles that
    meet only at a vertex fail, because the shared vertex has two
    disjoint edges as its link. A pure 2-complex with boundary (a
    single filled triangle, a disk) fails because some links are paths.
    """
    if K.dimension() != 2 or not K.is_pure(2) or not K.vertices:
        return False
    for vertex in K.vertices:
        if not _is_single_cycle(K.link(frozenset((vertex,)))):
            return False
    return True


def vertex_links_are_spheres_s2(K: SimplicialComplex) -> bool:
    """Whether every vertex link is a combinatorial 2-sphere.

    `K` must be pure of dimension 3. Each vertex link is accepted only
    when it is a connected closed combinatorial surface of Euler
    characteristic 2. In dimension 2 that is exact: by the
    classification of closed surfaces, that link is a 2-sphere. The
    boundary of a tetrahedron is the smallest example of such a link.
    The boundary of the 4-simplex is a tiny complex on which this
    predicate is true.

    Limits — this is not a 3-sphere recognizer:

    - Only vertex links are inspected. A combinatorial 3-manifold also
      needs the link of every edge to be a circle and the link of every
      triangle to be a 0-sphere (two points). Those faces are not
      checked here.
    - Every closed combinatorial 3-manifold has vertex links $S^2$, not
      only $S^3$. Lens spaces, connected sums, and disjoint unions can
      pass. Homeomorphism type is not decided.
    - Manifolds with boundary fail on purpose: a boundary vertex has a
      disk as its link, and a disk has Euler characteristic 1, not 2.
    - Returning true does not certify a production triangulation and
      does not implement Schleimer's certificate for $S^3$.
    """
    if K.dimension() != 3 or not K.is_pure(3) or not K.vertices:
        return False
    for vertex in K.vertices:
        if not _is_combinatorial_s2(K.link(frozenset((vertex,)))):
            return False
    return True


def _is_single_cycle(K: SimplicialComplex) -> bool:
    """One combinatorial 1-sphere: a single cycle, not a disjoint union."""
    if not is_disjoint_union_of_cycles(K):
        return False
    return _component_count(K) == 1


def _is_combinatorial_s2(K: SimplicialComplex) -> bool:
    r"""Connected closed combinatorial surface with $\chi = 2$.

    Used as the exact tutorial-scale test that a vertex link is $S^2$.
    Connectedness is required: a sphere disjoint from a torus has
    Euler characteristic 2 and circular vertex links, and is not $S^2$.
    """
    if K.dimension() != 2 or not K.is_pure(2):
        return False
    if _component_count(K) != 1:
        return False
    if not closed_surface_link_condition(K):
        return False
    return euler_characteristic(K) == 2


def _component_count(K: SimplicialComplex) -> int:
    """Components of the 1-skeleton, via its adjacency list."""
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
