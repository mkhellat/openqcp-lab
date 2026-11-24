# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Numerical and graph-theoretic invariants of a simplicial complex.

Euler characteristic reads the full simplex set (every dimension).
Degrees and the cycle test read the adjacency list of the 1-skeleton
once, then walk. They do not scan the facet list per vertex.
"""

from __future__ import annotations

from .complex import SimplicialComplex


def euler_characteristic(K: SimplicialComplex) -> int:
    r"""Ordinary Euler characteristic $\sum_d (-1)^d f_d$.

    `f_d` is the number of nonempty $d$-simplices from the downward
    closure, not the number of facets alone. The empty simplex is
    omitted, so a filled triangle has $\chi = 1$ and the boundary of a
    tetrahedron has $\chi = 2$.
    """
    return sum(
        ((-1) ** dim) * len(simplices)
        for dim, simplices in K.faces_by_dim().items()
    )


def vertex_degrees(K: SimplicialComplex) -> dict[int, int]:
    """Degree of each vertex in the 1-skeleton.

    Read off `adjacency_list()` (neighbour-set sizes). This is not a
    dense row-sum, and it does not rescan facets.
    """
    return {v: len(neighbours) for v, neighbours in K.adjacency_list().items()}


def is_disjoint_union_of_cycles(K: SimplicialComplex) -> bool:
    """Whether `K` is a pure 1-complex and a disjoint union of cycles.

    The test builds one adjacency list, requires every degree to be 2,
    then walks each component along that list until it closes. A finite
    undirected graph with all degrees 2 is a disjoint union of cycles;
    the walk is the explicit check, and it rejects a component that
    fails to close or is shorter than 3 (a simplicial cycle has no
    loops or digons).

    A filled triangle is not accepted: its 1-skeleton is a cycle, but
    the complex is not pure of dimension 1. Several disjoint cycles are
    accepted. The empty complex and a path are rejected.
    """
    if K.dimension() != 1 or not K.is_pure(1):
        return False
    adj = K.adjacency_list()
    if not adj:
        return False
    if any(len(neighbours) != 2 for neighbours in adj.values()):
        return False
    return _components_are_cycles(adj)


def _components_are_cycles(adj: dict[int, frozenset[int]]) -> bool:
    """Walk each component of a degree-2 graph and require a cycle.

    `adj` is the adjacency list. Every vertex is expected to have two
    neighbours; a vertex that does not is a failed walk, not a second
    facet scan.
    """
    unseen = set(adj)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        neighbours = adj[start]
        if len(neighbours) != 2:
            return False
        prev = start
        curr = min(neighbours)
        walked = 1
        while curr != start:
            if curr not in unseen:
                return False
            unseen.remove(curr)
            walked += 1
            onward = adj[curr] - {prev}
            if len(onward) != 1:
                return False
            prev, curr = curr, min(onward)
        if walked < 3 or prev not in adj[start]:
            return False
    return True
