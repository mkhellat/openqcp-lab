# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Finite abstract simplicial complexes.

The canonical store is a facet list: the inclusion-maximal simplices,
plus the vertex set (exactly the union of those facets). An isolated
vertex is stored as a singleton facet.

Two derived views are cached or rebuilt from that list:

- The full simplex set is the downward closure. Stars, links, and
  Euler counts need every face, not only the top-dimensional ones.
  `all_simplices` computes it on first use and caches it on the
  instance. The empty simplex is omitted, so Euler characteristic is
  the ordinary (unreduced) one.
- The adjacency list is the undirected 1-skeleton. Degree and cycle
  checks use it because those questions are local and the meshes here
  are sparse. A dense vertex-by-vertex matrix would store mostly
  zeros. The list is not written back into `facets`.

Half-edge / DCEL structures are not represented. Notebook 03 names
that mesh-native encoding; v1 `lib/` does not implement it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimplicialComplex:
    """Finite abstract simplicial complex.

    Canonical store: facet list (inclusion-maximal simplices). Use it
    when the complex is given by its top simplices — a triangulation,
    a filled simplex, a pinched identification.

    Derived views, not part of the stored data:

    - `all_simplices` / `faces_by_dim`: downward closure, cached on
      first use. Use these to count faces or to build a star or link.
    - `adjacency_list`: undirected 1-skeleton as `vertex -> neighbours`.
      Use this for degrees, walks, and $k=1$ cycle tests. Prefer it to
      a dense adjacency matrix: only existing edges are stored.

    `vertices` must equal the union of `facets`.
    """

    vertices: frozenset[int]
    facets: frozenset[frozenset[int]]

    def __post_init__(self) -> None:
        _validate_store(self.vertices, self.facets)
        # Lazy cache. None means "not computed"; the closure itself is
        # never None (the empty complex caches an empty frozenset).
        object.__setattr__(self, "_all_simplices_cache", None)

    def all_simplices(self) -> frozenset[frozenset[int]]:
        """Every nonempty face: the downward closure of the facet list.

        Why: a facet list does not contain edges or vertices as stored
        elements. Face counts, stars, and links need them.

        When: before Euler characteristic, and whenever the question
        is "which simplices contain this face?". For degree and cycle
        questions, use `adjacency_list` instead of scanning this set
        for edges by hand.

        The empty simplex is not included. Cached after the first call.
        """
        cached = self._all_simplices_cache
        if cached is None:
            cached = _downward_closure(self.facets)
            object.__setattr__(self, "_all_simplices_cache", cached)
        return cached

    def faces_by_dim(self) -> dict[int, frozenset[frozenset[int]]]:
        """Nonempty simplices grouped by dimension `|σ| - 1`.

        Keys run from 0 through `dimension()` with no gaps. The empty
        complex returns `{}`. Built from the cached full simplex set.
        """
        buckets: dict[int, set[frozenset[int]]] = {}
        for sigma in self.all_simplices():
            buckets.setdefault(len(sigma) - 1, set()).add(sigma)
        if not buckets:
            return {}
        top = max(buckets)
        return {dim: frozenset(buckets.get(dim, ())) for dim in range(top + 1)}

    def adjacency_list(self) -> dict[int, frozenset[int]]:
        """Undirected 1-skeleton: each vertex maps to its neighbours.

        Why an adjacency list: incidence of edges is a sparse, local
        relation. Building it once from the 1-simplices costs O(V + E)
        and answers degree and walk questions without another facet
        scan and without a dense V×V matrix.

        When: `vertex_degrees`, component walks, and the test that a
        pure 1-complex is a disjoint union of cycles. When the question
        needs triangles or tetrahedra, use the facet list or
        `all_simplices` instead. Isolated vertices map to an empty
        frozenset (they are singleton facets, not missing keys).
        """
        neighbours: dict[int, set[int]] = {v: set() for v in self.vertices}
        for edge in self.faces_by_dim().get(1, ()):
            a, b = tuple(edge)
            neighbours[a].add(b)
            neighbours[b].add(a)
        return {v: frozenset(neighbours[v]) for v in sorted(neighbours)}

    def star(self, sigma: frozenset[int]) -> frozenset[frozenset[int]]:
        """Cofaces of `sigma` in this complex, including `sigma` if it is a face.

        This is the set of simplices that contain `sigma`, not the
        smallest subcomplex containing them (faces of a coface that do
        not contain `sigma` are omitted).

        The empty set is not stored as a simplex. `star` of the empty
        frozenset still returns every nonempty simplex, since the empty
        set is contained in each of them. A non-face returns an empty
        frozenset.
        """
        return frozenset(tau for tau in self.all_simplices() if sigma <= tau)

    def link(self, sigma: frozenset[int]) -> SimplicialComplex:
        """Link of `sigma`, as a complex on the facet list of that link.

        The link is `{τ \\ σ | τ is a simplex and σ is a proper face of τ}`.
        Facets of the result are the inclusion-maximal sets in that
        collection. If `sigma` is not a face, the link is empty.

        Why a fresh facet list: the link is itself a complex, so later
        checks (circles, combinatorial 2-spheres) can call the same
        API. The parent complex is unchanged.
        """
        pieces: list[frozenset[int]] = []
        for tau in self.all_simplices():
            if sigma <= tau and tau != sigma:
                pieces.append(tau - sigma)
        maximal: set[frozenset[int]] = set()
        for piece in pieces:
            if any(piece < other for other in pieces):
                continue
            maximal.add(piece)
        if not maximal:
            return SimplicialComplex(frozenset(), frozenset())
        vertices = frozenset().union(*maximal)
        return SimplicialComplex(vertices, frozenset(maximal))

    def dimension(self) -> int:
        """Dimension of a largest facet, or -1 if there are no facets."""
        if not self.facets:
            return -1
        return max(len(facet) - 1 for facet in self.facets)

    def is_pure(self, dim: int | None = None) -> bool:
        """Whether every facet has the same dimension.

        With `dim=None`, that common dimension may be anything (the
        empty complex is pure). With `dim` set, every facet must have
        that dimension. The empty complex is pure of dimension -1 only.
        """
        if not self.facets:
            return dim is None or dim == -1
        dims = {len(facet) - 1 for facet in self.facets}
        if dim is None:
            return len(dims) == 1
        return dims == {dim}


def _validate_store(
    vertices: frozenset[int],
    facets: frozenset[frozenset[int]],
) -> None:
    if type(vertices) is not frozenset or any(type(v) is not int for v in vertices):
        raise TypeError("vertices must be a frozenset of int")
    if type(facets) is not frozenset or any(type(f) is not frozenset for f in facets):
        raise TypeError("facets must be a frozenset of frozenset")
    for facet in facets:
        if not facet:
            raise ValueError("facets must be nonempty")
        if any(type(v) is not int for v in facet):
            raise TypeError("each facet must be a frozenset of int")
    facet_list = list(facets)
    for i, left in enumerate(facet_list):
        for right in facet_list[i + 1 :]:
            if left < right or right < left:
                raise ValueError(
                    "facets must be inclusion-maximal; "
                    f"{set(left)} and {set(right)} are nested"
                )
    covered = frozenset().union(*facets) if facets else frozenset()
    if covered != vertices:
        raise ValueError(
            "vertices must be exactly the union of the facets "
            "(store an isolated vertex as a singleton facet)"
        )


def _downward_closure(
    facets: frozenset[frozenset[int]],
) -> frozenset[frozenset[int]]:
    simplices: set[frozenset[int]] = set()
    for facet in facets:
        ordered = list(facet)
        n = len(ordered)
        # Mask 0 is the empty simplex, which we do not store.
        for mask in range(1, 1 << n):
            simplices.add(
                frozenset(ordered[i] for i in range(n) if mask & (1 << i))
            )
    return frozenset(simplices)
