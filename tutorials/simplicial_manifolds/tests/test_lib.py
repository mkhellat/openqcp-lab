# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Tests for the simplicial-manifolds combinatorics library.

The hollow tetrahedron (boundary of the 3-simplex) is the combinatorial
2-sphere: Euler characteristic 2 and every vertex link a circle. The
predicate `vertex_links_are_spheres_s2` asks a different question — it
is the local closed 3-manifold check — so it is false on that surface
and true on the boundary of the 4-simplex, whose vertex links are
copies of the tetrahedron boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERIES_DIR = Path(__file__).resolve().parents[1]
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from lib import (  # noqa: E402
    SimplicialComplex,
    closed_surface_link_condition,
    euler_characteristic,
    is_disjoint_union_of_cycles,
    links_are_circles,
    vertex_degrees,
    vertex_links_are_spheres_s2,
)


def from_facets(facets: list[list[int]]) -> SimplicialComplex:
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen) if frozen else frozenset()
    return SimplicialComplex(vertices, frozen)


def simplex_boundary(dim: int) -> SimplicialComplex:
    """Boundary of the standard `dim`-simplex on vertices `0 .. dim`."""
    vertices = list(range(dim + 1))
    facets = [[v for v in vertices if v != omit] for omit in vertices]
    return from_facets(facets)


def test_filled_triangle_euler_characteristic() -> None:
    triangle = from_facets([[0, 1, 2]])
    assert triangle.dimension() == 2
    assert triangle.is_pure(2) is True
    by_dim = triangle.faces_by_dim()
    assert len(by_dim[0]) == 3
    assert len(by_dim[1]) == 3
    assert len(by_dim[2]) == 1
    assert len(triangle.all_simplices()) == 7
    assert euler_characteristic(triangle) == 1
    assert triangle.star(frozenset({0, 1})) == frozenset(
        {frozenset({0, 1}), frozenset({0, 1, 2})}
    )
    edge_link = triangle.link(frozenset({0, 1}))
    assert edge_link.vertices == frozenset({2})
    assert edge_link.facets == frozenset({frozenset({2})})


def test_hollow_tetrahedron_boundary_is_combinatorial_s2() -> None:
    sphere = simplex_boundary(3)
    assert sphere.dimension() == 2
    assert sphere.is_pure(2) is True
    assert euler_characteristic(sphere) == 2
    assert closed_surface_link_condition(sphere) is True
    # The surface is itself S^2. Its vertex links are circles, so the
    # 3-manifold predicate (links are S^2) does not hold.
    assert vertex_links_are_spheres_s2(sphere) is False
    vertex_link = sphere.link(frozenset({0}))
    assert euler_characteristic(vertex_link) == 0
    assert is_disjoint_union_of_cycles(vertex_link) is True
    assert links_are_circles(vertex_link) is True


def test_vertex_links_are_s2_on_four_simplex_boundary() -> None:
    three_sphere = simplex_boundary(4)
    assert three_sphere.dimension() == 3
    assert euler_characteristic(three_sphere) == 0
    assert vertex_links_are_spheres_s2(three_sphere) is True
    link = three_sphere.link(frozenset({0}))
    assert euler_characteristic(link) == 2
    assert closed_surface_link_condition(link) is True
    # A filled tetrahedron is a ball: vertex links are disks, not S^2.
    ball = from_facets([[0, 1, 2, 3]])
    assert vertex_links_are_spheres_s2(ball) is False


def test_cycle_c5_is_disjoint_union_of_cycles() -> None:
    cycle = from_facets([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    assert is_disjoint_union_of_cycles(cycle) is True
    assert links_are_circles(cycle) is True
    assert vertex_degrees(cycle) == {i: 2 for i in range(5)}

    union = from_facets(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [6, 7], [7, 5]]
    )
    assert is_disjoint_union_of_cycles(union) is True
    assert links_are_circles(union) is True


def test_path_of_length_2_is_not_a_cycle_union() -> None:
    path = from_facets([[0, 1], [1, 2]])
    assert is_disjoint_union_of_cycles(path) is False
    assert links_are_circles(path) is False
    assert vertex_degrees(path) == {0: 1, 1: 2, 2: 1}


def test_pinched_triangles_fail_closed_surface_link() -> None:
    pinched = from_facets([[0, 1, 2], [0, 3, 4]])
    assert pinched.is_pure(2) is True
    assert closed_surface_link_condition(pinched) is False
    shared = pinched.link(frozenset({0}))
    assert shared.facets == frozenset({frozenset({1, 2}), frozenset({3, 4})})
    assert is_disjoint_union_of_cycles(shared) is False


def test_adjacency_list_shape_and_degrees() -> None:
    path = from_facets([[0, 1], [1, 2]])
    adj = path.adjacency_list()
    assert list(adj) == [0, 1, 2]
    assert all(isinstance(neighbours, frozenset) for neighbours in adj.values())
    assert adj == {
        0: frozenset({1}),
        1: frozenset({0, 2}),
        2: frozenset({1}),
    }
    assert vertex_degrees(path) == {v: len(adj[v]) for v in adj}

    triangle = from_facets([[0, 1, 2]])
    assert triangle.adjacency_list() == {
        0: frozenset({1, 2}),
        1: frozenset({0, 2}),
        2: frozenset({0, 1}),
    }
