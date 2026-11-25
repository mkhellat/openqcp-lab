# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Gallery surfaces and the half-edge / Quad-Edge teaching sketch.

χ is the face-count integer. The sketches are checked only for the
operator identities the notebook traces: next³, twin², Rot⁴, and
Lnext reproducing next. RP² must refuse a classical twin.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERIES_DIR = Path(__file__).resolve().parents[1]
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from euler_surface_demo import (  # noqa: E402
    classification_label,
    closed_surface_link_condition,
    coherently_orientable,
    degree_defects,
    disjoint_union,
    euler_characteristic,
    face_counts,
    orientation_attempt,
    real_projective_plane,
    tetrahedron_boundary,
    torus_grid,
    torus_seven,
)
from lib import SimplicialComplex, boundary_matrix  # noqa: E402
from surface_tds_demo import (  # noqa: E402
    TeachingMeshError,
    build_halfedge,
    build_quadedge,
    trace_report,
)


def _assert_closed_triangulation(K, vertices, edges, faces, chi, orientable):
    assert face_counts(K) == (vertices, edges, faces)
    assert euler_characteristic(K) == chi
    assert 3 * faces == 2 * edges
    assert closed_surface_link_condition(K) is True
    assert coherently_orientable(K) is orientable
    assert sum(degree_defects(K).values()) == 6 * chi


def test_gallery_surfaces_match_the_classical_invariants():
    _assert_closed_triangulation(tetrahedron_boundary(), 4, 6, 4, 2, True)
    _assert_closed_triangulation(torus_grid(), 9, 27, 18, 0, True)
    _assert_closed_triangulation(torus_seven(), 7, 21, 14, 0, True)
    _assert_closed_triangulation(real_projective_plane(), 6, 15, 10, 1, False)
    assert classification_label(tetrahedron_boundary()) == "orientable, genus 0, χ=2"
    assert classification_label(torus_grid()) == "orientable, genus 1, χ=0"
    assert classification_label(torus_seven()) == "orientable, genus 1, χ=0"
    assert (
        classification_label(real_projective_plane())
        == "non-orientable, 1 crosscap, χ=1"
    )


def test_disjoint_spheres_add_euler_characteristic():
    pair = disjoint_union(tetrahedron_boundary(), tetrahedron_boundary())
    assert euler_characteristic(pair) == 4
    assert closed_surface_link_condition(pair) is True
    assert classification_label(pair) == "2 components, χ=4"


def test_boundary_columns_match_face_counts():
    sphere = tetrahedron_boundary()
    vertices, edges, faces = face_counts(sphere)
    assert boundary_matrix(sphere, 0).shape == (0, vertices)
    assert boundary_matrix(sphere, 1).shape == (vertices, edges)
    assert boundary_matrix(sphere, 2).shape == (edges, faces)


def test_halfedge_and_quadedge_on_the_sphere():
    sphere = tetrahedron_boundary()
    quad = build_quadedge(sphere)
    half = quad.half
    assert len(half.edges) == 12
    assert len(quad.records) == 24
    for index, edge in enumerate(half.edges):
        assert half.edges[edge.twin].twin == index
        assert half.edges[edge.twin].origin == edge.dest
        assert len(half.face_ring(index)) == 3
        ring = half.vertex_ring(index)
        assert len(ring) == 3
        assert all(half.edges[step].origin == edge.origin for step in ring)
    onext_image = [record.onext for record in quad.records]
    assert sorted(onext_image) == list(range(len(quad.records)))
    for index, record in enumerate(quad.records):
        cycle = quad.rot_cycle(index)
        assert len(cycle) == 4
        assert quad.records[quad.records[record.rot].rot].origin == record.dest
        assert len(quad.onext_ring(index)) == 3


def test_lnext_matches_halfedge_next_on_the_sphere():
    quad = build_quadedge(tetrahedron_boundary())
    for half_index, edge in enumerate(quad.half.edges):
        record = quad.record_of_half[half_index]
        assert quad.lnext(record) == quad.record_of_half[edge.nxt]


def test_torus_onext_lengths_are_vertex_degree_and_triangle():
    quad = build_quadedge(torus_grid())
    primal = {
        len(quad.onext_ring(index))
        for index, record in enumerate(quad.records)
        if record.primal
    }
    dual = {
        len(quad.onext_ring(index))
        for index, record in enumerate(quad.records)
        if not record.primal
    }
    assert primal == {6}
    assert dual == {3}


def test_rp2_refuses_a_classical_halfedge():
    plane = real_projective_plane()
    attempt = orientation_attempt(plane)
    assert attempt.closed is True
    assert attempt.coherent is False
    assert attempt.conflict_edge is not None
    edge = frozenset(attempt.conflict_edge)
    assert edge in plane.faces_by_dim()[1]
    try:
        build_halfedge(plane)
    except TeachingMeshError as err:
        assert "does not invent twins" in str(err)
    else:
        raise AssertionError("RP² built a half-edge sketch")


def test_boundary_triangle_has_no_twin():
    disk = SimplicialComplex(
        frozenset({0, 1, 2}),
        frozenset({frozenset({0, 1, 2})}),
    )
    try:
        build_halfedge(disk)
    except TeachingMeshError as err:
        assert "exactly two triangles" in str(err)
    else:
        raise AssertionError("a disk built twins")


def test_trace_report_names_the_operators():
    text = trace_report()
    assert "Rot cycle" in text
    assert "Lnext reproduces next: True" in text
    assert "does not rewire" in text
    assert "does not invent twins" in text
    assert "primal [6]" in text or "primal [6]" in text.replace(" ", "")
    assert "dual [3]" in text or "dual [3]" in text.replace(" ", "")
