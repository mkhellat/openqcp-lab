# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Tests for oriented simplices and sparse boundary matrices.

Positive orientation is the increasing vertex-id order. Signs live in
∂ only: a face obtained by deleting the vertex in position i (0-based)
of that increasing tuple has coefficient (-1)^i. The hollow
tetrahedron is a pure 2-complex, so there is no ∂_3 to compose; the
identity under test is ∂_1 ∂_2 = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SERIES_DIR = Path(__file__).resolve().parents[1]
if str(_SERIES_DIR) not in sys.path:
    sys.path.insert(0, str(_SERIES_DIR))

from lib import (  # noqa: E402
    SimplicialComplex,
    boundary_matrix,
    boundary_square_is_zero,
    oriented_simplices,
)
from lib.boundary import DictOfKeysMatrix  # noqa: E402


def _dense(matrix: object) -> np.ndarray:
    array = matrix.toarray()  # type: ignore[attr-defined]
    return np.asarray(array)


def from_facets(facets: list[list[int]]) -> SimplicialComplex:
    frozen = frozenset(frozenset(facet) for facet in facets)
    vertices = frozenset().union(*frozen) if frozen else frozenset()
    return SimplicialComplex(vertices, frozen)


def simplex_boundary(dim: int) -> SimplicialComplex:
    """Boundary of the standard `dim`-simplex on vertices `0 .. dim`."""
    vertices = list(range(dim + 1))
    facets = [[v for v in vertices if v != omit] for omit in vertices]
    return from_facets(facets)


def test_filled_triangle_boundary_signs_and_square() -> None:
    triangle = from_facets([[0, 1, 2]])
    assert oriented_simplices(triangle, 0) == [(0,), (1,), (2,)]
    assert oriented_simplices(triangle, 1) == [(0, 1), (0, 2), (1, 2)]
    assert oriented_simplices(triangle, 2) == [(0, 1, 2)]

    # Columns are edges (0,1), (0,2), (1,2). Rows are vertices 0, 1, 2.
    # ∂[0,1] = [1] - [0], and likewise for the other edges.
    expected_d1 = np.array(
        [
            [-1, -1, 0],
            [1, 0, -1],
            [0, 1, 1],
        ]
    )
    assert np.array_equal(_dense(boundary_matrix(triangle, 1)), expected_d1)

    # ∂[0,1,2] = [1,2] - [0,2] + [0,1].
    expected_d2 = np.array([[1], [-1], [1]])
    assert np.array_equal(_dense(boundary_matrix(triangle, 2)), expected_d2)
    assert boundary_square_is_zero(triangle, 2) is True


def test_unaugmented_vertex_boundary_has_no_rows() -> None:
    """The empty simplex is not stored, so ∂_0 lands in the zero module."""
    triangle = from_facets([[0, 1, 2]])
    assert boundary_matrix(triangle, 0).shape == (0, 3)
    assert boundary_square_is_zero(triangle, 1) is True


def test_hollow_tetrahedron_boundary_square_is_zero() -> None:
    sphere = simplex_boundary(3)
    assert sphere.dimension() == 2
    assert oriented_simplices(sphere, 3) == []
    assert oriented_simplices(sphere, 2) == [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    # Rows: edges in lex order. Columns: the four increasing triangles.
    # Each column is the deletion boundary of that tuple, not a global
    # flip that would make the four facets a coherently oriented cycle.
    expected_d2 = np.array(
        [
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, -1, -1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
        ]
    )
    assert np.array_equal(_dense(boundary_matrix(sphere, 2)), expected_d2)
    assert boundary_square_is_zero(sphere, 2) is True
    assert not np.all(_dense(boundary_matrix(sphere, 2)) == 0)


def test_sparse_backend_follows_scipy_availability() -> None:
    triangle = from_facets([[0, 1, 2]])
    matrix = boundary_matrix(triangle, 1)
    try:
        import scipy.sparse as sparse
    except ImportError:
        assert isinstance(matrix, DictOfKeysMatrix)
    else:
        assert sparse.isspmatrix_csr(matrix)


def test_dict_of_keys_fallback_squares_to_zero() -> None:
    """The pure-Python operator agrees with ∂_1 ∂_2 = 0 on the triangle."""
    d1 = DictOfKeysMatrix(
        {
            (0, 0): -1,
            (0, 1): -1,
            (1, 0): 1,
            (1, 2): -1,
            (2, 1): 1,
            (2, 2): 1,
        },
        (3, 3),
    )
    d2 = DictOfKeysMatrix({(0, 0): 1, (1, 0): -1, (2, 0): 1}, (3, 1))
    product = d1 @ d2
    assert product.shape == (3, 1)
    assert product.entries == {}
