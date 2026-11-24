# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Shared combinatorics for the simplicial-manifolds tutorials."""

from .boundary import (
    boundary_matrix,
    boundary_square_is_zero,
    oriented_simplices,
)
from .checks import (
    closed_surface_link_condition,
    links_are_circles,
    vertex_links_are_spheres_s2,
)
from .complex import SimplicialComplex
from .invariants import (
    euler_characteristic,
    is_disjoint_union_of_cycles,
    vertex_degrees,
)

__all__ = [
    "SimplicialComplex",
    "boundary_matrix",
    "boundary_square_is_zero",
    "closed_surface_link_condition",
    "euler_characteristic",
    "is_disjoint_union_of_cycles",
    "links_are_circles",
    "oriented_simplices",
    "vertex_degrees",
    "vertex_links_are_spheres_s2",
]
