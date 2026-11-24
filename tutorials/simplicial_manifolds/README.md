# Simplicial complexes and combinatorial manifolds

A graduate series on when a finite simplicial $k$-complex is a
combinatorial manifold, for $k = 1, \ldots, 5$. The dimension ladder
is the point: local link checks, Euler characteristic, and what
recognition can and cannot decide. Applications (scientific
computing, QFT renormalizability, cryptology-adjacent hardness) stay
in one notebook at v1.

Smooth charts, Lie groups, and the exponential map stay upstream in
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/).
This series does not rebuild that theory. It starts from a facet list
and asks when the complex is a manifold.


# Notebooks

Generators (`build_*.py`) are the source of truth; each `.ipynb` is
rebuilt from its generator. This directory ships the shared `lib/`
first. Notebooks 00–06 land with the later parts of the series.

| ID | Notebook | Arc |
|----|----------|-----|
| 00 | `00_series_guide.ipynb` | Prerequisites, reading paths, complexity cheat-sheet |
| 01 | `01_simplicial_complexes.ipynb` | A. Facets, faces, stars, pure complexes |
| 02 | `02_combinatorial_manifolds.ipynb` | A. Link condition, including $k=1$ |
| 03 | `03_surfaces_and_euler.ipynb` | A. $k=2$, $V-E+F$, short QFT appetizer |
| 04 | `04_three_manifolds_recognition.ipynb` | B. $k=3$, local $S^2$ links vs recognition |
| 05 | `05_dimension_four_and_higher.ipynb` | B. $k=4,5$, undecidability landscape |
| 06 | `06_applications.ipynb` | C. SciComp, QFT, cryptology-adjacent hardness |

Notebook 06 is unsplit at v1 (three banners in one file). A later
split into separate SciComp and cryptology notebooks is v2 only.

Interactive figures live in sibling `*_demo.py` modules. Notebook
cells only launch them, with the source hidden (`_nbutil.code(...,
hide_input=True)`).


# Reading paths

1. **Geometry.** 00 → 01 → 02 → 03
2. **Computation.** 00 → 01 → 02 → 04 → 05
3. **Applications.** 00 → 01 → 02 → 06
   - QFT vignette: read 03 first
   - Crypto vignette: read 04–05 first


# Software

- `numpy`
- `matplotlib`
- `ipywidgets` (launch-only demos)

`lib/` is pure Python for complexes, invariants, and link checks. It
stores the 1-skeleton as an adjacency list, not a dense matrix.
`boundary_matrix` returns a SciPy CSR matrix when `scipy` imports, and
a dict-of-keys matrix otherwise, so the chain checks run either way.
No Classiq, no Open Quantum Safe, no SnapPy for v1.

`numpy` and `matplotlib` are pinned in the top-level
`requirements.txt`. Demos follow the same widget stack as
`tutorials/manifolds_lie_groups_quantum/cylinder_group_demo.py`.

Library check:

```bash
pytest tutorials/simplicial_manifolds/tests/ -v
```


# Topological data structures

The canonical store is a **facet list**. Other views are derived and
named where a notebook uses them.

| Structure | Where | Why / when |
|-----------|--------|------------|
| Facet list | `SimplicialComplex.facets` | Compact description: the inclusion-maximal simplices. This is the constructor input. |
| Full simplex set | `all_simplices`, `faces_by_dim` | Downward closure. Stars, links, and Euler counts need every dimension, not only the top. Computed on first use and cached on the instance. |
| Adjacency list | `adjacency_list` | Undirected 1-skeleton, `vertex -> neighbours`. Degrees, component walks, and the $k=1$ cycle test. Sparse: $O(V+E)$, which is the right cost for these meshes. Prefer this over a dense $V \times V$ matrix. |
| Incidence / boundary $\partial_k$ | `lib/boundary.py` (`oriented_simplices`, `boundary_matrix`, `boundary_square_is_zero`) | Sparse chain map $C_k \to C_{k-1}$. Basis vectors are increasing vertex ids, listed lexicographically. Use this for incidence signs, $\partial_{k-1}\partial_k = 0$, and rank-nullity set next to a face-count Euler characteristic. Degrees and walks stay on the adjacency list. SciPy CSR when `scipy` imports; dict-of-keys otherwise. |
| Half-edge / DCEL | Notebook 03 (minimal teaching subset only if a lab needs walks) | Oriented circulation, twin edges, surface mesh surgery. |
| Quad-Edge (Guibas–Stolfi) | Notebook 03; SciComp reprise in 06 | Primal **and** dual edge algebra on surfaces. **Not** the same as half-edge. |

Pick the structure for the **question** (set / graph-local / chain /
oriented walk / primal+dual). Dense matrices are contrast only.

`vertex_degrees` and `is_disjoint_union_of_cycles` go through
`adjacency_list()` because those questions are graph-local. Chain
identities go through `boundary_matrix`. The adjacency list is the
wrong tool for $\partial^2 = 0$; $\partial_k$ is the wrong tool for a
degree or a walk.

An isolated vertex is a singleton facet. `vertices` is exactly the
union of the facets.


# Library

| Module | Role |
|--------|------|
| `lib/complex.py` | `SimplicialComplex`: facets, faces, star, link, pure, dimension |
| `lib/boundary.py` | Oriented simplices and sparse $\partial_k$ (CSR or dict-of-keys) |
| `lib/invariants.py` | Euler characteristic, degrees, disjoint unions of cycles |
| `lib/checks.py` | Closed 1-manifold check, closed-surface link condition, vertex links vs $S^2$ |
| `_nbutil.py` | `md` / `code` cell helpers, including `hide_input=True` |

`vertex_links_are_spheres_s2` checks that each vertex link is a
combinatorial 2-sphere. It is a local test for tutorial-scale
complexes, not a 3-sphere recognizer. Limits are in that function's
docstring.


# GNU GPL v3+

Copyright (C) 2026 Mohammadreza Khellat GNU GPL v3+

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

See also https://www.gnu.org/licenses/gpl.html
