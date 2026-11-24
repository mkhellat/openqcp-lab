#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build the simplicial-manifolds series guide (source of truth).

Run from the repository root or from this directory:

    python tutorials/simplicial_manifolds/build_00_series_guide.py

The generated notebook is ``00_series_guide.ipynb`` beside this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name("00_series_guide.ipynb")


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(r"""
# Simplicial complexes and combinatorial manifolds

A finite simplicial complex is a finite vertex set and a collection of
simplices closed under faces. This series asks when that complex is a
combinatorial manifold of dimension $k$, for $k = 1, \ldots, 5$, and
what a positive answer decides about homeomorphism type.

This notebook is the map: prerequisites, three reading paths, a
complexity cheat-sheet, and the data structures later notebooks name.
It does not build a complex, and it does not launch a figure.

> **Tier‑C pointer.** Smooth charts, tangent spaces, Lie groups, and
> the exponential map stay in the manifolds foundations notebook,
> [`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/)
> (Part I for the atlas language; Part II‑C for the Tier‑C invitations:
> Calabi–Yau, $G_2$, twistors, lattices). This series lives at
> `tutorials/simplicial_manifolds/`. It starts from a facet list and
> does not rebuild that smooth theory.
"""))

    cells.append(md(r"""
## Prerequisites

- Foundations Part I, if "manifold" still means a smooth atlas. The
  link above is the upstream notebook. Combinatorial and PL language
  begins here; the careful split among combinatorial, PL, and
  topological is notebook 02.
- Finite sets and graphs. Euler characteristic enters as an alternating
  face count; notebook 03 writes $V - E + F = \chi$ for surfaces.
- The series bounds are part of the subject: handmade examples,
  local checks, and citations. A SnapPy-scale census, a production
  $S^3$ recognizer, and a cryptographic protocol are out of scope.

## Learning objectives

1. Choose a reading path, and name which notebook owns $k=1$,
   surfaces, $S^3$ recognition, dimension $\ge 4$, and the applications.
2. Read the cheat-sheet as two columns — a local combinatorial check,
   and a global recognition or homeomorphism problem — with Schleimer's
   theorem on $S^3$ kept as membership in NP.
3. Name the facet list as the canonical store, the adjacency list as
   the 1-skeleton, and notebook 03 as the place that names half-edges.
"""))

    cells.append(md(r"""
## Notebooks

| ID | Notebook | What it owns |
|----|----------|----------------|
| 00 | `00_series_guide.ipynb` | This map |
| 01 | `01_simplicial_complexes.ipynb` | Facets, faces, stars, pure complexes. Stops before the manifold criterion |
| 02 | `02_combinatorial_manifolds.ipynb` | Link condition, including closed 1-manifolds ($k=1$) |
| 03 | `03_surfaces_and_euler.ipynb` | $k=2$, $V-E+F$, a short QFT appetizer |
| 04 | `04_three_manifolds_recognition.ipynb` | $k=3$: local links $\cong S^2$ versus recognition |
| 05 | `05_dimension_four_and_higher.ipynb` | $k=4,5$: undecidability landscape |
| 06 | `06_applications.ipynb` | SciComp, QFT renormalizability, cryptology-adjacent hardness |

Notebook 06 is unsplit at v1 (three banners in one file). Generators
(`build_*.py`) are the source of truth; rebuild a notebook by running
its generator. Interactive figures, when a later notebook has them,
live in a sibling `*_demo.py` and are launched from a hidden cell.
Software and the `lib/` layout are in `README.md` in this folder.
This guide has no demo.
"""))

    cells.append(md(r"""
## Reading paths

1. **Geometry.** 00 → 01 → 02 → 03
2. **Computation.** 00 → 01 → 02 → 04 → 05
3. **Applications.** 00 → 01 → 02 → 06
   - QFT vignette: read 03 first
   - Crypto vignette: read 04–05 first

Geometry stops at surfaces. Computation uses the same link language
for recognition in dimensions $3$, $4$, and higher. Applications defers
03 and 04–05 until the vignette needs them: the QFT vignette in 06
builds on the two-dimensional appetizer in 03, and the cryptology
vignette builds on the complexity table as stated in 04 and 05.
"""))

    cells.append(code(r"""
# Route card only. This notebook does not build a complex.
print("Geometry:     00 -> 01 -> 02 -> 03")
print("Computation:  00 -> 01 -> 02 -> 04 -> 05")
print("Applications: 00 -> 01 -> 02 -> 06")
print("QFT vignette: read 03 first")
print("Crypto vignette: read 04-05 first")
"""))

    cells.append(md(r"""
## Complexity cheat-sheet

Fix a dimension $k \ge 1$ and a finite simplicial complex $K$. The
series keeps two decisions apart.

**Local.** Is $K$ a combinatorial $k$-manifold? The working criterion,
stated in notebook 02, is: $K$ is pure of dimension $k$, and the link
of every nonempty face is a piecewise-linear sphere of the
complementary dimension. For a closed complex at $k=1$ that criterion
is a graph condition on the adjacency list: every vertex has degree
$2$, so $K$ is a disjoint union of cycles.

**Global.** When the local check passes, which manifold is the
geometric realization up to homeomorphism, and what does it cost to
decide?

| $k$ | Local "is it a $k$-manifold?" | Global recognition / homeomorphism |
|-----|-------------------------------|--------------------------------------|
| $1$ | Easy (degrees / cycles) | Easy |
| $2$ | Links + Euler / orientability | Classification: polynomial-scale story |
| $3$ | Links $\cong S^2$ | $S^3$ recognition $\in$ NP (Schleimer), not NP-hard; full homeomorphism harder |
| $4$ | Local PL checks still make sense | Homeomorphism undecidable (Markov-type); $S^4$ subtleties called out |
| $\ge 5$ | Local combinatorial checks | Sphere recognition undecidable (Novikov-type) |

Schleimer (2004) is the $k=3$ entry for the 3-sphere: recognition lies
in NP. NP-hard is a different predicate, and that theorem does not
assert it. The $k=4$ entry is undecidability of homeomorphism
(Markov-type). The $\ge 5$ entry is undecidability of sphere
recognition (Novikov-type). Notebooks 04 and 05 own the citations,
the hypotheses, and the correction of the slide from Schleimer's NP
membership to a claim that dimension $4$ is NP-hard. The table is the
index.

<details>
<summary><b>What 04 and 05 add</b> (click to expand)</summary>

Notebook 04 separates a vertex link $\cong S^2$ from a global
homeomorphism to $S^3$, and it records Schleimer's NP membership with
the hypotheses the paper uses. Notebook 05 records why dimension 4 is
special, the Markov-type result on homeomorphism of 4-manifolds, and
the Novikov-type result on sphere recognition in high dimensions.

</details>
"""))

    cells.append(md(r"""
## Topological data structures

Each later notebook says why its cell uses one representation. The
series lock is short:

| Structure | Role |
|-----------|------|
| Facet list | Canonical store: the inclusion-maximal simplices. This is the constructor input of `SimplicialComplex`. |
| Adjacency list | Undirected 1-skeleton, vertex $\mapsto$ neighbours. Degrees and the $k=1$ cycle test. Sparse, so it is the structure those checks use, ahead of a dense $V \times V$ matrix. |
| Half-edge (DCEL) | Named in notebook 03 for oriented surface meshes: walks, opposite edges, surgery. v1 `lib/` does not implement it. |

The full simplex set (downward closure) and a boundary operator
$\partial_k$ are derived views. A notebook calls them when it counts
faces or writes a chain. They are not a second canonical store.
"""))

    cells.append(md(r"""
## Summary

**Takeaway.** Smooth atlases stay in the manifolds foundations
notebook. This directory, `tutorials/simplicial_manifolds/`, is the
combinatorial series for $k = 1, \ldots, 5$.

- Geometry reads through surfaces (03). Computation reads through
  recognition (04, then 05). Applications (06) reads 03 before the QFT
  vignette and 04–05 before the cryptology vignette.
- The local column and the global column are different problems.
  $S^3$ recognition is in NP (Schleimer), which is not NP-hard.
  Dimension $4$ is a Markov-type undecidability statement; dimension
  $\ge 5$ is a Novikov-type statement about spheres. Citations sit in
  notebooks 04 and 05.
- Complexes are stored as facet lists. The 1-skeleton is an adjacency
  list. Half-edges are named in notebook 03.
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
