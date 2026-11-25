#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build the manifolds series guide (source of truth).

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_00_series_guide.py

The generated notebook is ``00_series_guide.ipynb`` beside this file.
The opening essay, prerequisites, learning objectives, depth map,
closing summary, and bookshelf are the text from the single notebook
this series replaced. This builder does not add a new development of
the geometry. A few phrases that named a section of that file now
name the series notebook that holds the section.
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
# Smooth Manifolds, Lie Groups, and the Exponential Map

Many sets that appear in mathematics and theoretical physics are not
vector spaces: if $p$ and $q$ are points of the set, $p+q$ or
$\alpha p$ need not lie in the set, and need not be meaningful.
Concrete examples are in notebook 01.

Such sets arise as domains of points on which geometry and physical
law are local and differential: they concern how structure and
quantities vary among neighbouring points. What the points *are*
depends on the subject — for example:

- in **mathematics**: the domains that forced the invention of
  manifolds — curved surfaces and higher-dimensional geometries that
  admit calculus only through local coordinates like open sets in
  $\mathbb{R}^n$ (not a single global vector-space chart), directions
  up to scaling, and continuous groups of transformations — so that
  derivatives, curvature, and differential equations can be stated
  intrinsically on those sets;
- in **special and general relativity**: **events**; the domain is
  spacetime itself, and the independent variables need not single out
  time as a privileged parameter of the set;
- in **quantum mechanics**, **quantum information**, and **quantum
  computing**: pure states up to phase, mixed states, and unitary
  gates (finite transformations), as well as control parameters of
  Hamiltonians;
- in **statistical physics** and **quantum statistical physics**:
  macroscopic states of ensembles, and parameters of families of
  probability distributions or Gibbs states;
- in **twistor theory**: points of the complex projective set
  $\mathbb{PT}=\mathbb{CP}^{3}$ that reorganise null structure in
  spacetime (incidence dictionary in notebook 06, §C3);
- in **post-quantum cryptography and cryptology**: geometric carriers
  such as elliptic curves and lattice quotients that underwrite hard
  problems (treated here only as a map; depth belongs in a dedicated
  notebook).

The language of that local variation is multivariable calculus: smooth
real-valued functions, smooth curves, and derivatives, written in local
coordinates. Manifolds were introduced in mathematics precisely to
carry that language on a set of points that is only locally like an
open set in $\mathbb{R}^n$, when a single global vector-space domain is
the wrong description of the points.

A **smooth manifold** is a set equipped with local coordinates in open
subsets of $\mathbb{R}^d$ so that those calculus notions are defined
via ordinary calculus in $\mathbb{R}^d$ (details in notebook 02). Differentiating
a curve through a point $p$ yields an element of a vector space
$T_pM$, the tangent space at $p$ — not an operation that adds points
of the set itself.

If a smooth manifold $G$ is also a group and the group operations are
smooth, $G$ is a **Lie group**. Its **Lie algebra** is the tangent
space at the identity, $\mathfrak{g}=T_eG$, which is a vector space.
For the matrix groups used in this notebook, the exponential series
$\exp(X)=\sum_{k\ge 0}X^k/k!$ maps $\mathfrak{g}$ into $G$, and
$t\mapsto\exp(tX)$ is the smooth curve in $G$ with derivative $X$ at
$t=0$.

The series develops that language with pictures and code, then
applications at depths A / B / C. It is a foundations module for the
geometry of states, transformations, and parameters used across the
tutorials and tools here. Part III bridges in particular to product
formulas and the coupled-oscillator Hamiltonian-simulation notebooks
(module 05); it does not replace them.

This notebook is the map: prerequisites, reading paths, the depth
map, the closing summary, the bookshelf, and links to the graduate
off-ramps. It does not build charts, and it does not launch a figure.
Notebooks 01–07 hold the labs. The old single file
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb` only
redirects here.
"""))

    cells.append(md(r"""
## Prerequisites

- Multivariable calculus (partial derivatives, chain rule).
- Linear algebra over \(\mathbb{R}\) and \(\mathbb{C}\) (matrices,
  eigenvalues, inner products).
- Undergraduate quantum mechanics: state vectors, Hermitian operators,
  unitary time evolution \(U(t)=\exp(-iHt/\hbar)\) at the level of a
  first course.

## Learning objectives

1. Explain, in plain language, what a smooth manifold is and why
   transition maps must be \(C^\infty\).
2. Construct charts on the circle and check transition smoothness in
   code.
3. State what a Lie group and its Lie algebra are, and compute the
   matrix exponential of generators in \(\mathfrak{gl}(n,\mathbb{C})\)
   and \(\mathfrak{u}(n)\).
4. Work through Tier‑A applications (Bloch sphere, projective state
   space, unitaries, symplectic Gaussians, statistical manifolds) with
   background, equations, executable checks, and application notes.
5. Read Tier‑B/C vignettes with enough background *and* geometry to know
   *what* the manifold is, *why it matters*, and where to go next
   (twistor incidence in II‑C §C3; PQC/crypto off‑ramp in §C4).
"""))

    cells.append(md(r"""
## Notebooks

| ID | Notebook | What it owns |
|----|----------|----------------|
| 00 | `00_series_guide.ipynb` | This map, closing summary, bookshelf |
| 01 | `01_extrinsic_sphere_and_dg_toolset.ipynb` | (M1)–(M3), (P1)–(P3), cylinder demo |
| 02 | `02_charts_atlases_tangent.ipynb` | Part I §2: atlas, charts, transitions |
| 03 | `03_lie_groups_exponential_map.ipynb` | Lie groups, Lie algebra, exponential map |
| 04 | `04_tier_a_applications.ipynb` | Part II‑A full labs |
| 05 | `05_tier_b_light_labs.ipynb` | Part II‑B light labs |
| 06 | `06_tier_c_offramps.ipynb` | Part II‑C thin map to the graduate notebooks |
| 07 | `07_bridge_product_formulas.ipynb` | Part III bridge to product formulas / module 05 |

Notebooks 00–07 are the series. Generators
(`build_00_series_guide.py` through
`build_07_bridge_product_formulas.py`) are the source of truth;
rebuild a notebook by running its generator. `build_notebook.py` is
deprecated and only writes the redirect at the old filename.
Interactive figures live in a sibling `*_demo.py` and are launched
from a hidden cell (`_nbutil.code(..., hide_input=True)`). The
cylinder demo is launched from notebook 01. Software requirements
are in `README.md` in this folder. This guide has no demo.

**Citation rule.** Every section of notebooks 01–07 names textbooks
or articles a reader can open. We do not invent references. The
compiled bookshelf is in this guide; the same works are named again
beside the section that uses them. Anchors are also listed in
`README.md`.
"""))

    cells.append(md(r"""
## Reading paths

1. **Geometry core.** 00 → 01 → 02 → 03
2. **Applications.** 00 → 04 → 05
3. **Off-ramps.** 00 → 06
4. **Bridge.** 00 → 07

Geometry core is Part I: the extrinsic sphere and the
differential-geometry toolset (01), charts, atlases, and tangent
spaces (02), then Lie groups and the exponential map (03).
Applications are the coded labs: Tier A in full (04), then Tier B
light labs (05). Off-ramps are the Tier‑C geometry-depth map (06).
The bridge (07) is Part III: product formulas and the
coupled-oscillator Hamiltonian-simulation notebooks (module 05); it
does not replace them.
"""))

    cells.append(code(r"""
# Route card only. This notebook does not build a chart.
print("Geometry core: 00 -> 01 -> 02 -> 03")
print("Applications:  00 -> 04 -> 05")
print("Off-ramps:     00 -> 06")
print("Bridge:        00 -> 07")
"""))

    cells.append(md(r"""
## Depth map

Each case is written as **background → geometry (with equations) →
where it shows up**, then code or a reading off-ramp. Tier A is
notebook 04, Tier B notebook 05, and Tier C notebook 06.

| Tier | Topics | Code depth |
|------|--------|------------|
| A | Bloch / $\mathbb{CP}^{1}$; $\mathbb{CP}^{d-1}$ & density operators; $\mathrm{U}(2^{n})$; $\mathrm{Sp}(2n,\mathbb{R})$; statistical manifolds | Full labs |
| B | Lorentzian Minkowski; $U(1)$ bundles / Berry phase | Light code |
| C | Calabi–Yau ($c_1=0$); $G_{2}$ (parallel $3$-form); twistors (incidence); elliptic curves & lattice tori | Geometry-depth + schematics; PQC/crypto off-ramp |

**Tier C (all of C1–C4).** Each section declares a working setup and
writes at least one explicit equation or dictionary (Yau / $c_1=0$;
model $\varphi_0$ on $\mathbb{R}^{7}$; $\omega^{A}=ix^{AA'}\pi_{A'}$;
Weierstrass $\Delta\neq 0$ and $\mathbb{R}^{n}/\Lambda$). Twistor
graduate notebook: [`../twistor_theory`](../twistor_theory/). PQC
engineering: [`../post_quantum_cryptography`](../post_quantum_cryptography/).

**Graduation:** PQC / cryptology continues in
[`../post_quantum_cryptography`](../post_quantum_cryptography/)
(NIST FIPS, hybrid TLS RFCs, soft-gated Open Quantum Safe). CY / $G_2$
holonomy deepen continues in
[`../special_holonomy`](../special_holonomy/).
"""))

    cells.append(md(r"""
## Links

Graduate off-ramps (Tier C, notebook 06):

- Twistors → [`../twistor_theory`](../twistor_theory/)
- CY / $G_2$ (holonomy deepen) →
  [`../special_holonomy`](../special_holonomy/)
- PQC / lattices / ECC engineering →
  [`../post_quantum_cryptography`](../post_quantum_cryptography/)
- Combinatorial manifolds ($k=1,\ldots,5$) →
  [`../simplicial_manifolds`](../simplicial_manifolds/)
  (mother notebook `00_series_guide.ipynb`)

**Bridge.** Part III (notebook 07) bridges to product formulas
and Hamiltonian simulation in
[`../coupled_harmonic_oscillators`](../coupled_harmonic_oscillators/)
(module 05). In those notebooks, §5.1 of the $N=2$ tutorial keeps a
thin reminder of the exponential map and then moves to Pauli
decomposition and Trotter. This foundations module is the place for
the geometry; module 05 is the place for the simulation pipeline. It
does not replace that pipeline.
"""))

    cells.append(md(r"""
## Summary

- Smooth manifolds let us do calculus where global addition fails;
  charts + \(C^\infty\) transitions make derivatives chart‑independent
  (Lee).
- Lie groups package manifolds of transformations; Lie algebras are
  their infinitesimal generators; the exponential map builds
  one‑parameter subgroups (Hall).
- Quantum information, Gaussian CV, and statistics supply Tier‑A
  manifolds you can plot and test in notebooks 04–06
  (Bengtsson–Życzkowski, de Gosson, Amari).
- Relativity and bundles appear at Tier‑B (Carroll/Wald, Nakahara,
  Berry).
- Tier‑C is geometry‑depth throughout: Calabi–Yau ($c_1=0$), $G_2$
  (parallel $3$‑form), twistors (incidence
  $\omega^{A}=ix^{AA'}\pi_{A'}$), elliptic curves / lattice tori.
  Graduate off‑ramps: [`twistor_theory`](../twistor_theory/);
  [`special_holonomy`](../special_holonomy/); PQC module for standards
  and OQS. Combinatorial manifolds ($k=1,\ldots,5$):
  [`simplicial_manifolds`](../simplicial_manifolds/)
  (`00_series_guide.ipynb`).

## Global further reading (bookshelf)

1. Lee, *Introduction to Smooth Manifolds*, 2nd ed., Springer, 2013.
2. Hall, *Lie Groups, Lie Algebras, and Representations*, 2nd ed.,
   Springer, 2015 (arXiv:math‑ph/0005032).
3. Bengtsson & Życzkowski, *Geometry of Quantum States*, 2nd ed., CUP,
   2017.
4. de Gosson, *Symplectic Geometry and Quantum Mechanics*, Birkhäuser,
   2006.
5. Amari & Nagaoka, *Methods of Information Geometry*, AMS, 2000.
6. Nakahara, *Geometry, Topology and Physics*, 2nd ed., 2003.
7. Joyce, *Compact Manifolds with Special Holonomy*, OUP, 2000.
8. Penrose & Rindler, *Spinors and Space‑Time*, Vol. 2, CUP, 1986.
9. Silverman, *The Arithmetic of Elliptic Curves*, 2nd ed., Springer,
   2009.
10. Open Quantum Safe: https://openquantumsafe.org/ and the `liboqs`
    / `oqs-provider` repositories.
"""))

    cells.append(md(r"""
## Where to go next

**Takeaway.** This file is the map. Notebooks 01–07 hold the labs.
`smooth_manifolds_lie_groups_and_the_exponential_map.ipynb` redirects
here.

- Geometry core reads 00 → 01 → 02 → 03.
- Applications reads 00 → 04 → 05 (Tier A, then Tier B).
- Off-ramps reads 00 → 06 (Tier C), then twistor theory, special
  holonomy, simplicial manifolds, or post-quantum cryptography.
- Bridge reads 00 → 07, then the coupled-oscillator notebooks
  (module 05). That pipeline stays there.
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
