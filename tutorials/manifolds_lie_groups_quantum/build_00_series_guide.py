#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Build the manifolds series guide (source of truth).

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_00_series_guide.py

The generated notebook is ``00_series_guide.ipynb`` beside this file.
Title, prerequisites, learning objectives, and the Tier A/B/C depth
map are the text already used by the mega notebook and this folder's
README. This builder does not add a new development of the geometry.
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

_MEGA = "smooth_manifolds_lie_groups_and_the_exponential_map.ipynb"


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []

    cells.append(md(rf"""
# Smooth Manifolds, Lie Groups, and the Exponential Map

A foundations tutorial: why many objects in quantum theory are
**smooth manifolds** (or **Lie groups**), how to do calculus on them
with charts and transition maps, and how the **exponential map**
connects a Lie algebra of generators to a Lie group of
transformations — with coded applications at three depths (A/B/C).

It is a foundations module for the geometry of states, transformations,
and parameters used across the tutorials and tools here. Part III
bridges in particular to product formulas and the coupled-oscillator
Hamiltonian-simulation notebooks (module 05); it does not replace them.

This notebook is the map: prerequisites, reading paths, the depth
map, and links to the graduate off-ramps. It does not build charts,
and it does not launch a figure.

> **Transitional.** The legacy mega notebook `{_MEGA}` still
> holds the full text (Parts I, II‑A, II‑B, II‑C, and III) while
> notebooks 01–07 are split out of it. Read that file for the labs
> until those notebooks exist. This series lives at
> `tutorials/manifolds_lie_groups_quantum/`.
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

    cells.append(md(rf"""
## Notebooks

| ID | Notebook | What it owns |
|----|----------|----------------|
| 00 | `00_series_guide.ipynb` | This map |
| 01 | `01_extrinsic_sphere_and_dg_toolset.ipynb` | (M1)–(M3) and the cylinder demo |
| 02 | `02_charts_atlases_tangent.ipynb` | Part I §2: atlas, charts, transitions |
| 03 | `03_lie_groups_exponential_map.ipynb` | Lie groups, Lie algebra, exponential map |
| 04 | `04_tier_a_applications.ipynb` | Part II‑A full labs |
| 05 | `05_tier_b_light_labs.ipynb` | Part II‑B light labs |
| 06 | `06_tier_c_offramps.ipynb` | Part II‑C thin map to the graduate notebooks |
| 07 | `07_bridge_product_formulas.ipynb` | Part III bridge to product formulas / module 05 |

Only 00 is split out. Notebooks 01–07 are the planned roster; their
cells still live in `{_MEGA}`. Generators (`build_*.py`) are the
source of truth; rebuild a notebook by running its generator.
Interactive figures, when a later notebook has them, live in a
sibling `*_demo.py` and are launched from a hidden cell
(`_nbutil.code(..., hide_input=True)`). The cylinder demo stays with
the mega notebook until notebook 01 is extracted. Software
requirements are in `README.md` in this folder. This guide has no
demo.

**Citation rule.** Every section of the mega notebook names textbooks
or articles a reader can open. We do not invent references. Full
bibliographic detail stays in that notebook; the anchors are listed
in `README.md`.
"""))

    cells.append(md(rf"""
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

Until 01–07 are extracted, each path is still the corresponding part
of `{_MEGA}`.
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
planned notebook 04, Tier B notebook 05, and Tier C notebook 06.
Those cells are still in the mega notebook.

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

Graduate off-ramps (Tier C, planned notebook 06), copied from Part II‑C
of the mega notebook:

- Twistors → [`../twistor_theory`](../twistor_theory/)
- CY / $G_2$ (holonomy deepen) →
  [`../special_holonomy`](../special_holonomy/)
- PQC / lattices / ECC engineering →
  [`../post_quantum_cryptography`](../post_quantum_cryptography/)
- Combinatorial manifolds ($k=1,\ldots,5$) →
  [`../simplicial_manifolds`](../simplicial_manifolds/)
  (mother notebook `00_series_guide.ipynb`)

**Bridge.** Part III (planned notebook 07) bridges to product formulas
and Hamiltonian simulation in
[`../coupled_harmonic_oscillators`](../coupled_harmonic_oscillators/)
(module 05). In those notebooks, §5.1 of the $N=2$ tutorial keeps a
thin reminder of the exponential map and then moves to Pauli
decomposition and Trotter. This foundations module is the place for
the geometry; module 05 is the place for the simulation pipeline. It
does not replace that pipeline.
"""))

    cells.append(md(rf"""
## Summary

**Takeaway.** This file is the map. The legacy mega notebook
`{_MEGA}` still holds Parts I, II‑A, II‑B, II‑C, and III until
notebooks 01–07 are extracted.

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
