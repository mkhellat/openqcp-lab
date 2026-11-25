# Smooth Manifolds, Lie Groups, and the Exponential Map

A foundations tutorial: why many objects in quantum theory are
**smooth manifolds** (or **Lie groups**), how to do calculus on them
with charts and transition maps, and how the **exponential map**
connects a Lie algebra of generators to a Lie group of
transformations — with coded applications at three depths (A/B/C).

It is a foundations module for the geometry of states, transformations,
and parameters used across the tutorials and tools here. The module is
being split into a series (00–07). Until that split finishes, the
legacy mega notebook still holds the full text, including Part III,
which bridges to product formulas and Hamiltonian simulation in
[`coupled_harmonic_oscillators`](../coupled_harmonic_oscillators/)
(module 05); it does **not** replace that pipeline.


# Notebooks

Generators (`build_*.py`) are the source of truth; each `.ipynb` is
rebuilt from its generator. The series is being split out of the
legacy mega notebook. That file is **transitional**: it still holds
Parts I, II‑A, II‑B, II‑C, and III, and it is not retired.

- `smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`
  (transitional mega notebook)

| ID | Notebook | Arc |
|----|----------|-----|
| 00 | `00_series_guide.ipynb` | Prerequisites, reading paths, depth map, off-ramps |
| 01 | `01_extrinsic_sphere_and_dg_toolset.ipynb` | (M1)–(M3) and the cylinder demo |
| 02 | `02_charts_atlases_tangent.ipynb` | Part I §2 atlas, charts, transitions |
| 03 | `03_lie_groups_exponential_map.ipynb` | Lie groups and the exponential map |
| 04 | `04_tier_a_applications.ipynb` | Part II‑A full labs |
| 05 | `05_tier_b_light_labs.ipynb` | Part II‑B light labs |
| 06 | `06_tier_c_offramps.ipynb` | Part II‑C thin map |
| 07 | `07_bridge_product_formulas.ipynb` | Part III → coupled oscillators, module 05 |

Notebooks 00 and 01 are split out. Rebuild notebook 01 with
`build_01_extrinsic_sphere_and_dg_toolset.py`. Notebooks 02–07 are
the planned roster; their cells still live in the mega notebook.

Interactive figures live in sibling `*_demo.py` modules. Notebook
cells only launch them, with the source hidden (`_nbutil.code(...,
hide_input=True)`). The cylinder demo is launched from notebook 01.


# Reading paths

1. **Geometry core.** 00 → 01 → 02 → 03
2. **Applications.** 00 → 04 → 05
3. **Off-ramps.** 00 → 06
4. **Bridge.** 00 → 07

Until 02–07 exist as their own notebooks, each of those paths is
still the corresponding part of the mega notebook.


# Software Requirements

- `numpy`
- `scipy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit` required here.


# Part I toolset (before applications)

Examples (M1)–(M3) build an extrinsic differential-geometry kit in
order: embedded constraint → tangent space; differential of a map /
vector field; charts + chain rule; projective quotients; matrix Lie
groups and Lie-algebra velocities at the identity. Those examples
and the cylinder demo are in
`01_extrinsic_sphere_and_dg_toolset.ipynb`. Section 2 then
makes the atlas language precise, and that section stays in the
mega notebook until notebooks 02–03 are extracted.

# Depth map (applications)

Each case is written as **background → geometry (with equations) →
where it shows up**, then code or a reading off-ramp.

| Tier | Topics | Code depth |
|------|--------|------------|
| A | Bloch / $\mathbb{CP}^{1}$; $\mathbb{CP}^{d-1}$ & density operators; $\mathrm{U}(2^{n})$; $\mathrm{Sp}(2n,\mathbb{R})$; statistical manifolds | Full labs |
| B | Lorentzian Minkowski; $U(1)$ bundles / Berry phase | Light code |
| C | Calabi–Yau ($c_1=0$); $G_{2}$ (parallel $3$-form); twistors (incidence); elliptic curves & lattice tori | Geometry-depth + schematics; PQC/crypto off-ramp |

Tier A is planned notebook 04, Tier B notebook 05, and Tier C
notebook 06. Those cells are still in the mega notebook.

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


# Further reading (primary)

Listed with full bibliographic detail inside the mega notebook. Anchors:

- J. M. Lee, *Introduction to Smooth Manifolds*, 2nd ed., Springer, 2013.
- B. C. Hall, *Lie Groups, Lie Algebras, and Representations*, 2nd ed.,
  Springer, 2015.
- I. Bengtsson and K. Życzkowski, *Geometry of Quantum States*, 2nd ed.,
  Cambridge, 2017.


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
