# Smooth Manifolds, Lie Groups, and the Exponential Map

A foundations tutorial: why many objects in quantum theory are
**smooth manifolds** (or **Lie groups**), how to do calculus on them
with charts and transition maps, and how the **exponential map**
connects a Lie algebra of generators to a Lie group of
transformations — with coded applications at three depths (A/B/C).

It is a foundations module for the geometry of states, transformations,
and parameters used across the tutorials and tools here. Part III of
the notebook bridges in particular to product formulas and Hamiltonian
simulation in
[`coupled_harmonic_oscillators`](../coupled_harmonic_oscillators/)
(module 05); it does **not** replace that pipeline.


# Notebook

- `smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`


# Software Requirements

- `numpy`
- `scipy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit` required here.


# Depth map (applications)

Each case is written as **background → geometry (with equations) →
where it shows up**, then code or a reading off-ramp.

| Tier | Topics | Code depth |
|------|--------|------------|
| A | Bloch / $\mathbb{CP}^{1}$; $\mathbb{CP}^{d-1}$ & density operators; $\mathrm{U}(2^{n})$; $\mathrm{Sp}(2n,\mathbb{R})$; statistical manifolds | Full labs |
| B | Lorentzian Minkowski; $U(1)$ bundles / Berry phase | Light code |
| C | Calabi–Yau; $G_{2}$; twistors; elliptic curves & lattices | Prose + schematics; PQC/crypto off-ramp |

**Graduation:** PQC / cryptology (and other topics that outgrow a
vignette) are meant for **dedicated** follow-on notebooks using RFCs
and Open Quantum Safe (`liboqs`, `oqs-provider`), not for deep crypto
implementation inside this foundations file.


# Further reading (primary)

Listed with full bibliographic detail inside the notebook. Anchors:

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
