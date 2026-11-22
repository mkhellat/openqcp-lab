# Special Holonomy (Calabi–Yau and $G_2$)

A foundations tutorial that graduates the manifolds notebook's Tier‑C
**§C1 (Calabi–Yau)** and **§C2 ($G_2$)** invitations: holonomy as a
differential-geometry tool, the Calabi–Yau package ($c_1=0$, Ricci-flat
Kähler, $\mathrm{SU}(m)$), and $G_2$ via a parallel stable 3-form —
without constructing Ricci-flat metrics or full string compactification
pipelines here.

Upstream geometry (charts, embedded manifolds, Tier‑C working setups)
lives in
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/).
This module does **not** redo Lie theory, twistors, or PQC.


# Notebook

- `special_holonomy_calabi_yau_and_g2.ipynb`

Rebuild from `build_notebook.py` after editing the generator.


# Software Requirements

- `numpy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit`, no Open Quantum Safe.


# Depth map

| Part | Content | Code depth |
|------|---------|------------|
| I | Holonomy of a metric connection; Berger list as a map | Prose + small checks |
| II | Calabi–Yau: $c_1=0$, Yau, $\mathrm{SU}(m)$, Hodge sketch | Labs + schematic |
| III | $G_2$: model $\varphi_0$, parallel 3-form, Joyce existence | Labs + schematic |
| IV | $\mathrm{Spin}(7)$ map; bridges; further reading | Prose |

**Upstream:**
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/)
(§C1–C2).


# Further reading (primary)

Full bibliographic detail is inside the notebook. Anchors:

- D. Joyce, *Compact Manifolds with Special Holonomy*, OUP, 2000.
- S.-T. Yau, “On the Ricci curvature of a compact Kähler manifold…,”
  *Comm. Pure Appl. Math.* **31** (1978) 339–411.
- M. Gross, D. Huybrechts, D. Joyce, *Calabi–Yau Manifolds and Related
  Geometries*, Springer, 2003.


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
