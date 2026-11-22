# Twistor Theory (geometry first)

A foundations tutorial that takes the manifolds notebook's Tier‑C
**§C3 incidence dictionary** one step further: 2‑spinor coordinates,
reality conditions selecting real Minkowski space, and the **Penrose
transform** sketch for massless fields — without building a full
amplitude or integrable‑systems course here.

Upstream geometry (charts, Minkowski as a manifold, the
$\mathbb{PT}=\mathbb{CP}^{3}$ dictionary) lives in
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/).
This module does **not** redo Lie theory or PQC.


# Notebook

- `twistor_theory_geometry_and_penrose_transform.ipynb`

Rebuild from `build_notebook.py` after editing the generator.


# Software Requirements

- `numpy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit`, no Open Quantum Safe.


# Depth map

| Part | Content | Code depth |
|------|---------|------------|
| I | Recap incidence; 2‑spinor blocks; null lines from $\pi$ | Full labs (numpy) |
| II | Reality conditions; real $M\subset\mathbb{CM}$ | Prose + checks |
| III | Penrose transform sketch (helicity / contour idea) | Light code + reading |
| IV | Amplitudes / integrable systems as maps; further reading | Prose |

**Upstream:**
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/)
(§B1 Minkowski; §C3 incidence).


# Further reading (primary)

Full bibliographic detail is inside the notebook. Anchors:

- R. Penrose, “Twistor algebra,” *J. Math. Phys.* **8** (1967) 345–366.
- R. Penrose and W. Rindler, *Spinors and Space‑Time*, Vol. 2,
  Cambridge, 1986.
- S. A. Huggett and K. P. Tod, *An Introduction to Twistor Theory*,
  2nd ed., Cambridge, 1994.


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
