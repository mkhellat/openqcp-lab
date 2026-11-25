#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
"""Deprecated redirect builder for the old manifolds filename.

The series generators ``build_00_series_guide.py`` through
``build_07_bridge_product_formulas.py`` are the source of truth.
This script only rewrites
``smooth_manifolds_lie_groups_and_the_exponential_map.ipynb`` as a
short redirect to ``00_series_guide.ipynb``. It does not contain the
labs, the physics examples, or the bookshelf.

Run from the repository root or from this directory:

    python tutorials/manifolds_lie_groups_quantum/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _nbutil import code, md

OUT = Path(__file__).with_name(
    "smooth_manifolds_lie_groups_and_the_exponential_map.ipynb"
)


def build_cells() -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    cells.append(md(r"""
# Smooth Manifolds, Lie Groups, and the Exponential Map

This filename is a redirect. The foundations text is the series in
this folder. Start at [`00_series_guide.ipynb`](00_series_guide.ipynb).

| ID | Notebook | What it owns |
|----|----------|----------------|
| 00 | `00_series_guide.ipynb` | Map, closing summary, bookshelf |
| 01 | `01_extrinsic_sphere_and_dg_toolset.ipynb` | (M1)–(M3), (P1)–(P3), cylinder demo |
| 02 | `02_charts_atlases_tangent.ipynb` | Atlas, charts, transitions |
| 03 | `03_lie_groups_exponential_map.ipynb` | Lie groups and the exponential map |
| 04 | `04_tier_a_applications.ipynb` | Tier A labs |
| 05 | `05_tier_b_light_labs.ipynb` | Tier B labs |
| 06 | `06_tier_c_offramps.ipynb` | Tier C off-ramps |
| 07 | `07_bridge_product_formulas.ipynb` | Product formulas / module 05 |

`build_notebook.py` is deprecated. It writes only this redirect.
The series generators are the source of truth.
"""))
    cells.append(code(r"""
# Redirect card. The labs are in notebooks 00-07.
print("Start at 00_series_guide.ipynb")
print("Series: 00 -> 01 -> 02 -> 03; applications 04 -> 05; 06; 07")
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
    print(
        f"Wrote {OUT} with {len(cells)} cells "
        f"({n_md} markdown, {n_code} code)"
    )


if __name__ == "__main__":
    main()
