# Discrete-Time Quantum Walk

Quantum walk operator for a path graph with 16 nodes ($P_{16}$).


# Implementation

Diffuser part of the $C$ operator is deployed using `zero_diffuzer(x:
QNum)`. For the full $C$, one needs to control the operation of the
zero diffuzer using the first register and also to iterate through all
vertices. This is done in `W_iteration(i: int, vertices: QNum,
adjacent_vertices: QNum)`.

For $S$, one needs to check if two vertices are connected using
`edge_oracle(res: Output[QBit], vertices: QNum, adjacent_vertices:
QNum)` and then to apply a bit-wise swap in `S_operator(vertices: QNum,
adjacent_vertices: QNum)` on all connected vertices.


# Software Requirements

The following Python packages are required to run this notebook:

- `numpy`
- `classiq`

**Note:** These packages are included in the top-level `requirements.txt`.
If you have set up the base environment as described in the main README,
no additional installation is needed.


# GNU GPL v3+

Copyright (C) 2024 Mohammadreza Khellat GNU GPL v3+

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
USA.

See also https://www.gnu.org/licenses/gpl.html
