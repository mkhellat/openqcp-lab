# QUBO and Variational Quantum Eigensolvers (VQE)

This notebook demonstrates the formulation of Quadratic Unconstrained
Binary Optimization (QUBO) problems as ground state problems and their
solution using Variational Quantum Eigensolvers (VQE).


# Implementation

The notebook implements a VQE solver for the Max-Cut problem using
Qiskit's `EfficientSU2` ansatz and `COBYLA` optimizer, and compares
the result against a brute-force classical baseline. The QUBO problem
is formulated as a ground state problem and solved using variational
quantum algorithms.


# Software Requirements

The following Python packages are required to run this notebook:

- `numpy` (for numerical operations)
- `networkx` (for graph construction)
- `qiskit` (for quantum circuit construction)
- `qiskit-algorithms` (for the VQE algorithm and optimizers)
- `qiskit-optimization` (for QUBO problem formulation)

**Note:** All of the above are listed in the top-level
`requirements.txt` and installed by `./bootstrap` (or `make env`), so
no separate installation step is needed.


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
