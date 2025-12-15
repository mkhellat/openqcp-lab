# QUBO and Variational Quantum Eigensolvers (VQE)

This notebook demonstrates the formulation of Quadratic Unconstrained
Binary Optimization (QUBO) problems as ground state problems and their
solution using Variational Quantum Eigensolvers (VQE).


# Implementation

The notebook implements a VQE solver for the Max-Cut problem using
Qiskit's `EfficientSU2` ansatz and `GSLS` optimizer. The QUBO problem
is formulated as a ground state problem and solved using variational
quantum algorithms.


# Software Requirements

The following Python packages are required to run this notebook:

- `numpy` (for numerical operations)
- `qiskit` (for quantum circuit construction and VQE implementation)
- `qiskit-optimization` (for QUBO problem formulation)

**Note:** The `numpy` package is included in the top-level
`requirements.txt`. The Qiskit packages (`qiskit` and
`qiskit-optimization`) are optional dependencies that will be
automatically installed by the notebook if not already present. If you
prefer to install them beforehand, you can run:

```bash
pip install qiskit qiskit-optimization
```

Alternatively, if you have set up the base environment as described in
the main README, the notebook will handle the installation of Qiskit
packages automatically when executed.


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
