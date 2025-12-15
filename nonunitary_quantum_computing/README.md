# Linear Combination of Unitaries

An example LCU implementation of a $2 \times 2$ non-unitary matrix.


# Implementation

Coefficients $\alpha_i$ are determined performing a PauliTerms
decomposition.

Consequently, SELECT and PREPARE operators are prepared so that 
$\langle 0| \text{PREPARE}^{\dagger}.\text{SELECT}.\text{PREPARE} |0\rangle = \hat{U} |\psi\rangle$.
Here, $\hat{U}$ is the quantum circuit. For our problem 
$\hat{U} = 0.5\hat{I} + 0.5\hat{Z}$. This means applying $\hat{I}$ when
controller qubit is in state $|0\rangle$ and applying $\hat{Z}$ when
controller qubit is in state $|1\rangle$. The required operations are 
defined in `lcu_controllers(controller: QNum, psi: QNum)`.

The rest is simply preparing the state $|\psi\rangle$ which is to be
operated by our non-unitary operator and applying the
`lcu_controllers` on it.


# Software Requirements

The following Python packages are required to run this notebook:

- `numpy`
- `classiq`

**Note:** These packages are included in the top-level `requirements.txt`.
If you have set up the base environment as described in the main README,
no additional installation is needed.


# Outputs

The notebook may generate the following file when executed:

- `lcu-2x2.qmod` - Quantum model file (generated if the `write_qmod` cell
  is executed)

This file is saved in the `nonunitary_quantum_computing/` directory.
The notebook also displays measurement results showing the application of
the non-unitary operator, but these are displayed inline and not saved to
files by default.


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
