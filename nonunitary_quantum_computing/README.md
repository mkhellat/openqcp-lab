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

To run the codes within the notebook, you would need the following
python modules:

```
pip install numpy classiq
```


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
