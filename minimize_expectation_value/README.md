# Minimize Expectation Value of a 2-qubit Quantum Observable

This is a commented version of PennyLane beginner level optimization
challenge [Keeping Expectations Low](https://pennylane.ai/challenges/keeping_expectations_low/).


# Implementation

The quantum circuit for the computation is composed of a
`StronglyEntanglingLayers` circuitry which would prepare 2-qubit
states to be operated on by a Hermitian operator `hamiltonian` after
which measurements would be performed on the output.

The parameters for the `StronglyEntanglingLayers` would be optimized
by a gradient descent optimizer with maximum 100 steps and a precision
of `1e-07`.


# Software Requirements

To run the codes within the notebook, you would need the following
python modules:

```
pip install json pennylane
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
