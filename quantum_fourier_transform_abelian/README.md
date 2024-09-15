# Quantum Fourier Transform (Abelian)

This is a rather comprehensive review of quantum Fourier transform
over finite Abelian groups with use cases in phase estimation and
Hadamard test.

# Theory

The main point of this tutorial is a review of the mathematical
foundations of quantum Fourier transform (for Abelian groups) as a
linear transformation between $\mathbb{C}[G]$ and
$\mathbb{C}[\widehat{G}]$. Both being vector spaces, over the field of
complex numbers $\mathbb{C}$, of complex-valued functions
$f : G \to \mathbb{C}$ and $\widehat{f} : \widehat{G} \to \mathbb{C}$
where $G$ is a finite Abelian group and $\hat{G}$ is the complete set
of its characters.

# Implementation

The quantum circuit for the Hadamard test example is quite simple and
self-explanatory.


# Software Requirements

To run the codes within the notebook, you would need the following
python modules:

```
pip install numpy matplotlib classiq
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
