# openqcp-lab

A collection of educational Jupyter notebooks mainly on quantum algorithms.

# Run

Running Jupyter notebooks require a conda installation such as
`miniconda`. You could start a Jupyter notebook server on a
unix-like operating system using:

````
$ jupyter notebook
````

or if you have migrated to jupyterlab using

````
$ jupyter lab
````

From the web portal, Jupyter notebooks could be opened and executed.
In newer versions of Jupyter server, it is possible to open an
`.ipynb` file in `NbClassic`, `JupyterLab` or `Notebook`.

Most Jupyter notebooks contain python codes and hence Jupyter would
use the preinstalled `Python 3 (pykernel)` kernel to execute codes
within them. However, in case a notebook requires running codes in
other languages such as `Julia`, `R`, `SageMath`, `C`, and ..., it is
possible to install the [relevant
kernel](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels) or
even to [make
kernels](https://jupyter-client.readthedocs.io/en/stable/kernels.html)
on need basis. Having the required kernel, one could then switch to
that kernel from the corresponding Jupyter web-portal. Also, to run
different types of codes within a single notebook, one could take
advantage of _magic_.

Furthermore, each notebook in this project is supposed to have a
section in which any required modules/libraries/packages would be
checked and installed if necessary.

Good luck and have fun using these tutorials!

- [( 00 ) - Quantum Fourier Transform - Abelian groups case](quantum_fourier_transform_abelian)
- [( 01 ) - Quantum Machine Learning - minimize expectation value](minimize_expectation_value)
- [( 02 ) - Discrete-Time Quantum Walk - path graph](quantum_walk)
- [( 03 ) - Non-Unitary Quantum Computing - lcu](nonunitary_quantum_computing)
- [( 04 ) - Qunatum Optimization - qubo and vqe](quantum_variational_algorithms)
- [( 05 ) - Qunatum Simulation - coupled harmonic oscillators](coupled_harmonic_oscillators)

# GNU GPL v3+

Copyright (C) 2023 Mohammadreza Khellat GNU GPL v3+

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
