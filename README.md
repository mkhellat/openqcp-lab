# openqcp-lab

A collection of educational Jupyter notebooks focused on quantum
algorithms, plus standalone tools developed in support of them (see
[Tools](#tools) below).

## Environment Setup

We recommend setting up a dedicated Python environment before running
the notebooks. The project includes a `requirements.txt` file that
specifies all necessary dependencies.

**Note:** `classiq` does not yet publish wheels for Python 3.13+, so
the environment must use Python 3.12. Setting up a plain `venv` with
whatever `python3` your system resolves to may not satisfy this if
your system's default Python is newer.

### Using `./bootstrap` (recommended)

The provided `bootstrap` script provisions a Python 3.12 interpreter
(via [uv](https://docs.astral.sh/uv/) by default, with automatic
fallback to pyenv or building from source) and creates `venv/` with
all dependencies installed:

```bash
./bootstrap
```

Run `./bootstrap --help` for provisioning options (`--with-python=uv|pyenv|source`).

### Using `Makefile` targets

Equivalently, if you have `make` installed:

```bash
make env
```

This runs `./bootstrap` under the hood. See `make help` for other
targets, including `make test` for running the test suite.

### Manual setup

If you already have a Python 3.12 interpreter available, you can set
up the environment yourself:

```bash
python3.12 -m venv venv
. venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

After setting up the environment, start a Jupyter notebook server:

```bash
jupyter notebook
```

or if you have migrated to JupyterLab:

```bash
jupyter lab
```

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

**Note:** If you have installed the dependencies from `requirements.txt`,
the notebooks should run without requiring additional package
installations. Each notebook's README also documents its specific
software requirements for reference.

For information on reproducing figures and results from the notebooks,
see [REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md).

Good luck and have fun using these tutorials!

## Tutorials

All tutorial notebooks live under [`tutorials/`](tutorials).

- [( 00 ) - Quantum Fourier Transform - Abelian groups case](tutorials/quantum_fourier_transform_abelian)  
  Learn the mathematical foundations of QFT over finite Abelian groups
  and its applications in phase estimation and Hadamard test.

- [( 01 ) - Quantum Machine Learning - minimize expectation value](tutorials/minimize_expectation_value)  
  Optimize variational quantum circuits using gradient descent to minimize
  expectation values of quantum observables.

- [( 02 ) - Discrete-Time Quantum Walk - path graph](tutorials/quantum_walk)  
  Implement quantum walk operators on graphs using coin and shift operators
  for a path graph with 16 nodes.

- [( 03 ) - Non-Unitary Quantum Computing - lcu](tutorials/nonunitary_quantum_computing)  
  Represent and manipulate non-unitary operations using Linear Combination
  of Unitaries (LCU) decomposition.

- [( 04 ) - Quantum Optimization - qubo and vqe](tutorials/quantum_variational_algorithms)  
  Solve Quadratic Unconstrained Binary Optimization (QUBO) problems using
  Variational Quantum Eigensolvers (VQE).

- [( 05 ) - Quantum Simulation - coupled harmonic oscillators](tutorials/coupled_harmonic_oscillators)  
  Simulate the dynamics of coupled classical harmonic oscillators using
  quantum Hamiltonian simulation with exponential speedup.

## Tools

Standalone, installable Python packages developed in support of the
tutorials above, kept separate from the notebooks themselves since
they're independently versioned software rather than lesson material.

- [`paulikit`](tools/paulikit) — performance-engineering tools for
  Pauli decomposition of Hermitian and non-Hermitian operators.
  Built to scale module ( 05 )'s Hamiltonian simulation beyond its
  original symbolic decomposition's practical limit (~N=4). Install
  with `pip install -e tools/paulikit` and see that package's own
  README for usage.

## GNU GPL v3+

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
