# openqcp-lab

A collection of educational Jupyter notebooks focused on quantum algorithms.

## Environment Setup

We recommend setting up a dedicated Python environment before running
the notebooks. The project includes a `requirements.txt` file that
specifies all necessary dependencies.

### Using pip (recommended)

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
. venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Using conda

Alternatively, if you prefer conda:

```bash
conda create -n openqcp-lab python=3.9
conda activate openqcp-lab
pip install -r requirements.txt
```

### Using `setup_env.sh` helper script

You can also use the provided POSIX shell script to create a virtual
environment and install dependencies in one step:

```bash
./setup_env.sh
```

This will create a `venv` directory (if it does not already exist),
install the required packages, and print instructions on how to
activate the environment.

### Using `Makefile` targets

If you have `make` installed, you can use the provided `Makefile`:

- **Create a virtual environment and install dependencies**:

  ```bash
  make env
  ```

- **Install dependencies in the current environment**:

  ```bash
  make install
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

Good luck and have fun using these tutorials!

- [( 00 ) - Quantum Fourier Transform - Abelian groups case](quantum_fourier_transform_abelian)
- [( 01 ) - Quantum Machine Learning - minimize expectation value](minimize_expectation_value)
- [( 02 ) - Discrete-Time Quantum Walk - path graph](quantum_walk)
- [( 03 ) - Non-Unitary Quantum Computing - lcu](nonunitary_quantum_computing)
- [( 04 ) - Quantum Optimization - qubo and vqe](quantum_variational_algorithms)
- [( 05 ) - Quantum Simulation - coupled harmonic oscillators](coupled_harmonic_oscillators)

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
