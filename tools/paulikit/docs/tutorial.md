# Tutorial

This page walks through using `paulikit` end to end: building a
Hamiltonian, decomposing it, interpreting and verifying the result,
and using the command-line interface. For the "why" behind each step,
see {doc}`background` and {doc}`theory`; for the full function
signatures, see the {doc}`API reference <api/index>`.

## Installation

```bash
cd tools/paulikit   # from the openqcp-lab repository root
pip install -e . --no-build-isolation
```

Everything in this tutorial works with just the core install (`numpy`
is the only runtime dependency). The command-line examples below
assume the `paulikit` console script is on your `PATH`, which the
install above sets up automatically.

The install above also compiles a native (Cython/C++) label-generation
kernel if a C++ toolchain and oneTBB are available, for faster
`fwht_pauli_terms` — falling back to pure Python automatically
otherwise. See the project README's "Native extension" section for
details; nothing in this tutorial depends on which path is active.

## 1. Building a Hamiltonian

`paulikit.hamiltonian.build_hamiltonian` constructs the
coupled-oscillator Hamiltonian matrix from physical parameters: a
dictionary of spring constants and a list of masses.

```python
from paulikit.hamiltonian import build_hamiltonian

spring_constants = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
masses = [1.0, 2.0]

H = build_hamiltonian(n_oscillators=2, spring_constants=spring_constants, masses=masses)
```

`spring_constants` maps `(i, j)` with `i <= j` to the spring constant
$k_{ij}$: diagonal entries are each oscillator's own spring constant
(coupling to a fixed wall), off-diagonal entries are the coupling
strength between oscillator `i` and oscillator `j`. `masses[i]` is
oscillator `i`'s mass.

The result is a $5 \times 5$ matrix (for $N=2$, the dimension is
$N + N(N+1)/2$):

```
[[ 0.          0.         -1.          0.         -1.41421356]
 [ 0.          0.          0.         -1.22474487  1.        ]
 [-1.          0.          0.          0.          0.        ]
 [ 0.         -1.22474487  0.          0.          0.        ]
 [-1.41421356  1.          0.          0.          0.        ]]
```

## 2. Padding to a power-of-two dimension

Pauli decomposition requires the matrix dimension to be exactly
$2^n$ for some integer $n$ (the number of qubits). Real
coupled-oscillator Hamiltonians rarely have a power-of-two dimension
already, so pad first:

```python
from paulikit.hamiltonian import pad_to_power_of_two

H_padded, n_qubits = pad_to_power_of_two(H)
# H_padded.shape == (8, 8), n_qubits == 3
```

This zero-pads $H$ into the top-left block of an $8 \times 8$ matrix
(since $\lceil \log_2 5 \rceil = 3$). The padding entries are all
zero, so they contribute nothing to the physics — they exist only to
satisfy the tensor-product structure quantum circuits require.

## 3. Decomposing into Pauli terms

```python
from paulikit.algorithms.fwht import fwht_pauli_terms

terms = fwht_pauli_terms(H_padded)
```

`terms` is a dictionary mapping Pauli-string labels to their
coefficients:

```
{
    'IXI': -0.556186, 'IXZ': 0.056186,
    'XII': -0.353553, 'XIX': 0.250000, 'XIZ': -0.353553,
    'XZI': -0.353553, 'XZX': 0.250000, 'XZZ': -0.353553,
    'YIY': 0.250000, 'YZY': 0.250000,
    'ZXI': -0.556186, 'ZXZ': 0.056186,
}
```

Each label is a string of length `n_qubits`, read left-to-right as
qubit 0, 1, 2, ... — e.g. `'XIZ'` means $X$ on qubit 0, $I$
(identity) on qubit 1, $Z$ on qubit 2, i.e. the operator
$X \otimes I \otimes Z$. Only nonzero terms are included (12 of the
$4^3 = 64$ possible 3-qubit Pauli strings, here); the threshold is
controlled by `fwht_pauli_terms`'s `atol` parameter.

By default (`assume_hermitian=True`), coefficients are returned as
real `float`s, and a `ValueError` is raised if the input wasn't
actually Hermitian (a useful sanity check — see {doc}`non_hermitian`
for when and how to decompose non-Hermitian operators instead).

## 4. Verifying the result

`paulikit.pauli_utils.reconstruct_from_terms` rebuilds the dense
matrix from a term dictionary, useful both as a sanity check and for
programmatically confirming a decomposition is correct:

```python
from paulikit.pauli_utils import reconstruct_from_terms

reconstructed = reconstruct_from_terms(terms, n_qubits)
error = (reconstructed.real - H_padded)
print(abs(error).max())  # 0.0 - exact to floating-point precision
```

This is exactly the check `paulikit`'s own test suite runs against
every fixture (see `tests/test_fwht.py`), and it's good practice to
run it yourself whenever decomposing a new Hamiltonian you haven't
validated before.

## 5. Using the command-line interface

For quick exploration without writing a script, the `paulikit`
console command wraps the same functionality:

```console
$ paulikit decompose --n-oscillators 4 --show-terms
N=4 oscillators, 4 qubits, 16x16 padded Hamiltonian
Decomposition time: 0.0005s
Nonzero Pauli terms: 56
  IXII: -0.5470915958155509
  IXIZ: 0.015053558406667805
  IXZI: 0.02983035390312644
  ...
```

`paulikit decompose` builds a synthetic Hamiltonian internally (a
fixed, deterministic — not physically calibrated — set of spring
constants and masses that scale with $N$), so it's meant for quickly
checking behavior and timing at a given size, not for physically
meaningful results; use the library API (above) with your own
parameters for real work.

`paulikit benchmark` sweeps multiple $N$ values and reports timing:

```console
$ paulikit benchmark --n-oscillators 2 4 8 16 30
    N  qubits    dim    terms   time (s)
    2       3      8       12     0.0004
    4       4     16       56     0.0004
    8       6     64      928     0.0043
   16       8    256    15360     0.0850
   30       9    512   112384     0.4867
```

`paulikit regenerate-fixtures` recomputes the expected Pauli terms
used by the test suite's correctness fixtures, using PennyLane as an
independent oracle — see the {doc}`API reference <api/testing>` for
`paulikit.testing.fixtures` if you're extending the test suite itself
rather than just using the library.

Run `paulikit --help` or `paulikit <subcommand> --help` for full
argument details on any of these.
