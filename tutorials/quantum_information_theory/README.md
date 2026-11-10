# Quantum Information and Information Theory

A foundations tutorial on **classical and quantum information
measures** — entropy, relative entropy, channels, and
distinguishability — with coded checks on small alphabets and qubit
systems. It feeds later cryptology work (including code-based PQC such
as HQC) without implementing cryptographic primitives here.

Cross-links the manifolds foundations notebook for density-operator
geometry and the Fisher metric; it does **not** redo Lie theory.


# Notebook

- `quantum_information_and_information_theory.ipynb`

Rebuild from `build_notebook.py` after editing the generator.


# Software Requirements

- `numpy`
- `scipy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit`, no Open Quantum Safe in this module.


# Depth map

| Part | Content | Code depth |
|------|---------|------------|
| I | Shannon entropy, mutual information, KL, classical channels (BSC), Hamming coding bridge | Full labs |
| II | von Neumann entropy, quantum relative entropy, fidelity / trace distance, Kraus CPTP, Holevo sketch | Full labs |
| III | Why crypto cares; code-based idea; **HQC** as destination; OQS/RFCs off-ramp | Prose + schematic |
| IV | Bridges to manifolds (Fisher) and the wider lab | Short |

**Graduation:** NIST PQC standards, RFCs, and Open Quantum Safe
(`liboqs`, `oqs-provider`) continue in
[`../post_quantum_cryptography`](../post_quantum_cryptography/) —
not in this file.


# Further reading (primary)

Full bibliographic detail is inside the notebook. Anchors:

- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd
  ed., Wiley, 2006.
- M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum
  Information*, Cambridge, 2000.
- M. M. Wilde, *Quantum Information Theory*, 2nd ed., Cambridge, 2017.


# GNU GPL v3+

Copyright (C) 2026 Mohammadreza Khellat GNU GPL v3+

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

See also https://www.gnu.org/licenses/gpl.html
