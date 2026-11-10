# Post-Quantum Cryptography and Cryptology

A foundations tutorial on **post-quantum cryptography (PQC)**: why
classical public-key trapdoors fail against a large quantum adversary,
how NIST's PQC portfolio is organised (KEMs and signatures), and how
engineers touch those algorithms through standards, hybrid TLS, and
Open Quantum Safe — without re-implementing primitives from scratch.

Cross-links the QI notebook for the coding / HQC bridge and the
manifolds foundations notebook (Tier C) for lattice / ECC geometry; it
does **not** redo information measures or Lie theory.


# Notebook

- `post_quantum_cryptography_and_cryptology.ipynb`

Rebuild from `build_notebook.py` after editing the generator.


# Software Requirements

**Core (CI / default env):**

- `numpy`
- `matplotlib`

These are in the top-level `requirements.txt`. No Classiq, no
`paulikit` / `trotterkit`.

**Optional (local OQS labs):**

- [`liboqs-python`](https://github.com/open-quantum-safe/liboqs-python)
  (needs a system `liboqs` shared library)

The notebook soft-gates `import oqs`: without it, prose and size
tables still run; with it, ML-KEM / ML-DSA round-trips execute.


# Depth map

| Part | Content | Code depth |
|------|---------|------------|
| I | Why PQC: classical trapdoors, Shor threat, transition framing | Prose + checklist |
| II | NIST portfolio: FIPS 203/204/205; HQC as second KEM family | Tables + schematics |
| III | KEM / signature APIs; parameter-set size comparison | Full labs (numpy) |
| IV | Open Quantum Safe soft-gated round-trips | Full when OQS present |
| V | PQ/T hybrid TLS (RFC map); `oqs-provider` | Prose + checklist |
| VI | Bridges to QI and manifolds; further reading | Short |

**Upstream:** [`../quantum_information_theory`](../quantum_information_theory/)
(coding / HQC destination);
[`../manifolds_lie_groups_quantum`](../manifolds_lie_groups_quantum/)
(Tier C geometry invitation).


# Further reading (primary)

Full bibliographic detail is inside the notebook. Anchors:

- NIST Post-Quantum Cryptography project pages and FIPS 203, 204, 205.
- NIST news / IR on HQC fourth-round selection; IR 8547 transition.
- RFC 10024 (hybrid ECDHE–ML-KEM key agreement for TLS 1.3).
- Open Quantum Safe: `liboqs`, `liboqs-python`, `oqs-provider`.


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
