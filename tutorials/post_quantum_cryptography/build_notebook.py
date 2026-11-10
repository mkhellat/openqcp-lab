#!/usr/bin/env python3
"""Build the PQC / cryptology notebook (run; commit the .ipynb)."""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name(
    "post_quantum_cryptography_and_cryptology.ipynb"
)


def fix_math(source: str) -> str:
    """Use $ / $$ like the rest of openqcp-lab (MathJax-reliable)."""
    source = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "$$" + m.group(1) + "$$",
        source,
        flags=re.S,
    )
    source = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + m.group(1) + "$",
        source,
        flags=re.S,
    )
    return source


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(fix_math(source.strip() + "\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


cells: list[nbf.NotebookNode] = []

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Post-Quantum Cryptography and Cryptology

Public-key cryptography needs a **hard problem with a trapdoor**: easy
with a secret, hard without one. For decades that meant factoring and
discrete logarithms (including on elliptic curves). A cryptanalytically
relevant quantum computer breaks those via Shor's algorithm, so
**post-quantum cryptography (PQC)** replaces them with problems believed
hard for both classical and quantum adversaries — primarily on
**lattices**, **error-correcting codes**, and **hash** constructions —
and then **standardises** concrete KEMs and signatures.

This module is **standards- and interface-first**. Geometry of lattices
and ECC lives in the manifolds foundations notebook (Tier C);
coding / decoding hardness and the HQC destination live in the QI
notebook. Here we map NIST's portfolio, KEM/signature APIs, hybrid TLS,
and Open Quantum Safe. We do **not** re-implement ML-KEM, HQC, or
ML-DSA by hand.

> **Further reading (anchors).** NIST Post-Quantum Cryptography project:
> https://csrc.nist.gov/Projects/post-quantum-cryptography/ — FIPS 203
> (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA). Open Quantum Safe:
> https://openquantumsafe.org/
"""))

cells.append(md(r"""
## Prerequisites

- QI notebook Parts I–III recommended (Hamming bridge; HQC named).
- Manifolds foundations Tier C recommended (lattice / ECC geometry).
- Comfort reading FIPS / RFC abstracts.

## Learning objectives

1. State why RSA / ECC public-key schemes are quantum-vulnerable and
   what PQC replaces.
2. Name the published NIST PQC FIPS (ML-KEM, ML-DSA, SLH-DSA) and the
   role of HQC as a second, code-based KEM.
3. Distinguish KEM vs signature APIs; compare public-key / ciphertext /
   signature sizes for representative parameter sets.
4. When Open Quantum Safe is installed, run an ML-KEM encapsulate /
   decapsulate and an ML-DSA sign / verify round-trip; otherwise follow
   the install off-ramp without failing the notebook.
5. Explain PQ/T hybrid key agreement in TLS 1.3 at the named-group
   level and point to `oqs-provider` for OpenSSL wiring.
6. Cross-link QI and manifolds C4 as conceptual feeds without re-deriving
   them.
"""))

cells.append(md(r"""
## How this notebook is organised

- **Part I.** Why PQC: trapdoors, Shor, transition framing.
- **Part II.** NIST portfolio map (FIPS + HQC).
- **Part III.** KEM / signature interfaces and sizes.
- **Part IV.** Soft-gated Open Quantum Safe labs.
- **Part V.** Hybrid TLS and `oqs-provider`.
- **Part VI.** Bridges back to the lab.

**Citation rule.** Every major block names primary sources a reader can
open. We do not invent FIPS numbers, RFC numbers, or NIST status.
**Security note.** Demos are educational; production needs vetted
libraries, side-channel awareness, and institutional transition plans.
"""))

cells.append(code(r"""
# Setup — PQC / cryptology notebook (numpy / matplotlib; OQS optional)
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.figsize": (6.0, 4.0), "font.size": 11})


def require_oqs():
    # Soft-gate: never fail the notebook solely because OQS is absent.
    try:
        import oqs
    except ImportError:
        print(
            "OQS not installed — prose and size tables still valid.\n"
            "Optional: pip install liboqs-python "
            "(needs a liboqs shared library on the system).\n"
            "See https://github.com/open-quantum-safe/liboqs-python"
        )
        return None
    return oqs


print("numpy", np.__version__)
"""))

# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part I — Why post-quantum

## I.1 Classical trapdoors

A **key-encapsulation mechanism (KEM)** lets two parties establish a
shared secret over a public channel: one side publishes a public key;
the other encapsulates a secret to that key; only the holder of the
secret key recovers it. A **signature** scheme authenticates a message
relative to a public key.

Classical public-key KEMs and signatures typically rely on factoring
(RSA) or discrete logarithms in finite fields or on elliptic curves
(ECDH, ECDSA, …). Those problems are believed hard classically at
current sizes — and are polynomial-time for a large enough quantum
computer via Shor's algorithm.

> **Further reading (geometry).** Manifolds foundations notebook §C4
> (elliptic curves as groups / manifolds; lattice tori). Silverman,
> *The Arithmetic of Elliptic Curves*, 2nd ed., Springer, 2009.
"""))

cells.append(md(r"""
## I.2 Shor threat model

**Shor's algorithm** finds periods / orders in abelian groups efficiently
on a fault-tolerant quantum computer, which yields factoring and
discrete-log breaks. That is a change of **asymptotic** hardness, not a
claim that today's noisy devices break production RSA.

**Grover's algorithm** gives a quadratic speedup for unstructured search.
For symmetric crypto that usually means larger keys / digests — not a
full replacement of AES by an unrelated primitive. PQC is about
**asymmetric** algorithms whose security assumptions survive Shor.

> **Further reading.** Nielsen & Chuang, Ch. 5 (Shor / order-finding);
> NIST PQC project overview pages.
"""))

cells.append(md(r"""
## I.3 Transition framing

Deployments rarely flip a switch. **Hybrid** (PQ/T) constructions
combine a classical and a post-quantum share so that breaking only one
component does not yield the session key. NIST's transition report
discusses deprecation and disallow timelines for quantum-vulnerable
public-key algorithms in federal standards — read the primary text for
dates rather than memorising secondary summaries.

> **Further reading.** NIST IR 8547, *Transition to Post-Quantum
> Cryptography Standards*
> (https://csrc.nist.gov/pubs/ir/8547/final — confirm current URL when
> reading). SP 800-227 (KEM recommendations).
"""))

cells.append(code(r"""
# Transition checklist (engineering map — not legal advice)

checklist = [
    "Inventory: where do we use RSA / ECDH / ECDSA / similar?",
    "Prefer standardised PQC: ML-KEM (FIPS 203), ML-DSA (FIPS 204), "
    "SLH-DSA (FIPS 205); track HQC as second KEM when its FIPS lands.",
    "Prefer hybrid KEMs in TLS during transition (see Part V / RFC 10024).",
    "Use vetted libraries (e.g. OQS / FIPS modules) — do not hand-roll.",
    "Plan key sizes, Cert/CSR profiles, and middlebox ClientHello limits.",
]
for i, item in enumerate(checklist, 1):
    print(f"{i}. {item}")
"""))

# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part II — NIST portfolio map

## II.1 Key-encapsulation mechanisms

| Algorithm | Standard / status | Family (intuition) |
|-----------|-------------------|--------------------|
| **ML-KEM** | FIPS 203 (published Aug 2024) | Module lattices (Kyber lineage) |
| **HQC** | Selected for standardization (Mar 2025); FIPS draft in progress | Code-based (links to QI Hamming bridge) |

NIST positions ML-KEM as the primary general-purpose PQC KEM. HQC was
selected in the fourth round as a second KEM on a different hardness
assumption (coding), so a cryptanalytic break of lattice KEMs would not
leave deployments without an alternative once HQC's FIPS is published.

Parameter sets for ML-KEM include ML-KEM-512, ML-KEM-768, and
ML-KEM-1024 (see FIPS 203). NIST often recommends ML-KEM-768 as a
default security/performance balance — confirm in the current FIPS /
SP guidance when shipping product.

> **Further reading.** FIPS 203; NIST news
> https://csrc.nist.gov/News/2025/hqc-announced-as-a-4th-round-selection ;
> NIST IR 8545 (fourth-round status report).
"""))

cells.append(md(r"""
## II.2 Signatures

| Algorithm | Standard | Family (intuition) |
|-----------|----------|--------------------|
| **ML-DSA** | FIPS 204 | Module lattices (Dilithium lineage) |
| **SLH-DSA** | FIPS 205 | Stateless hash-based (SPHINCS+ lineage) |

ML-DSA is the primary general-purpose PQC signature. SLH-DSA rests only
on hash assumptions — conservative, with larger signatures. Parameter
sets are named in the FIPS (e.g. ML-DSA-44/65/87; SLH-DSA hash/size
variants).

> **Further reading.** FIPS 204; FIPS 205.
"""))

cells.append(md(r"""
## II.3 Families (intuition only)

- **Lattices.** Discrete subgroups $\Lambda\subset\mathbb{R}^n$; short /
  close vector problems underpin much of the lattice portfolio. Geometry
  pointer: manifolds §C4 (tori $\mathbb{R}^n/\Lambda$).
- **Codes.** Public scrambled parity-check structure; decoding without
  the trapdoor is hard. QI notebook: Hamming $[7,4]$ toy + HQC named.
- **Hashes.** Few assumptions beyond the hash; cost is signature size
  and speed (SLH-DSA).

We do **not** develop LWE proofs or decoder algorithms here.

## II.4 Explicitly deferred

Falcon (selected for ongoing signature standardization work), NIST
additional-signature on-ramp candidates, and Classic McEliece history
are named only as map entries — not labs in this notebook.

> **Further reading.** NIST PQC project “Selected Algorithms” / round
> pages (verify live).
"""))

cells.append(code(r"""
# Portfolio sketch (names only — not an implementation)

portfolio = {
    "KEMs": ["ML-KEM (FIPS 203)", "HQC (selected; FIPS in progress)"],
    "signatures": ["ML-DSA (FIPS 204)", "SLH-DSA (FIPS 205)"],
    "deferred_here": ["Falcon", "on-ramp signatures", "Classic McEliece deep dive"],
}
for k, v in portfolio.items():
    print(f"{k}:")
    for item in v:
        print(f"  - {item}")
"""))

# ---------------------------------------------------------------------------
# Part III
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part III — Interfaces and sizes

## III.1 KEM API

Abstractly:

1. $\mathsf{KeyGen}\to (pk, sk)$
2. $\mathsf{Encaps}(pk)\to (ct, ss)$
3. $\mathsf{Decaps}(sk, ct)\to ss'$

Correctness: $ss'=ss$ (except with negligible failure probability for
some schemes). Security goals are stated as IND-CCA (and related)
notions in the FIPS / SP documents — we do not prove them here.

## III.2 Signature API

1. $\mathsf{KeyGen}\to (pk, sk)$
2. $\mathsf{Sign}(sk, m)\to \sigma$
3. $\mathsf{Verify}(pk, m, \sigma)\to \{\mathsf{true},\mathsf{false}\}$
"""))

cells.append(md(r"""
## III.3 Representative sizes

Sizes below are **byte lengths** from the cited standards (encapsulation
key / decapsulation key / ciphertext for KEMs; public key / secret key /
signature for signatures). Re-check the FIPS tables if you need a
different parameter set.

| Object | Sizes (bytes) | Source |
|--------|---------------|--------|
| ML-KEM-768 | ek $1184$, dk $2400$, ct $1088$, ss $32$ | FIPS 203 |
| ML-DSA-65 | pk $1952$, sk $4032$, sig $3309$ | FIPS 204 |
| SLH-DSA-SHA2-128s | pk $32$, sk $64$, sig $7856$ | FIPS 205 |

HQC sizes depend on the parameter set and the exact specification /
library build; when OQS is present, Part IV prints live `details`.
Do not treat blog round-ups as authoritative over the FIPS PDF.
"""))

cells.append(code(r"""
# Size comparison chart (FIPS tables — cited in the markdown cell above)

sizes = {
    "ML-KEM-768\n(ct)": 1088,
    "ML-KEM-768\n(ek)": 1184,
    "ML-DSA-65\n(sig)": 3309,
    "ML-DSA-65\n(pk)": 1952,
    "SLH-DSA-SHA2-128s\n(sig)": 7856,
    "SLH-DSA-SHA2-128s\n(pk)": 32,
}

labels = list(sizes.keys())
values = [sizes[k] for k in labels]

fig, ax = plt.subplots()
ax.bar(range(len(values)), values, color="C0")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("bytes")
ax.set_title("Representative PQC object sizes (FIPS 203/204/205)")
fig.tight_layout()
plt.show()

assert sizes["ML-KEM-768\n(ct)"] == 1088
assert sizes["ML-DSA-65\n(sig)"] == 3309
print("size table OK (FIPS-sourced constants)")
"""))

# ---------------------------------------------------------------------------
# Part IV
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part IV — Open Quantum Safe (soft-gated)

[Open Quantum Safe](https://openquantumsafe.org/) provides `liboqs` (C)
and language wrappers. The Python wrapper is imported as `oqs`. This
part **must not** crash CI when `liboqs` is absent: we detect, demo, or
print an install off-ramp.

> **Further reading.** https://github.com/open-quantum-safe/liboqs ;
> https://github.com/open-quantum-safe/liboqs-python
"""))

cells.append(code(r"""
# IV.1 Detect OQS

oqs = require_oqs()
if oqs is not None:
    print("liboqs version:", oqs.oqs_version())
    kems = list(oqs.get_enabled_kem_mechanisms())
    sigs = list(oqs.get_enabled_sig_mechanisms())
    print("enabled KEM count:", len(kems))
    print("enabled sig count:", len(sigs))
    for name in ("ML-KEM-768", "ML-DSA-65"):
        pool = kems if name.startswith("ML-KEM") else sigs
        print(f"  {name} enabled:", name in pool)
"""))

cells.append(code(r"""
# IV.2 ML-KEM round-trip (when enabled)

oqs = require_oqs()
if oqs is None:
    print("skip ML-KEM lab")
else:
    kem_name = "ML-KEM-768"
    enabled = list(oqs.get_enabled_kem_mechanisms())
    if kem_name not in enabled:
        print(kem_name, "not enabled in this liboqs build; skip")
    else:
        with oqs.KeyEncapsulation(kem_name) as server:
            public_key = server.generate_keypair()
            with oqs.KeyEncapsulation(kem_name) as client:
                ciphertext, ss_client = client.encap_secret(public_key)
            ss_server = server.decap_secret(ciphertext)
        assert ss_client == ss_server
        print(
            kem_name,
            "OK; ct bytes =",
            len(ciphertext),
            "; ss bytes =",
            len(ss_client),
        )
"""))

cells.append(code(r"""
# IV.3 ML-DSA round-trip (when enabled)

oqs = require_oqs()
if oqs is None:
    print("skip ML-DSA lab")
else:
    sig_name = "ML-DSA-65"
    message = b"openqcp-lab PQC notebook demo"
    enabled = list(oqs.get_enabled_sig_mechanisms())
    if sig_name not in enabled:
        print(sig_name, "not enabled in this liboqs build; skip")
    else:
        with oqs.Signature(sig_name) as signer:
            public_key = signer.generate_keypair()
            signature = signer.sign(message)
        with oqs.Signature(sig_name) as verifier:
            ok = verifier.verify(message, signature, public_key)
        assert ok
        print(
            sig_name,
            "OK; sig bytes =",
            len(signature),
            "; pk bytes =",
            len(public_key),
        )
"""))

cells.append(code(r"""
# IV.4 HQC probe — sizes only if the mechanism is enabled

oqs = require_oqs()
if oqs is None:
    print("skip HQC probe")
else:
    kems = list(oqs.get_enabled_kem_mechanisms())
    hqc = [k for k in kems if "HQC" in k.upper()]
    if not hqc:
        print(
            "No HQC mechanism enabled in this liboqs build.\n"
            "HQC is selected for NIST standardization; enable/track "
            "it in your liboqs version when you need live sizes."
        )
    else:
        name = sorted(hqc)[0]
        with oqs.KeyEncapsulation(name) as kem:
            d = kem.details
        print("HQC mechanism:", name)
        for key in (
            "length_public_key",
            "length_secret_key",
            "length_ciphertext",
            "length_shared_secret",
        ):
            if key in d:
                print(f"  {key}: {d[key]}")
"""))

cells.append(md(r"""
## IV.5 Security hygiene

- Shared secrets and signing keys are **sensitive bytes** in process
  memory; notebook demos are not a key-management system.
- Do not commit secret keys, ciphertexts-with-secrets, or production
  seeds to git.
- Prefer FIPS-validated modules and side-channel-aware implementations
  for real deployments; OQS is excellent for prototyping and learning.
"""))

cells.append(code(r"""
print(
    "Hygiene: demo secrets stay in RAM for this session only; "
    "do not log or commit them."
)
"""))

# ---------------------------------------------------------------------------
# Part V
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part V — Protocols and providers

## V.1 PQ/T hybrid idea

A **post-quantum / traditional (PQ/T) hybrid** key agreement combines a
classical share (e.g. ECDHE) with a PQC KEM share so that an attacker
must break both to recover the session key (under the combiner's
assumptions). During transition this hedges against cryptanalysis of
either component.

> **Further reading.** RFC 9794 (PQ/T hybrid terminology — verify on
> rfc-editor); RFC 9954 (TLS 1.3 hybrid framework — verify); RFC 10024
> (concrete ECDHE–ML-KEM groups for TLS 1.3).
"""))

cells.append(md(r"""
## V.2 TLS 1.3 named groups (RFC 10024)

RFC 10024 specifies hybrid key agreement mechanisms for TLS 1.3,
including:

- **X25519MLKEM768** — X25519 + ML-KEM-768 (widely practical default)
- **SecP256r1MLKEM768** — P-256 + ML-KEM-768
- **SecP384r1MLKEM1024** — P-384 + ML-KEM-1024

Read the RFC for code points, encodings, and FIPS-related discussion.
This notebook does not run a TLS stack.
"""))

cells.append(code(r"""
# Named groups from RFC 10024 (names only)

rfc10024_groups = [
    "X25519MLKEM768",
    "SecP256r1MLKEM768",
    "SecP384r1MLKEM1024",
]
print("RFC 10024 hybrid groups:")
for g in rfc10024_groups:
    print(" ", g)
"""))

cells.append(md(r"""
## V.3 `oqs-provider`

[`oqs-provider`](https://github.com/open-quantum-safe/oqs-provider) is an
OpenSSL 3 provider that exposes liboqs algorithms through OpenSSL's
provider API (TLS 1.3 KEMs, signatures, related encoding). Wiring it is
an OpenSSL / packaging task — not a cell in this notebook.

Typical checks on a machine where the provider is installed:

```bash
openssl list -kem-algorithms -provider oqsprovider
openssl list -signature-algorithms -provider oqsprovider
```

> **Further reading.** oqs-provider `USAGE.md` on GitHub.
"""))

cells.append(md(r"""
## V.4 Production off-ramp checklist

1. Prefer FIPS 203/204/205 algorithms for general use; track HQC's FIPS.
2. Prefer hybrid groups from RFC 10024 (or successors) in TLS 1.3.
3. Use maintained libraries / providers (`liboqs`, `oqs-provider`, or
   other vetted FIPS modules) — not notebook code.
4. Follow institutional transition guidance (NIST IR 8547 and your
   sector's profiles).
5. Test ClientHello size / fragmentation on real paths.
"""))

cells.append(code(r"""
print("Protocol off-ramp: RFC 10024 + oqs-provider + IR 8547 — not this notebook's TLS.")
"""))

# ---------------------------------------------------------------------------
# Part VI
# ---------------------------------------------------------------------------
cells.append(md(r"""
# Part VI — Bridge back to the lab

- **QI notebook** — classical coding bridge and HQC as destination; this
  module continues with NIST status and OQS interfaces.
- **Manifolds foundations §C4** — elliptic curves and lattice tori as
  geometric carriers; this module does not re-plot them.
- **Non-goals here** — hand-rolled Kyber/HQC, Classiq, trotterkit /
  paulikit, full OpenSSL rebuilds in CI.

> **Further reading (primary URLs).**
> - https://csrc.nist.gov/Projects/post-quantum-cryptography/
> - FIPS 203 / 204 / 205 final publications on csrc.nist.gov
> - https://csrc.nist.gov/News/2025/hqc-announced-as-a-4th-round-selection
> - NIST IR 8547 transition report
> - https://www.rfc-editor.org/rfc/rfc10024
> - https://openquantumsafe.org/
"""))

cells.append(code(r"""
print("PQC module complete: standards map + soft-gated OQS + hybrid TLS off-ramp.")
"""))

cells.append(md(r"""
**Takeaway.** PQC replaces Shor-broken public-key trapdoors with
standardised lattice-, code-, and hash-based KEMs and signatures.
Learn the FIPS names and APIs; run OQS when you can; deploy via hybrid
TLS and vetted providers — not via re-implemented notebook crypto.
"""))

# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook(cells=cells)
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
nbf.write(nb, OUT)
print("wrote", OUT)
