# Migration plan: a fresh, professionally-staged paulikit repository

Drafted 2026-09-10. **This file stays with the monorepo** — it is the
plan for the move, not part of what moves.

Supersedes the "what moves" audit in `MIGRATION.md`, which remains the
record of *why* each directory ships or stays. This document covers
*how*: the commit sequence, the layout, and the checklist.

---

## 0. The decision that shapes everything: fresh history, staged

`extract-standalone` preserves the monorepo's 133 filtered commits.
That is one valid option. **This plan proposes the other**: a fresh
repository built from a deliberate sequence of commits.

**Why fresh, given that preserved history is normally the better
default:**

- The monorepo history interleaves paulikit with tutorials, module 05
  work, and Classiq notebooks. Filtered, it reads as a stream of
  fixes to code the reader has not seen introduced.
- Commit messages there were written for a research log, not for
  someone learning the codebase.
- A reader arriving at a new package wants to see the architecture
  assembled in dependency order. That is a different artifact from a
  development record, and it is the one that serves them.
- **The development record is not lost.** It stays in
  `openqcp-lab`, which keeps `profiling/` (297 tracked files, 6.6 MB
  of measurement evidence), `PLAN.md`, and the full commit history.
  The new repository's README will link to it explicitly.

**The cost, stated plainly:** the new repository will show ~20 commits
on day one rather than a year of development. Some software-paper
venues weigh sustained-development evidence. The monorepo remains the
citable record of that, and `CITATION.cff` can reference both.

If you would rather keep the filtered history instead, say so — the
`extract-standalone` path is already written and tested (verified
380 → 282 → 133 commits, 214 tests passing in the extracted tree).
The two are mutually exclusive.

---

## 1. Location and naming

**Local:** a new directory named `paulikit`, at a location you
specify. Not inside `openqcp-lab`.

**Codeberg:** `codeberg.org/beavernets/paulikit` — the org exists
(HTTP 200) and SSH auth is verified working.

**GitHub:** `github.com/mkhellat/paulikit` initially.

The `beavernets` GitHub name is **not** available and will not be for
~88 more days. GitHub documents a **90-day hold** after account
deletion ("Your username will be available for anyone to use after 90
days" — GitHub Docs, *Personal account reference*). Waiting overnight
will not help.

Recommendation: publish under `mkhellat/paulikit` now and transfer to
the organisation when the name frees. GitHub transfers preserve
stars, issues, and set up redirects from the old URL, so nothing is
lost. Choosing a different permanent org name would lock in a
compromise for a temporary problem.

One risk to check: if any repository under the old `beavernets`
account crossed GitHub's popularity thresholds (a Marketplace Action,
>100 clones or Action-uses in its final week, or a container image
with >5,000 downloads), the `OWNER/REPO` namespace is **permanently
retired** for repojacking protection and never returns. If the account
was new and low-traffic, this does not apply.

---

## 2. Layout of the new repository

```
paulikit/
├── .github/workflows/tests.yml   CI, monorepo path filters stripped
├── .gitignore
├── AUTHORS                       GNU convention; four authors
├── CHANGELOG.md                  starts at 0.1.0
├── CITATION.cff                  DOI added only once archived
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md               paulikit-specific, not inherited
├── LICENSE                       GPL-3.0-or-later, full text
├── Makefile.in                   configure-generated build entry
├── README.md
├── configure                     cache probes, SIMD detection
├── meson.build                   optional C/C++/Cython languages
├── meson.options                 native / cache_probe / wht_kernel
├── pyproject.toml                meson-python backend
├── docs/                         Sphinx source (no _build, no superpowers)
│   ├── api/                      7 files
│   ├── background.md  conf.py  index.md  installation.md
│   ├── non_hermitian.md  package_layout.md  theory.md  tutorial.md
│   └── Makefile
├── src/paulikit/
│   ├── __init__.py  cli.py  hamiltonian.py  pauli_utils.py
│   ├── algorithms/   __init__.py  autotune.py  fwht.py
│   ├── _native/      wht.{c,h}  coeffs.{c,h}  pauli_label.{c,h}
│   │                 pauli_label_parallel.{cpp,h}  cache_probe.{c,h}
│   │                 *.pyx  test_*.c  meson.build
│   └── testing/      __init__.py  fixtures.py
├── tests/                        16 files, 312 tests
└── verification/                 exhaustive per-term evidence
    ├── FINDINGS.md  README.md
    ├── exhaustive_projection.py  run_verification.py
    └── results/                  6 JSON artifacts
```

**Explicitly absent, and why:**

| not shipped | reason |
|---|---|
| `profiling/` | 297 tracked files of research log; several harnesses are machine-bound (hardcoded venv paths, `/sys`, this machine's topology). Stays as the monorepo's research record. |
| `bindings/` | Phase 3a's four-binding comparison — a historical artifact, superseded by the shipped Cython path. |
| `PLAN.md` | the phased research plan; a process document. |
| `REVIEW_NOTES.md`, `MIGRATION.md`, `MIGRATION_PLAN.md` | migration and review scaffolding. |
| `docs/superpowers/` | internal design specs and implementation plans. |
| `docs/_build/` | 15 MB of generated HTML; never tracked. |
| `.mailmap-proposed` | a decision memo, not a `.mailmap`. |
| **`PRIVATE_PERF_PLAN.md`** | **private. Never to appear on any remote.** |

### PRIVATE_PERF_PLAN.md — verified, and guarded

Confirmed clean before this plan was written:

- **Zero commits** touch it anywhere in the monorepo history.
- It appears in **no tree on any branch**.
- It is gitignored (`.gitignore:66`).

So it has never reached Codeberg or GitHub. It is additionally listed
in `extract-standalone`'s prune set, and that script now **fails
hard** if the file is tracked or appears in history after extraction —
a guard that costs nothing and cannot be forgotten.

---

## 3. Commit sequence

Twenty commits in dependency order, so a reader can follow the build
from the ground up. GNU-style messages throughout: a short imperative
subject line under 50 characters, a blank line, then a body explaining
*why* — wrapped at 72 characters.

Each commit is intended to leave the tree in a coherent state; tests
arrive alongside the code they cover rather than in a lump at the end.

| # | commit | contents |
|---|---|---|
| 1 | Add project scaffolding and licence | `LICENSE`, `.gitignore`, `README.md` (stub) |
| 2 | Add packaging and build configuration | `pyproject.toml`, `meson.build`, `meson.options` |
| 3 | Add the configure script | `configure`, `Makefile.in` |
| 4 | Add the package skeleton | `src/paulikit/__init__.py`, `src/paulikit/meson.build` |
| 5 | Add Hamiltonian construction | `hamiltonian.py` + `tests/test_sparse_hamiltonian.py` |
| 6 | Add Pauli utilities | `pauli_utils.py` |
| 7 | Add the FWHT decomposition core | `algorithms/{__init__,fwht}.py` + `tests/test_fwht.py` |
| 8 | Add streaming and chunked decomposition | streaming API + `test_streaming.py`, `test_chunked_accumulator.py` |
| 9 | Add the binary checkpoint format | checkpoint code + `test_checkpoint_format.py`, `test_progress_marker.py` |
| 10 | Add cache-aware auto-tuning | `algorithms/autotune.py` + `test_autotune.py` |
| 11 | Add the cache-latency probe extension | `_native/cache_probe.*` + `test_cache_probe.py` |
| 12 | Add the Pauli label C kernel | `_native/pauli_label*` (C, C++, pyx) |
| 13 | Add the Walsh-Hadamard butterfly kernel | `_native/wht.{c,h}`, `wht_native.pyx` + `test_wht_kernel.py` |
| 14 | Add the fused coefficient kernel | `_native/coeffs.{c,h}`, `coeffs_native.pyx` |
| 15 | Add multi-core decomposition | `parallel_decompose*` + `test_parallel_decompose.py`, `test_array_yielding.py` |
| 16 | Add the threaded drain | `executor=` + `test_threaded_drain.py` |
| 17 | Add the command-line interface | `cli.py` + `test_cli_parallel.py` |
| 18 | Add test fixtures and their generator | `testing/`, `test_fixtures.py`, `test_benchmark_reference.py` |
| 19 | Add exhaustive verification evidence | `verification/` + `test_exhaustive_verification.py` |
| 20 | Add documentation | `docs/`, final `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `AUTHORS`, `CHANGELOG.md` |
| 21 | Add continuous integration | `.github/workflows/tests.yml` |

**Why this order.** Build config before code, so the tree is
installable from commit 2. Pure-Python core before native kernels, so
the fallback path is established first and each kernel arrives as a
visible accelerator with its own tests. Parallelism after the kernels,
because the threaded drain's justification depends on them releasing
the GIL. CLI late, since it composes everything. Documentation last,
so it describes a finished system.

**Verification gate at each step:** commits 5 onward must leave
`pytest` green for the tests present at that point. Commit 21 is where
CI takes over.

---

## 4. Checklist — what must exist on the new remote, and why

### Blocking before the first push

- [ ] **Repository URLs.** `pyproject.toml`'s three `[project.urls]`
      entries and `CITATION.cff`'s `repository-code` currently name
      the monorepo. Both carry a `TODO` at the relevant line.
- [ ] **`AUTHORS`** — GNU convention. Khellat, Masoumi, S. Nasouri,
      S. Nasouri, all Beavernets Technologies.
- [ ] **`CONTRIBUTING.md`** — paulikit-specific: `make build`,
      `make test`, the optional-extension model, commit conventions.
- [ ] **`CODE_OF_CONDUCT.md`** — conventional for a public project.
- [ ] **`CHANGELOG.md`** — starting at 0.1.0.
- [ ] **CI workflow** relocated, monorepo path filters and
      `working-directory` stripped.
- [ ] **Link check** — every relative link resolves after the move;
      no dangling references to `profiling/` or `PLAN.md`.

### Verify before pushing

- [ ] `PRIVATE_PERF_PLAN.md` absent from tree **and** history.
- [ ] `docs/_build/`, `bindings/`, `profiling/`, `docs/superpowers/`
      all absent.
- [ ] `pytest` green — 312 tests.
- [ ] `make build` succeeds from a clean checkout.
- [ ] Sphinx builds without new warnings.
- [ ] Package installs and `paulikit --help` works.

### Deliberately deferred

- [ ] **DOI** — added to `CITATION.cff` only once a release is
      actually archived, with `version:` matching the archived tag. A
      CITATION naming an unreleased version is worse than none.
- [ ] **PyPI publication** — after the repository is public and CI is
      green.
- [ ] **GitHub org transfer** — once `beavernets` frees (~88 days).

### Decisions still open

- **Licence.** GPL-3.0-or-later is a considered choice, but most of
  this ecosystem is Apache-2.0/MIT/BSD, and copyleft measurably
  suppresses adoption where downstream projects are permissive.
  Changing it after release requires every copyright holder's
  agreement, so this is the moment.
- **`.mailmap`.** All monorepo commits carry `mkhellat@gmail.com`
  while `CITATION.cff` credits four authors at `@beavernets.com`. On
  a fresh-history repository this largely evaporates — the new
  commits can carry the institutional address from the start.
- **Fresh history vs filtered** — section 0.

---

## 5. Execution order

1. You confirm: fresh-history staging (this plan) or filtered history
   (`extract-standalone`), the local directory location, and the
   licence question.
2. Create the local repository and the two remotes (`gh` is
   authenticated as `mkhellat`; Codeberg SSH is verified).
3. Write the missing conventional files (AUTHORS, CONTRIBUTING,
   CODE_OF_CONDUCT, CHANGELOG) and fix the URL TODOs.
4. Stage commits 1–21, running the verification gate at each step.
5. Push to both remotes via `proxychains4`.
6. Confirm CI green, then tag `v0.1.0`.
