# Migration to a standalone repository

Working checklist for extracting `paulikit` from the `openqcp-lab`
research monorepo into its own public repository, which becomes the
source for PyPI releases and the archived DOI.

This file does **not** travel with the package. Delete it once the
migration is complete.

---

## What moves

Only what a user or a packaging system needs:

```
src/paulikit/        the library
tests/               the test suite
docs/                tutorial, theory, background, installation, layout
verification/        exhaustive correctness runs and their artifacts
meson.build          build rules
pyproject.toml       packaging metadata
README.md
LICENSE
CITATION.cff
configure            build configuration script
Makefile.in
```

Plus the CI workflow, relocated from the monorepo's
`.github/workflows/paulikit-tests.yml` to the new repository's own
`.github/workflows/` (dropping the `tools/paulikit` path filters and
`working-directory`, which exist only because of the monorepo layout).

`verification/` moves because it is the evidence behind the
correctness claims in the README, it is reproducible on any machine,
and it is small.

## What stays behind

```
profiling/           every measurement harness and research note
PLAN.md              the phased research plan
docs/superpowers/    design specs and implementation plans
```

`profiling/` stays for three reasons: it is ~250 files of research log
irrelevant to anyone installing the package; several of its harnesses
are machine-bound (hardcoded venv paths, Linux-only interfaces, this
machine's topology - see `profiling/README.md`); and its value is as a
research record, which the monorepo already provides.

Consequence: any README or docstring reference to `profiling/` or
`PLAN.md` breaks on arrival and must be resolved before the first
release. If a document under `docs/` needs to cite a measured figure,
either inline the figure with its provenance or cite the archived
research record by DOI - not a relative path that will not exist.

## Conventional files still to add

The new repository needs the usual GNU/FOSS set. Present or absent:

| file | status |
|---|---|
| `README.md` | present |
| `LICENSE` | present (GPL-3.0-or-later, full text) |
| `CITATION.cff` | present |
| `CHANGELOG.md` | **missing** |
| `CONTRIBUTING.md` | **missing** - the monorepo has one, but it is not paulikit-specific |
| `CODE_OF_CONDUCT.md` | **missing** |
| `AUTHORS` | **missing** - conventional for GNU projects; `CITATION.cff` carries the same list but is not a substitute by convention |
| `NEWS` or release notes | optional; `CHANGELOG.md` is the modern equivalent |

## Blocking, before the first release

1. **Repository URLs.** `pyproject.toml`'s three `[project.urls]`
   entries and `CITATION.cff`'s `repository-code` all name the
   monorepo. They must name the new repository. Both files carry a
   `TODO` at the relevant line.
2. **DOI.** `CITATION.cff` deliberately has no `doi:` field. Add it
   only once a release is actually archived, and make the `version:`
   field match the tag that was archived - a CITATION.cff naming a
   version that was never released is worse than none.
3. **Version.** Currently `0.1.0` in both `pyproject.toml` and
   `CITATION.cff`. Keep them in step.
4. **Broken relative links.** Run a link check after the move; the
   README currently links to `profiling/phase13/MEASUREMENT_METHODOLOGY.md`
   and `profiling/`, neither of which will exist.

## Worth deciding deliberately

**Licence.** GPL-3.0-or-later is a considered choice, but most tooling
in this ecosystem is Apache-2.0, MIT, or BSD, and copyleft measurably
suppresses adoption where downstream projects are permissively
licensed. If uptake matters for the work's reach, this is the moment
to decide - changing it after release requires the agreement of every
copyright holder.

**History.** A fresh repository with a clean initial commit loses the
development history; a filtered extraction (`git filter-repo
--subdirectory-filter tools/paulikit`) preserves it. Preserved history
is evidence of sustained development, which some software-paper venues
weigh explicitly.
