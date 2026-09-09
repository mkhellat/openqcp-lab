# Migration to a standalone repository

Working checklist for extracting `paulikit` from the `openqcp-lab`
research monorepo into its own public repository, which becomes the
source for PyPI releases and the archived DOI.

This file does **not** travel with the package. Delete it once the
migration is complete.

---

## What moves, measured

| directory | files | size | decision |
|---|---|---|---|
| `src/` | 24 | 248K | **ships** - the library |
| `tests/` | 13 | 132K | **ships** - 214 tests, no hardcoded paths |
| `verification/` | 10 | 64K | **ships** - see below |
| `docs/` | 23 | 240K | **ships, minus `superpowers/`** |
| `profiling/` | 248 | **6.3M** | **stays** |

Plus root files: `README.md`, `LICENSE`, `CITATION.cff`,
`pyproject.toml`, `meson.build`, `meson.options`, `configure`,
`Makefile.in`, `.gitignore`. And the CI workflow, relocated from the
monorepo's `.github/workflows/paulikit-tests.yml` into the new
repository's own `.github/workflows/` (dropping the `tools/paulikit`
path filters and `working-directory`, which exist only because of the
monorepo layout).

### Why `verification/` ships

1. It is the evidence for the README's central claim - "every term
   verified individually, not sampled". Shipping the claim without the
   artifacts is what a reviewer objects to.
2. It is portable: zero hardcoded paths, no `/sys`, no `/proc`, no
   venv assumptions - unlike `profiling/`, anyone can run it.
3. It is 64K, about 1% of `profiling/`. Six JSON artifacts recording
   command, git commit, dependency versions, machine, and results.
4. It runs fast at small N (N=20 in ~0.02s), so a reviewer can
   verify the verifier without a long wait.

### Why `profiling/` stays

248 files and 6.3M of research log irrelevant to installing the
package; several harnesses are machine-bound (hardcoded venv paths,
Linux-only interfaces, this machine's topology - see
`profiling/README.md`); and its value is as a research record, which
the monorepo already provides.

### `docs/` needs surgery before it moves

- **`docs/plan.md` was a one-line `include` of `../PLAN.md`**, which
  stays behind. Left in place it breaks the Sphinx build, since
  `docs/index.md`'s toctree referenced it. Already removed, and the
  toctree now lists `installation` and `package_layout`, which were
  orphaned pages. Build verified.
- **`docs/superpowers/` (6 files)** holds internal design specs and
  implementation plans - the process record of how features were
  built. Research material, not user documentation. It should stay
  with the monorepo.

### Root files that stay

`PLAN.md` (the phased research plan), `REVIEW_NOTES.md`, and
`MIGRATION.md` itself.

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
4. **Broken relative links.** The README's links into `profiling/`
   and `PLAN.md` have been removed and every remaining relative link
   verified to resolve. Re-run a link check after the move anyway -
   `docs/` still has to be re-checked once `superpowers/` is dropped.

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
