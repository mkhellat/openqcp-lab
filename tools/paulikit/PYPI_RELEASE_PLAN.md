# PyPI release plan

Drafted 2026-09-10. **Stays with the monorepo** — this is release
scaffolding, not part of what ships.

Companion to `MIGRATION_PLAN.md`. Nothing here can be executed until
the new repository exists, because the sdist defect in section 2 is
fixed by the migration itself.

---

## 1. What gets published: sdist **and** wheels, not sdist alone

PyPA guidance is explicit: *"When publishing a package on PyPI (or
elsewhere), you should always upload both an sdist and one or more
wheel."* ([packaging.python.org, Package formats](https://packaging.python.org/en/latest/discussions/package-formats/))

For a package with compiled extensions this is not a formality —
`pip` cannot install from an sdist without invoking the build backend,
so an sdist-only release blocks every user without a C toolchain.

For scale, NumPy 2.5.3 currently ships **1 sdist + 60 wheels**
(manylinux and musllinux on x86-64/aarch64, macOS on both
architectures, Windows on three, across supported CPythons).

**paulikit's target for 0.1.0**, proportionate to a first release:

| artifact | why |
|---|---|
| sdist | required; the fallback for any platform without a wheel |
| manylinux x86-64 wheels, CPython 3.10–3.13 | the primary target |
| musllinux x86-64 | cheap once cibuildwheel is configured; Alpine users |
| aarch64 manylinux | `configure` already has ARM64 cache probes and NEON detection (QEMU-verified), so the claim is real — but only if CI actually builds it |

macOS and Windows are **deliberately out of scope for 0.1.0**. The
cache probing and CPU-feature detection are Linux-specific with
documented fallbacks; shipping wheels for platforms exercised by
nothing would be claiming more than has been tested. The sdist covers
them, with the pure-Python fallback.

---

## 2. The sdist defect — measured, and why migration fixes it

**Built the sdist and looked inside.** Current contents: **401
entries**, including

- **297 profiling files** — machine-bound research harnesses
- `bindings/`, `PLAN.md`, `REVIEW_NOTES.md`, `MIGRATION.md`,
  `MIGRATION_PLAN.md`, `extract-standalone`, `.mailmap-proposed`

meson-python builds the sdist via `meson dist`, which uses git's
archival mechanism: **every tracked file is included**. So today a
`pip download paulikit` would hand someone the entire research log.

The wheel is already correct — 5 `dist-info` entries plus the package,
with `LICENSE` correctly placed at
`paulikit-0.1.0.dist-info/licenses/LICENSE` (PEP 639 behaviour,
verified).

**After migration this resolves by construction**: those files will not
be tracked in the new repository. Projected sdist ≈ **82 entries**
(30 src, 16 tests, 16 docs, 10 verification, 9 root).

**Verification gate before the first upload:** build the sdist, list
it, and confirm no `profiling/`, `bindings/`, `PLAN.md` or private
material. Do not rely on the projection.

**Keep tests in the sdist.** NumPy and SciPy both do, so downstream
packagers (conda-forge, Linux distributions) can run the suite against
the built artifact. Docs are the arguable case; at 16 files they cost
little and make the sdist self-documenting, so keep them too.

**If a tracked file ever needs excluding**, the mechanism is
`.gitattributes` with `export-ignore` — meson-python honours it
because `meson dist` uses `git archive`. There is no meson-python
specific exclude key.

---

## 3. Metadata — already modern, one thing to fix

Audited `pyproject.toml` against PEP 639 (status: **Final**).

**Already correct:**

```toml
license = "GPL-3.0-or-later"     # bare SPDX expression, the modern form
license-files = ["LICENSE"]      # PEP 639 native
```

The old `license = {file = "..."}` table form is deprecated, and we do
not use it. The comment above `license-files` records a real bug this
caught: without it the built artifacts carried the licence *claim* but
no licence *text*, because the repository's GPLv3 file sits at the
monorepo root, outside what meson packages for the subproject.

Also correct: `name`, `version`, `description`, `readme`,
`requires-python = ">=3.10"`, `keywords`, `authors`,
`[project.scripts]`, and `[project.optional-dependencies]`.

**To fix before release:**

- [ ] **`[project.urls]`** — all three still point at the research
      monorepo. A consumer following "Repository" lands where paulikit
      is one tool among several. Already carries a `TODO`.
- [ ] **Classifiers** — no `License ::` entry is present, which is
      correct (PEP 639: *"New license classifiers MUST NOT be added to
      PyPI"*). Two changes worth making:
      - `Operating System :: OS Independent` **overclaims**. The cache
        probe and CPU-feature detection are Linux-specific. Replace
        with `Operating System :: POSIX :: Linux`, and add
        `Operating System :: OS Independent` back only if the
        pure-Python fallback is actually exercised on another platform
        in CI.
      - Add `Programming Language :: C` (legitimate and used by NumPy
        and SciPy for exactly this — it signals implementation
        language and is unrelated to the licence-classifier
        deprecation), plus
        `Programming Language :: Python :: Implementation :: CPython`.
- [ ] **`Typing :: Typed`** only if a `py.typed` marker is shipped.
      The code is annotated but the marker is absent; either add the
      marker and the classifier together, or neither.

---

## 4. Building the wheels

cibuildwheel is PEP 517 backend-agnostic, so it needs no special
integration with meson-python. Standard shape: a matrix job running
`pypa/cibuildwheel`, a separate sdist job, then
`pypa/gh-action-pypi-publish`.

**The gotcha that matters for us.** Every wheel build re-runs the
Meson configure step inside a manylinux container whose toolchain is
older than this development machine. Our extensions are *optional* by
design, and that design has to survive there:

- `meson.build` already uses
  `add_languages('c'/'cpp'/'cython', required: false)` and guards each
  extension on `have_cython`/`have_c`/`have_cpp`. This was fixed
  earlier precisely because required languages made the documented
  pure-Python fallback unreachable and broke CI.
- oneTBB will very likely be **absent** in the manylinux image, so
  `pauli_label_native` will not build there. That is correct
  behaviour, not a failure — but it means the published wheels may
  ship *without* the oneTBB label kernel while still carrying
  `wht_native`, `coeffs_native` and `cache_probe`, which need only C
  and Cython.
- **Therefore:** verify explicitly that a manylinux wheel contains
  `wht_native` and `coeffs_native`. Those two carry the performance
  story; a wheel missing them would silently be the slow path.

**Test matrix before the first upload:**

- [ ] Wheel installs and `pytest` passes in a clean container.
- [ ] `paulikit --help` works from the installed console script.
- [ ] `wht_native` and `coeffs_native` import from the wheel.
- [ ] sdist installs **on a machine with no compiler** and the
      pure-Python fallback engages, with results still correct.

---

## 5. Upload procedure

1. **Check the name.** `paulikit` is already PEP 503-normalised
   (lowercase, no separators), so there is no hyphen/underscore
   ambiguity. Confirm `pypi.org/project/paulikit/` is free before the
   first upload — names are first-come.
2. **TestPyPI first.** Upload sdist + wheels to `test.pypi.org` and
   install from it. TestPyPI does not mirror real PyPI, so
   dependencies need `--extra-index-url https://pypi.org/simple/`.
3. **Trusted Publishing (OIDC), not an API token.** This is PyPI's
   current recommendation
   ([docs.pypi.org/trusted-publishers](https://docs.pypi.org/trusted-publishers/)):
   the publisher relationship is configured on PyPI against the
   repository and workflow filename, and a short-lived token is minted
   per run. Nothing long-lived is stored in CI secrets.

   This matters here specifically: two Codeberg tokens were pasted
   into a chat session earlier in this project and had to be treated
   as compromised. Trusted Publishing removes the class of secret that
   can leak that way.
4. **Tag and release** only after CI is green on the new repository.
5. **DOI last.** Archive the tagged release, then add `doi:` to
   `CITATION.cff` with `version:` matching the archived tag. A
   CITATION naming a version that was never released is worse than
   none.

---

## 6. Order of operations

The dependency is strict:

```
migration (new repo, profiling/ and plan files absent)
  └─> sdist becomes clean (~82 entries, verified not projected)
        └─> URLs and classifiers fixed
              └─> cibuildwheel CI green, wheels contain the kernels
                    └─> TestPyPI dry run
                          └─> PyPI release + tag
                                └─> archive + DOI in CITATION.cff
```

Nothing before the migration is worth doing, because the sdist cannot
be made correct while the research log is tracked alongside the
package.

---

## 7. Sources

- [PyPA, Package formats](https://packaging.python.org/en/latest/discussions/package-formats/) — sdist *and* wheels
- [PEP 639](https://peps.python.org/pep-0639/) — Final; SPDX `license`,
  `license-files`, licence classifiers deprecated
- [PyPA, pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- meson-python: sdist via `meson dist` → git archive → tracked files
  only; `.gitattributes export-ignore` is the exclusion mechanism
- NumPy on PyPI — 1 sdist + 60 wheels, checked live

The exact cibuildwheel workflow YAML should be read from
[cibuildwheel's CI-services page](https://cibuildwheel.pypa.io/en/stable/ci-services/)
when the workflow is written, rather than reproduced from memory here —
that page could not be fetched in full during this research and CI
templates change.
