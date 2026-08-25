# Cache locality investigation — does -O3 / -march=native matter here?

Follow-on to the other `cache_locality/` docs. Directly prompted by a
challenge: are we sure `-O3` (and by extension loop unrolling,
vectorization, `-march=native`) has nothing to do with the measured
cache behavior? That's a fair question to actually test, not assert -
"O3 isn't automatically better than O2, it's a decision that must be
made with open eyes" is a real engineering principle independent of
this specific result.

## What's actually being built right now (checked, not assumed)

`meson.build`/`meson.options` set **no explicit compiler flags at
all** - no `-O2`/`-O3`, no `-march`, no vectorization flags. The
project only sets `default_options: ['cpp_std=c++17']`. Checking the
live build directory's `meson-info/intro-buildoptions.json`:
`buildtype=release`, `optimization=3`, `debug=false` - Meson's
`release` buildtype implies `-O3` by default. Confirmed directly
against `compile_commands.json`: `pauli_label_parallel.cpp` is
compiled with `... -std=c++17 -O3 -fPIC ...`, no `-march=native`, no
other vectorization flags.

**This means `-O3` is already active, but only as an implicit
buildtype default - never a documented, deliberate decision in
`PLAN.md` or `meson.build` comments.** That gap itself is worth
closing regardless of what the numbers below show.

## Method

Reconfigured the local dev build in place with `meson configure` +
`ninja`, confirmed each rebuild actually changed the compiled `.so`
(different file size for `-O2` [103984 bytes] vs. `-O3`
[119832 bytes] - not a stale/cached artifact), ran the same N=50
`perf stat` protocol as `stall_cycles_n50_findings.md` (3 runs each),
then restored the build to its original committed `-O3` state and
confirmed via `git status` that nothing under version control was
touched (`build/` is gitignored).

```
meson configure build/cp312 -Doptimization=2   # or -Dcpp_args=/-Dc_args="-march=native"
ninja -C build/cp312
perf stat -e cycles,cache-references,cache-misses,cycle_activity.stalls_total,cycle_activity.stalls_mem_any \
  paulikit decompose --n-oscillators 50
```

Ad hoc for now, not yet wrapped into a checked-in script - reordering
build configuration mid-script is riskier to automate safely than the
read-only `perf` wrapping done so far. If this becomes a recurring
check (e.g. re-verified once the dense-array fix lands), it should
get a proper script with restore-on-exit guarantees. Not yet done.

## Results (3 runs each, N=50)

| build | cache-miss ratio | total-stall / cycles | mem-stall / cycles |
|---|---|---|---|
| `-O3` (committed default) | 54.0-57.0% | 30.0-30.3% | 18.7-18.8% |
| `-O2` | 53.3-54.3% | 30.1-30.2% | 18.7-18.9% |
| `-O3` + `-march=native` | 54.2-54.7%\* | 30.0-30.6%\* | 18.6-18.9%\* |

\* computed from the 3 runs in this session; see raw numbers in the
session's `perf stat` output if reproducing.

**All three configurations are statistically indistinguishable** -
differences are within normal run-to-run noise (compare to the
existing 3-run spreads in `stall_cycles_n50_findings.md`, which show
similar noise bands even with a single fixed build).

## Why this result makes sense, not just "nothing to see here"

This isn't a case of compiler flags being irrelevant to cache
locality in general - that would contradict real hardware behavior
(vectorization width, unrolling, and branch prediction absolutely
affect memory-access patterns and cache behavior in compute-bound
code). It's specific to *this* measurement: `perf_record_n50_findings.md`
already localized the sampled cache-miss events to symbols like
`CDOUBLE_subtract_X86_V3`, `CDOUBLE_add_X86_V3`, and
`generic_wrapped_legacy_loop` - **NumPy's own compiled ufunc code**,
which ships as a separate, pre-built binary (NumPy's own wheel) that
paulikit's `meson.build` compiler flags have no influence over
whatsoever. Compiler flags on `pauli_label_native`/`pauli_label_parallel.cpp`
can only affect *paulikit's own* compiled code - and since that code
wasn't where the sampled misses were concentrated, changing its
optimization level had nothing to bite on. This is a mechanistic
explanation for the null result, not just an empirical shrug - and it
is itself a further, independent piece of evidence supporting
`perf_record_n50_findings.md`'s localization (if that localization
were wrong and the native/TBB kernel actually dominated cache misses,
`-march=native`'s wider vector loads should have shown *some* signal
here, and it didn't).

## What this does NOT settle

- This tests N=50 only. If the dense-array-densification fix (still
  not implemented) changes *where* the hot path actually is - e.g.
  shifting real weight into `pauli_label_native`/TBB code - compiler
  flags could become relevant again after that fix, and should be
  re-tested then, not assumed to stay irrelevant forever.
- `-march=native` specifically is also a real portability hazard for
  prebuilt wheels (Phase 5) - a wheel built with `-march=native` on
  CI would crash (`SIGILL`) on end-user hardware lacking that exact
  instruction set. That's a separate, real reason to avoid it in the
  wheel-building config regardless of today's null performance
  result - already noted as a constraint in `PLAN.md` Phase 5, and
  this finding doesn't change that.
- Whether `-O3`'s *implicit* status (undocumented buildtype default
  rather than a deliberate, recorded choice) should be made explicit
  in `meson.build` with a comment explaining why, even though it
  doesn't change measured behavior today - a documentation/hygiene
  item independent of the performance question, not yet done.

## Bottom line

Directly answering the challenge: yes, we tested it, not just
asserted it. `-O2` vs `-O3` and adding `-march=native` produce no
measurable change in cache-miss ratio or stall cycles at N=50, and
there's a mechanistic reason why (the hot cache-miss path lives in
NumPy's own binary, outside paulikit's build). This doesn't mean
compiler flags never matter for this project - it means they don't
explain *this specific* measured bottleneck, and that conclusion
should be re-checked once the dense-array fix changes where the hot
path actually is.
