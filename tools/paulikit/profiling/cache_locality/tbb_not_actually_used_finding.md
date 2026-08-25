# TBB is not actually invoked in the production call path - correcting an assumption

Recorded 2026-08-25, prompted by a direct question: "we have TBB there
as well who is supposed to aggressively distribute loads between the
workers" - raised in the context of whether OpenBLAS's 8-thread pool
(see `stall_floor_mystery_solved.md`) might be contending with TBB's
own worker threads for the same 8 cores. Checking this directly
revealed something more fundamental: **TBB's parallel kernel is not
called by `fwht_pauli_terms` at all**, so the OpenBLAS/TBB contention
question doesn't apply - there's no active TBB thread pool in the
current production code path to contend with anything.

## What was checked

Sampled OS-level thread count (`/proc/<pid>/task`, the correct
ground-truth measurement per `stall_floor_mystery_solved.md`'s update)
before, during, and after a `fwht_pauli_terms` call at N=100 (a size
that definitely exercises the native extension). Thread count stayed
at exactly 8 throughout - the same 8 threads OpenBLAS's pool created
at `import numpy` time, no additional threads appeared during the
native-kernel call.

## Why: read the source, don't assume from the name

`src/paulikit/_native/pauli_label_native.pyx` exposes two batch
entry points:

- `pauli_label_batch` - calls the **serial** C kernel
  (`pauli_label.c`).
- `pauli_label_batch_parallel` - calls the **oneTBB-parallelized**
  C++ kernel (`pauli_label_parallel.cpp`).

`src/paulikit/algorithms/fwht.py`'s `_pauli_label_batch` (the only
caller inside the actual decomposition pipeline) calls
`_native.pauli_label_batch` - **the serial one**. The parallel
entry point is compiled, linked, tested, and available, but not
wired into `fwht_pauli_terms` by default.

The `.pyx` file's own docstring (written during Phase 3a/3b, not new
information - just something this investigation hadn't re-surfaced
until now) explains why:

> "Not currently used by `fwht_pauli_terms` - Phase 3a found
> parallelizing this specific loop barely helps end to end, since
> Python str construction (not the C loop) dominates wall-clock time
> (see `bindings/README.md`'s oneTBB section) - kept available for
> completeness, not wired in by default."

This is a real, previously-established finding from Phase 3a - this
document isn't discovering something new about the code, it's
correcting an assumption this investigation (and the earlier Google
AI Mode transcript that kicked off the whole cache-locality thread)
had been carrying without re-checking: that TBB is actively
"aggressively distributing loads" in the hot path today. It is not.

## Correcting PLAN.md Phase 6's framing

`PLAN.md`'s Phase 6 section (scoped earlier in this same session)
says: "this is not motivated by any finding in this investigation and
would be scope creep" regarding the native kernel - that statement
itself is still correct (the dense-array bug is genuinely unrelated
to TBB), but the section's surrounding language implicitly assumed
TBB is part of the active hot path when discussing what's out of
scope. Worth a small follow-up correction to make explicit that TBB
isn't just "not the bottleneck" but "not even invoked" - a stronger,
more precise statement. Not yet applied to PLAN.md as of this doc's
writing - tracked here first.

## Does this change anything about the dense-array fix (Phase 6)?

No, and it doesn't need to. The root-cause finding
(`perf_record_n50_findings.md`) already localized the cache misses to
NumPy ufunc code and CPython object churn, not to
`pauli_label_native`/TBB - which is now even more clearly explained,
since the TBB path was never running to begin with. Phase 6's fix
target (`fwht_pauli_coefficients`'s dense-array densification) is
unaffected by this finding either way.

## Open question this raises, not yet investigated

If TBB parallelization was found "barely helpful" for the label-
generation loop specifically (a string-construction-bound workload),
would it be more effective if wired into a *different* part of the
pipeline - e.g. as part of a Phase 6 sparse-representation fix, if
that fix ends up restructuring how `active_x`/`active_coefficients`
are processed? This is speculative, not something to chase now, but
worth flagging as a possible consideration during Phase 6's
"prototype and measure both real candidates" step (see PLAN.md), not
a settled recommendation.
