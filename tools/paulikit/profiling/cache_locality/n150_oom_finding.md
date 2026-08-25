# N=150 OOM-kills on this machine - unmodified code, existing bug

Recorded 2026-08-25 while attempting to extend the standard
cache-locality sweep to N=25/50/100/150 (per project convention as of
this date). **No source code was modified before or during this
test** - `git diff --stat tools/paulikit/src/` was empty at the time
this was recorded. This is the existing, unmodified Phase 3c
`fwht_pauli_coefficients`/`fwht_pauli_terms` code (see
`perf_record_n50_findings.md` for where the dense-array behavior
lives), hitting a real limit on its own - not something introduced by
this investigation's profiling scripts, and specifically **not**
caused by removing or weakening Phase 3b's sparse-computation
optimization, which is untouched.

## What happened

`steady_state_decompose.py --n-oscillators 150 --reps 1` (see that
script for why steady-state, in-process timing was introduced) was
killed by the OOM killer (exit code 137) after 4 minutes 56 seconds,
having swapped 2.8 GiB, without completing even the single untimed
warm-up call. Machine: 15 GiB RAM, ~10 GiB available at start (per
`free -h` immediately before the run).

## Why: the same dense-array bug, now quantified at N=150

`perf_record_n50_findings.md` identified `fwht_pauli_coefficients`
allocating a dense `(dim, dim)` complex128 array, then
`fwht_pauli_terms` allocating a same-size `np.abs()` float64
temporary to scan it. At N=150 (14 qubits, dim=16384):

- `coefficients` array: 16384^2 x 16 bytes = **4.00 GiB**
- `np.abs(coefficients)` temporary: 16384^2 x 8 bytes = **2.00 GiB**
- peak concurrent (both live before either is freed): **6.00 GiB**

That 6 GiB is just these two arrays - it doesn't include the input
Hamiltonian, `gathered_active`, WHT intermediate buffers, or the
label-string/dict construction that follows. Against ~10 GiB
available (and less once the interpreter, NumPy, and other resident
processes are accounted for), a single call's real peak plausibly
exceeds what was actually free, which matches the observed result:
it died before even completing the *first* (untimed, warm-up) call -
this looks like a single-call OOM, not a multi-rep accumulation
effect from the driver script's repetition.

## Why this matters for the fix

This is not merely a cache-locality nicety anymore - it's evidence
that the dense-array densification identified in
`perf_record_n50_findings.md` is a genuine **robustness/correctness
risk**, not just a performance one. A user with this problem's
natural next scaling step (N=150, or realistically anyone on a
laptop-class machine going somewhat further) can have the process
killed outright, with no graceful error - just a `SIGKILL` from the
kernel. That's a materially stronger argument for prioritizing the
fix than "the cache-miss ratio is elevated."

## What this does NOT show

- This is a single data point on one machine (15 GiB RAM) - it does
  not establish an exact N threshold in general; a machine with more
  RAM would push the failure point higher, and this should not be
  read as "N=150 is universally impossible."
- Not yet tested: N=150 after the dense-array fix (not yet designed
  or implemented) - the expectation is that a sparse-output fix
  should make N=150 memory-safe (proportional to `n_active x dim`
  rather than `dim^2`), but this is a prediction to verify, not yet a
  measured result.
- The standard N=25/50/100/150 sweep (per project convention as of
  2026-08-25) cannot currently include N=150 safely on this machine
  until either the fix lands or the sweep script gains a memory
  precheck/guard - tracked as open, not silently dropped from the
  convention.
