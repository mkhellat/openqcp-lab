# N=150 still OOMs after the Phase 6 sparse fix

Recorded 2026-08-26, immediately after `n150_oom_finding.md`'s dense-path
OOM and alongside `phase6_dense_vs_sparse_findings.md`'s N=25/50/100 A/B
comparison. This finding fills the gap that comparison left open: it
explicitly excluded N=150 ("N=150 remains open and is tracked
separately"), and this is that separate check - does `sparse=True` alone
make N=150 survive?

## What happened

`steady_state_decompose.py --n-oscillators 150 --reps 1` run with
`fwht_pauli_coefficients(operator, sparse=True)` (Phase 6's fix, no other
change) under four conditions:

| condition | result |
|---|---|
| `ulimit -v 8388608` (8 GiB) | `ArrayMemoryError` at `gathered_active = np.zeros((11189, 16384), complex)` — 2.73 GiB |
| `ulimit -v 10485760` (10 GiB) | `ArrayMemoryError` at `_walsh_hadamard_transform_rows`'s `array.copy()` — same 2.73 GiB array, one call later |
| `ulimit -v 12582912` (12 GiB) | same as the 10 GiB case |
| no cap (real machine, ~10 GiB available) | killed by the OOM killer, exit code 137, no traceback |

Raw `perf stat` + traceback output for all four:
[`phase6_n150_sparse_capped_20260826T045021Z.txt`](phase6_n150_sparse_capped_20260826T045021Z.txt),
[`phase6_n150_sparse_capped_10g_20260826T045105Z.txt`](phase6_n150_sparse_capped_10g_20260826T045105Z.txt),
[`phase6_n150_sparse_capped_12g_20260826T045155Z.txt`](phase6_n150_sparse_capped_12g_20260826T045155Z.txt),
[`phase6_n150_sparse_only_attempt2_20260826T044742Z.txt`](phase6_n150_sparse_only_attempt2_20260826T044742Z.txt)
(the fifth file,
[`phase6_n150_sparse_only_20260826T044412Z.txt`](phase6_n150_sparse_only_20260826T044412Z.txt),
is a one-line note from an earlier attempt in the same session, skipping
the dense leg since it was already confirmed to OOM by
`n150_oom_finding.md`).

## Why sparse alone wasn't enough

`sparse=True` avoids the dense `(dim, dim)` allocation
(`n150_oom_finding.md`'s 4 GiB `coefficients` array), but
`gathered_active` is still shaped `(n_active, dim)` -
`(11189, 16384)` complex128 = 2.73 GiB - and
`_walsh_hadamard_transform_rows` immediately `.copy()`s it, briefly
holding two such buffers live (~5.5 GiB) before the first is freed. At
N=150, `n_active` (11189) is already large enough relative to `dim`
(16384) that the sparse array is not meaningfully smaller than the
memory pressure that killed the dense path - both `gathered_active`'s
allocation and the transform's own working copy are large enough,
individually, to exceed what was available under the tested caps and
on the unconstrained machine.

This matches `phase6_dense_vs_sparse_findings.md`'s conclusion that
Phase 6 is a crash-avoidance fix, not a cache-locality one - but shows
its crash-avoidance guarantee has a limit: it moves the OOM boundary,
it does not remove it. `sparse=True` was necessary but not sufficient
for N=150.

## What resolved it

Not this fix. N=150 was later made to work via the streaming API
(`fwht_pauli_terms_iter`, chunk_size-bounded), landed in Phases 8-10 -
see [`../phase10/full_pipeline_n150_findings.md`](../phase10/full_pipeline_n150_findings.md),
which measures the real, working N=150 pipeline these four capped runs
could not reach. The fix was bounding `gathered_active`/transform
buffer size to `chunk_size` rows at a time instead of `n_active` rows
at once, not anything about the dense/sparse coefficient-array choice
itself.
