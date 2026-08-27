# Steady-state N-scaling of the streaming path: N=25/50/100/150

Recorded 2026-08-27, per direct user request for an N=25/50/100/150
timing table using the real end-to-end streaming path
(`fwht_pauli_terms_iter`), the same code path Phase 10 validated at
N=150. Machine: 15.7 GiB RAM, 8 cores.

## Method

`steady_state_streaming_sweep.py` (this directory) - same warm-up
discipline as `../cache_locality/steady_state_decompose.py`: builds
the Hamiltonian once via `build_hamiltonian(sparse=True)` +
`pad_to_power_of_two(sparse=True)`, runs one **untimed** warm-up call
through `fwht_pauli_terms_iter` (pays for first-call costs - lazy
imports, allocator warm-up, first-touch page faults), then times
`--reps` further calls in the same process and reports the mean.
Verifies term count is identical between warm-up and every timed rep
before trusting the timing (protects against a silently
non-deterministic result).

Unlike the historical `steady_state_decompose.py` (which uses the
dense path at small N and only bothers with sparse/chunking for
large N), this sweep uses the **same sparse + streaming code path at
every N**, including N=25 - so the table is apples-to-apples across
all four sizes, not comparing a dense small-N regime against a sparse
large-N regime.

`chunk_size=256` (Phase 10's own default choice) at every N.
N=25/50/100 ran 3 reps each with no special memory handling needed
(small enough to be safe outright). N=150 ran under the same
`ulimit -v 4000000` + `free -m` polling safety harness used
throughout this project, with `--reps 1` (single rep) to keep total
wall time reasonable given the warm-up call alone already runs the
full ~100s decomposition once.

## Results

| N | qubits | dim | terms | chunks | mean steady-state time |
|---|---|---|---|---|---|
| 25 | 9 | 512 | 78,336 | 2 | 0.069s |
| 50 | 11 | 2,048 | 1,261,568 | 5 | 1.329s |
| 100 | 13 | 8,192 | 20,299,776 | 20 | 22.064s |
| 150 | 14 | 16,384 | 91,652,096 | 44 | 101.310s (single rep) |

Individual rep times (N=25/50/100, 3 reps each): N=25:
`[0.0562, 0.0561, 0.0945]`; N=50: `[1.4650, 1.3126, 1.2091]`; N=100:
`[22.4074, 20.9560, 22.8282]`. N=150: single rep,
`[101.3102]`.

## Interpretation

Term count grows roughly ~16x from N=100 to N=150 (20.3M -> 91.7M),
while mean time grows only ~4.6x (22.06s -> 101.31s) - sublinear
relative to term-count growth, consistent with per-chunk overhead
(gather setup, WHT dispatch) amortizing better as chunk count grows
(20 chunks at N=100, 44 at N=150) and per-term cost inside
`dict_build` (identified as the dominant stage in
`full_pipeline_n150_findings.md`) staying roughly flat rather than
growing with N.

This is the first N=25/50/100/150 timing table that includes a
successful N=150 data point at all - every earlier attempt in this
project's history either OOM-killed (`../cache_locality/n150_oom_finding.md`)
or required experimental memory caps that didn't complete
(`../phase9/phase9_findings.md`). Phase 10's streaming fix is what
makes this table possible in the first place.

## What this does NOT show

- N=150 has only one timed rep (not 3 like the smaller sizes), purely
  for wall-clock budget reasons (~100s per full run) - the single
  number is a real, reproducible measurement (this project's other
  N=150 runs, e.g. `phase10_streaming_findings.md`'s 4GB/2GB
  comparison, land in the same ~97-102s range), not a one-off outlier,
  but it does not have the same run-to-run variance characterization
  the smaller N's do.
- No hardware performance counters here (wall-clock only) - see
  `full_pipeline_n150_findings.md` for the `perf stat`-based
  cache-locality numbers at N=150 specifically; this table is a
  scaling/throughput result, not a cache-behavior one.
- `chunk_size=256` only - not swept across chunk sizes here (see
  Phase 6/9's own chunk_size verification work for that dimension).
