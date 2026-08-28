# paulikit correctness verification

This directory holds an independent, exhaustive correctness check for
`paulikit.algorithms.fwht.fwht_pauli_terms`, separate from the
existing benchmark comparison in
`tests/test_benchmark_reference.py` (which only checks term *counts*,
not coefficient values, at large N — see that file's docstring for
why).

## Why this exists

PennyLane's `qml.pauli_decompose` is O(n · 4^n) **regardless of input
sparsity** — confirmed by reading its source (see `FINDINGS.md`). It
cannot reach paulikit's target scale (N~150 oscillators). So a
coefficient-level correctness check at that scale needs an
independent method that isn't PennyLane. `exhaustive_projection.py`
computes each Pauli label's coefficient directly from the projection
formula `Tr(H @ P_label^dagger) / dim`, without materializing the
full Pauli matrix — this is mathematically independent of paulikit's
own FWHT-based algorithm, not a re-derivation of the same shortcut,
and it scales with `nnz(H) * n_terms`, not `4^n`.

**Coverage is exhaustive, not sampled**: every term paulikit reports
for a given input is individually verified — this was an explicit,
non-negotiable requirement (see `FINDINGS.md`'s process notes).

## Reproducing any result

Every result file under `results/` records the exact command that
produced it in its `"command"` field. To reproduce:

```bash
cd tools/paulikit/verification
../../../tools/paulikit/.venv/bin/python run_verification.py --n <N> --hermitian --against projection
# or, with the venv from ./configure:
~/.venvs/paulikit/bin/python run_verification.py --n <N> --hermitian --against projection
```

Each run also records: paulikit git commit, Python/numpy/scipy/
pennylane versions, platform string, and the exact synthetic
spring-constant/mass generator used (deterministic, no randomness for
`--hermitian` runs; `--seed` for `--non-hermitian` runs, default 42).
Rerunning the same command should reproduce the same `max_abs_error`
exactly (floating-point deterministic) and `wall_time_s` approximately
(machine-load-dependent).

## Usage

```bash
# Large N: projection-only (PennyLane cannot finish at this scale)
python run_verification.py --n 150 --hermitian --against projection

# Small N: dual-oracle (PennyLane AND projection both required to agree)
python run_verification.py --n 20 --hermitian --against both

# Non-Hermitian input
python run_verification.py --n 20 --non-hermitian --against both
```

Results are written to `results/N<n>_<hermitian|nonhermitian>_<date>.json`
and also printed to stdout.

## Files

- `exhaustive_projection.py` — the verification method itself (real,
  importable module — not a scratch script). See its module docstring
  for the math and the 4-iteration performance history.
- `run_verification.py` — the single reproducible CLI entry point.
  Every number in `FINDINGS.md` or project memory must trace back to
  a JSON file this script produced.
- `results/` — one JSON file per run, committed to the repo so past
  results remain inspectable without rerunning.
- `FINDINGS.md` — narrative writeup: PennyLane's O(n·4^n) proof, the
  4 iterations of the projection method (with real timings at each
  stage, including the ones that didn't scale), and the final
  measured results across N.

## Relationship to the test suite

`tests/test_exhaustive_verification.py` imports `exhaustive_projection.py`
directly and runs it as real pytest cases: fast small-N cases run in
CI, large-N cases are marked `slow` (same convention as
`test_benchmark_reference.py`).
