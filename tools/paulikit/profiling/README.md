# Phase 2 profiling results

Follows [PLAN.md](../PLAN.md) Section 5, Phase 2: `cProfile` +
`snakeviz` for an initial pass, `line_profiler` for per-line detail on
whatever cProfile flags as hot, `py-spy` as a no-instrumentation
sampling cross-check. All three tools agree on the same hot spot.

## Setup

`profile_target.py` builds the same matched-N real coupled-oscillator
Hamiltonian used by `tests/test_benchmark_reference.py` (so results
are directly comparable to the recorded PennyLane benchmark), then
calls `fwht_pauli_terms` on it. N=50 (2048x2048, 11 qubits, ~6.2s
plain run time) was chosen as the profiling target: the largest
matched-benchmark size that still runs in single-digit seconds, so it
can be profiled repeatedly. N=100 (~8192x8192) was timed once for
reference but is too slow to iterate on directly.

Reproduce:

```bash
# cProfile
python3 -m cProfile -o profiling/cprofile_n50.prof profiling/profile_target.py
snakeviz profiling/cprofile_n50.prof   # interactive flame graph, browser GUI

# line_profiler (requires @profile-decorated copy or kernprof -l)
kernprof -l -v <script importing fwht.pauli_label / fwht.fwht_pauli_terms>

# py-spy
py-spy record -o profiling/pyspy_n50.svg -- python3 profiling/profile_target.py
```

## Finding: `pauli_label` dominates, not the FWHT math

At N=50, `fwht_pauli_terms` takes 11.7s total. Breakdown by
**cumulative** time (`cprofile_n50.prof`, sorted by `cumulative`):

| function | cumtime | % of total |
|---|---|---|
| `fwht_pauli_terms` (whole call) | 11.708s | 100% |
| `pauli_label` (1,261,568 calls) | 6.980s | **59.6%** |
| `fwht_pauli_coefficients` (the actual FWHT: gather + WHT + phase) | 1.812s | 15.5% |
| `_walsh_hadamard_transform_rows` (the WHT butterfly itself) | 1.042s | 8.9% |
| `_popcount_array` | 0.463s | 4.0% |

By **self** time (`tottime`, excludes callees) the gap is even more
stark: `pauli_label` alone costs 4.902s of self time — more than
2x the *entire* `fwht_pauli_coefficients` core algorithm's self time
(0.274s self, i.e. almost all of its 1.812s cumulative is the WHT and
gather steps it calls, not label formatting).

**The algorithmic core (FWHT itself) is not the bottleneck.** The
bottleneck is the pure-Python bookkeeping step that turns numeric
`(x, z)` bitmask coefficients into `"IXYZ"`-style string dict keys,
run once per nonzero term (1.26M times at N=50).

## Line-level detail (`line_profiler`, `kernprof -l -v`)

Inside `fwht_pauli_terms`, one line accounts for 88.1% of the
function's own measured time:

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
...
1261568   32265959.6     25.6     88.1      real_terms[pauli_label(x, z, n_qubits)] = float(c.real)
```

Inside `pauli_label` itself (called 1,261,568 times, `n_qubits=11`,
so 13,877,248 inner-loop iterations):

```
Line #      Hits         Time  Per Hit   % Time  Line Contents
...
   1261568     650627.3      0.5      2.8      letters = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
   1261568     349118.4      0.3      1.5      chars = []
  15138816    4184285.3      0.3     18.2      for qubit in range(n_qubits):
  13877248    3843422.5      0.3     16.7          bit = n_qubits - 1 - qubit
  13877248    4012536.9      0.3     17.5          xj = (x_mask >> bit) & 1
  13877248    3992917.2      0.3     17.4          zj = (z_mask >> bit) & 1
  13877248    5104560.2      0.4     22.2          chars.append(letters[(xj, zj)])
   1261568     820816.5      0.7      3.6      return "".join(chars)
```

Two contributing causes, both fixable without touching the FWHT
algorithm:
1. **`letters` dict is rebuilt on every call** (2.8% of `pauli_label`'s
   own time) instead of being a module-level constant.
2. **The per-qubit Python loop itself** (lines 208-212, ~92% combined)
   is pure per-bit interpreted-Python work repeated 11-13 times per
   term, 1.26M times over. This is the real cost: bit-shifting,
   masking, and a dict lookup per qubit per term, none of which is
   vectorizable in the current one-call-per-term design.

## Cross-check (`py-spy`, sampling, ~0 instrumentation overhead)

`py-spy record` (100 Hz sampling) on the same N=50 target agrees with
the instrumented profilers without their overhead. Self-time samples
landing inside `pauli_label`, broken down by exact source line
(`pyspy_n50.svg` flame graph, leaf-frame percentages):

| line | content | % of all samples |
|---|---|---|
| fwht.py:212 | `chars.append(letters[(xj, zj)])` | 16.75% |
| fwht.py:210 | `xj = (x_mask >> bit) & 1` | 7.74% |
| fwht.py:211 | `zj = (z_mask >> bit) & 1` | 6.48% |
| fwht.py:213 | `return "".join(chars)` | 1.90% |

These four lines alone account for **~33% of every sample taken
across the entire program run** (not just within `pauli_label`) —
independent confirmation, via a fundamentally different measurement
method, of the same hot spot line_profiler found.

## Conclusion

The FWHT algorithm itself (`fwht_pauli_coefficients`: the XOR gather,
the Walsh-Hadamard butterfly, the phase-factor multiplication) is
fast and already vectorized in NumPy — it is not the target for any
future optimization work. The actual bottleneck is
`pauli_label`'s per-term, per-qubit, pure-Python string-building loop
in the `fwht_pauli_terms` convenience wrapper, called once per
nonzero coefficient.

This matters for any future Phase 3 (C porting) decision per
PLAN.md: a native port of the WHT butterfly would speed up only
~15% of current runtime, and *none* of the label-formatting cost
(~60-90% of runtime depending on term count) — porting `pauli_label`
(or vectorizing/batching label generation in NumPy, or making label
generation lazy so it only runs for terms the caller actually
inspects) is the higher-leverage target if wall-clock time on the
convenience `fwht_pauli_terms` API is the concern.
`fwht_pauli_coefficients` (the numeric-array API, no label strings)
does not pay this cost at all.

## Artifacts in this directory

- `profile_target.py` — shared setup, builds the matched-N Hamiltonian.
- `cprofile_n50.prof` — raw cProfile output, open with `snakeviz` for
  an interactive flame graph.
- `cprofile_n50_tottime.txt` — text dump, sorted by self time.
- `pyspy_n50.svg` — py-spy sampling flame graph (open in a browser).
- `pyspy_n50.speedscope.json` — same py-spy run, speedscope format
  (upload to https://www.speedscope.app/ for an interactive view).
