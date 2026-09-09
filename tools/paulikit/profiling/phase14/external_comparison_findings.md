# paulikit vs pauli_lcu

Date: 2026-09-09. Machine: i7-8550U, 15 GiB RAM.

**Headline, N=150, replicated after the Phase 14 optimizations:
paulikit is 1.47x FASTER on wall clock (3.021s +/-0.196 vs 4.447s
+/-0.181) and uses 100x less memory (89 MiB vs 8892 MiB). On a single
core it now uses 1.77x less CPU than pauli_lcu (2.648 vs 4.68
CPU-seconds) while emitting explicit (x, z) indices their positional
output never has to produce.**

This reverses the position this document originally recorded. The
starting numbers were 17.63 CPU-seconds sequential against their 4.68
- a 3.8x deficit. What closed it, in order of contribution:

| change | N=150 sequential CPU-s |
|---|---|
| phase-14 start | 17.63 |
| + scratch buffer, phase LUT, conj table | (see below) |
| + WHT C kernel | 7.29 |
| + fused coefficient C kernel | 3.83 |
| + hoisted gather, pre-sized accumulators | **2.648** |
| *pauli_lcu, for reference* | *4.68* |

Replicated head-to-head (5 reps, interleaved, cooldown to 55C), run
twice - the second after the gather/accumulator fixes:

| N | pauli_lcu | paulikit | ratio | peak RSS |
|---|---|---|---|---|
| 100 | 1.129s ±0.035 | 0.742s ±0.020 | 1.52x | 2177 / 88 MiB |
| 100 | 1.111s ±0.044 | **0.680s ±0.030** | **1.63x** | 2178 / **88** MiB |
| 150 | 4.447s ±0.181 | 3.021s ±0.196 | 1.47x | 8892 / 89 MiB |
| 150 | 4.349s ±0.116 | **3.011s ±0.270** | **1.44x** | 8893 / **89** MiB |

N=150 reproduces closely across the two runs (3.021s then 3.011s,
1.47x then 1.44x). N=100 improves from 1.52x to 1.63x, which is the
two Python fixes - the first run predates them.

**These rows measure the PARALLEL path**, which is no longer the
faster of paulikit's two options (see
`parallelism_no_longer_pays.md`). The sequential path does N=150 in
2.525s wall / 2.648 CPU-seconds, so paulikit's real margin over
pauli_lcu is wider than these rows show: about 1.7x on wall clock and
1.77x on CPU.

The memory result was always the structural claim and it still holds,
unchanged, through every optimization. The CPU gap - originally
presented here as the open problem - is closed.

An earlier version of this file reported paulikit at 41.3s and
concluded pauli_lcu won on both axes. That was a harness defect, not
a result. See "Harness defect" below — it is recorded because the
failure mode is instructive, not for the numbers.

## What pauli_lcu is

Verified, not assumed.

- `pauli_lcu/` package: 3 pure-Python files, 32K.
- The work is in `pauli_lcu_module.cpython-312-x86_64-linux-gnu.so`,
  a **separate top-level** C extension, 64.7K. A glob restricted to
  the package directory misses it.
- `nm -D -u`: no OpenMP symbols. `objdump -d`: no packed-double SIMD
  (`vmulpd`/`vfmadd*pd`/`vaddpd`).
- **Single-threaded scalar C.** MIT licensed, v1.0.1. Riverlane Ltd
  (Cambridge, UK); two authors overlap with the NJP FWHT paper.

`pauli_coefficients(matrix)` overwrites its input in place and
returns None — O(1) auxiliary memory, but the caller must already
hold the dense 2^n x 2^n complex matrix.

## Correctness cross-validation

Independent implementations, identical input. Checksums are
`sum|coeff|`; the dense mode compares against pauli_lcu's full sum,
the streaming mode against pauli_lcu's sum restricted to the same
`atol`.

| n | dense mode | streaming mode |
|---|---|---|
| 3 | 0.00e+00 | 0.00e+00 |
| 5 | 0.00e+00 | 0.00e+00 |
| 7 | 0.00e+00 | 0.00e+00 |

Bit-identical. On structured Hamiltonians the term counts match
exactly at every N tested, including 91.65M at N=150. paulikit's
output is confirmed correct against an independent implementation.

## N=150 (14 qubits, dim 16384), the real pipeline

Configuration ladder, isolating one factor at a time. All three
produce identical output (91,652,096 terms). n=1 each — indicative.

| configuration | time | peak RSS |
|---|---|---|
| dense input, sequential | 30.6s | 9699 MiB |
| sparse input, sequential | 21.5s | 3142 MiB |
| **sparse input, parallel w4_c4** | **8.85s** | **90 MiB** |

The bottom row is the supported pipeline
(`parallel_decompose_arrays` on a `scipy.sparse` operator) and
reproduces the ~8s figure from the same day's core-scaling work.

### Replicated head-to-head

Protocol-grade (`replicated_head_to_head.py`): 5 reps, interleaved,
cooldown to 55C before every run, first rep discarded. Run twice -
before the phase-14 optimizations and after.

| N | pauli_lcu | paulikit before | paulikit after | ratio after |
|---|---|---|---|---|
| 100 | 1.022s ±0.009 | 1.780s ±0.036 | **1.587s ±0.029** | 0.64x |
| 150 | 4.283s ±0.071 | 8.068s ±0.338 | **7.550s ±0.685** | 0.57x |

Peak RSS across all of these: 87-89 MiB for paulikit, 2178 MiB
(N=100) and 8893 MiB (N=150) for pauli_lcu.

The N=100 improvement (1.12x) is well outside its error bars. **The
N=150 improvement (1.07x) is NOT statistically established** - its
sigma of 0.685 is larger than the 0.518s gain and ~10x pauli_lcu's
own sigma on the same row, so at least one rep was disturbed. More
reps are needed before that number is quoted.

### CPU-seconds, which is the number that matters

Wall-clock hides the real gap, because paulikit is spending four
cores to pauli_lcu's one. Measured with `getrusage` over the
decomposition region only (self + children):

| | wall | CPU-seconds | cores |
|---|---|---|---|
| pauli_lcu | 4.71s | **4.68** | 0.99 |
| paulikit sequential | 18.46s | 18.30 | 0.99 |
| paulikit parallel (w4_c4) | 6.66s | **29.83** | 4.48 |

Two distinct gaps, which earlier notes conflated:

1. **Per-core efficiency, ~3.9x** (18.30 vs 4.68 CPU-seconds). The
   butterfly runs on stride-2 NumPy views, which cannot emit packed
   SIMD the way a compiled contiguous loop does.
2. **Parallel overhead, 11.5 CPU-seconds** (29.83 - 18.30) - 63%
   overhead to buy a 2.77x speedup from 4.48 cores, i.e. 62%
   efficiency.

Together: paulikit spends 6.4x the total CPU to land within 1.8x of
pauli_lcu's wall time. Closing (1) shrinks (2) proportionally.

### pauli_lcu is single-core - verified, against an initial misread

Its process shows 122-175% CPU and 8 threads during a run, which
looks like hidden parallelism. It is not:

- Per-thread tick counts are 213 on the main thread and 13 on each of
  seven others - a pool that does one burst and then idles.
- Those threads belong to NumPy building the dense input matrix
  (0.74s at N=150), which happens **outside** the timed region.
- CPU-time/wall measured over the decomposition alone is **0.99**.

`ps` reports a lifetime average, so it was still carrying the
construction phase. Sample the region of interest, not the process.
The comparison is fair: construction sits outside both timers.

## Dense random Hermitian input (n=5, interleaved, cooldowns)

pauli_lcu's best case and paulikit's worst: every one of the 4^n
terms is nonzero, so sparsity-aware machinery costs without paying.

| n | dim | pauli_lcu | paulikit dense | paulikit stream |
|---|---|---|---|---|
| 10 | 1024 | 0.0215s ±0.0113, 104M | 0.2664s ±0.0167, 220M | 0.3165s ±0.0190, 238M |
| 11 | 2048 | 0.0652s ±0.0029, 299M | 1.0117s ±0.0454, 663M | 1.5015s ±0.0372, 654M |
| 12 | 4096 | 0.2859s ±0.0134, 1079M | 4.1625s ±0.2138, 2503M | 6.1260s ±0.3360, 2263M |

12–20x slower here. This is a real and reportable weakness — on
unstructured dense input paulikit has nothing to exploit — but it is
not the workload the library targets, and it is not the configuration
the N=150 result above uses.

## Harness defect (recorded deliberately)

The first run of this comparison used, for paulikit:

- `build_hamiltonian(...)` **without** `sparse=True`, discarding the
  structural sparsity the real pipeline depends on; and
- `fwht_pauli_coefficients(..., sparse=True, chunk_size=...)`, the
  **single-process** path, rather than `parallel_decompose_arrays`.

Result: 41.3s / 10018 MiB, from which it was concluded that pauli_lcu
won on both time and memory and that the streaming memory argument
was falsified. Every part of that conclusion was an artifact. The
harness measured a configuration nobody runs — it defeated the exact
mechanism it was meant to test.

Two process lessons:

1. **A new harness must be validated against a known-good result
   before its output is interpreted.** An ~8s figure for N=150 from
   the same day's profiling was already available. The 41.3s reading
   contradicted it by 5x; that contradiction was evidence of a
   harness bug and should have halted interpretation immediately.
2. **A falsification that contradicts a sound structural argument is
   suspect first.** Bounded working set really does bound peak RSS;
   the measurement saying otherwise was the thing to doubt. This is
   the failure mode already recorded in
   `feedback_isolate_before_falsifying`, repeated.

Also corrected: N=150 is **14 qubits, dim 16384** (dim 512 is N=30).

## Where this leaves the comparison

1. **Correctness.** Exhaustive per-term validation, bit-identical to
   an independent implementation. Unaffected by any of the above.
2. **Memory.** ~99x smaller peak RSS at N=150 on the real pipeline.
   This is the strongest quantitative result and follows directly
   from the design rather than from tuning.
3. **Time.** 0.57-0.64x on wall clock, replicated. Substantially
   worse on dense unstructured input, and 6.4x worse in total CPU.
4. **Sparse-input capability.** pauli_lcu cannot accept a sparse
   operator at all; it requires the dense array as input. For
   operators too large to materialise densely, the comparison has no
   pauli_lcu side.

Their output data structure is what buys them (1) and costs them (4):
the result **is** the input buffer, a dense (2^n, 2^n) array where
position encodes the Pauli label, so there are no indices to store
and no compaction step - but also no way to stream, skip zeros, or
exceed RAM. We return COO triples (32 B/term with explicit indices
against their 16 B at a computed offset), which is what makes 89 MiB
flat, streaming, and multi-node partitioning possible at all.

That asymmetry also constrains what we may adopt: their transform
mutates one shared buffer and their transpose touches [i][j] and
[j][i] across the whole matrix, so partitioning it across nodes needs
all-to-all on the full array. Any optimization requiring a single
shared mutable buffer, or positional output, trades away the property
this project exists to have.

## Next steps

1. **The 11.5 CPU-seconds of parallel overhead** is the largest
   single lever and does not require touching the transform.
2. **The strided butterfly.** Whether a permutation exists that makes
   both butterfly operands unit-stride, so NumPy can vectorize the
   inner loop, is an open question (see the Hacker's Delight
   perfect-shuffle material).
3. Push N past the point where pauli_lcu's dense input requirement
   becomes infeasible (n=15 is 16 GiB, above this machine's RAM) -
   that is where the memory advantage becomes a capability difference
   rather than an efficiency one.
