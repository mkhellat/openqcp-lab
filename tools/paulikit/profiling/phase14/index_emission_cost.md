# What does COO output actually cost us?

2026-09-09. pauli_lcu's output is positional - the array offset *is*
the Pauli label - so it never emits indices. We return explicit
`(x, z, coeff)` triples. Before treating the whole gap as inefficiency,
that structural difference has to be priced.

Measured on a **real** N=150 chunk (`chunk_size=2`, `dim=16384`,
16,384 surviving terms, 50% block density), median of 40 reps per
stage.

| stage | µs | share | index emission? |
|---|---|---|---|
| gather (sparse -> dense block) | 68.3 | 3.0% | |
| **butterfly (WHT)** | **1592.6** | **69.5%** | |
| phase factor | 259.0 | 11.3% | |
| scale by phase | 49.9 | 2.2% | |
| threshold / `nonzero` | 212.7 | 9.3% | yes |
| gather x, coeff | 84.8 | 3.7% | yes |
| narrow dtype | 25.7 | 1.1% | yes |
| **TOTAL** | **2293.0** | | |
| index emission subtotal | 323.2 | **14.1%** | |

Per output term:

| | ns/term |
|---|---|
| paulikit total | 140.0 |
| — of which index emission | **19.7** |
| — of which everything else | 120.2 |
| pauli_lcu, total | **51.06** |

## Conclusions

1. **Index emission costs 19.7 ns/term, 14.1% of the work.** This is
   the real, quantified price of COO output. It is not waste - it is
   what buys streaming, bounded memory, and chunk independence. It is
   the one part of the gap that should *not* be optimized away.

2. **It is not the problem.** Even with index emission free, the
   transform alone is 120.2 ns/term against pauli_lcu's 51.06 for the
   entire decomposition - still 2.4x slower.

3. **The butterfly is 69.5% of per-chunk work** and is the only stage
   whose improvement can close the gap. Everything else combined is
   30.5%; eliminating *all* of it entirely would leave us above
   pauli_lcu's total.

## Status of the native extension, checked rather than assumed

The `_native` package contains `pauli_label` (label string generation)
and `cache_probe`. Both are built and importable in the current venv
(`build/cp312/…*.so`).

**The transform has never been ported.** The only native call site in
`fwht.py` is inside `_pauli_label_batch` - and
`parallel_decompose_arrays`, the path every N=150 measurement uses,
never generates labels at all. So the compiled kernel contributes
**nothing** to the numbers in this phase: 140 ns/term is pure NumPy.

That is the gap between where the C infrastructure exists (labeling,
already fast, and skipped by the array API) and where the time
actually goes (the butterfly, 69.5%, pure NumPy on stride-2 views).

## Constraints any butterfly kernel must preserve

Non-negotiable, from the design goals this project exists to serve:

- **Peak RSS must stay ~89 MiB and flat in N.** The kernel operates on
  one `(chunk_size, dim)` block at a time - 512 KiB at N=150 - and
  must not allocate anything proportional to `dim²` or to the total
  term count. In-place on the caller's block, one reusable scratch
  buffer at most.
- **Chunk independence must survive.** No shared mutable state across
  chunks, no cross-chunk reduction, no global buffer. This is what
  makes `ProcessPoolExecutor` work today and MPI possible later.
- **Output stays COO.** Positional output is disqualified regardless
  of speed - it is exactly the thing that makes pauli_lcu unable to
  stream or exceed RAM.
- **GIL released during the transform**, so worker processes (and any
  future threading) are not serialized on it.

A kernel meeting all four is a drop-in replacement for
`_walsh_hadamard_transform_rows`: same signature, same in-place
semantics, operating on one chunk's block. Nothing else in the
pipeline changes, and the memory and parallelism properties are
structurally untouched because the kernel never sees more than one
chunk.

## Note on the fair comparison

pauli_lcu is **strictly sequential** (no OpenMP symbols, no SIMD
runtime, CPU-time/wall = 0.99 over the decomposition region). The
like-for-like comparison is therefore against paulikit's **sequential**
path: 17.63 vs 4.68 CPU-seconds, a **3.8x** gap. Our parallel wall
time of 7.55s against their 4.28s understates the deficit, because it
is bought with 4.5 cores.
