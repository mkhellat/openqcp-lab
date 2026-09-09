# Is there algorithmic headroom in the operator's sparsity?

2026-09-09. The question: paulikit knows which rows are nonzero and
returns COO triples; pauli_lcu must transform every row because its
output is positional. Is that asymmetry worth an algorithm?

**Answer: no, not via the obvious route. The measurements below close
the sparse-WHT idea and reframe the target as an execution-efficiency
problem, not an algorithmic one.**

## Active-row fraction across N

`active_x` = rows with at least one nonzero gathered entry. Those are
the rows paulikit transforms; pauli_lcu transforms all `dim`.

| N | q | dim | nnz | nnz/dim² | active | active/dim | skipped |
|---|---|---|---|---|---|---|---|
| 10 | 7 | 128 | 200 | 0.01221 | 48 | 0.375 | 62.5% |
| 20 | 8 | 256 | 800 | 0.01221 | 192 | 0.750 | 25.0% |
| 30 | 9 | 512 | 1800 | 0.00687 | 440 | 0.859 | 14.1% |
| 50 | 11 | 2048 | 5000 | 0.00119 | 1233 | 0.602 | 39.8% |
| 75 | 12 | 4096 | 11250 | 0.00067 | 2789 | 0.681 | 31.9% |
| 100 | 13 | 8192 | 20000 | 0.00030 | 4957 | 0.605 | 39.5% |
| 125 | 13 | 8192 | 31250 | 0.00047 | 7761 | 0.947 | 5.3% |
| 150 | 14 | 16384 | 45000 | 0.00017 | 11189 | 0.683 | 31.7% |

**The active fraction does not shrink with N.** It oscillates between
0.60 and 0.95 depending on how close N sits to a power-of-two padding
boundary - 31.7% of rows skipped at N=150, but only 5.3% at N=125.
This is a modest constant-factor advantage, not an asymptotic one, and
it is not dependable across N.

The striking number is elsewhere: `nnz/dim²` is **0.017%** at N=150.
The operator is extraordinarily sparse, but the XOR-mask partition
scatters those 45,000 nonzeros across 68% of all rows.

## The waste, stated precisely

| N | dim | active | nnz per active row | input utilization |
|---|---|---|---|---|
| 20 | 256 | 192 | 4.2 | 1.63% |
| 30 | 512 | 440 | 4.1 | 0.80% |
| 50 | 2048 | 1233 | 4.1 | 0.20% |
| 100 | 8192 | 4957 | 4.0 | 0.05% |
| 150 | 16384 | 11189 | 4.0 | **0.02%** |

Every active row carries on average **4.0 nonzeros**, and we run a
full 16384-point Walsh-Hadamard on each. That looked like the
algorithmic opening.

## The sparse-WHT idea, and why it fails

The WHT of a k-sparse row has a closed form: for nonzeros `v_j` at
positions `q_j`,

    W[z] = sum_j v_j * (-1)^popcount(q_j & z)

so the cost is O(k·dim) rather than O(dim·log dim). At dim=16384, k=4:
65K ops against 229K - **3.5x on paper**.

Implemented and verified correct (`np.allclose` against the butterfly
at dim = 256, 4096, 16384). Measured:

| dim | rows | dense | sparse | ratio |
|---|---|---|---|---|
| 256 | 8 | 0.155ms | 0.958ms | 0.16x |
| 4096 | 8 | 1.911ms | 2.838ms | 0.67x |
| 16384 | 8 | 9.679ms | 8.084ms | 1.20x |
| 4096 | 64 | 15.6ms | 26.6ms | 0.59x |
| 16384 | 64 | 146.0ms | 123.8ms | 1.18x |
| 16384 | 256 | 457.1ms | 489.0ms | 0.93x |

**Never a real win, and it degrades with row count.** Two reasons:

1. **The ops are not comparable.** A butterfly op is one add or
   subtract. A sparse op requires `(-1)^popcount(q & z)` - a popcount
   plus sign conversion, ~4-8x the cost. The 3.5x operation-count
   advantage is spent on making each operation more expensive.
2. **No mask reuse.** The sign mask for position `q` could be
   amortized if many rows shared `q` values. They do not: N=150 has
   11,475 distinct `q` for 45,000 nonzeros, ~4 entries per mask.
   Generating a `(nnz, dim)` sign matrix touches more data than the
   dense butterfly does.

## The floor that governs everything

**The output is not sparse.** At N=150, 91,652,096 terms survive from
a `dim²` = 268M grid - **34% density**. A 0.017%-sparse input produces
a 34%-dense output; the transform densifies. Any correct algorithm
must emit 91.65M coefficients, so O(output) is a hard floor.

Work per output term is already near it:

    dense WHT ops = dim² · log₂(dim) = 3.758e9
    output terms  = 9.165e7
    ops per term  = 41

## The real target, in the right units

| | CPU-s at N=150 | ns per output term |
|---|---|---|
| pauli_lcu | 4.68 | **51.06** |
| paulikit sequential | 17.63 | 192.36 |
| paulikit parallel (4w) | 43.48 | 474.40 |

pauli_lcu sustains **19.6M terms per CPU-second**.

At ~4 GHz, 51 ns/term is ~200 cycles for 41 operations - about 5
cycles per op, near memory-bound for this access pattern. We spend
~770 cycles for the same 41 operations.

**Conclusion: this is not an algorithmic gap.** Both implementations
run the same O(dim² log dim) transform and both are within a small
factor of the O(output) floor. The entire 3.8x is in how each
operation executes - strided NumPy views that cannot vectorize, versus
a contiguous compiled loop that GCC turns into packed SIMD.

## What this rules in and out

**Ruled out.** Sparse-WHT reformulation; exploiting the active-row
fraction (not asymptotic, not dependable across N); any hope that a
cleverer transform reduces the work materially, since we are already
at ~41 ops per output term against an O(output) floor.

**Ruled in.** Execution efficiency is the whole problem. Getting from
192 to below 51 ns/term needs the butterfly to run as a contiguous
compiled loop - the existing C/Cython kernel infrastructure applied to
our own chunked, COO-producing structure. NumPy on stride-2 views
cannot close a 3.8x gap.

A caveat worth keeping: pauli_lcu's 51 ns/term buys a *positional*
output. Ours includes emitting explicit indices, which is real work
they do not do and which is what makes streaming and bounded memory
possible. Some fraction of the gap is that structural choice rather
than inefficiency, and it should be quantified before the remaining
difference is treated as pure waste.
