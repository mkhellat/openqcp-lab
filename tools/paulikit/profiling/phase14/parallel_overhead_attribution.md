# Where the parallel CPU-seconds go

2026-09-09. N=150 (14 qubits, dim 16384, 91,652,096 terms) unless
stated. All figures are CPU-seconds (`getrusage`, SELF + CHILDREN)
alongside wall, because wall alone hides work spent on extra cores.

## The measurement that reframes the problem

Fixed `chunk_size=2`, varying only the worker count:

| workers | wall | CPU-s | cores |
|---|---|---|---|
| 1 | 22.220s | 24.576 | 1.11 |
| 2 | 13.479s | 29.751 | 2.21 |
| 4 | 9.836s | 43.475 | 4.42 |

**Total CPU grows 77% (24.6 -> 43.5) for identical total work.** That
is the decisive fact. Fixed per-task dispatch overhead cannot do this
- it would be constant per chunk regardless of how many workers are
running. Cost that rises with the number of *concurrently active*
workers is the signature of contention for a shared resource: memory
bandwidth and the shared 8 MiB L3, which four workers each streaming
`chunk_size * dim` blocks will thrash.

Separately, `workers=1` costs 24.58 CPU-s against the in-process
sequential path's 17.63 - so roughly 7 CPU-seconds is pool machinery
(fork/spawn, task dispatch, result transfer) even with no parallelism
at all.

## A falsified hypothesis: "fewer, larger chunks"

The reasoning was that 5595 chunks x ~2.25 ms/chunk of overhead should
amortize away with bigger blocks. Measured on the parallel path:

| chunk_size | wall | CPU-s | chunks |
|---|---|---|---|
| 2 | 7.763s | **34.276** | 5595 |
| 8 | 12.337s | 51.951 | 1399 |
| 32 | 14.623s | 60.853 | 350 |
| 128 | 14.527s | 60.759 | 88 |

**Falsified, and inverted.** 16x fewer tasks costs 78% *more* CPU.
The per-chunk cost is not amortizable setup - it scales with
`chunk_size * dim` data volume, and larger blocks worsen cache
behaviour. The autotuner's choice of 2 is correct on the parallel path
too, for a reason it did not model.

This also matches the sequential sweep (`chunk_size` 3/16/64/256/1024
at N=100: 2.22s/3.40s/4.01s/4.00s/4.62s) - monotonically worse with
size, same underlying cause.

## Microbenchmark attribution, and why it misleads

Isolated `ProcessPoolExecutor` costs, 4 workers, 512 KiB payload:

| | ms/task |
|---|---|
| empty task | 0.103 |
| tiny return | 0.084 |
| generate payload, return small | 0.329 |
| generate payload, return 512 KiB | 0.653 |

So dispatch ~0.10 ms and 512 KiB transfer ~0.32 ms, totalling ~0.42 ms
against the ~2.25 ms/chunk seen in production. The microbenchmark
accounts for under a fifth of it. The gap is contention, which a
microbenchmark with no memory pressure cannot reproduce.

Shared memory was probed as a fix (workers write into a shared buffer,
only a descriptor crosses the pipe): **1.96x on transfer** (743 ->
1461 MiB/s for 100 MiB). Real, but it addresses ~0.32 ms of a 2.25 ms
problem - a ~14% dent at best. Not the lever.

## Consequence for the plan

The "parallel overhead" and "per-core efficiency" gaps recorded in
`external_comparison_findings.md` are **not two independent problems**.
Roughly 7 CPU-s is genuine pool machinery; the rest of the growth is
the same per-core inefficiency, paid simultaneously on 4.5 cores
against a shared memory hierarchy. Making the per-chunk work cheaper
and more cache-resident therefore attacks both at once, and is the
only lever that does.

Reducing bytes touched per term is the direction. paulikit carries
32 B/term at the API boundary (intp x + intp z + complex128 coeff);
the index arrays are pure bookkeeping that pauli_lcu does not pay at
all, because position encodes identity in its dense output.

**Already done, and worth noting so it is not re-proposed:**
`_index_dtype_for_dim` narrows both index arrays to the minimum width
`dim` requires (uint16 through n=16), and `_parallel_worker_chunk`
already applies it before returning - so the IPC payload is 20 B/term,
not 32. That lever has been pulled.

What remains is the coefficient array itself (16 of the 20 bytes) and
the per-chunk intermediate `(chunk_size, dim)` complex block, which is
what four concurrent workers are actually contending over.

## Method note

`ps`/`top` `%CPU` cannot be used for any of this - it is a lifetime
average and includes process startup and input construction. Every
figure here is a `getrusage` delta around the timed region only. See
the same point in `external_comparison_findings.md`, where it
initially caused a single-core implementation to be misread as
parallel.
