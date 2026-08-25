# Cache locality investigation — does the signal scale with N as predicted?

Follow-on to `perf_record_n50_findings.md` and `stall_cycles_n50_findings.md`.
Directly prompted by a good challenge: if the dense-`(dim,dim)`-array
densification in `fwht_pauli_coefficients` (see
`perf_record_n50_findings.md`) is really the dominant cause of the
measured cache misses, the effect should track the array's size
relative to cache, not be a flat property of "the code." Testing at
N=25 (array fits in L3) and N=100 (array is 128x larger than L3), in
addition to the existing N=50 (8x larger than L3) baseline, is a
direct way to check that - not just assert it.

## Method

```
perf stat -e cycles,cache-references,cache-misses,cycle_activity.stalls_total,cycle_activity.stalls_mem_any \
  paulikit decompose --n-oscillators <N>
```

Ad hoc for now (single run per N, not yet the 3-run protocol used for
N=50 in the other docs) - this was a quick scaling check, not a final
measurement. If N=25/N=100 become part of the standard baseline, they
should get the same 3-run treatment and be folded into
`run_baseline_perf_stat.sh`. Not yet done.

## Results

| N | qubits | dense array size | vs. 8 MiB L3 | cache-miss ratio | total-stall / cycles | mem-stall / cycles |
|---|---|---|---|---|---|---|
| 25  | 9  | 4 MiB    | fits (0.5x)  | **17.2%** | 33.0% | 13.9% |
| 50  | 11 | 64 MiB   | 8x over      | ~55%\*    | 30.0% | 18.7% |
| 100 | 13 | 8192x8192 = 1024 MiB | 128x over | **59.1%** | 31.7% | 24.5% |

\* N=50's cache-miss ratio is from `baseline_perf_stat.md`'s separate
3-run measurement (54.0-57.0%, using the standard event set); this
single ad hoc run didn't request `cache-references`/`cache-misses`
alongside the stall counters due to PMU counter-slot limits (see
"Method" - `perf` multiplexes when too many events are requested at
once, which is one reason these numbers are split across two
measurement passes rather than one).

## What this confirms

**The cache-miss ratio scales almost exactly as the dense-array
hypothesis predicts**: 17.2% when the array fits comfortably in L3,
jumping to ~55-59% once it's meaningfully larger than L3, whether 8x
(N=50) or 128x (N=100) over. This is a real, non-trivial confirmation
- if the misses were coming from somewhere size-independent (e.g. a
fixed per-call Python/CPython overhead, or the native kernel's own
memory pattern), the N=25 number would not have dropped this
sharply. It directly supports treating `fwht_pauli_coefficients`'s
dense-array materialization (line 216-217, see
`perf_record_n50_findings.md`) as a real, size-scaling contributor to
the cache-miss ratio - not a red herring.

**Mem-stall (cycles stalled on outstanding loads specifically) also
grows with N** (13.9% -> 18.7% -> 24.5%), consistent with the same
story: as the dense array grows further past cache capacity, more
cycles are spent waiting on it.

## What this does NOT confirm - the honest complication

**Total-stall stays nearly flat across a 128x range in array size**
(33.0% at N=25, 30.0% at N=50, 31.7% at N=100). If the dense array
were the *only* thing driving stalls, we'd expect total-stall to grow
with N the same way mem-stall did. It doesn't. The most likely
reading: there's a roughly constant ~30% stall budget from *something
else* (present even at N=25, where the array easily fits in cache) -
possibly other memory traffic (the input Hamiltonian matrix itself,
label-string construction, Python object overhead - all scale with N
too, just not the same way as the O(dim^2) dense array does), branch
mispredicts, or resource stalls unrelated to memory at all. As N
grows, memory-specific stalls (mem-stall) claim a *larger share* of
that same roughly-fixed total-stall budget, rather than the total
growing on top of it.

This means: fixing the dense-array densification is very likely to
reduce the cache-miss *ratio* substantially (strong evidence for
that) and probably reduce mem-stall cycles at larger N, but should
**not** be expected to reduce *total* stall cycles by anywhere near
that much, since total-stall wasn't the thing that scaled with array
size in the first place. This is a real, useful correction to make
before scoping a fix - overclaiming "removing the dense array will
fix ~30% of stalled cycles" would not be supported by this data;
the honest claim is closer to "removing it should shrink the
mem-stall-specific slice (13.9-24.5% depending on N), and its
disproportionate growth at larger N."

## Not yet done

- Localizing what's driving the flat ~30% total-stall baseline that's
  present even at N=25 - a `perf record` run at N=25 (not yet done;
  only N=50 has been localized so far) would show whether it's the
  same NumPy/CPython-overhead symbols seen at N=50, or something else
  entirely.
- The 3-run statistical protocol for N=25/N=100 (this was a single
  ad hoc run each, for a quick scaling sanity check).
- A combined single perf invocation that gets cache-miss and
  stall-cycle counters in the same run without multiplexing artifacts
  (may require `--group` or reducing the requested event count).
