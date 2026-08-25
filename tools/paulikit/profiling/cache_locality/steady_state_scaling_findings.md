# Cache locality investigation — corrected N-scaling with steady-state timing

Follow-on to `n_scaling_findings.md`, redone with the methodology fix
that emerged from investigating this session's flat ~30% total-stall
puzzle: `perf record` at N=25 showed a large fraction of sampled
cache misses came from **process startup** (`kernel_init_pages`,
`get_mem_cgroup_from_mm`/`charge_memcg` kernel memory accounting,
`gc_collect_main`/`visit_decref` Python GC), not the algorithm -
proportionally worse at small N because the real work finishes fast.
`n_scaling_findings.md`'s one-shot-CLI numbers were confounded by this.

`steady_state_decompose.py` fixes this: build the Hamiltonian once,
run one untimed warm-up call (pays for first-call costs), then time 5
further calls in the same process. `run_steady_state_sweep.sh` wraps
this under `perf stat` across the project's now-standard N=25/50/100/150
set (see `n150_oom_finding.md` for why N=150 isn't included below -
it OOM-killed this machine before completing even the warm-up call,
on the current unmodified code).

## Results (3 runs each at N=25/50, 1 run at N=100 - see note)

| N | dense array | cache-miss ratio | total-stall / cycles | mem-stall / cycles |
|---|---|---|---|---|
| 25  | 4 MiB (fits in 8 MiB L3) | 22.7-24.6% | 29.7-30.8% | 14.8-15.7% |
| 50  | 64 MiB (8x over L3)      | 57.6-58.5% | 28.6-29.4% | 21.5-22.3% |
| 100 | 1024 MiB (128x over L3)  | 58.5%\*    | 31.8%\*    | 25.1%\*    |

\* N=100 only completed 1 of 3 planned runs - each run is 5 reps at
~34s/rep (~170s), and 2 of 3 runs hit this session's tool timeout.
Single-run numbers are reported as-is, not averaged from insufficient
data - a second/third run to confirm stability is a reasonable
follow-up, not yet done.

## What changed vs. the uncorrected (one-shot CLI) measurement

`n_scaling_findings.md`'s N=25 cache-miss ratio was 17.2%. The
corrected, steady-state N=25 ratio is **22.7-24.6% - higher, not
lower**, once process-startup noise is removed. This says something
real: the startup overhead wasn't just adding noise on top of the
algorithm's own cache behavior, it was partly *masking* it by
diluting the sample with unrelated kernel/GC misses that don't scale
with N the way the algorithm's own memory access does. The corrected
number is the more trustworthy one for reasoning about the algorithm
itself.

**Cache-miss ratio scaling is now even cleaner** than before:
~23% (fits in L3) -> ~58% (8x over) -> ~58.5% (128x over) - the jump
from "fits" to "doesn't fit" is stark and clearly the dominant
effect; going from 8x-over to 128x-over doesn't move the ratio much
further, suggesting the ratio saturates once the array is
meaningfully larger than cache (makes sense: once most accesses miss,
making the array even bigger doesn't create more misses per access,
it's already close to worst-case).

**Total-stall stays flat, more convincingly now**: 29.7-30.8% (N=25)
vs. 28.6-31.8% (N=100) - overlapping ranges, no real trend, across
the same 32x range in array size that showed a huge cache-miss-ratio
swing. This confirms `n_scaling_findings.md`'s honest complication
was real, not a startup-noise artifact: **something other than the
dense array accounts for a roughly constant ~30% of stall cycles**,
independent of N in this range. Not yet identified.

**Mem-stall now scales cleanly and monotonically**: 14.8-15.7% (N=25)
-> 21.5-22.3% (N=50) -> 25.1% (N=100) - a clear, steady increase, more
convincing than the earlier noisier numbers. This is good, direct
evidence that the memory-load-specific cost (as opposed to total
stalls from all causes) does track the dense array's growing
misalignment with cache size.

## Revised understanding

Splitting "cache-miss ratio" and "mem-stall" from "total-stall" turns
out to matter a lot: the first two scale cleanly with the dense
array's size relative to cache (strong support for treating the
densification as the real, fixable driver of *those* metrics). Total
stall cycles do not - there's an apparently N-independent ~30% floor
from something else entirely, still unidentified. A fix to the
densification should be expected to meaningfully shrink the
cache-miss ratio and the mem-stall share, but very likely will not
move total-stall by nearly as much - reiterating (with much better
data now) the same caution `n_scaling_findings.md` raised.

## Not yet done

- N=100's second and third runs (timed out this session).
- N=150 remains untested (see `n150_oom_finding.md`) - needs either
  more RAM, a smaller test problem as a stand-in, or the fix itself
  before it can be safely measured.
- Identifying the source of the flat ~30% total-stall floor - a
  `perf record` pass focused specifically on non-memory stall causes
  (e.g. `resource_stalls.*`, branch-misprediction-related events)
  would be the natural next step, not yet done.
