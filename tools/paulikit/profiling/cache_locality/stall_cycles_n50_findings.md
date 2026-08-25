# Cache locality investigation — stall-cycle quantification

Follow-on to `baseline_perf_stat.md` and `perf_record_n50_findings.md`.
Answers the question those two left open: how much of the wall time
is *actually* lost to memory stalls, as opposed to just having a high
miss *ratio* that the out-of-order core mostly hides?

## Method

```
perf stat -e cycles,cycle_activity.stalls_total,cycle_activity.stalls_l2_miss,\
cycle_activity.stalls_l3_miss,cycle_activity.stalls_mem_any \
  paulikit decompose --n-oscillators 50
```

`cycle_activity.stalls_*` are Skylake-family PMU events (confirmed
available via `perf list` on this CPU) that directly count cycles the
core's execution was stalled with a load of the given kind
outstanding - this is a direct measurement of wasted time, not an
inferred one. Not yet wrapped into `run_baseline_perf_stat.sh` (that
script's event list is fixed to the original baseline metrics) - ad
hoc for now; wrapping it into a dedicated script is a candidate
follow-up, not done here.

## Results (3 runs, N=50)

| metric | run 1 | run 2 | run 3 |
|---|---|---|---|
| total-stall / cycles | 30.0% | 30.3% | 30.3% |
| L2-miss-stall / cycles | 10.2% | 10.2% | 10.0% |
| L3-miss-stall / cycles | 8.6% | 8.6% | 8.4% |
| mem-any-stall / cycles | 18.7% | 18.7% | 18.8% |

## Correcting the earlier framing

`baseline_perf_stat.md` reported a 54-57% **cache-miss ratio**
(misses / references - i.e. "of the times we checked a cache level,
just over half were misses"). That is a real, reproducible number,
but it is not the same thing as "half the runtime is wasted on cache
misses," and it would be dishonest to let a reader conflate the two.

The direct stall measurement here says: **roughly 30% of all
execution cycles are stalled on some resource, and about 19% of all
cycles are specifically stalled on an outstanding memory load**
(L2/L3-miss-specific stalls are a subset of that, ~8-10%). The
remaining ~70% of cycles are cycles where the out-of-order core found
other useful work to do (or was waiting on something other than
memory - e.g. a dependent computation, a branch, a resource
conflict). That's the honest picture: a real, worth-fixing cost, not
a "half the program is wasted" story.

## Implication for the identified fix (dense-array scatter)

`perf_record_n50_findings.md` identified `fwht_pauli_coefficients`
materializing a `(dim, dim)` dense array (64 MiB at N=50, 8x this
machine's L3) as a likely dominant cause of the cache-miss activity.
Given ~19% of cycles are memory-stalled overall, a fix that
eliminates most of that unnecessary dense-array traffic is a
plausible route to a real (if not dramatic) wall-time improvement -
something meaningfully less than "cut the runtime in half," and this
document should not be read as promising more than that. The actual
number can only come from measuring before/after once a fix exists -
not yet done.

## Not yet done

- Before/after stall-cycle comparison once a fix to the densification
  is implemented (next step).
- These stall counters have not yet been correlated with a
  `perf record`+annotate pass to confirm the L2/L3-miss stalls
  specifically land inside `fwht_pauli_coefficients`/`fwht_pauli_terms`
  rather than elsewhere (e.g. the native kernel, TBB, or fixture/CLI
  setup code) - `perf_record_n50_findings.md`'s cache-miss *sample*
  localization is suggestive but is a different metric (miss count,
  not stall cycles) than what's measured here.
