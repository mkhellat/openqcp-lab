# Traffic-intensity suspect: tested, and dense WHT traffic alone is NOT enough

Recorded 2026-09-04. Follows the user's question whether cache
coherence is the multi-core roadblock (answered: unlikely) and the
agreed primary suspect: **per-chunk memory traffic intensity** (gather
+ flat full-width WHT), with IPC as the visible stall surface.

## Question under test

Is matching paulikit's per-chunk **dense working-set size** and
**14 full-array stage touches** (with the real
`_walsh_hadamard_transform_rows`) sufficient to reproduce the
2-physical-core ceiling (`w8_c4` no faster / slower than `w2_c1`)?

If yes → traffic intensity of the WHT buffer is a sufficient cause.
If no → something else in the full `parallel_decompose` path is
required (irregular gather, operator footprint, term filtering, …).

## Method

New targets (`traffic_intensity_target.py`), same pool shape as
`synthetic_ipc_control.py` / `parallel_decompose` (ProcessPoolExecutor,
pinned CPUs via `condition_table`, bounded `max_in_flight`,
pickle-over-pipe). `N_CHUNKS=5595` matches the real N=150 /
`chunk_size=2` outer DAG.

| workload | per-task body | IPC payload |
|---|---|---|
| `wht_small` | real WHT on fresh `(2,16384)` complex128 | 64 floats |
| `touch_small` | 14 RMW passes on same shape (no butterfly) | 64 floats |
| `wht_large` | same WHT as `wht_small` | full `(2,16384)` complex (~512 KiB) |

Sweep (`traffic_intensity_sweep.py`): `{wht_small,touch_small,wht_large}
× {w2_c1,w8_c4}`, **5 reps**, cooldown to 55°C package temp, child-
inheriting `perf stat` (same events/protocol as
`bandwidth_hypothesis_sweep.py`). Raw data:
`traffic_intensity_results.jsonl`.

Welch's unequal-variance t-test on wall-clock, n=5 per cell.

## Results

| workload | mean w2_c1 (s) | mean w8_c4 (s) | speedup w2→w8 | Welch p | eff. P @ w8 |
|---|---|---|---|---|---|
| `wht_small` | 14.745 | 6.719 | **2.19×** (w8 faster) | 5.8e-8 | **7.59** |
| `touch_small` | 7.957 | 2.963 | **2.69×** | 1.6e-8 | **6.91** |
| `wht_large` | 17.633 | 7.919 | **2.23×** | 1.2e-12 | **5.80** |
| paulikit (prior) | 22.068 | 24.801 | **0.89×** (w8 slower) | 8.0e-3 | **2.33** |

All three traffic controls **scale**. Even returning the full 512 KiB
transformed buffer per task (`wht_large`) does not recreate paulikit's
ceiling. Effective concurrency at w8 reaches ~6–7.6, like the old
48×48 FLOP synthetic — not like paulikit's stuck ~2.3.

Cache-miss % does rise under w8 for these controls (e.g. `wht_small`
0.75% → 12.1%), and wall-clock still improves. So “miss ratio goes up
with more cores” is **not** by itself the failure mode.

## Interpretation (careful)

1. **Refuted as a sufficient cause:** dense `(chunk_size, dim)` WHT
   traffic + 14 stage touches + ProcessPoolExecutor IPC, even with
   large pickled results, **does not** impose the observed paulikit
   multi-core collapse.

2. **Still consistent with a memory-system story, but a narrower one:**
   on w2, paulikit's cache-miss ratio is ~9% while `wht_small` is
   ~0.75% despite similar or higher `cache_refs_per_ms` on the
   synthetic WHT. The full pipeline's **access pattern** (irregular
   gather/scatter from the sparse operator into the dense buffer,
   plus living with full operator / setup arrays in each worker) is
   not reproduced by allocating a dense random buffer and transforming
   it. Traffic *volume* of the WHT array alone is the wrong dial;
   **irregular / gather traffic + resident footprint of the operator**
   remain open.

3. **Large result IPC alone is not the roadblock** under this pool
   shape: `wht_large` still gets 2.2× from packing onto 4 cores.

4. **Coherence (MESIF) as the simple explanation stays weak:** same
   coherent CPU, same multi-process pool, WHT-sized buffers — scales.
   Paulikit-specific work in the chunk body is required for the
   failure.

## What this does NOT show

- Does not prove gather is the cause (not yet isolated).
- Does not re-measure full paulikit in this script (uses prior
  bandwidth-hypothesis paulikit cells as contrast only).
- Does not claim cache-blocked WHT is useless — a blocked WHT might
  still help the *real* gather+WHT pipeline; this experiment only
  shows that mimicking dense WHT traffic outside that pipeline is
  not enough to reproduce the bug.

## Actual next isolation step

Build a fourth control that keeps tiny IPC but reproduces the
**gather pattern**: each task zeros a `(2,16384)` buffer and scatters
`nnz`-like irregular `(row, q)` writes from a precomputed index
pattern matching the coupled-oscillator FWHT gather, optionally
followed by WHT. Same w2_c1 vs w8_c4, 5 reps, Welch.

- If that **fails to scale** → gather/irregular traffic is sufficient.
- If that **still scales** → look at operator-sized resident set /
  multi-GiB per-worker footprint / label-filter path next.

## Artifacts

- `traffic_intensity_target.py`
- `traffic_intensity_sweep.py`
- `traffic_intensity_results.jsonl` (30 runs)
