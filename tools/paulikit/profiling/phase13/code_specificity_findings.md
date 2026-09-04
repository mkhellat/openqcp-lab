# Is the 2-physical-core ceiling paulikit-specific, or generic multiprocessing/OS contention?

Recorded 2026-09-04. Follows directly from `full_optimum_sweep_findings.md`
(2 workers on 1 physical core beats every wider configuration,
statistically confirmed via ANOVA+Tukey HSD) and
`contended_chunk_size_screen_results.jsonl` (chunk_size retuning does
**not** close the gap - screened 12 multi-core configs x 4 chunk_sizes,
none beat `w2_c1`'s record). This investigation is the second deferred
overreach question: given tuning doesn't fix it, is the ceiling itself
a property of paulikit's own code (WHT butterfly, gather/scatter,
label construction), or a generic property of this hardware/Python's
multiprocessing under contention?

## Method

Two real conditions from the same `full_matrix_target.py` used
throughout this investigation, N=150 (dim=16384), chunk_size=2:
`w1_c1` (1 worker, no contention - baseline) vs `w8_c4` (8 workers on
4 physical cores - maximum contention in the sweep). For each, one
real worker process (a `ProcessPoolExecutor` child, verified via
`pstree -p` to be a **direct child** of the launching process for
every condition including `w1_c1` - `parallel_decompose` always uses a
real pool, not a special-cased sequential path) was profiled two ways:

1. **`py-spy dump`** (root privilege required, `ptrace_scope=1`):
   3 stack-sample snapshots per condition, ~3s apart, mid-run - shows
   exactly which Python function is on-CPU at each sample.
2. **`strace -f -c`** (root privilege required for full-tree `-f`):
   whole-process-tree syscall time/count summary for the complete run
   - shows total time and per-call cost spent in each syscall class,
   which is where OS-level contention (lock waits, scheduling,
   pipe I/O) would show up if the effect is not application-code-bound.

Deliberately a screening pass, not a publication-grade repeated
measurement (direct user decision, same reasoning as the chunk_size
screen): 1 run per condition, 3 py-spy samples each. If this points
at a specific mechanism, a proper repeated/statistical confirmation is
the natural next step before treating the conclusion as final.

A real bug was found and fixed while building the capture script
(`code_specificity_capture`): the first version sampled the **root**
process for the `w1_c1`/no-contention case, on the assumption a real
worker would be a grandchild. `pstree -p` showed this was wrong - the
worker is always a **direct child** - so the first capture attempt
caught the mostly-idle main thread waiting on the pool, not the
worker doing the real math. Fixed before the results below were
captured; the corrected script always resolves to the real worker
process, with retries since it may not have forked yet exactly at the
sampling boundary.

## Results

### py-spy: where is the worker's CPU time going?

| condition | sample 1 | sample 2 | sample 3 |
|---|---|---|---|
| `w1_c1` (lone) | `_walsh_hadamard_transform_rows` | `_walsh_hadamard_transform_rows` | `dumps` (result pickling) |
| `w8_c4` (contended) | `multiprocessing.synchronize.__enter__` (lock wait) | `connection._send` (pipe write) | `multiprocessing.synchronize.__enter__` (lock wait) |

**Lone worker: 2/3 samples inside real paulikit math** (the WHT
butterfly transform itself, `fwht.py:162/164`), 1/3 in result
serialization (also paulikit/stdlib-adjacent - `dumps`/`put`, handing
a finished chunk back to the main process).

**Contended worker: 0/3 samples inside any paulikit computation.**
All 3 are blocked inside Python's own `multiprocessing` internals -
acquiring a lock to safely write to the shared results queue
(`synchronize.__enter__`, twice) and blocked inside a raw pipe
`send()` call (`connection._send`, once). Small n (3 samples), but a
completely clean, one-sided split - every single contended sample
landed in IPC/synchronization code, not one in application math.

### strace: syscall time and per-call cost, whole tree

| syscall | w1_c1 time (% of total) | w8_c4 time (% of total) | per-call cost w1_c1 -> w8_c4 |
|---|---|---|---|
| `futex` | 20.33s (44.8%) | 24.19s (49.3%) | 72.3us -> 54.0us |
| `wait4` | 9.83s (21.7%) | 11.64s (23.7%) | 41.3ms -> 50.6ms |
| `poll` | 3.72s (8.2%) | 8.25s (16.8%) | **130.6us -> 478.8us (3.7x)** |

`futex` + `wait4` alone already account for ~66-73% of all traced time
in BOTH conditions - even the lone-process baseline spends most of its
traced syscall time in synchronization/wait primitives, not raw
compute syscalls (`futex` here is standard `ProcessPoolExecutor`/
`multiprocessing.Queue` internals: the pool's manager thread and
worker both use locks and condition variables to coordinate task
handoff, present even with `n_workers=1`). `poll`'s share of total
time nearly doubled (8.2% -> 16.8%), and its **per-call cost roughly
quadrupled** under contention - each individual poll for pipe
readiness now takes measurably longer, consistent with more processes
contending for the same scheduler/memory-bandwidth resources.

Raw call *counts* are not directly comparable between the two runs
(`w8_c4`'s tree has 8 workers vs `w1_c1`'s 1, so more processes
naturally issue more total syscalls) - the informative signal is
**time share and per-call cost**, both of which point the same
direction as the py-spy evidence.

### Where does this activity live in the code?

`parallel_decompose`'s result-collection loop
(`fwht.py:1558-1577`) is a standard `concurrent.futures`
pattern - `wait(in_flight, return_when=FIRST_COMPLETED)`, then
`future.result()` per completed future. This is stdlib
`ProcessPoolExecutor`/`multiprocessing` machinery, not custom
paulikit synchronization code - the `futex`/`poll`/queue-lock activity
observed is coming from Python's own IPC implementation, which every
`ProcessPoolExecutor`-based Python program pays, not something unique
to paulikit's design.

## Interpretation

Both signals agree, and the mechanism they point at is NOT
paulikit-specific: under 8-way contention, a worker spends its time
blocked trying to hand a finished result back through the OS pipe/lock
machinery `ProcessPoolExecutor` is built on, not computing. The lone
worker, by contrast, spends the large majority of its sampled time in
the real WHT math. This is consistent with the chunk_size screen's
own finding that the gap scales with physical-core count (not worker
count) - more concurrently-scheduled processes competing for the same
shared L3/memory-bandwidth and kernel scheduler naturally slows down
every process's syscalls, including the completely generic
lock-acquire/pipe-write path any `ProcessPoolExecutor` program uses,
regardless of what work the workers are actually doing.

This reframes the earlier "2-physical-core ceiling" as more likely a
property of **this machine's multiprocessing/scheduling overhead under
contention** than of paulikit's own decomposition algorithm - the
algorithm itself (WHT, gather, filtering) was the dominant cost in the
uncontended case and essentially never showed up as the bottleneck in
the contended sampled data.

## What this does NOT show

- **Small n.** 3 py-spy samples per condition, 1 run each (not a
  repeated/statistical measurement) - a real screening signal, not a
  publication-grade result. The clean 3/3-vs-0/3 split is suggestive
  but should be confirmed with more samples (a longer, denser
  `py-spy record` capture, if `perf_event_paranoid`/ptrace access can
  be arranged more conveniently than one-off `sudo` runs) before being
  treated as final.
- **Does not identify which OTHER program would show the same
  behavior.** The finding is "this bottleneck lives in generic
  `ProcessPoolExecutor`/`multiprocessing` code, not paulikit's own
  algorithm" - it does NOT independently confirm that literally any
  other CPU-bound multiprocessing Python program on this exact machine
  would show the identical ceiling. That would need a second,
  paulikit-free synthetic workload (e.g. a trivial CPU-bound function
  parallelized the same way) run through the same w2_c1-vs-w8_c4
  comparison - not done here, a natural follow-up if this needs to be
  fully nailed down.
- **Does not, by itself, suggest a fix.** Even having identified the
  mechanism (IPC/synchronization contention under `ProcessPoolExecutor`),
  no alternative IPC strategy (shared memory instead of pickled
  results, fewer/larger chunk handoffs, a different parallelism
  primitive entirely) has been tried or measured here.
- **Does not generalize to other hardware.** Same caveat as the
  optimum-sweep findings - this is one machine's measured behavior.
