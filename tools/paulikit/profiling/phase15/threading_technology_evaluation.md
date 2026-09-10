# Choosing a threading technology

Phase 15 item 3. 2026-09-10. Gated on items 1 and 2, both of which
are now measured.

**Conclusion: the choice is not a performance question.** Dispatch
overhead is under 0.07% of a chunk for every candidate, so they are
indistinguishable on speed at this granularity. The decision is
packaging and maintenance, and on those grounds a **Python
`ThreadPoolExecutor` drain is the right first move**, with pthreads
inside the C layer as the fallback if the drain itself becomes the
limit. OpenMP and OpenCilk are rejected.

## What item 2 established, and why it opens this up

The GIL-held fraction of a real N=150 chunk is **f = 0.0336** - 13.0
µs of `np.zeros` + value slice + scatter against 374.6 µs inside the
two GIL-releasing C kernels. Amdahl ceilings: 1.94x at 2 threads,
3.63x at 4, 6.48x at 8.

So the sparse gather is **not** a blocking serial fraction, and does
not need porting to C before threading is attempted. That was the
open question gating this item; it is answered.

## Dispatch overhead is irrelevant at our granularity

Measured directly: 5595 tasks (the real chunk count at N=150), with a
deliberately tiny work body so the number isolates dispatch rather
than compute.

| threads | OpenMP µs/task | pthread µs/task |
|---|---|---|
| 1 | 0.2664 | 0.2441 |
| 2 | 0.1562 | 0.1436 |
| 4 | 0.1018 | 0.0898 |
| 8 | 0.0559 | 0.2454 |

**One real chunk is ~390 µs of work.** So dispatch costs between
0.014% and 0.068% of a chunk. Even the worst cell in that table
(pthreads at 8 threads, 0.2454 µs - static split showing thread
create/join cost) is under 0.07%.

This is the opposite of the process-pool situation, where a ~1.33 ms
round trip dominated ~0.5 ms of compute. Thread dispatch is roughly
**four orders of magnitude** cheaper than the IPC it would replace.

Consequence: **no candidate can win on dispatch performance.** Any
argument for one over another has to come from elsewhere.

## Toolchain availability on this machine

| technology | status |
|---|---|
| OpenMP (GCC) | available, `gcc -fopenmp` links |
| pthreads | available, in libc |
| oneTBB | 2023.1.0 via pkg-config (already an optional dep) |
| OpenCilk | **not installed** - needs a custom LLVM build |
| Python `ThreadPoolExecutor` | stdlib, no build step |

## The packaging constraint, which decides it

This project targets the NumPy/SciPy model: extension always
required, shipped as prebuilt wheels (see the wheel-model note in
PLAN.md). Under that model a runtime dependency is a real cost.

Checked directly: an OpenMP shared object links **`libgomp.so.1`**,
and **NumPy's own wheels carry no OpenMP dependency at all**
(scanned its bundled `.so` files - no `omp` in any `ldd` output).
That is a deliberate choice by the project whose packaging model we
are copying.

Against that:

- **OpenMP** adds `libgomp.so.1` on Linux and needs `libomp`
  separately installed under Apple clang - a per-platform wheel
  problem for a gain measured at under 0.07%.
- **OpenCilk** requires a custom LLVM toolchain. For a project that
  must stay `pip install`-able and already treats a C++ compiler and
  oneTBB as *optional*, making a bespoke compiler mandatory is not
  proportionate. Its work-stealing scheduler is genuinely good, but
  it solves load-imbalance problems we do not have: our chunks are
  uniform by construction (`chunk_size` rows each).
- **oneTBB** is already wired in as an optional dependency for
  `pauli_label_parallel`, so the precedent exists - but it is C++,
  and both of the kernels this would drive are C. Introducing a C++
  dependency to the C path re-creates exactly the build fragility
  that made `wht_native` deliberately independent of the `native`
  option.
- **pthreads** is in libc everywhere POSIX. No new dependency, no
  wheel implications, works under both GCC and clang.
- **Python `ThreadPoolExecutor`** needs no build change at all. Both
  kernels already release the GIL, so the threads run the C work
  genuinely concurrently.

## Recommendation

**Start with `ThreadPoolExecutor`.** Rationale:

1. Zero packaging cost and zero new dependency - it is stdlib.
2. Both kernels already release the GIL, which is the entire
   precondition. Nothing in the C layer has to change.
3. It reuses the existing chunk-independent structure directly, so
   the streaming contract, checkpointing, and bounded memory carry
   over unchanged.
4. If it proves insufficient, the fallback is a *narrow* one: move
   the drain loop into C and use pthreads there, keeping the same
   chunk decomposition. That is an implementation swap behind the
   same API, not a redesign.

**Reject OpenMP** on packaging grounds - a `libgomp` runtime
dependency and an Apple-clang special case, to buy under 0.07%.
**Reject OpenCilk** as disproportionate: a mandatory custom compiler
for a load-balancing benefit our uniform chunks do not need.
**Do not extend oneTBB** to the C kernels - it would couple the
independent C path back to the C++/TBB build.

Note this is a *different* conclusion from the one the pauli_lcu
paper reached (they used OpenMP), and reasonably so: they parallelise
a monolithic in-place C routine with no Python layer and no wheel to
ship, so `#pragma omp parallel for` is the natural fit and costs them
nothing. Our constraints differ.

## What must be preserved, whichever is chosen

Unchanged from the Phase 14 kernel constraints:

- Peak RSS bounded by `chunk_size`, not `dim²` or term count.
- Chunk independence - no shared mutable state, so multi-node stays
  possible even though it is deferred.
- COO output, not positional.
- Bit-identical results.

A threaded drain touches none of these: it changes who calls
`do_chunk`, not what a chunk is.
