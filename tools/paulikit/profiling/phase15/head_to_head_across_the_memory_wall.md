# Head to head across the memory wall

2026-09-10. Machine: i7-8550U, 4 physical cores / 8 threads,
**15.4 GiB RAM**. `pauli_lcu` 1.0.1 as released (single core - that is
what ships; see `paper_claims_audit.md`).

Chunks are consumed and discarded, never accumulated: at 15-16 qubits
the result is 6-27 GiB and retaining it is not the use case. Peak RSS
is therefore the pipeline's own.

## N=150 — 14 qubits, dim 16384, 91,652,096 terms

Both implementations run. Median of 3, cooldown to 55 °C between runs,
cycles from `perf` (wall clock on this machine carries a ~2x frequency
artifact - see `wall_clock_unusable_on_this_machine.md`).

| implementation | wall | cycles | peak RSS | terms |
|---|---|---|---|---|
| pauli_lcu | 4.778s | 17.43 B | **8892 MiB** | 91,652,096 |
| paulikit sequential | 2.367s | 11.29 B | 66 MiB | 91,652,096 |
| **paulikit threaded** | **0.989s** | 14.41 B | **72 MiB** | 91,652,096 |
| paulikit process pool | 3.140s | 26.49 B | 67 MiB | 91,652,096 |

**4.83x faster at 1/123rd the memory.** Term counts identical across
all four.

Two things worth reading off this table:

- Even the **sequential** path beats pauli_lcu by 2.0x, on one core,
  in fewer cycles (11.29 B vs 17.43 B) - so this is not a
  core-count win.
- The **process pool** is the slowest paulikit configuration and
  costs 26.49 B cycles, nearly double the threaded path's 14.41 B.
  That is the pickling and IPC the threaded drain removes.

## N=200 — 15 qubits, dim 32768, 326,134,272 terms

pauli_lcu's dense input is **16 GiB** on a 15.4 GiB machine.

| implementation | wall | cycles | peak RSS | terms |
|---|---|---|---|---|
| pauli_lcu | **OOM-killed** | — | — | — |
| paulikit sequential | 8.544s | 37.97 B | 72 MiB | 326,134,272 |
| **paulikit threaded** | **3.211s** | 48.06 B | **89 MiB** | 326,134,272 |
| paulikit process pool | 10.851s | 83.55 B | 73 MiB | 326,134,272 |

The process was killed by the OOM reaper before reaching the
transform - it cannot allocate the input its API requires. This is not
a slow result; it is no result.

## N=300 — 16 qubits, dim 65536, 1,470,021,632 terms

pauli_lcu's dense input is **64 GiB**, 4.2x this machine's total RAM.

| implementation | wall | peak RSS | terms |
|---|---|---|---|
| pauli_lcu | **cannot run** (needs 64 GiB) | — | — |
| **paulikit threaded** | **19.59s** | **123 MiB** | 1,470,021,632 |

**1.47 billion Pauli coefficients in under 20 seconds, in 123 MiB.**

## What the numbers actually show

**Peak RSS is flat in problem size.** 72 MiB at 14 qubits, 89 MiB at
15, 123 MiB at 16 - while the problem grows 16x. pauli_lcu's grows
with the operator: 8.9 GiB at 14 qubits, 16 GiB required at 15,
64 GiB at 16.

That is the difference between `O(1)` *additional* memory and `O(1)`
memory. Their claim is true as stated and verified (+0.0 MiB
auxiliary), but it is auxiliary space on top of a mandatory dense
input, and the input is what runs out.

**The advantage is not a core-count trick.** paulikit's sequential
path uses fewer cycles than pauli_lcu at N=150 (11.29 B vs 17.43 B)
on one core. Threading then converts that into wall-clock time. The
two contributions are separable and both real.

**Their algorithm is not the limitation.** As recorded in
`paper_claims_audit.md`, the paper reports an OpenMP implementation
achieving 7x on 8 cores, and their rows are independent exactly as
ours are. What fails at 15 and 16 qubits is not the mathematics - it
is the decision to require the caller to materialise the whole
operator. Every artifact they published makes that decision.

This is the distinction worth stating carefully in any write-up: the
gap measured here is between two *systems*, not two algorithms. A
decomposition into independent rows says nothing about whether the
result can be produced on hardware anyone owns. Streaming, bounded
chunk state, COO output and a drain that does not copy are engineering
choices, and they are what puts 16 qubits within reach of a laptop.

## Reproducing

`profiling/phase15/bench_child.py` (one measurement per process) and
`profiling/phase15/bench.py` (interleaved driver with cooldowns and
`perf` counters). `N=300` was run separately for the threaded path
only, since the comparison there has no pauli_lcu side.
