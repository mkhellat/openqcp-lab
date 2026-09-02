# Scoping: portable physical-core/hyperthread detection beyond Linux

Recorded 2026-09-02, per direct user instruction to plan this rather
than leave it implicit. Not yet implemented - this is a design pass
only.

## What already exists, and why it's already microarch-portable

`_physical_core_representative_cpus()` (added in the CPU-pinning fix,
`cpu_pinning_findings.md`) reads ONLY
`/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list` - a
generic Linux kernel abstraction, not a CPU-vendor or microarch-
specific interface. The kernel itself resolves physical-core/
hyperthread-sibling topology for whatever CPU is actually installed
(Intel, AMD, ARM big.LITTLE, RISC-V multi-thread designs, etc.) and
exposes the answer through this one sysfs path uniformly. **No
microarch branching exists in this function today, and none is
needed** - the "don't need to do branchings based on different CPU
microarchs" requirement is already satisfied for every CPU vendor on
Linux, by construction, not as something still to be added.

## The real remaining gap: non-Linux platforms, not non-Intel CPUs

The actual portability gap is `os.sched_getaffinity`/the
`thread_siblings_list` sysfs path being Linux-specific APIs. On
macOS/BSD, `_physical_core_representative_cpus()` already returns
`None` gracefully (per its existing docstring/return contract), and
`parallel_decompose` already falls back to the logical-CPU count in
that case - but that fallback provides NO physical-core awareness at
all on those platforms, unlike the graceful-with-real-data path Linux
gets. This is the gap worth scoping, not a microarch-detection gap.

## Design options for non-Linux physical-core detection (not yet implemented)

1. **macOS**: `sysctl hw.physicalcpu` and `sysctl hw.logicalcpu` (or
   the C-level `sysctlbyname` equivalent) directly report physical and
   logical core counts - no per-CPU sibling-group topology like
   Linux's sysfs gives, just aggregate counts. This would let
   `parallel_decompose` cap `n_workers` to the physical count on
   macOS too, but WITHOUT the specific CPU-id-to-physical-core mapping
   needed for `_pin_current_process_to_cpu`-style explicit pinning
   (macOS does not expose a portable, stable thread-affinity API at
   all in the way Linux's `sched_setaffinity` does - Apple's own
   affinity API, `thread_affinity_policy`, is a *hint* the scheduler
   is free to ignore, not a hard pin). So on macOS, only the
   "auto-detected `n_workers` should use the physical count" half of
   this fix could be ported cleanly - the explicit-pinning half likely
   cannot be, for a real platform-API reason, not a missing-effort
   reason.
2. **BSD** (FreeBSD/OpenBSD/etc.): `sysctl hw.ncpu` gives logical count
   only on most BSDs; physical-core-aware sysctls vary by BSD flavor
   and are less standardized than either Linux's sysfs or macOS's
   `hw.physicalcpu` - would need per-BSD-flavor research before
   committing to an approach, a real unknown not yet investigated.
3. **Windows** (not currently a supported platform for this package at
   all - no existing Windows-specific code path anywhere in
   `paulikit`): `GetLogicalProcessorInformation`/`SetThreadAffinityMask`
   exist but are out of scope unless Windows support is separately
   requested; not scoped further here.

## Recommended approach, consistent with existing project conventions

Mirror the exact pattern `_detect_available_worker_count` and
`autotune.py`'s memory-detection chain already use: a POSIX-first,
best-effort detection chain with an explicit, documented fallback at
each tier, never a hard failure -
1. Linux: existing sysfs-based `_physical_core_representative_cpus()`
   (unchanged, already correct and vendor-portable).
2. macOS: `sysctl hw.physicalcpu`/`hw.logicalcpu` (via
   `subprocess.run(["sysctl", ...])`, POSIX-available, no third-party
   dependency, matching this project's own "POSIX-compliant tools
   only" discipline - `configure`'s own Slurm/PBS detection already
   sets this precedent) for the auto-`n_workers` cap ONLY - explicit
   pinning support deliberately NOT attempted there, documented as a
   known platform limitation rather than silently no-op'd.
3. Everything else (BSD flavors, unknown platforms): fall back to the
   existing logical-CPU-count behavior, unchanged from today - no
   regression, just no improvement, exactly like today's already-
   graceful `None`-return fallback path.

## What this document does NOT do

- Does not implement anything - `fwht.py` is unchanged by this
  document.
- Does not resolve the BSD-flavor sysctl question - flagged as
  unresearched, not decided.
- Does not scope Windows support - out of scope unless separately
  requested, no existing precedent in this package to extend from.
- Does not change the conclusion that microarch-specific branching
  (Intel vs. AMD vs. ARM) is unnecessary and was never present -
  confirmed, not just assumed, by reading the existing
  implementation's own logic.
