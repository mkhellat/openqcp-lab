# Research: how are L1/L2 caches actually shared between hyperthread siblings?

Recorded 2026-09-02, prompted directly by the user's own critical
question during the CPU-pinning investigation
(`cpu_pinning_findings.md`): "with two logical cores on the same
physical core, are L1 and L2 caches of that physical core shared
between them and how? is it 50/50? what is the algorithm?" This
document records the answer with citations, checked against Intel's
own documentation rather than stated from unverified general
knowledge.

## Answer: dynamic, competitive/demand-based sharing - NOT a fixed 50/50 split

Both logical processors (hyperthreads) on one physical core share the
same physical L1 data cache, L1 instruction cache, and L2 cache
structures. There is no static per-thread reservation or fixed
capacity quota. Allocation and eviction are governed by the cache's
normal replacement policy (an LRU-family policy) applied uniformly to
all cache lines regardless of which logical processor issued the
access that created or last touched the line - a thread with a
larger/hotter working set can and does claim more of the shared
cache's capacity than the other thread, dynamically, based on actual
access patterns, not by design allocation.

**Sources**:
- Intel® 64 and IA-32 Architectures Software Developer's Manual,
  Volume 3 (system programming), the section on L1 data cache sharing
  under Hyper-Threading Technology: "the L1 data cache is
  competitively shared between logical processors... processors
  compete for cache resources, which reduces the effective size of
  the cache for each logical processor." -
  https://xem.github.io/minix86/manual/intel-x86-and-64-manual-vol3/o_fe12b1e2a880e0ce-430.html
- Intel Community technical discussion (consistent with the manual and
  general Hyper-Threading resource-sharing literature, e.g. NASA/NAS
  technical reports on Hyper-Threading performance impact): "L1 and L2
  caches are competitively shared, meaning if one thread uses a
  portion of cache, the other thread can use the full remaining cache
  for itself." -
  https://community.intel.com/t5/Intel-Moderncode-for-Parallel/Shared-memory-in-Hyperthreading/m-p/994001

## A caveat found during research, and why it does NOT apply to this project's dev machine

Some of the same sources also describe an "adaptive mode / shared
mode" L1 data-cache **context mode** flag, keyed on whether the two
logical processors' CR3 (page-table base) register values match:

- **Adaptive mode** (the default): if both logical processors' CR3
  values (and paging mode) are identical, "the entire L1 data cache is
  available to each logical processor instead of being competitively
  shared" - i.e. no competition at all in that specific case. If CR3
  values differ, the processors fall back to competing for cache
  resources.
- **Shared mode**: always competitively shared regardless of CR3
  match, described in the sources as generally discouraged since it
  can cause thrashing.

**This adaptive/shared mode mechanism is specific to the Pentium 4
(NetBurst microarchitecture) generation of Hyper-Threading, not to
this project's dev machine** (an Intel i7-8550U, Kaby Lake R - a
Skylake-family core, several architecture generations newer). On
Skylake-family cores, L1/L2 sharing between hyperthreads is simply
competitive/dynamic by design; there is no exposed CR3-based mode
switch. This distinction matters because conflating the two would
misattribute a legacy, no-longer-relevant configurability to the
actual hardware this project measures on.

## Honesty note on citation completeness

A Skylake-family-specific paragraph from Intel's current Optimization
Reference Manual (the document that would carry the most precise,
up-to-date wording for this exact microarchitecture) was NOT
successfully pulled directly in this research pass - the full current
manual PDF exceeded the fetch tool's size limit, and a smaller,
architecture-specific section was not isolated. The citations above
are the general x86/64 architectural manual (Volume 3, which describes
the mechanism as implemented across the Hyper-Threading-capable
architecture family, including Skylake) plus a corroborating
community/technical summary - not a Skylake-specific optimization-guide
paragraph. If a more precise citation is needed for a specific
decision, that would require fetching the current Optimization
Reference Manual in smaller chunks (by chapter/page range) rather than
as one large PDF.

## Relevance to this project's Phase 13 findings

This research directly informs the interpretation in
`cpu_pinning_findings.md`: CPU pinning (one worker per physical core)
eliminates hyperthread-*sibling* L1/L2 competition entirely (each
physical core's L1/L2 is now used by only one process, so there is
nothing left to compete for at that level) - and the measured result
was that pinning did not meaningfully change wall-clock or cache-miss
ratio. Combined with `l3_contention_direct_evidence_findings.md`'s
DIRECT (not inferred) evidence that cross-core contention alone -
with zero L1/L2 sharing - produces a real, large effect (wall-clock
3.2x slower, cache-miss/LLC-miss ratios roughly tripled), the overall
picture is now evidence-based rather than assumed: L1/L2 hyperthread
sharing is a real, dynamically-competitive mechanism (confirmed by
Intel's own documentation), but it is not the dominant contention
source for this specific workload on this specific machine - cross-
core (L3/memory-bandwidth) contention is, and that conclusion now
rests on a direct controlled measurement, not elimination-by-absence.
