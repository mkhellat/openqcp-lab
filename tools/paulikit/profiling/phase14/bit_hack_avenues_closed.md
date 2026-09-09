# Bit-level avenues: what was checked and ruled out

2026-09-09. Sources: Warren, *Hacker's Delight* (2nd ed., 2012), and
*Bit Twiddling Hacks*. Recorded so these are not re-explored.

Two of the four questions were posed on a false premise. That is the
useful part of the record.

## 1. Perfect shuffle to make the butterfly contiguous — NO

**Question.** The butterfly operates on stride-2 NumPy views, which
cannot emit packed SIMD. Is there a standard permutation (perfect
shuffle / outer unshuffle, HD Ch. 7 §7-2, pp. 138-140) that makes both
operands unit-stride, so that permute -> all stages contiguous ->
unpermute beats staying strided?

**Answer: the technique is real but operates on the wrong scope.** HD's
outer (un)shuffle is a sequence of masked shift-XOR passes over the bit
positions *within a single machine word* (5 passes for 32 bits). It is
not an element-index permutation across a `dim`-length array. Applied
at array level it degrades into an explicit gather/scatter, O(dim) to
O(dim log dim) per direction, performed twice around the transform -
and those passes are strided-access operations of exactly the character
being eliminated. Unless the permutation can be expressed as a no-copy
NumPy reshape/transpose, it relocates the cost rather than removing it.

(Row-local application would have been fine against the
no-shared-buffer constraint - that was not the blocker.)

## 2. Cheaper popcount mod 4 — NO, none exists

**Question.** We need only `popcount & 3`. Can the divide-and-conquer
popcount (HD Ch. 5 §5-1, p. 90) be truncated to produce just the low
two bits more cheaply than a full count?

**Answer: no operation-count saving is available.** The reduction is a
carry-save cascade (`0x55555555` -> `0x33333333` -> `0x0F0F0F0F` ->
shifts -> final mask). Truncating the *output* only changes the final
mask from `& 0x3F` to `& 3`; no reduction stage may be skipped,
because carries out of the lower stages ripple into bits 0-1 of the
sum. The uint8-accumulator + `& 3` form already in
`_phase_from_popcount` is the cheapest correct form of this algorithm.

## 3. Independent cheap bit0 and bit1 of popcount — the premise is false

**Question.** Parity (bit 0) is cheap. If bit 1 were separately cheap,
composing them would give popcount mod 4 directly.

**Answer: bit 1 has no such shortcut, and cannot have one.** Parity is
genuinely cheap - a 9-10 instruction XOR-reduction tree (HD Ch. 5 §5-2,
pp. 99-101) against ~21 for full popcount. But bit 1 is structurally
the carry-out of the bit-0 summation, and XOR-reduction discards
carries *by construction*. Obtaining bit 1 requires carry-aware
(addition-based) reduction, i.e. essentially the cost of a full
popcount. The proposed composition does not exist in either book or in
the underlying bit arithmetic.

## 4. Walsh-Hadamard / butterfly material — absent

**Not present in either book.** Full-text search: zero hits for
"Hadamard", "Walsh", or "butterfly" in both. *Bit Twiddling Hacks* has
no transform content at all. *Hacker's Delight* mentions FFT twice, in
Ch. 7's bit-reversed-counter-increment passage (~p. 139), concerning
how to increment a bit-reversed loop index - not the butterfly's
memory-access or vectorization pattern.

## Conclusion

No usable NumPy-level bit-level fix for the strided butterfly, and no
further gain available in the popcount/phase path beyond what Phase 14
already landed. The phase step is finished as a target.

The remaining levers are (a) reducing bytes touched per term, which
`parallel_overhead_attribution.md` identifies as the contended
resource, and (b) compiled code for the butterfly using our own
chunked, COO-producing structure.
