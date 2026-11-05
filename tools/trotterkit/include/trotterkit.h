/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * The trotterkit C ABI. Every backend (C, C++, Rust) implements
 * exactly these functions, with the semantics stated here; a backend
 * can be written from this header alone.
 *
 * Data layout
 *   Only the state vector psi is interleaved: 2 * 2^n_qubits doubles,
 *   amplitude j at psi[2j] (re) and psi[2j + 1] (im) - numpy
 *   complex128 memory. Coefficients (c_re, c_im) and matrix values
 *   (v_re, v_im) are split arrays: one double array per part.
 *
 * Pauli convention (paulikit's)
 *   A Pauli string is a pair of masks (x, z). For n qubits, label
 *   character j (qubit j) is bit n-1-j of both masks. The operator is
 *     P = i^popcount(x & z) X^x Z^z,
 *   so that
 *     P|r> = i^popcount(x & z) (-1)^popcount(r & z) |r ^ x>.
 *   Every bit of x and z must lie below n_qubits.
 *
 * Product formula
 *   S2(dt) applies every fragment with angle theta = dt/2 in order,
 *   then every fragment again with angle dt/2 in reverse order (the
 *   last fragment is applied twice in a row). A schedule runs its
 *   stage list n_steps times, stage k being S2(dt[k]):
 *     psi <- [S2(dt[n_stages-1]) ... S2(dt[0])]^n_steps psi.
 *   n_steps = 0 or n_stages = 0 is the identity (psi untouched,
 *   TK_OK). Coefficients and matrix entries may be complex: the
 *   evolution need not be unitary.
 *
 * Errors
 *   Every int function returns TK_OK or one of the codes below; none
 *   aborts, prints or exits. On any error psi is left bit-for-bit
 *   unchanged. When several errors apply, which code is returned is
 *   unspecified.
 *
 * Cross-language duties
 *   - The ABI assumes a 64-bit size_t (dimensions up to 2^32 and
 *     their products must be representable).
 *   - Strings returned by tk_backend_name and tk_strerror are static,
 *     NUL-terminated and must not be freed.
 *   - A tk_groups handle may only be used with, and freed by, the
 *     library that built it.
 *   - A const tk_groups * may be used by several concurrent
 *     tk_evolve_grouped calls; the functions keep no global mutable
 *     state.
 *   - No panic or exception may cross this boundary. Rust: wrap each
 *     entry point in catch_unwind and use checked arithmetic for
 *     sizes; C++: catch std::bad_alloc (and size overflow) and return
 *     TK_ENOMEM, and let nothing else escape.
 */
#ifndef TROTTERKIT_H
#define TROTTERKIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes.
 *   TK_OK      success.
 *   TK_EINVAL  NULL pointer where data is required, mask bit or
 *              matrix index out of range, n_qubits not matching a
 *              groups handle, or an option value this backend does
 *              not support.
 *   TK_ENOMEM  allocation failure, or a size computation that would
 *              overflow size_t.
 *   TK_EQUBITS n_qubits outside 1..TK_MAX_QUBITS. */
enum { TK_OK = 0, TK_EINVAL = 1, TK_ENOMEM = 2, TK_EQUBITS = 3 };

/* Per-term strategy (tk_evolve_terms).
 *   TK_AUTO     backend chooses.
 *   TK_ONTHEFLY signs and partner indices computed in the kernel.
 *   TK_TABLES   precomputed index/sign tables.
 * A backend returns TK_EINVAL for a strategy it does not support
 * (the c, cpp and rust backends: TK_AUTO and TK_ONTHEFLY only). */
enum { TK_AUTO = 0, TK_ONTHEFLY = 1, TK_TABLES = 2 };

#define TK_MAX_QUBITS 32u

/* A Pauli sum H = sum_g c[g] P(x[g], z[g]), c[g] = c_re[g] + i c_im[g].
 * x, z, c_re and c_im each hold n entries; the backend cannot verify
 * this and reading past them is the caller's fault. When n = 0 the
 * pointers may be NULL; otherwise all four must be non-NULL. Terms
 * are applied in index order g = 0..n-1 (then reversed, see S2). */
typedef struct {
    const uint64_t *x;
    const uint64_t *z;
    const double *c_re;
    const double *c_im;
    size_t n;
} tk_terms;

/* dt holds n_stages step sizes (NULL allowed when n_stages = 0); the
 * stage list is repeated n_steps times. Order 2 is one stage dt;
 * order 4 (Suzuki) is five stages. */
typedef struct {
    const double *dt;
    size_t n_stages;
    size_t n_steps;
} tk_schedule;

/* Opaque grouped form of a matrix H: its XOR-offset slices. Holds H
 * only, not a schedule. */
typedef struct tk_groups tk_groups;

/* Evolve psi (in place) under the per-term product formula.
 *
 * Each fragment is one term exp(-i theta c P), theta = dt/2. With
 * P^2 = I this is
 *   psi'[j] = a psi[j] + k s(j) psi[j ^ x],
 *   a = cos(theta c),
 *   k = -i sin(theta c) i^popcount(x & z),
 *   s(j) = (-1)^popcount((j ^ x) & z)
 * (the sign uses the SOURCE index j ^ x); c is complex, cos and sin
 * are the complex functions. x = 0 gives a diagonal update.
 *
 * Options:
 *   strategy   TK_AUTO, TK_ONTHEFLY or TK_TABLES (see above).
 *   tile_bits  0 = auto, < 0 = no cache blocking, > 0 = tiles of
 *              2^tile_bits amplitudes: consecutive terms with
 *              x < 2^tile_bits are applied tile by tile.
 *   n_threads  0 = auto, 1 = serial, k > 1 = k threads; < 0 invalid.
 * The result does not depend on tile_bits or n_threads. A value the
 * backend does not support returns TK_EINVAL. The c, cpp and rust
 * backends support strategy AUTO/ONTHEFLY, tile_bits 3..n_qubits-1
 * (or <= 0), and n_threads > 1 (c and cpp only when built with
 * OpenMP). Their auto choices are measured on one machine: serial
 * below 10 qubits, 2 threads at 10, up to 4 from 11, with tiles of
 * 2^(n_qubits-2).
 *
 * Returns TK_EINVAL if psi, terms or s is NULL, if n > 0 with a NULL
 * term array, if n_stages > 0 with dt NULL, if any x[g] or z[g] has a
 * bit at or above n_qubits, or for an unsupported option; TK_EQUBITS
 * if n_qubits is outside 1..32; TK_ENOMEM on allocation failure or
 * size overflow. psi is unchanged on every error. */
int tk_evolve_terms(double *psi, uint32_t n_qubits,
                    const tk_terms *terms, const tk_schedule *s,
                    int strategy, int tile_bits, int n_threads);

/* Build the grouped form of the 2^n_qubits square matrix H given as
 * nnz COO entries H[row[e], col[e]] = v_re[e] + i v_im[e].
 *
 *   - Duplicate (row, col) entries are summed (summation order
 *     unspecified); entries with v == 0 exactly are ignored.
 *   - Any row[e] or col[e] >= 2^n_qubits returns TK_EINVAL.
 *   - The diagonal (x = 0) holds d_i = H[i, i].
 *   - Every offset x = row ^ col != 0 is one fragment, made of 2x2
 *     blocks {i, i ^ x} with i < i ^ x, a = H[i, i ^ x] and
 *     b = H[i ^ x, i] (either may be zero: H need not be Hermitian
 *     or symmetric).
 *   - Fragment order: the diagonal first (only if it has an entry),
 *     then the offsets x in ascending order.
 *
 * On success *out receives a handle to free with tk_groups_free. On
 * any failure *out is set to NULL (if out is non-NULL). Returns
 * TK_EINVAL if out is NULL, if nnz > 0 with a NULL array, or for an
 * index out of range; TK_EQUBITS if n_qubits is outside 1..32;
 * TK_ENOMEM on allocation failure or size overflow. When nnz = 0 the
 * arrays may be NULL and H = 0. */
int tk_groups_build(uint32_t n_qubits, const uint64_t *row,
                    const uint64_t *col, const double *v_re,
                    const double *v_im, size_t nnz, tk_groups **out);

/* Evolve psi (in place) under the grouped product formula of g.
 *
 * Each S2(dt) stage applies, at theta = dt/2: the diagonal, the
 * fragments in ascending x, the fragments in descending x, the
 * diagonal again. The diagonal multiplies psi[i] by
 * exp(-i theta d_i). A block maps (p, q) = (psi[i], psi[i ^ x]) by
 *   exp(-i theta M), M = [[0, a], [b, 0]],
 *   = cos(theta w) I - i (sin(theta w) / w) M,  w^2 = ab,
 * both functions being even in w; for |theta^2 ab| < 1e-6 the C
 * backend uses the series 1 - u/2 + u^2/24 and
 * theta (1 - u/6 + u^2/120), u = theta^2 ab.
 * These exponentials are precomputed once per call here, per stage
 * dt; the handle is not modified.
 *
 * n_threads: 0 = auto, 1 = serial, k > 1 = k threads; unsupported
 * values return TK_EINVAL (c, cpp and rust backends: 0 or 1).
 *
 * Returns TK_EINVAL if psi, g or s is NULL, if n_stages > 0 with dt
 * NULL, if n_qubits differs from the n_qubits g was built with
 * (including n_qubits = 0), or for an unsupported n_threads;
 * TK_ENOMEM on allocation failure or size overflow. psi is unchanged
 * on every error. */
int tk_evolve_grouped(double *psi, uint32_t n_qubits,
                      const tk_groups *g, const tk_schedule *s,
                      int n_threads);

/* Free a handle from tk_groups_build. tk_groups_free(NULL) is a
 * no-op. */
void tk_groups_free(tk_groups *g);

/* Static backend name: "c", "cpp" or "rust". */
const char *tk_backend_name(void);

/* Static, human-readable message for a return code; unknown codes
 * give a generic message, never NULL. */
const char *tk_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif
