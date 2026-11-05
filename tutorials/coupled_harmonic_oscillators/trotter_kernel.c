/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Trotter-step inner loop: apply one 2nd-order (Strang) Suzuki-Trotter
 * step to a statevector, given precomputed per-term permutation
 * (src_idx) and phase tables.
 *
 * Built because the pure-Python/NumPy version of this same loop is
 * too slow to run interactively at N=30's ~15,000 Pauli-term count
 * (the per-term Python loop overhead dominates; see the notebook cell
 * that compiles and loads this kernel for the measured numbers). This
 * mirrors paulikit's own compiled-kernel-with-NumPy-fallback design:
 * if a C compiler is not available, the notebook falls back to the
 * (correct, just much slower) pure-NumPy version automatically.
 *
 * psi: complex128 statevector, length dim, modified IN PLACE.
 * coeffs: real coefficient per Pauli term, length n_terms.
 * src_idx: int64 source-index table, shape (n_terms, dim), row-major -
 *          src_idx[g*dim + i] is the index psi reads from for term g,
 *          output position i.
 * phase: complex128 phase table, same shape, phase[g*dim+i] is the
 *        phase factor for term g at output position i.
 * dim, n_terms, dt: as named.
 *
 * One call does a FULL step: forward pass over all n_terms (0..n-1),
 * then a reverse pass (n-1..0) - the palindromic Strang splitting,
 * exactly matching the Python reference this kernel replaces.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

void trotter_step(
    double complex *psi,
    const double *coeffs,
    const int64_t *src_idx,
    const double complex *phase,
    int64_t dim,
    int64_t n_terms,
    double dt
) {
    double complex *scratch = (double complex *) malloc(sizeof(double complex) * dim);
    int64_t g, i;

    /* forward pass: g = 0 .. n_terms-1 */
    for (g = 0; g < n_terms; g++) {
        double theta = coeffs[g] * dt / 2.0;
        double c = cos(theta);
        double s = sin(theta);
        const int64_t *src_row = src_idx + g * dim;
        const double complex *phase_row = phase + g * dim;
        for (i = 0; i < dim; i++) {
            scratch[i] = c * psi[i] - I * s * phase_row[i] * psi[src_row[i]];
        }
        memcpy(psi, scratch, sizeof(double complex) * dim);
    }

    /* reverse pass: g = n_terms-1 .. 0 */
    for (g = n_terms - 1; g >= 0; g--) {
        double theta = coeffs[g] * dt / 2.0;
        double c = cos(theta);
        double s = sin(theta);
        const int64_t *src_row = src_idx + g * dim;
        const double complex *phase_row = phase + g * dim;
        for (i = 0; i < dim; i++) {
            scratch[i] = c * psi[i] - I * s * phase_row[i] * psi[src_row[i]];
        }
        memcpy(psi, scratch, sizeof(double complex) * dim);
    }

    free(scratch);
}
