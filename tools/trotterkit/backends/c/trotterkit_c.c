/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * trotterkit C backend (C99). Plan 1: scalar, serial, on-the-fly
 * strategy; SIMD, cache blocking and threads arrive in Plan 2 behind
 * the same ABI.
 */
#define _POSIX_C_SOURCE 200112L

#include "trotterkit.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int parity64(uint64_t v)
{
#if defined(__GNUC__)
    return __builtin_parityll((unsigned long long)v);
#else
    v ^= v >> 32;
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    return (int)(v & 1u);
#endif
}

static int popcount64(uint64_t v)
{
#if defined(__GNUC__)
    return __builtin_popcountll((unsigned long long)v);
#else
    int n = 0;
    while (v != 0u) {
        v &= v - 1u;
        n++;
    }
    return n;
#endif
}

/* Index of the highest set bit; v must be nonzero. */
static unsigned top_bit(uint64_t v)
{
    unsigned h = 0;
    while ((v >>= 1) != 0u)
        h++;
    return h;
}

static void *alloc64(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes == 0 ? 64 : bytes) != 0)
        return NULL;
    return p;
}

/* exp(-i theta c P) = a I - i b P, a = cos(theta c), b = sin(theta c).
 * Stores (a_re, a_im, k_re, k_im) with k = -i b phase(x, z), so that
 * psi'[j] = a psi[j] + k s(j) psi[j ^ x],
 * s(j) = (-1)^popcount((j ^ x) & z). */
static void term_constants(double theta, double c_re, double c_im,
                           int phase_pow, double out[4])
{
    static const double ph_re[4] = {1.0, 0.0, -1.0, 0.0};
    static const double ph_im[4] = {0.0, 1.0, 0.0, -1.0};
    double u = theta * c_re, v = theta * c_im;
    double b_re = sin(u) * cosh(v), b_im = cos(u) * sinh(v);
    double p_re = b_re * ph_re[phase_pow] - b_im * ph_im[phase_pow];
    double p_im = b_re * ph_im[phase_pow] + b_im * ph_re[phase_pow];

    out[0] = cos(u) * cosh(v);
    out[1] = -sin(u) * sinh(v);
    out[2] = p_im;
    out[3] = -p_re;
}

static void apply_term(double *re, double *im, size_t dim, uint64_t x,
                       uint64_t z, const double *k4)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    size_t j;

    if (x == 0u) {
        for (j = 0; j < dim; j++) {
            double sg = 1.0 - 2.0 * (double)parity64((uint64_t)j & z);
            double mr = ar + sg * kr, mi = ai + sg * ki;
            double pr = re[j], pi = im[j];
            re[j] = mr * pr - mi * pi;
            im[j] = mr * pi + mi * pr;
        }
        return;
    }
    {
        unsigned h = top_bit(x);
        size_t low = ((size_t)1 << h) - 1u, half = dim >> 1, k;

        for (k = 0; k < half; k++) {
            size_t j1 = ((k >> h) << (h + 1u)) | (k & low);
            size_t j2 = j1 ^ (size_t)x;
            double s1 = 1.0 - 2.0 * (double)parity64((uint64_t)j2 & z);
            double s2 = 1.0 - 2.0 * (double)parity64((uint64_t)j1 & z);
            double pr = re[j1], pi = im[j1];
            double qr = re[j2], qi = im[j2];

            re[j1] = ar * pr - ai * pi + s1 * (kr * qr - ki * qi);
            im[j1] = ar * pi + ai * pr + s1 * (kr * qi + ki * qr);
            re[j2] = ar * qr - ai * qi + s2 * (kr * pr - ki * pi);
            im[j2] = ar * qi + ai * qr + s2 * (kr * pi + ki * pr);
        }
    }
}

int tk_evolve_terms(double *psi, uint32_t n_qubits,
                    const tk_terms *terms, const tk_schedule *s,
                    int strategy, int tile_bits, int n_threads)
{
    size_t dim, n, n_st, g, st, step, j;
    double *re, *im, *consts;

    if (psi == NULL || terms == NULL || s == NULL)
        return TK_EINVAL;
    if (n_qubits < 1u || n_qubits > TK_MAX_QUBITS)
        return TK_EQUBITS;
    if (strategy != TK_AUTO && strategy != TK_ONTHEFLY)
        return TK_EINVAL;
    if (tile_bits > 0 || n_threads < 0 || n_threads > 1)
        return TK_EINVAL;
    n = terms->n;
    n_st = s->n_stages;
    if (n_st > 0u && s->dt == NULL)
        return TK_EINVAL;
    if (n > 0u && (terms->x == NULL || terms->z == NULL
                   || terms->c_re == NULL || terms->c_im == NULL))
        return TK_EINVAL;
    for (g = 0; g < n; g++)
        if (((terms->x[g] | terms->z[g]) >> n_qubits) != 0u)
            return TK_EINVAL;

    dim = (size_t)1 << n_qubits;
    re = alloc64(dim * sizeof *re);
    im = alloc64(dim * sizeof *im);
    consts = alloc64(n_st * n * 4u * sizeof *consts);
    if (re == NULL || im == NULL || consts == NULL) {
        free(re);
        free(im);
        free(consts);
        return TK_ENOMEM;
    }
    for (st = 0; st < n_st; st++)
        for (g = 0; g < n; g++)
            term_constants(0.5 * s->dt[st], terms->c_re[g],
                           terms->c_im[g],
                           popcount64(terms->x[g] & terms->z[g]) & 3,
                           consts + 4u * (st * n + g));

    for (j = 0; j < dim; j++) {
        re[j] = psi[2u * j];
        im[j] = psi[2u * j + 1u];
    }
    for (step = 0; step < s->n_steps; step++)
        for (st = 0; st < n_st; st++) {
            const double *cs = consts + 4u * st * n;
            for (g = 0; g < n; g++)
                apply_term(re, im, dim, terms->x[g], terms->z[g],
                           cs + 4u * g);
            for (g = n; g-- > 0u;)
                apply_term(re, im, dim, terms->x[g], terms->z[g],
                           cs + 4u * g);
        }
    for (j = 0; j < dim; j++) {
        psi[2u * j] = re[j];
        psi[2u * j + 1u] = im[j];
    }
    free(re);
    free(im);
    free(consts);
    return TK_OK;
}

const char *tk_backend_name(void)
{
    return "c";
}

const char *tk_strerror(int code)
{
    switch (code) {
    case TK_OK:
        return "success";
    case TK_EINVAL:
        return "invalid argument";
    case TK_ENOMEM:
        return "out of memory";
    case TK_EQUBITS:
        return "qubit count out of range (1..32)";
    default:
        return "unknown error";
    }
}

struct tk_groups {
    uint32_t n_qubits;
};

int tk_groups_build(uint32_t n_qubits, const uint64_t *row,
                    const uint64_t *col, const double *v_re,
                    const double *v_im, size_t nnz, tk_groups **out)
{
    (void)n_qubits; (void)row; (void)col; (void)v_re; (void)v_im;
    (void)nnz; (void)out;
    return TK_EINVAL;
}

int tk_evolve_grouped(double *psi, uint32_t n_qubits,
                      const tk_groups *g, const tk_schedule *s,
                      int n_threads)
{
    (void)psi; (void)n_qubits; (void)g; (void)s; (void)n_threads;
    return TK_EINVAL;
}

void tk_groups_free(tk_groups *g)
{
    free(g);
}
