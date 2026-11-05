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
    size_t n_diag;
    uint64_t *diag_i;
    double *diag_re, *diag_im;
    size_t n_frag;
    size_t *frag_off;          /* n_frag + 1 offsets into blocks */
    size_t n_blk;
    uint64_t *blk_i, *blk_j;   /* blk_j = blk_i ^ x */
    double *a_re, *a_im, *b_re, *b_im;
};

typedef struct {
    uint64_t x, i;
    int lower;                 /* entry is H[i ^ x, i] (b) */
    double re, im;
} tk_entry;

static int entry_cmp(const void *pa, const void *pb)
{
    const tk_entry *a = pa, *b = pb;
    if (a->x != b->x)
        return a->x < b->x ? -1 : 1;
    if (a->i != b->i)
        return a->i < b->i ? -1 : 1;
    return 0;
}

void tk_groups_free(tk_groups *g)
{
    if (g == NULL)
        return;
    free(g->diag_i);
    free(g->diag_re);
    free(g->diag_im);
    free(g->frag_off);
    free(g->blk_i);
    free(g->blk_j);
    free(g->a_re);
    free(g->a_im);
    free(g->b_re);
    free(g->b_im);
    free(g);
}

static size_t at_least_one(size_t n)
{
    return n == 0u ? 1u : n;
}

int tk_groups_build(uint32_t n_qubits, const uint64_t *row,
                    const uint64_t *col, const double *v_re,
                    const double *v_im, size_t nnz, tk_groups **out)
{
    size_t dim, e, m = 0, n_diag = 0, n_blk = 0, n_frag = 0;
    tk_entry *ent;
    tk_groups *g;

    if (out == NULL)
        return TK_EINVAL;
    *out = NULL;
    if (n_qubits < 1u || n_qubits > TK_MAX_QUBITS)
        return TK_EQUBITS;
    if (nnz > 0u && (row == NULL || col == NULL || v_re == NULL
                     || v_im == NULL))
        return TK_EINVAL;
    dim = (size_t)1 << n_qubits;
    ent = malloc(at_least_one(nnz) * sizeof *ent);
    if (ent == NULL)
        return TK_ENOMEM;
    for (e = 0; e < nnz; e++) {
        uint64_t r = row[e], c = col[e];
        if (r >= dim || c >= dim) {
            free(ent);
            return TK_EINVAL;
        }
        if (v_re[e] == 0.0 && v_im[e] == 0.0)
            continue;
        ent[m].x = r ^ c;
        ent[m].i = r < c ? r : c;
        ent[m].lower = r > c;
        ent[m].re = v_re[e];
        ent[m].im = v_im[e];
        m++;
    }
    qsort(ent, m, sizeof *ent, entry_cmp);

    for (e = 0; e < m; e++) {
        int same_x = e > 0u && ent[e].x == ent[e - 1u].x;
        int fresh = !(same_x && ent[e].i == ent[e - 1u].i);
        if (ent[e].x == 0u) {
            if (fresh)
                n_diag++;
        } else {
            if (!same_x)
                n_frag++;
            if (fresh)
                n_blk++;
        }
    }

    g = calloc(1, sizeof *g);
    if (g == NULL) {
        free(ent);
        return TK_ENOMEM;
    }
    g->n_qubits = n_qubits;
    g->n_diag = n_diag;
    g->n_frag = n_frag;
    g->n_blk = n_blk;
    g->diag_i = malloc(at_least_one(n_diag) * sizeof *g->diag_i);
    g->diag_re = malloc(at_least_one(n_diag) * sizeof *g->diag_re);
    g->diag_im = malloc(at_least_one(n_diag) * sizeof *g->diag_im);
    g->frag_off = malloc((n_frag + 1u) * sizeof *g->frag_off);
    g->blk_i = malloc(at_least_one(n_blk) * sizeof *g->blk_i);
    g->blk_j = malloc(at_least_one(n_blk) * sizeof *g->blk_j);
    g->a_re = calloc(at_least_one(n_blk), sizeof *g->a_re);
    g->a_im = calloc(at_least_one(n_blk), sizeof *g->a_im);
    g->b_re = calloc(at_least_one(n_blk), sizeof *g->b_re);
    g->b_im = calloc(at_least_one(n_blk), sizeof *g->b_im);
    if (g->diag_i == NULL || g->diag_re == NULL || g->diag_im == NULL
        || g->frag_off == NULL || g->blk_i == NULL || g->blk_j == NULL
        || g->a_re == NULL || g->a_im == NULL || g->b_re == NULL
        || g->b_im == NULL) {
        free(ent);
        tk_groups_free(g);
        return TK_ENOMEM;
    }

    {
        size_t d = 0, b = 0, f = 0;
        for (e = 0; e < m; e++) {
            int same_x = e > 0u && ent[e].x == ent[e - 1u].x;
            int fresh = !(same_x && ent[e].i == ent[e - 1u].i);
            if (ent[e].x == 0u) {
                if (fresh) {
                    g->diag_i[d] = ent[e].i;
                    g->diag_re[d] = 0.0;
                    g->diag_im[d] = 0.0;
                    d++;
                }
                g->diag_re[d - 1u] += ent[e].re;
                g->diag_im[d - 1u] += ent[e].im;
                continue;
            }
            if (!same_x)
                g->frag_off[f++] = b;
            if (fresh) {
                g->blk_i[b] = ent[e].i;
                g->blk_j[b] = ent[e].i ^ ent[e].x;
                b++;
            }
            if (ent[e].lower) {
                g->b_re[b - 1u] += ent[e].re;
                g->b_im[b - 1u] += ent[e].im;
            } else {
                g->a_re[b - 1u] += ent[e].re;
                g->a_im[b - 1u] += ent[e].im;
            }
        }
        g->frag_off[f] = b;
    }
    free(ent);
    *out = g;
    return TK_OK;
}

/* (c, A, B) for one block and angle theta:
 * exp(-i theta [[0,a],[b,0]]) = [[c, A], [B, c]],
 * c = cos(theta w), A = -i (sin(theta w)/w) a, B likewise with b,
 * w^2 = ab; the series keeps it exact as ab -> 0. */
static void block_constants(double theta, double a_re, double a_im,
                            double b_re, double b_im, double out[6])
{
    double complex a = a_re + I * a_im, b = b_re + I * b_im;
    double complex u = theta * theta * a * b, c, sv, A, B;

    if (cabs(u) < 1e-6) {
        c = 1.0 - u / 2.0 + u * u / 24.0;
        sv = theta * (1.0 - u / 6.0 + u * u / 120.0);
    } else {
        double complex w = csqrt(a * b);
        c = ccos(theta * w);
        sv = csin(theta * w) / w;
    }
    A = -I * sv * a;
    B = -I * sv * b;
    out[0] = creal(c);
    out[1] = cimag(c);
    out[2] = creal(A);
    out[3] = cimag(A);
    out[4] = creal(B);
    out[5] = cimag(B);
}

static void apply_diag(double *re, double *im, const tk_groups *g,
                       const double *dm)
{
    size_t d;
    for (d = 0; d < g->n_diag; d++) {
        size_t i = (size_t)g->diag_i[d];
        double mr = dm[2u * d], mi = dm[2u * d + 1u];
        double pr = re[i], pi = im[i];
        re[i] = mr * pr - mi * pi;
        im[i] = mr * pi + mi * pr;
    }
}

static void apply_frag(double *re, double *im, const tk_groups *g,
                       size_t f, const double *bm)
{
    size_t b;
    for (b = g->frag_off[f]; b < g->frag_off[f + 1u]; b++) {
        const double *k = bm + 6u * b;
        size_t i = (size_t)g->blk_i[b], j = (size_t)g->blk_j[b];
        double pr = re[i], pi = im[i], qr = re[j], qi = im[j];
        re[i] = k[0] * pr - k[1] * pi + k[2] * qr - k[3] * qi;
        im[i] = k[0] * pi + k[1] * pr + k[2] * qi + k[3] * qr;
        re[j] = k[4] * pr - k[5] * pi + k[0] * qr - k[1] * qi;
        im[j] = k[4] * pi + k[5] * pr + k[0] * qi + k[1] * qr;
    }
}

int tk_evolve_grouped(double *psi, uint32_t n_qubits,
                      const tk_groups *g, const tk_schedule *s,
                      int n_threads)
{
    size_t dim, n_st, st, step, j, d, b, f;
    double *re, *im, *dm, *bm;

    if (psi == NULL || g == NULL || s == NULL)
        return TK_EINVAL;
    if (n_qubits != g->n_qubits || n_threads < 0 || n_threads > 1)
        return TK_EINVAL;
    n_st = s->n_stages;
    if (n_st > 0u && s->dt == NULL)
        return TK_EINVAL;
    dim = (size_t)1 << n_qubits;
    re = alloc64(dim * sizeof *re);
    im = alloc64(dim * sizeof *im);
    dm = alloc64(n_st * g->n_diag * 2u * sizeof *dm);
    bm = alloc64(n_st * g->n_blk * 6u * sizeof *bm);
    if (re == NULL || im == NULL || dm == NULL || bm == NULL) {
        free(re);
        free(im);
        free(dm);
        free(bm);
        return TK_ENOMEM;
    }
    for (st = 0; st < n_st; st++) {
        double theta = 0.5 * s->dt[st];
        for (d = 0; d < g->n_diag; d++) {
            double grow = exp(theta * g->diag_im[d]);
            double ang = theta * g->diag_re[d];
            dm[2u * (st * g->n_diag + d)] = grow * cos(ang);
            dm[2u * (st * g->n_diag + d) + 1u] = -grow * sin(ang);
        }
        for (b = 0; b < g->n_blk; b++)
            block_constants(theta, g->a_re[b], g->a_im[b], g->b_re[b],
                            g->b_im[b], bm + 6u * (st * g->n_blk + b));
    }
    for (j = 0; j < dim; j++) {
        re[j] = psi[2u * j];
        im[j] = psi[2u * j + 1u];
    }
    for (step = 0; step < s->n_steps; step++)
        for (st = 0; st < n_st; st++) {
            const double *ds = dm + 2u * st * g->n_diag;
            const double *bs = bm + 6u * st * g->n_blk;
            apply_diag(re, im, g, ds);
            for (f = 0; f < g->n_frag; f++)
                apply_frag(re, im, g, f, bs);
            for (f = g->n_frag; f-- > 0u;)
                apply_frag(re, im, g, f, bs);
            apply_diag(re, im, g, ds);
        }
    for (j = 0; j < dim; j++) {
        psi[2u * j] = re[j];
        psi[2u * j + 1u] = im[j];
    }
    free(re);
    free(im);
    free(dm);
    free(bm);
    return TK_OK;
}
