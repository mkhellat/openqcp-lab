/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * trotterkit C backend (C99), on-the-fly strategy.
 *
 * Per-term mode processes amplitudes in aligned groups of 4 (AVX2
 * when available, otherwise the same arithmetic in scalar code), with
 * a scalar loop for small masks. With OpenMP, threads split each term
 * (one barrier per term) or, for runs of tile-local terms, split the
 * tiles (one barrier per run). Every path gives the same bits.
 */
#define _POSIX_C_SOURCE 200112L

#include "trotterkit.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

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

/* Kernels over index ranges, so threads and tiles can split them.
 * The unit of a range depends on the path:
 *   P_DIAG1  element j            (x = 0, scalar)
 *   P_PAIR1  pair k, j1 = k with a zero inserted at bit h
 *   P_DIAG4  group of 4 elements  (x = 0)
 *   P_PAIR4  group of 4 pairs     (top bit of x >= 2)
 * The scalar paths are the reference: P_PAIR1 also serves masks
 * below 4, states below 8 amplitudes, and TROTTERKIT_FORCE_SCALAR=1
 * (tests compare the paths). */
enum { P_DIAG1, P_PAIR1, P_DIAG4, P_PAIR4 };

static int path_of(uint64_t x, size_t dim, int scalar)
{
    if (scalar || dim < 8u)
        return x == 0u ? P_DIAG1 : P_PAIR1;
    if (x == 0u)
        return P_DIAG4;
    return x < 4u ? P_PAIR1 : P_PAIR4;
}

static size_t units_of(int path, size_t dim)
{
    switch (path) {
    case P_DIAG1:
        return dim;
    case P_PAIR1:
        return dim / 2u;
    case P_DIAG4:
        return dim / 4u;
    default:
        return dim / 8u;
    }
}

static void apply_diag1(double *re, double *im, uint64_t z,
                        const double *k4, size_t lo, size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    size_t j;

    for (j = lo; j < hi; j++) {
        double sg = 1.0 - 2.0 * (double)parity64((uint64_t)j & z);
        double mr = ar + sg * kr, mi = ai + sg * ki;
        double pr = re[j], pi = im[j];
        re[j] = mr * pr - mi * pi;
        im[j] = mr * pi + mi * pr;
    }
}

static void apply_pair1(double *re, double *im, uint64_t x, uint64_t z,
                        const double *k4, size_t lo, size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    unsigned h = top_bit(x);
    size_t low = ((size_t)1 << h) - 1u, k;

    for (k = lo; k < hi; k++) {
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

/* Grouped paths. For x with top bit h >= 2, j1 = G + q (G a multiple
 * of 4 with bit h clear, q = 0..3) pairs with
 * j2 = j1 ^ x = G2 + (q ^ p), G2 = G ^ (x & ~3), p = x & 3. With
 * sigma(B) = (-1)^popcount(B & z) and the per-term lane pattern
 * pat[q] = (-1)^popcount((q ^ p) & z & 3):
 *   s(G + q)  = sigma(G2) pat[q]   (sign for the j1 update)
 *   s(G2 + r) = sigma(G)  pat[r]   (sign for the j2 update)
 * so each group costs two parities, and the partner group is one
 * load plus a fixed lane permutation. The arithmetic per element is
 * the scalar path's, in the same order, so results are identical. */
static double sigma(size_t b, uint64_t z)
{
    return 1.0 - 2.0 * (double)parity64((uint64_t)b & z);
}

static void lane_pattern(uint64_t x, uint64_t z, double pat[4])
{
    unsigned q;
    for (q = 0; q < 4u; q++)
        pat[q] = 1.0 - 2.0 * (double)parity64((q ^ (x & 3u)) & z & 3u);
}

static void apply_diag4(double *re, double *im, uint64_t z,
                        const double *k4, size_t lo, size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    double pat[4];
    size_t i;
    unsigned q;

    lane_pattern(0u, z, pat);
    for (i = lo; i < hi; i++) {
        size_t g = 4u * i;
        double sb = sigma(g, z);
        for (q = 0; q < 4u; q++) {
            double sg = sb * pat[q];
            double mr = ar + sg * kr, mi = ai + sg * ki;
            double pr = re[g + q], pi = im[g + q];
            re[g + q] = mr * pr - mi * pi;
            im[g + q] = mr * pi + mi * pr;
        }
    }
}

/* Base of the i-th group of 4 pairs: 4i with a zero inserted at bit
 * h (h >= 2, so the low two bits stay clear). */
static size_t group_base(size_t i, unsigned h)
{
    size_t f = 4u * i;
    return ((f >> h) << (h + 1u)) | (f & (((size_t)1 << h) - 1u));
}

#if defined(__AVX2__)
/* v[q ^ p] for each lane q. */
static __m256d permute_lanes(__m256d v, unsigned p)
{
    switch (p) {
    case 1:
        return _mm256_permute_pd(v, 0x5);
    case 2:
        return _mm256_permute4x64_pd(v, 0x4E);
    case 3:
        return _mm256_permute4x64_pd(v, 0x1B);
    default:
        return v;
    }
}

/* pat with every lane's sign flipped when par (0 or 1) is 1:
 * branch-free, by XOR-ing the sign bit (a select on the parity
 * mispredicts about half the time and cost 32% more cycles). */
static __m256d flip_if(__m256d pat, uint64_t par)
{
    return _mm256_xor_pd(pat, _mm256_castsi256_pd(
        _mm256_set1_epi64x((long long)(par << 63))));
}

/* One group of 4 pairs: lanes g..g+3 against g2..g2+3 permuted by p,
 * with sign vectors s1 (j1 updates) and s2 (j2 updates). */
static void pair4_group(double *re, double *im, size_t g, size_t g2,
                        unsigned p, __m256d s1, __m256d s2, __m256d ar,
                        __m256d ai, __m256d kr, __m256d ki)
{
    __m256d pr = _mm256_load_pd(re + g), pi = _mm256_load_pd(im + g);
    __m256d qr = _mm256_load_pd(re + g2), qi = _mm256_load_pd(im + g2);
    __m256d qrp = permute_lanes(qr, p), qip = permute_lanes(qi, p);
    __m256d prp = permute_lanes(pr, p), pip = permute_lanes(pi, p);
    __m256d n1r = _mm256_add_pd(
        _mm256_sub_pd(_mm256_mul_pd(ar, pr), _mm256_mul_pd(ai, pi)),
        _mm256_mul_pd(s1, _mm256_sub_pd(_mm256_mul_pd(kr, qrp),
                                        _mm256_mul_pd(ki, qip))));
    __m256d n1i = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ar, pi), _mm256_mul_pd(ai, pr)),
        _mm256_mul_pd(s1, _mm256_add_pd(_mm256_mul_pd(kr, qip),
                                        _mm256_mul_pd(ki, qrp))));
    __m256d n2r = _mm256_add_pd(
        _mm256_sub_pd(_mm256_mul_pd(ar, qr), _mm256_mul_pd(ai, qi)),
        _mm256_mul_pd(s2, _mm256_sub_pd(_mm256_mul_pd(kr, prp),
                                        _mm256_mul_pd(ki, pip))));
    __m256d n2i = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ar, qi), _mm256_mul_pd(ai, qr)),
        _mm256_mul_pd(s2, _mm256_add_pd(_mm256_mul_pd(kr, pip),
                                        _mm256_mul_pd(ki, prp))));
    _mm256_store_pd(re + g, n1r);
    _mm256_store_pd(im + g, n1i);
    _mm256_store_pd(re + g2, n2r);
    _mm256_store_pd(im + g2, n2i);
}

/* Groups [lo, hi) of term x, block by block so the inner loop is a
 * plain stride-4 walk (a flat loop over group indices cost about 5
 * more instructions per group). */
static void apply_pair4(double *re, double *im, uint64_t x, uint64_t z,
                        const double *k4, size_t lo, size_t hi)
{
    const __m256d ar = _mm256_set1_pd(k4[0]), ai = _mm256_set1_pd(k4[1]);
    const __m256d kr = _mm256_set1_pd(k4[2]), ki = _mm256_set1_pd(k4[3]);
    unsigned h = top_bit(x), p = (unsigned)(x & 3u);
    size_t hi_mask = (size_t)(x & ~(uint64_t)3), left = hi - lo;
    size_t span = (size_t)1 << h, g = group_base(lo, h);
    size_t end = (g & ~(2u * span - 1u)) + span;
    double pat[4];
    __m256d pos;

    lane_pattern(x, z, pat);
    pos = _mm256_loadu_pd(pat);
    while (left > 0u) {
        size_t cnt = (end - g) / 4u < left ? (end - g) / 4u : left;
        size_t stop = g + 4u * cnt;
        left -= cnt;
        for (; g < stop; g += 4u) {
            size_t g2 = g ^ hi_mask;
            pair4_group(re, im, g, g2, p,
                        flip_if(pos, (uint64_t)parity64((uint64_t)g2 & z)),
                        flip_if(pos, (uint64_t)parity64((uint64_t)g & z)),
                        ar, ai, kr, ki);
        }
        g += span;               /* skip the partner half */
        end += 2u * span;
    }
}
#else
/* One group of 4 pairs, scalar: the AVX2 arithmetic lane by lane. */
static void pair4_group(double *re, double *im, size_t g, size_t g2,
                        unsigned p, double sg1, double sg2,
                        const double pat[4], const double *k4)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    double pr[4], pi[4], qr[4], qi[4];
    unsigned q;

    for (q = 0; q < 4u; q++) {
        pr[q] = re[g + q];
        pi[q] = im[g + q];
        qr[q] = re[g2 + q];
        qi[q] = im[g2 + q];
    }
    for (q = 0; q < 4u; q++) {
        double s1 = sg1 * pat[q], s2 = sg2 * pat[q];
        unsigned r = q ^ p;
        re[g + q] = ar * pr[q] - ai * pi[q]
                    + s1 * (kr * qr[r] - ki * qi[r]);
        im[g + q] = ar * pi[q] + ai * pr[q]
                    + s1 * (kr * qi[r] + ki * qr[r]);
        re[g2 + q] = ar * qr[q] - ai * qi[q]
                     + s2 * (kr * pr[r] - ki * pi[r]);
        im[g2 + q] = ar * qi[q] + ai * qr[q]
                     + s2 * (kr * pi[r] + ki * pr[r]);
    }
}

static void apply_pair4(double *re, double *im, uint64_t x, uint64_t z,
                        const double *k4, size_t lo, size_t hi)
{
    unsigned h = top_bit(x), p = (unsigned)(x & 3u);
    size_t hi_mask = (size_t)(x & ~(uint64_t)3), left = hi - lo;
    size_t span = (size_t)1 << h, g = group_base(lo, h);
    size_t end = (g & ~(2u * span - 1u)) + span;
    double pat[4];

    lane_pattern(x, z, pat);
    while (left > 0u) {
        size_t cnt = (end - g) / 4u < left ? (end - g) / 4u : left;
        size_t stop = g + 4u * cnt;
        left -= cnt;
        for (; g < stop; g += 4u) {
            size_t g2 = g ^ hi_mask;
            pair4_group(re, im, g, g2, p, sigma(g2, z), sigma(g, z),
                        pat, k4);
        }
        g += span;
        end += 2u * span;
    }
}
#endif

static void apply_range(int path, double *re, double *im, uint64_t x,
                        uint64_t z, const double *k4, size_t lo,
                        size_t hi)
{
    switch (path) {
    case P_DIAG1:
        apply_diag1(re, im, z, k4, lo, hi);
        break;
    case P_PAIR1:
        apply_pair1(re, im, x, z, k4, lo, hi);
        break;
    case P_DIAG4:
        apply_diag4(re, im, z, k4, lo, hi);
        break;
    default:
        apply_pair4(re, im, x, z, k4, lo, hi);
    }
}

/* Everything one evolution needs, shared by the threads. */
typedef struct {
    double *re, *im;
    size_t dim, n, n_st, n_steps;
    const uint64_t *x, *z;
    const double *consts;          /* 4 per (stage, term) */
    int scalar;
    unsigned tile_bits;            /* 0: no tiles */
    const unsigned char *local;    /* x < 2^tile_bits */
    const size_t *run_end;         /* end of the local run at g */
    const size_t *run_begin;       /* start of the local run at g */
    struct spin_barrier *bar;      /* team barrier, NULL if serial */
} drive_ctx;

/* Team barrier after every term or tile-local run. A spinning barrier
 * on atomics, the same algorithm as the C++ and Rust backends: OpenMP's
 * barrier cost 26% more wall time at 12 qubits (chain N=80, 4 threads)
 * and the wait policy made no difference. C99 has no atomics, so this
 * uses GCC's builtins and falls back to the OpenMP barrier elsewhere. */
struct spin_barrier {
    size_t arrived, gen;
};

static void team_barrier(const drive_ctx *c, int T)
{
    if (T <= 1)
        return;
#if defined(__GNUC__)
    {
        struct spin_barrier *b = c->bar;
        size_t gen = __atomic_load_n(&b->gen, __ATOMIC_ACQUIRE);
        if (__atomic_add_fetch(&b->arrived, 1, __ATOMIC_ACQ_REL)
            == (size_t)T) {
            /* Reset before releasing: the next phase's arrivals only
             * start after they observe the new generation. */
            __atomic_store_n(&b->arrived, 0, __ATOMIC_RELAXED);
            __atomic_add_fetch(&b->gen, 1, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(&b->gen, __ATOMIC_ACQUIRE) == gen) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
        }
    }
#elif defined(_OPENMP)
    (void)c;
    {
#pragma omp barrier
    }
#else
    (void)c;
#endif
}

/* Thread tid of T applies its share of term g to the whole state. */
static void term_share(const drive_ctx *c, size_t g, const double *k4,
                       int tid, int T)
{
    int path = path_of(c->x[g], c->dim, c->scalar);
    size_t u = units_of(path, c->dim), lo = 0, hi = u;
    if (T > 1) {                 /* two divisions per term otherwise */
        lo = u * (size_t)tid / (size_t)T;
        hi = u * (size_t)(tid + 1) / (size_t)T;
    }
    apply_range(path, c->re, c->im, c->x[g], c->z[g], k4, lo, hi);
}

/* Terms [b, e) are all tile-local; thread tid of T takes its share of
 * the tiles and applies the run to each, forward or in reverse. In a
 * tile at offset off every sign is the whole-state sign times
 * (-1)^popcount(off & z), which is the same as negating the term's
 * off-diagonal constant; negation is exact, so results match the
 * untiled evolution bit for bit. */
static void run_tiles(const drive_ctx *c, size_t b, size_t e,
                      const double *cs, int reverse, int tid, int T)
{
    size_t len = (size_t)1 << c->tile_bits, n_tiles = c->dim / len;
    size_t t, t_lo = n_tiles * (size_t)tid / (size_t)T;
    size_t t_hi = n_tiles * (size_t)(tid + 1) / (size_t)T;

    for (t = t_lo; t < t_hi; t++) {
        size_t off = t * len, i;
        for (i = 0; i < e - b; i++) {
            size_t g = reverse ? e - 1u - i : b + i;
            const double *k4 = cs + 4u * g;
            double kt[4];
            int path = path_of(c->x[g], len, c->scalar);
            if (parity64((uint64_t)off & c->z[g])) {
                kt[0] = k4[0];
                kt[1] = k4[1];
                kt[2] = -k4[2];
                kt[3] = -k4[3];
                k4 = kt;
            }
            apply_range(path, c->re + off, c->im + off, c->x[g],
                        c->z[g], k4, 0, units_of(path, len));
        }
    }
}

static void drive(const drive_ctx *c, int tid, int T)
{
    size_t step, st, g;

    for (step = 0; step < c->n_steps; step++)
        for (st = 0; st < c->n_st; st++) {
            const double *cs = c->consts + 4u * st * c->n;
            for (g = 0; g < c->n;) {
                if (c->tile_bits != 0u && c->local[g]) {
                    run_tiles(c, g, c->run_end[g], cs, 0, tid, T);
                    g = c->run_end[g];
                } else {
                    term_share(c, g, cs + 4u * g, tid, T);
                    g++;
                }
                team_barrier(c, T);
            }
            for (g = c->n; g > 0u;) {
                size_t last = g - 1u;
                if (c->tile_bits != 0u && c->local[last]) {
                    run_tiles(c, c->run_begin[last], g, cs, 1, tid, T);
                    g = c->run_begin[last];
                } else {
                    term_share(c, last, cs + 4u * last, tid, T);
                    g = last;
                }
                team_barrier(c, T);
            }
        }
}

/* Automatic choices (n_threads = 0, tile_bits = 0), measured with
 * bench/quick on an i7-8550U (4 cores, 2 threads each); other
 * machines may want explicit values. Threads lost at 9 qubits (one
 * barrier per term outweighs a 0.4 us term) and won from 11; 4
 * threads beat 8 (hyperthreads share the SIMD units this kernel
 * saturates); tiles of 2^(n-2) amplitudes were best at 11 and 12
 * qubits. */
#define TK_AUTO_MIN_QUBITS 11u
#define TK_AUTO_MAX_THREADS 4

static int auto_threads(uint32_t n_qubits)
{
#if defined(_OPENMP)
    int procs = omp_get_num_procs();
    if (n_qubits < TK_AUTO_MIN_QUBITS)
        return 1;
    return procs < TK_AUTO_MAX_THREADS ? procs : TK_AUTO_MAX_THREADS;
#else
    (void)n_qubits;
    return 1;
#endif
}

int tk_evolve_terms(double *psi, uint32_t n_qubits,
                    const tk_terms *terms, const tk_schedule *s,
                    int strategy, int tile_bits, int n_threads)
{
    size_t dim, n, n_st, g, st, j;
    double *re, *im, *consts;
    unsigned char *local = NULL;
    size_t *run_end = NULL, *run_begin = NULL;
    const char *force = getenv("TROTTERKIT_FORCE_SCALAR");
    drive_ctx c;
    int T;

    if (psi == NULL || terms == NULL || s == NULL)
        return TK_EINVAL;
    if (n_qubits < 1u || n_qubits > TK_MAX_QUBITS)
        return TK_EQUBITS;
    if (strategy != TK_AUTO && strategy != TK_ONTHEFLY)
        return TK_EINVAL;
    if (n_threads < 0)
        return TK_EINVAL;
#if !defined(_OPENMP)
    if (n_threads > 1)
        return TK_EINVAL;
#endif
    /* Tiles need at least 8 amplitudes and must split the state. */
    if (tile_bits > 0 && (tile_bits < 3 || (uint32_t)tile_bits >= n_qubits))
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

    T = n_threads == 0 ? auto_threads(n_qubits) : n_threads;
    if (tile_bits == 0)
        tile_bits = T > 1 && n_qubits >= 5u ? (int)n_qubits - 2 : -1;
    dim = (size_t)1 << n_qubits;
    re = alloc64(dim * sizeof *re);
    im = alloc64(dim * sizeof *im);
    consts = alloc64(n_st * n * 4u * sizeof *consts);
    if (tile_bits > 0) {
        local = malloc(n == 0u ? 1u : n);
        run_end = malloc((n == 0u ? 1u : n) * sizeof *run_end);
        run_begin = malloc((n == 0u ? 1u : n) * sizeof *run_begin);
    }
    if (re == NULL || im == NULL || consts == NULL
        || (tile_bits > 0
            && (local == NULL || run_end == NULL || run_begin == NULL))) {
        free(re);
        free(im);
        free(consts);
        free(local);
        free(run_end);
        free(run_begin);
        return TK_ENOMEM;
    }
    if (tile_bits > 0) {
        for (g = 0; g < n; g++)
            local[g] = (terms->x[g] >> tile_bits) == 0u;
        for (g = n; g-- > 0u;)
            run_end[g] = local[g] && g + 1u < n && local[g + 1u]
                         ? run_end[g + 1u] : g + 1u;
        for (g = 0; g < n; g++)
            run_begin[g] = local[g] && g > 0u && local[g - 1u]
                           ? run_begin[g - 1u] : g;
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

    c.re = re;
    c.im = im;
    c.dim = dim;
    c.n = n;
    c.n_st = n_st;
    c.n_steps = s->n_steps;
    c.x = terms->x;
    c.z = terms->z;
    c.consts = consts;
    c.scalar = force != NULL && strcmp(force, "1") == 0;
    c.tile_bits = tile_bits > 0 ? (unsigned)tile_bits : 0u;
    c.local = local;
    c.run_end = run_end;
    c.run_begin = run_begin;
    {
        struct spin_barrier bar = {0, 0};
        c.bar = &bar;
#if defined(_OPENMP)
        if (T > 1) {
#pragma omp parallel num_threads(T)
            drive(&c, omp_get_thread_num(), omp_get_num_threads());
        } else
#endif
            drive(&c, 0, 1);
    }
    (void)T;                     /* unused without OpenMP */

    for (j = 0; j < dim; j++) {
        psi[2u * j] = re[j];
        psi[2u * j + 1u] = im[j];
    }
    free(re);
    free(im);
    free(consts);
    free(local);
    free(run_end);
    free(run_begin);
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

/* Complex arithmetic written out on (re, im) pairs from real libm
 * functions only, so the C++ and Rust backends can repeat the same
 * operations in the same order and agree to the bit (C's complex.h
 * functions have no Rust counterpart with identical rounding). */
typedef struct {
    double re, im;
} cplx;

static cplx cmul(cplx a, cplx b)
{
    cplx r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

/* Principal square root, the branch chosen to avoid cancellation. */
static cplx csqrt_h(cplx a)
{
    double r = hypot(a.re, a.im);
    cplx w;
    if (a.re >= 0.0) {
        w.re = sqrt((r + a.re) * 0.5);
        w.im = a.im / (2.0 * w.re);
    } else {
        w.im = sqrt((r - a.re) * 0.5);
        if (a.im < 0.0)
            w.im = -w.im;
        w.re = a.im / (2.0 * w.im);
    }
    return w;
}

/* (c, A, B) for one block and angle theta:
 * exp(-i theta [[0,a],[b,0]]) = [[c, A], [B, c]],
 * c = cos(theta w), A = -i (sin(theta w)/w) a, B likewise with b,
 * w^2 = ab; the series keeps it exact as ab -> 0. */
static void block_constants(double theta, double a_re, double a_im,
                            double b_re, double b_im, double out[6])
{
    cplx a, b, ab, u, c, sv, sa, sb;
    double t2 = theta * theta;

    a.re = a_re;
    a.im = a_im;
    b.re = b_re;
    b.im = b_im;
    ab = cmul(a, b);
    u.re = t2 * ab.re;
    u.im = t2 * ab.im;
    if (hypot(u.re, u.im) < 1e-6) {
        cplx u2 = cmul(u, u);
        c.re = (1.0 - u.re / 2.0) + u2.re / 24.0;
        c.im = (0.0 - u.im / 2.0) + u2.im / 24.0;
        sv.re = theta * ((1.0 - u.re / 6.0) + u2.re / 120.0);
        sv.im = theta * ((0.0 - u.im / 6.0) + u2.im / 120.0);
    } else {
        cplx w = csqrt_h(ab), sn;
        double xr = theta * w.re, xi = theta * w.im;
        double d = w.re * w.re + w.im * w.im;
        c.re = cos(xr) * cosh(xi);
        c.im = -(sin(xr) * sinh(xi));
        sn.re = sin(xr) * cosh(xi);
        sn.im = cos(xr) * sinh(xi);
        sv.re = (sn.re * w.re + sn.im * w.im) / d;
        sv.im = (sn.im * w.re - sn.re * w.im) / d;
    }
    sa = cmul(sv, a);
    sb = cmul(sv, b);
    out[0] = c.re;
    out[1] = c.im;
    out[2] = sa.im;            /* -i (x + iy) = y - ix */
    out[3] = -sa.re;
    out[4] = sb.im;
    out[5] = -sb.re;
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
