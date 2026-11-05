// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
//
// trotterkit C++ backend (C++17). The C backend's algorithm, layout
// and arithmetic order (see backends/c/trotterkit_c.c), written in
// C++: the lane permutation of the grouped per-term path is a template
// parameter, buffers are aligned std::vectors, and no exception
// crosses the C ABI (allocation failure returns TK_ENOMEM). Results
// equal the C backend's to the bit.

#include "trotterkit.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

int parity64(std::uint64_t v)
{
    return __builtin_parityll(static_cast<unsigned long long>(v));
}

int popcount64(std::uint64_t v)
{
    return __builtin_popcountll(static_cast<unsigned long long>(v));
}

unsigned top_bit(std::uint64_t v)
{
    unsigned h = 0;
    while ((v >>= 1) != 0u)
        h++;
    return h;
}

// 64-byte aligned storage for the state and constants.
template <class T>
struct Aligned {
    using value_type = T;
    Aligned() = default;
    template <class U>
    Aligned(const Aligned<U> &) noexcept {}
    T *allocate(std::size_t n)
    {
        void *p = nullptr;
        if (posix_memalign(&p, 64, n == 0 ? 64 : n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
};
template <class T, class U>
bool operator==(const Aligned<T> &, const Aligned<U> &) { return true; }
template <class T, class U>
bool operator!=(const Aligned<T> &, const Aligned<U> &) { return false; }

using dvec = std::vector<double, Aligned<double>>;

// exp(-i theta c P) = a I - i b P, a = cos(theta c), b = sin(theta c):
// (a_re, a_im, k_re, k_im), k = -i b phase(x, z).
void term_constants(double theta, double c_re, double c_im,
                    int phase_pow, double out[4])
{
    static const double ph_re[4] = {1.0, 0.0, -1.0, 0.0};
    static const double ph_im[4] = {0.0, 1.0, 0.0, -1.0};
    double u = theta * c_re, v = theta * c_im;
    double b_re = std::sin(u) * std::cosh(v);
    double b_im = std::cos(u) * std::sinh(v);
    double p_re = b_re * ph_re[phase_pow] - b_im * ph_im[phase_pow];
    double p_im = b_re * ph_im[phase_pow] + b_im * ph_re[phase_pow];

    out[0] = std::cos(u) * std::cosh(v);
    out[1] = -std::sin(u) * std::sinh(v);
    out[2] = p_im;
    out[3] = -p_re;
}

// Paths and their range units, as in the C backend.
enum Path { P_DIAG1, P_PAIR1, P_DIAG4, P_PAIR4 };

Path path_of(std::uint64_t x, std::size_t dim, bool scalar)
{
    if (scalar || dim < 8u)
        return x == 0u ? P_DIAG1 : P_PAIR1;
    if (x == 0u)
        return P_DIAG4;
    return x < 4u ? P_PAIR1 : P_PAIR4;
}

std::size_t units_of(Path path, std::size_t dim)
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

void apply_diag1(double *re, double *im, std::uint64_t z,
                 const double *k4, std::size_t lo, std::size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    for (std::size_t j = lo; j < hi; j++) {
        double sg = 1.0 - 2.0 * static_cast<double>(parity64(j & z));
        double mr = ar + sg * kr, mi = ai + sg * ki;
        double pr = re[j], pi = im[j];
        re[j] = mr * pr - mi * pi;
        im[j] = mr * pi + mi * pr;
    }
}

void apply_pair1(double *re, double *im, std::uint64_t x,
                 std::uint64_t z, const double *k4, std::size_t lo,
                 std::size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    unsigned h = top_bit(x);
    std::size_t low = (std::size_t{1} << h) - 1u;
    for (std::size_t k = lo; k < hi; k++) {
        std::size_t j1 = ((k >> h) << (h + 1u)) | (k & low);
        std::size_t j2 = j1 ^ static_cast<std::size_t>(x);
        double s1 = 1.0 - 2.0 * static_cast<double>(parity64(j2 & z));
        double s2 = 1.0 - 2.0 * static_cast<double>(parity64(j1 & z));
        double pr = re[j1], pi = im[j1], qr = re[j2], qi = im[j2];
        re[j1] = ar * pr - ai * pi + s1 * (kr * qr - ki * qi);
        im[j1] = ar * pi + ai * pr + s1 * (kr * qi + ki * qr);
        re[j2] = ar * qr - ai * qi + s2 * (kr * pr - ki * pi);
        im[j2] = ar * qi + ai * qr + s2 * (kr * pi + ki * pr);
    }
}

double sigma(std::size_t b, std::uint64_t z)
{
    return 1.0 - 2.0 * static_cast<double>(parity64(b & z));
}

void lane_pattern(std::uint64_t x, std::uint64_t z, double pat[4])
{
    for (unsigned q = 0; q < 4u; q++)
        pat[q] = 1.0 - 2.0 * static_cast<double>(
                     parity64((q ^ (x & 3u)) & z & 3u));
}

void apply_diag4(double *re, double *im, std::uint64_t z,
                 const double *k4, std::size_t lo, std::size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    double pat[4];
    lane_pattern(0u, z, pat);
    for (std::size_t i = lo; i < hi; i++) {
        std::size_t g = 4u * i;
        double sb = sigma(g, z);
        for (unsigned q = 0; q < 4u; q++) {
            double sg = sb * pat[q];
            double mr = ar + sg * kr, mi = ai + sg * ki;
            double pr = re[g + q], pi = im[g + q];
            re[g + q] = mr * pr - mi * pi;
            im[g + q] = mr * pi + mi * pr;
        }
    }
}

std::size_t group_base(std::size_t i, unsigned h)
{
    std::size_t f = 4u * i;
    return ((f >> h) << (h + 1u)) | (f & ((std::size_t{1} << h) - 1u));
}

#if defined(__AVX2__)
// v[q ^ P] for each lane q, chosen at compile time.
template <unsigned P>
__m256d permute_lanes(__m256d v)
{
    if constexpr (P == 1)
        return _mm256_permute_pd(v, 0x5);
    else if constexpr (P == 2)
        return _mm256_permute4x64_pd(v, 0x4E);
    else if constexpr (P == 3)
        return _mm256_permute4x64_pd(v, 0x1B);
    else
        return v;
}

// pat with every lane's sign flipped when par (0 or 1) is 1.
__m256d flip_if(__m256d pat, std::uint64_t par)
{
    return _mm256_xor_pd(pat, _mm256_castsi256_pd(_mm256_set1_epi64x(
                                  static_cast<long long>(par << 63))));
}

template <unsigned P>
void pair4_group(double *re, double *im, std::size_t g, std::size_t g2,
                 __m256d s1, __m256d s2, __m256d ar, __m256d ai,
                 __m256d kr, __m256d ki)
{
    __m256d pr = _mm256_load_pd(re + g), pi = _mm256_load_pd(im + g);
    __m256d qr = _mm256_load_pd(re + g2), qi = _mm256_load_pd(im + g2);
    __m256d qrp = permute_lanes<P>(qr), qip = permute_lanes<P>(qi);
    __m256d prp = permute_lanes<P>(pr), pip = permute_lanes<P>(pi);
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

// Groups [lo, hi), block by block with a plain stride-4 inner loop.
template <unsigned P>
void apply_pair4_p(double *re, double *im, std::uint64_t x,
                   std::uint64_t z, const double *k4, std::size_t lo,
                   std::size_t hi)
{
    const __m256d ar = _mm256_set1_pd(k4[0]), ai = _mm256_set1_pd(k4[1]);
    const __m256d kr = _mm256_set1_pd(k4[2]), ki = _mm256_set1_pd(k4[3]);
    unsigned h = top_bit(x);
    std::size_t hi_mask = static_cast<std::size_t>(x & ~std::uint64_t{3});
    std::size_t left = hi - lo, span = std::size_t{1} << h;
    std::size_t g = group_base(lo, h);
    std::size_t end = (g & ~(2u * span - 1u)) + span;
    double pat[4];
    lane_pattern(x, z, pat);
    const __m256d pos = _mm256_loadu_pd(pat);
    while (left > 0u) {
        std::size_t cnt = std::min((end - g) / 4u, left);
        std::size_t stop = g + 4u * cnt;
        left -= cnt;
        for (; g < stop; g += 4u) {
            std::size_t g2 = g ^ hi_mask;
            pair4_group<P>(re, im, g, g2,
                           flip_if(pos, static_cast<std::uint64_t>(
                                            parity64(g2 & z))),
                           flip_if(pos, static_cast<std::uint64_t>(
                                            parity64(g & z))),
                           ar, ai, kr, ki);
        }
        g += span;  // skip the partner half
        end += 2u * span;
    }
}
#else
template <unsigned P>
void apply_pair4_p(double *re, double *im, std::uint64_t x,
                   std::uint64_t z, const double *k4, std::size_t lo,
                   std::size_t hi)
{
    const double ar = k4[0], ai = k4[1], kr = k4[2], ki = k4[3];
    unsigned h = top_bit(x);
    std::size_t hi_mask = static_cast<std::size_t>(x & ~std::uint64_t{3});
    std::size_t left = hi - lo, span = std::size_t{1} << h;
    std::size_t g = group_base(lo, h);
    std::size_t end = (g & ~(2u * span - 1u)) + span;
    double pat[4];
    lane_pattern(x, z, pat);
    while (left > 0u) {
        std::size_t cnt = std::min((end - g) / 4u, left);
        std::size_t stop = g + 4u * cnt;
        left -= cnt;
        for (; g < stop; g += 4u) {
            std::size_t g2 = g ^ hi_mask;
            double sg1 = sigma(g2, z), sg2 = sigma(g, z);
            double pr[4], pi[4], qr[4], qi[4];
            for (unsigned q = 0; q < 4u; q++) {
                pr[q] = re[g + q];
                pi[q] = im[g + q];
                qr[q] = re[g2 + q];
                qi[q] = im[g2 + q];
            }
            for (unsigned q = 0; q < 4u; q++) {
                double s1 = sg1 * pat[q], s2 = sg2 * pat[q];
                unsigned r = q ^ P;
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
        g += span;
        end += 2u * span;
    }
}
#endif

void apply_range(Path path, double *re, double *im, std::uint64_t x,
                 std::uint64_t z, const double *k4, std::size_t lo,
                 std::size_t hi)
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
        switch (x & 3u) {
        case 0:
            apply_pair4_p<0>(re, im, x, z, k4, lo, hi);
            break;
        case 1:
            apply_pair4_p<1>(re, im, x, z, k4, lo, hi);
            break;
        case 2:
            apply_pair4_p<2>(re, im, x, z, k4, lo, hi);
            break;
        default:
            apply_pair4_p<3>(re, im, x, z, k4, lo, hi);
        }
    }
}

// Team barrier after every term or tile-local run: a spinning barrier
// on atomics, the same algorithm as the C and Rust backends (OpenMP's
// barrier cost 26% more wall time at 12 qubits, 4 threads).
struct SpinBarrier {
    std::atomic<std::size_t> arrived{0}, gen{0};

    void wait(std::size_t n)
    {
        std::size_t g = gen.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            // Reset before releasing: the next phase's arrivals only
            // start after they observe the new generation.
            arrived.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
        }
    }
};

struct Drive {
    double *re, *im;
    std::size_t dim, n, n_st, n_steps;
    const std::uint64_t *x, *z;
    const double *consts;
    bool scalar;
    unsigned tile_bits;  // 0: no tiles
    const unsigned char *local;
    const std::size_t *run_end, *run_begin;
    SpinBarrier *bar;

    void barrier(int T) const
    {
        if (T > 1)
            bar->wait(static_cast<std::size_t>(T));
    }

    void term_share(std::size_t g, const double *k4, int tid, int T) const
    {
        Path path = path_of(x[g], dim, scalar);
        std::size_t u = units_of(path, dim), lo = 0, hi = u;
        if (T > 1) {
            lo = u * static_cast<std::size_t>(tid)
                 / static_cast<std::size_t>(T);
            hi = u * static_cast<std::size_t>(tid + 1)
                 / static_cast<std::size_t>(T);
        }
        apply_range(path, re, im, x[g], z[g], k4, lo, hi);
    }

    // Tile-local run [b, e): sign in a tile at off is the whole-state
    // sign times (-1)^popcount(off & z) = negated off-diagonal constant.
    void run_tiles(std::size_t b, std::size_t e, const double *cs,
                   bool reverse, int tid, int T) const
    {
        std::size_t len = std::size_t{1} << tile_bits;
        std::size_t n_tiles = dim / len;
        std::size_t t_lo = n_tiles * static_cast<std::size_t>(tid)
                           / static_cast<std::size_t>(T);
        std::size_t t_hi = n_tiles * static_cast<std::size_t>(tid + 1)
                           / static_cast<std::size_t>(T);
        for (std::size_t t = t_lo; t < t_hi; t++) {
            std::size_t off = t * len;
            for (std::size_t i = 0; i < e - b; i++) {
                std::size_t g = reverse ? e - 1u - i : b + i;
                const double *k4 = cs + 4u * g;
                double kt[4];
                Path path = path_of(x[g], len, scalar);
                if (parity64(off & z[g])) {
                    kt[0] = k4[0];
                    kt[1] = k4[1];
                    kt[2] = -k4[2];
                    kt[3] = -k4[3];
                    k4 = kt;
                }
                apply_range(path, re + off, im + off, x[g], z[g], k4, 0,
                            units_of(path, len));
            }
        }
    }

    void run(int tid, int T) const
    {
        for (std::size_t step = 0; step < n_steps; step++)
            for (std::size_t st = 0; st < n_st; st++) {
                const double *cs = consts + 4u * st * n;
                for (std::size_t g = 0; g < n;) {
                    if (tile_bits != 0u && local[g]) {
                        run_tiles(g, run_end[g], cs, false, tid, T);
                        g = run_end[g];
                    } else {
                        term_share(g, cs + 4u * g, tid, T);
                        g++;
                    }
                    barrier(T);
                }
                for (std::size_t g = n; g > 0u;) {
                    std::size_t last = g - 1u;
                    if (tile_bits != 0u && local[last]) {
                        run_tiles(run_begin[last], g, cs, true, tid, T);
                        g = run_begin[last];
                    } else {
                        term_share(last, cs + 4u * last, tid, T);
                        g = last;
                    }
                    barrier(T);
                }
            }
    }
};

// Automatic choices, the C backend's (measured on an i7-8550U):
// serial below 10 qubits, 2 threads at 10, up to 4 from 11.
constexpr std::uint32_t AUTO_MIN_QUBITS = 10;
constexpr int AUTO_MAX_THREADS = 4;

int auto_threads(std::uint32_t n_qubits)
{
#if defined(_OPENMP)
    if (n_qubits < AUTO_MIN_QUBITS)
        return 1;
    int want = n_qubits == AUTO_MIN_QUBITS ? 2 : AUTO_MAX_THREADS;
    return std::min(omp_get_num_procs(), want);
#else
    (void)n_qubits;
    return 1;
#endif
}

bool force_scalar()
{
    const char *f = std::getenv("TROTTERKIT_FORCE_SCALAR");
    return f != nullptr && std::strcmp(f, "1") == 0;
}

int evolve_terms(double *psi, std::uint32_t n_qubits, const tk_terms *t,
                 const tk_schedule *s, int strategy, int tile_bits,
                 int n_threads)
{
    if (psi == nullptr || t == nullptr || s == nullptr)
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
    if (tile_bits > 0
        && (tile_bits < 3 || static_cast<std::uint32_t>(tile_bits)
                                 >= n_qubits))
        return TK_EINVAL;
    const std::size_t n = t->n, n_st = s->n_stages;
    if (n_st > 0u && s->dt == nullptr)
        return TK_EINVAL;
    if (n > 0u && (t->x == nullptr || t->z == nullptr
                   || t->c_re == nullptr || t->c_im == nullptr))
        return TK_EINVAL;
    for (std::size_t g = 0; g < n; g++)
        if (((t->x[g] | t->z[g]) >> n_qubits) != 0u)
            return TK_EINVAL;

    int T = n_threads == 0 ? auto_threads(n_qubits) : n_threads;
    if (tile_bits == 0)
        tile_bits = T > 1 && n_qubits >= 5u
                        ? static_cast<int>(n_qubits) - 2 : -1;
    const std::size_t dim = std::size_t{1} << n_qubits;

    dvec re(dim), im(dim), consts(n_st * n * 4u);
    std::vector<unsigned char> local;
    std::vector<std::size_t> run_end, run_begin;
    if (tile_bits > 0) {
        local.resize(n);
        run_end.resize(n);
        run_begin.resize(n);
        for (std::size_t g = 0; g < n; g++)
            local[g] = (t->x[g] >> tile_bits) == 0u;
        for (std::size_t g = n; g-- > 0u;)
            run_end[g] = local[g] && g + 1u < n && local[g + 1u]
                             ? run_end[g + 1u] : g + 1u;
        for (std::size_t g = 0; g < n; g++)
            run_begin[g] = local[g] && g > 0u && local[g - 1u]
                               ? run_begin[g - 1u] : g;
    }
    for (std::size_t st = 0; st < n_st; st++)
        for (std::size_t g = 0; g < n; g++)
            term_constants(0.5 * s->dt[st], t->c_re[g], t->c_im[g],
                           popcount64(t->x[g] & t->z[g]) & 3,
                           consts.data() + 4u * (st * n + g));
    for (std::size_t j = 0; j < dim; j++) {
        re[j] = psi[2u * j];
        im[j] = psi[2u * j + 1u];
    }

    Drive d{re.data(), im.data(), dim, n, n_st, s->n_steps, t->x, t->z,
            consts.data(), force_scalar(),
            tile_bits > 0 ? static_cast<unsigned>(tile_bits) : 0u,
            local.data(), run_end.data(), run_begin.data(), nullptr};
    SpinBarrier bar;
    d.bar = &bar;
#if defined(_OPENMP)
    if (T > 1) {
#pragma omp parallel num_threads(T)
        d.run(omp_get_thread_num(), omp_get_num_threads());
    } else
#endif
        d.run(0, 1);

    for (std::size_t j = 0; j < dim; j++) {
        psi[2u * j] = re[j];
        psi[2u * j + 1u] = im[j];
    }
    return TK_OK;
}

}  // namespace

// Grouped form: the XOR-offset slices of H.
struct tk_groups {
    std::uint32_t n_qubits = 0;
    std::vector<std::uint64_t> diag_i;
    std::vector<double> diag_re, diag_im;
    std::vector<std::size_t> frag_off;  // n_frag + 1 offsets into blocks
    std::vector<std::uint64_t> blk_i, blk_j;
    std::vector<double> a_re, a_im, b_re, b_im;
};

namespace {

struct Entry {
    std::uint64_t x, i;
    bool lower;  // entry is H[i ^ x, i] (b)
    double re, im;
};

int groups_build(std::uint32_t n_qubits, const std::uint64_t *row,
                 const std::uint64_t *col, const double *v_re,
                 const double *v_im, std::size_t nnz, tk_groups **out)
{
    if (out == nullptr)
        return TK_EINVAL;
    *out = nullptr;
    if (n_qubits < 1u || n_qubits > TK_MAX_QUBITS)
        return TK_EQUBITS;
    if (nnz > 0u && (row == nullptr || col == nullptr || v_re == nullptr
                     || v_im == nullptr))
        return TK_EINVAL;
    const std::size_t dim = std::size_t{1} << n_qubits;
    std::vector<Entry> ent;
    ent.reserve(nnz);
    for (std::size_t e = 0; e < nnz; e++) {
        std::uint64_t r = row[e], c = col[e];
        if (r >= dim || c >= dim)
            return TK_EINVAL;
        if (v_re[e] == 0.0 && v_im[e] == 0.0)
            continue;
        ent.push_back({r ^ c, r < c ? r : c, r > c, v_re[e], v_im[e]});
    }
    std::sort(ent.begin(), ent.end(), [](const Entry &a, const Entry &b) {
        return a.x != b.x ? a.x < b.x : a.i < b.i;
    });

    auto g = std::make_unique<tk_groups>();
    g->n_qubits = n_qubits;
    for (std::size_t e = 0; e < ent.size(); e++) {
        bool same_x = e > 0u && ent[e].x == ent[e - 1u].x;
        bool fresh = !(same_x && ent[e].i == ent[e - 1u].i);
        if (ent[e].x == 0u) {
            if (fresh) {
                g->diag_i.push_back(ent[e].i);
                g->diag_re.push_back(0.0);
                g->diag_im.push_back(0.0);
            }
            g->diag_re.back() += ent[e].re;
            g->diag_im.back() += ent[e].im;
            continue;
        }
        if (!same_x)
            g->frag_off.push_back(g->blk_i.size());
        if (fresh) {
            g->blk_i.push_back(ent[e].i);
            g->blk_j.push_back(ent[e].i ^ ent[e].x);
            g->a_re.push_back(0.0);
            g->a_im.push_back(0.0);
            g->b_re.push_back(0.0);
            g->b_im.push_back(0.0);
        }
        if (ent[e].lower) {
            g->b_re.back() += ent[e].re;
            g->b_im.back() += ent[e].im;
        } else {
            g->a_re.back() += ent[e].re;
            g->a_im.back() += ent[e].im;
        }
    }
    g->frag_off.push_back(g->blk_i.size());
    *out = g.release();
    return TK_OK;
}

// Complex arithmetic on (re, im) pairs from real libm functions, the
// same operations in the same order as the C backend.
struct cplx {
    double re, im;
};

cplx cmul(cplx a, cplx b)
{
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

cplx csqrt_h(cplx a)
{
    double r = std::hypot(a.re, a.im);
    cplx w;
    if (a.re >= 0.0) {
        w.re = std::sqrt((r + a.re) * 0.5);
        w.im = a.im / (2.0 * w.re);
    } else {
        w.im = std::sqrt((r - a.re) * 0.5);
        if (a.im < 0.0)
            w.im = -w.im;
        w.re = a.im / (2.0 * w.im);
    }
    return w;
}

void block_constants(double theta, double a_re, double a_im,
                     double b_re, double b_im, double out[6])
{
    cplx a{a_re, a_im}, b{b_re, b_im}, c, sv;
    double t2 = theta * theta;
    cplx ab = cmul(a, b);
    cplx u{t2 * ab.re, t2 * ab.im};
    if (std::hypot(u.re, u.im) < 1e-6) {
        cplx u2 = cmul(u, u);
        c.re = (1.0 - u.re / 2.0) + u2.re / 24.0;
        c.im = (0.0 - u.im / 2.0) + u2.im / 24.0;
        sv.re = theta * ((1.0 - u.re / 6.0) + u2.re / 120.0);
        sv.im = theta * ((0.0 - u.im / 6.0) + u2.im / 120.0);
    } else {
        cplx w = csqrt_h(ab);
        double xr = theta * w.re, xi = theta * w.im;
        double d = w.re * w.re + w.im * w.im;
        c.re = std::cos(xr) * std::cosh(xi);
        c.im = -(std::sin(xr) * std::sinh(xi));
        cplx sn{std::sin(xr) * std::cosh(xi), std::cos(xr) * std::sinh(xi)};
        sv.re = (sn.re * w.re + sn.im * w.im) / d;
        sv.im = (sn.im * w.re - sn.re * w.im) / d;
    }
    cplx sa = cmul(sv, a), sb = cmul(sv, b);
    out[0] = c.re;
    out[1] = c.im;
    out[2] = sa.im;  // -i (x + iy) = y - ix
    out[3] = -sa.re;
    out[4] = sb.im;
    out[5] = -sb.re;
}

void apply_diag(double *re, double *im, const tk_groups &g,
                const double *dm)
{
    for (std::size_t d = 0; d < g.diag_i.size(); d++) {
        std::size_t i = static_cast<std::size_t>(g.diag_i[d]);
        double mr = dm[2u * d], mi = dm[2u * d + 1u];
        double pr = re[i], pi = im[i];
        re[i] = mr * pr - mi * pi;
        im[i] = mr * pi + mi * pr;
    }
}

void apply_frag(double *re, double *im, const tk_groups &g,
                std::size_t f, const double *bm)
{
    for (std::size_t b = g.frag_off[f]; b < g.frag_off[f + 1u]; b++) {
        const double *k = bm + 6u * b;
        std::size_t i = static_cast<std::size_t>(g.blk_i[b]);
        std::size_t j = static_cast<std::size_t>(g.blk_j[b]);
        double pr = re[i], pi = im[i], qr = re[j], qi = im[j];
        re[i] = k[0] * pr - k[1] * pi + k[2] * qr - k[3] * qi;
        im[i] = k[0] * pi + k[1] * pr + k[2] * qi + k[3] * qr;
        re[j] = k[4] * pr - k[5] * pi + k[0] * qr - k[1] * qi;
        im[j] = k[4] * pi + k[5] * pr + k[0] * qi + k[1] * qr;
    }
}

int evolve_grouped(double *psi, std::uint32_t n_qubits,
                   const tk_groups *g, const tk_schedule *s,
                   int n_threads)
{
    if (psi == nullptr || g == nullptr || s == nullptr)
        return TK_EINVAL;
    if (n_qubits != g->n_qubits || n_threads < 0 || n_threads > 1)
        return TK_EINVAL;
    const std::size_t n_st = s->n_stages;
    if (n_st > 0u && s->dt == nullptr)
        return TK_EINVAL;
    const std::size_t dim = std::size_t{1} << n_qubits;
    const std::size_t n_diag = g->diag_i.size(), n_blk = g->blk_i.size();
    const std::size_t n_frag = g->frag_off.size() - 1u;
    dvec re(dim), im(dim), dm(n_st * n_diag * 2u), bm(n_st * n_blk * 6u);
    for (std::size_t st = 0; st < n_st; st++) {
        double theta = 0.5 * s->dt[st];
        for (std::size_t d = 0; d < n_diag; d++) {
            double grow = std::exp(theta * g->diag_im[d]);
            double ang = theta * g->diag_re[d];
            dm[2u * (st * n_diag + d)] = grow * std::cos(ang);
            dm[2u * (st * n_diag + d) + 1u] = -grow * std::sin(ang);
        }
        for (std::size_t b = 0; b < n_blk; b++)
            block_constants(theta, g->a_re[b], g->a_im[b], g->b_re[b],
                            g->b_im[b], bm.data() + 6u * (st * n_blk + b));
    }
    for (std::size_t j = 0; j < dim; j++) {
        re[j] = psi[2u * j];
        im[j] = psi[2u * j + 1u];
    }
    for (std::size_t step = 0; step < s->n_steps; step++)
        for (std::size_t st = 0; st < n_st; st++) {
            const double *ds = dm.data() + 2u * st * n_diag;
            const double *bs = bm.data() + 6u * st * n_blk;
            apply_diag(re.data(), im.data(), *g, ds);
            for (std::size_t f = 0; f < n_frag; f++)
                apply_frag(re.data(), im.data(), *g, f, bs);
            for (std::size_t f = n_frag; f-- > 0u;)
                apply_frag(re.data(), im.data(), *g, f, bs);
            apply_diag(re.data(), im.data(), *g, ds);
        }
    for (std::size_t j = 0; j < dim; j++) {
        psi[2u * j] = re[j];
        psi[2u * j + 1u] = im[j];
    }
    return TK_OK;
}

}  // namespace

extern "C" {

int tk_evolve_terms(double *psi, uint32_t n_qubits, const tk_terms *terms,
                    const tk_schedule *s, int strategy, int tile_bits,
                    int n_threads)
{
    try {
        return evolve_terms(psi, n_qubits, terms, s, strategy, tile_bits,
                            n_threads);
    } catch (const std::bad_alloc &) {
        return TK_ENOMEM;
    } catch (const std::length_error &) {
        return TK_ENOMEM;
    }
}

int tk_groups_build(uint32_t n_qubits, const uint64_t *row,
                    const uint64_t *col, const double *v_re,
                    const double *v_im, size_t nnz, tk_groups **out)
{
    try {
        return groups_build(n_qubits, row, col, v_re, v_im, nnz, out);
    } catch (const std::bad_alloc &) {
        return TK_ENOMEM;
    } catch (const std::length_error &) {
        return TK_ENOMEM;
    }
}

int tk_evolve_grouped(double *psi, uint32_t n_qubits, const tk_groups *g,
                      const tk_schedule *s, int n_threads)
{
    try {
        return evolve_grouped(psi, n_qubits, g, s, n_threads);
    } catch (const std::bad_alloc &) {
        return TK_ENOMEM;
    } catch (const std::length_error &) {
        return TK_ENOMEM;
    }
}

void tk_groups_free(tk_groups *g)
{
    delete g;
}

const char *tk_backend_name(void)
{
    return "cpp";
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

}  // extern "C"
