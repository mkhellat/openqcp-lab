// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! Per-term evolution: the C backend's algorithm and arithmetic order
//! (backends/c/trotterkit_c.c), so results agree to the bit. The lane
//! permutation of the grouped path is a const generic. The state is
//! reached through raw pointers because the thread team writes
//! disjoint index ranges of one buffer, which the algorithm (not the
//! borrow checker) guarantees.

#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;

#[inline]
pub fn parity(v: u64) -> u64 {
    (v.count_ones() & 1) as u64
}

#[inline]
pub fn top_bit(v: u64) -> u32 {
    63 - v.leading_zeros()
}

/// exp(-i theta c P) = a I - i b P: (a_re, a_im, k_re, k_im) with
/// k = -i b phase(x, z).
pub fn term_constants(theta: f64, c_re: f64, c_im: f64, phase_pow: u32)
    -> [f64; 4]
{
    const PH_RE: [f64; 4] = [1.0, 0.0, -1.0, 0.0];
    const PH_IM: [f64; 4] = [0.0, 1.0, 0.0, -1.0];
    let p = phase_pow as usize;
    let (u, v) = (theta * c_re, theta * c_im);
    let b_re = u.sin() * v.cosh();
    let b_im = u.cos() * v.sinh();
    let p_re = b_re * PH_RE[p] - b_im * PH_IM[p];
    let p_im = b_re * PH_IM[p] + b_im * PH_RE[p];
    [u.cos() * v.cosh(), -u.sin() * v.sinh(), p_im, -p_re]
}

#[derive(Clone, Copy)]
pub enum Path {
    Diag1,
    Pair1,
    Diag4,
    Pair4,
}

pub fn path_of(x: u64, dim: usize, scalar: bool) -> Path {
    if scalar || dim < 8 {
        return if x == 0 { Path::Diag1 } else { Path::Pair1 };
    }
    if x == 0 {
        Path::Diag4
    } else if x < 4 {
        Path::Pair1
    } else {
        Path::Pair4
    }
}

pub fn units_of(path: Path, dim: usize) -> usize {
    match path {
        Path::Diag1 => dim,
        Path::Pair1 => dim / 2,
        Path::Diag4 => dim / 4,
        Path::Pair4 => dim / 8,
    }
}

#[inline]
fn sign(v: u64) -> f64 {
    1.0 - 2.0 * parity(v) as f64
}

unsafe fn apply_diag1(re: *mut f64, im: *mut f64, z: u64, k: &[f64; 4],
                      lo: usize, hi: usize) {
    let (ar, ai, kr, ki) = (k[0], k[1], k[2], k[3]);
    for j in lo..hi {
        let sg = sign(j as u64 & z);
        let (mr, mi) = (ar + sg * kr, ai + sg * ki);
        let (pr, pi) = (*re.add(j), *im.add(j));
        *re.add(j) = mr * pr - mi * pi;
        *im.add(j) = mr * pi + mi * pr;
    }
}

unsafe fn apply_pair1(re: *mut f64, im: *mut f64, x: u64, z: u64,
                      k: &[f64; 4], lo: usize, hi: usize) {
    let (ar, ai, kr, ki) = (k[0], k[1], k[2], k[3]);
    let h = top_bit(x);
    let low = (1usize << h) - 1;
    for kk in lo..hi {
        let j1 = ((kk >> h) << (h + 1)) | (kk & low);
        let j2 = j1 ^ x as usize;
        let s1 = sign(j2 as u64 & z);
        let s2 = sign(j1 as u64 & z);
        let (pr, pi) = (*re.add(j1), *im.add(j1));
        let (qr, qi) = (*re.add(j2), *im.add(j2));
        *re.add(j1) = ar * pr - ai * pi + s1 * (kr * qr - ki * qi);
        *im.add(j1) = ar * pi + ai * pr + s1 * (kr * qi + ki * qr);
        *re.add(j2) = ar * qr - ai * qi + s2 * (kr * pr - ki * pi);
        *im.add(j2) = ar * qi + ai * qr + s2 * (kr * pi + ki * pr);
    }
}

fn lane_pattern(x: u64, z: u64) -> [f64; 4] {
    let mut pat = [0.0; 4];
    for (q, p) in pat.iter_mut().enumerate() {
        *p = sign((q as u64 ^ (x & 3)) & z & 3);
    }
    pat
}

unsafe fn apply_diag4(re: *mut f64, im: *mut f64, z: u64, k: &[f64; 4],
                      lo: usize, hi: usize) {
    let (ar, ai, kr, ki) = (k[0], k[1], k[2], k[3]);
    let pat = lane_pattern(0, z);
    for i in lo..hi {
        let g = 4 * i;
        let sb = sign(g as u64 & z);
        for (q, &pq) in pat.iter().enumerate() {
            let sg = sb * pq;
            let (mr, mi) = (ar + sg * kr, ai + sg * ki);
            let (pr, pi) = (*re.add(g + q), *im.add(g + q));
            *re.add(g + q) = mr * pr - mi * pi;
            *im.add(g + q) = mr * pi + mi * pr;
        }
    }
}

/// Base of the i-th group of 4 pairs: 4i with a zero inserted at bit
/// h (h >= 2).
fn group_base(i: usize, h: u32) -> usize {
    let f = 4 * i;
    ((f >> h) << (h + 1)) | (f & ((1usize << h) - 1))
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn permute_lanes<const P: u32>(v: __m256d) -> __m256d {
    match P {
        1 => _mm256_permute_pd::<0x5>(v),
        2 => _mm256_permute4x64_pd::<0x4E>(v),
        3 => _mm256_permute4x64_pd::<0x1B>(v),
        _ => v,
    }
}

/// pat with every lane's sign flipped when par (0 or 1) is 1.
#[cfg(target_feature = "avx2")]
#[inline(always)]
unsafe fn flip_if(pat: __m256d, par: u64) -> __m256d {
    _mm256_xor_pd(pat, _mm256_castsi256_pd(
        _mm256_set1_epi64x((par << 63) as i64)))
}

#[cfg(target_feature = "avx2")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pair4_group<const P: u32>(re: *mut f64, im: *mut f64, g: usize,
                                    g2: usize, s1: __m256d, s2: __m256d,
                                    ar: __m256d, ai: __m256d, kr: __m256d,
                                    ki: __m256d) {
    let pr = _mm256_load_pd(re.add(g));
    let pi = _mm256_load_pd(im.add(g));
    let qr = _mm256_load_pd(re.add(g2));
    let qi = _mm256_load_pd(im.add(g2));
    let (qrp, qip) = (permute_lanes::<P>(qr), permute_lanes::<P>(qi));
    let (prp, pip) = (permute_lanes::<P>(pr), permute_lanes::<P>(pi));
    let n1r = _mm256_add_pd(
        _mm256_sub_pd(_mm256_mul_pd(ar, pr), _mm256_mul_pd(ai, pi)),
        _mm256_mul_pd(s1, _mm256_sub_pd(_mm256_mul_pd(kr, qrp),
                                        _mm256_mul_pd(ki, qip))));
    let n1i = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ar, pi), _mm256_mul_pd(ai, pr)),
        _mm256_mul_pd(s1, _mm256_add_pd(_mm256_mul_pd(kr, qip),
                                        _mm256_mul_pd(ki, qrp))));
    let n2r = _mm256_add_pd(
        _mm256_sub_pd(_mm256_mul_pd(ar, qr), _mm256_mul_pd(ai, qi)),
        _mm256_mul_pd(s2, _mm256_sub_pd(_mm256_mul_pd(kr, prp),
                                        _mm256_mul_pd(ki, pip))));
    let n2i = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ar, qi), _mm256_mul_pd(ai, qr)),
        _mm256_mul_pd(s2, _mm256_add_pd(_mm256_mul_pd(kr, pip),
                                        _mm256_mul_pd(ki, prp))));
    _mm256_store_pd(re.add(g), n1r);
    _mm256_store_pd(im.add(g), n1i);
    _mm256_store_pd(re.add(g2), n2r);
    _mm256_store_pd(im.add(g2), n2i);
}

/// Groups [lo, hi), block by block with a plain stride-4 inner loop.
#[cfg(target_feature = "avx2")]
unsafe fn apply_pair4<const P: u32>(re: *mut f64, im: *mut f64, x: u64,
                                    z: u64, k: &[f64; 4], lo: usize,
                                    hi: usize) {
    let (ar, ai) = (_mm256_set1_pd(k[0]), _mm256_set1_pd(k[1]));
    let (kr, ki) = (_mm256_set1_pd(k[2]), _mm256_set1_pd(k[3]));
    let h = top_bit(x);
    let hi_mask = (x & !3) as usize;
    let span = 1usize << h;
    let mut left = hi - lo;
    let mut g = group_base(lo, h);
    let mut end = (g & !(2 * span - 1)) + span;
    let pat = lane_pattern(x, z);
    let pos = _mm256_loadu_pd(pat.as_ptr());
    while left > 0 {
        let cnt = ((end - g) / 4).min(left);
        let stop = g + 4 * cnt;
        left -= cnt;
        while g < stop {
            let g2 = g ^ hi_mask;
            pair4_group::<P>(re, im, g, g2,
                             flip_if(pos, parity(g2 as u64 & z)),
                             flip_if(pos, parity(g as u64 & z)),
                             ar, ai, kr, ki);
            g += 4;
        }
        g += span; // skip the partner half
        end += 2 * span;
    }
}

#[cfg(not(target_feature = "avx2"))]
unsafe fn apply_pair4<const P: u32>(re: *mut f64, im: *mut f64, x: u64,
                                    z: u64, k: &[f64; 4], lo: usize,
                                    hi: usize) {
    let (ar, ai, kr, ki) = (k[0], k[1], k[2], k[3]);
    let h = top_bit(x);
    let hi_mask = (x & !3) as usize;
    let span = 1usize << h;
    let mut left = hi - lo;
    let mut g = group_base(lo, h);
    let mut end = (g & !(2 * span - 1)) + span;
    let pat = lane_pattern(x, z);
    while left > 0 {
        let cnt = ((end - g) / 4).min(left);
        let stop = g + 4 * cnt;
        left -= cnt;
        while g < stop {
            let g2 = g ^ hi_mask;
            let (sg1, sg2) = (sign(g2 as u64 & z), sign(g as u64 & z));
            let mut v = [[0.0f64; 4]; 4]; // pr, pi, qr, qi
            for q in 0..4 {
                v[0][q] = *re.add(g + q);
                v[1][q] = *im.add(g + q);
                v[2][q] = *re.add(g2 + q);
                v[3][q] = *im.add(g2 + q);
            }
            let [pr, pi, qr, qi] = v;
            for q in 0..4 {
                let (s1, s2) = (sg1 * pat[q], sg2 * pat[q]);
                let r = q ^ P as usize;
                *re.add(g + q) = ar * pr[q] - ai * pi[q]
                    + s1 * (kr * qr[r] - ki * qi[r]);
                *im.add(g + q) = ar * pi[q] + ai * pr[q]
                    + s1 * (kr * qi[r] + ki * qr[r]);
                *re.add(g2 + q) = ar * qr[q] - ai * qi[q]
                    + s2 * (kr * pr[r] - ki * pi[r]);
                *im.add(g2 + q) = ar * qi[q] + ai * qr[q]
                    + s2 * (kr * pi[r] + ki * pr[r]);
            }
            g += 4;
        }
        g += span;
        end += 2 * span;
    }
}

/// Apply units [lo, hi) of one term along path.
///
/// # Safety
/// re and im must address the whole range the path touches.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_range(path: Path, re: *mut f64, im: *mut f64, x: u64,
                          z: u64, k: &[f64; 4], lo: usize, hi: usize) {
    match path {
        Path::Diag1 => apply_diag1(re, im, z, k, lo, hi),
        Path::Pair1 => apply_pair1(re, im, x, z, k, lo, hi),
        Path::Diag4 => apply_diag4(re, im, z, k, lo, hi),
        Path::Pair4 => match x & 3 {
            0 => apply_pair4::<0>(re, im, x, z, k, lo, hi),
            1 => apply_pair4::<1>(re, im, x, z, k, lo, hi),
            2 => apply_pair4::<2>(re, im, x, z, k, lo, hi),
            _ => apply_pair4::<3>(re, im, x, z, k, lo, hi),
        },
    }
}
