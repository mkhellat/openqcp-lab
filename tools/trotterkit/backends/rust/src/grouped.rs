// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! Grouped mode: the XOR-offset slices of H as 2x2 blocks, each
//! exponentiated exactly, as in the C backend (same operations, same
//! order, constants from real libm functions only).

pub struct Groups {
    pub n_qubits: u32,
    pub diag_i: Vec<u64>,
    pub diag_re: Vec<f64>,
    pub diag_im: Vec<f64>,
    pub frag_off: Vec<usize>, // n_frag + 1 offsets into blocks
    pub blk_i: Vec<u64>,
    pub blk_j: Vec<u64>,
    pub a_re: Vec<f64>,
    pub a_im: Vec<f64>,
    pub b_re: Vec<f64>,
    pub b_im: Vec<f64>,
}

struct Entry {
    x: u64,
    i: u64,
    lower: bool, // entry is H[i ^ x, i] (b)
    re: f64,
    im: f64,
}

/// Build from COO entries already checked to lie inside the matrix.
pub fn build(n_qubits: u32, entries: impl Iterator<Item = (u64, u64, f64,
             f64)>, nnz: usize) -> Option<Groups> {
    let mut ent: Vec<Entry> = Vec::new();
    ent.try_reserve_exact(nnz).ok()?;
    for (r, c, re, im) in entries {
        if re == 0.0 && im == 0.0 {
            continue;
        }
        ent.push(Entry { x: r ^ c, i: r.min(c), lower: r > c, re, im });
    }
    ent.sort_unstable_by(|a, b| (a.x, a.i).cmp(&(b.x, b.i)));

    let mut g = Groups {
        n_qubits,
        diag_i: Vec::new(),
        diag_re: Vec::new(),
        diag_im: Vec::new(),
        frag_off: Vec::new(),
        blk_i: Vec::new(),
        blk_j: Vec::new(),
        a_re: Vec::new(),
        a_im: Vec::new(),
        b_re: Vec::new(),
        b_im: Vec::new(),
    };
    for e in 0..ent.len() {
        let same_x = e > 0 && ent[e].x == ent[e - 1].x;
        let fresh = !(same_x && ent[e].i == ent[e - 1].i);
        let en = &ent[e];
        if en.x == 0 {
            if fresh {
                g.diag_i.push(en.i);
                g.diag_re.push(0.0);
                g.diag_im.push(0.0);
            }
            *g.diag_re.last_mut()? += en.re;
            *g.diag_im.last_mut()? += en.im;
            continue;
        }
        if !same_x {
            g.frag_off.push(g.blk_i.len());
        }
        if fresh {
            g.blk_i.push(en.i);
            g.blk_j.push(en.i ^ en.x);
            g.a_re.push(0.0);
            g.a_im.push(0.0);
            g.b_re.push(0.0);
            g.b_im.push(0.0);
        }
        if en.lower {
            *g.b_re.last_mut()? += en.re;
            *g.b_im.last_mut()? += en.im;
        } else {
            *g.a_re.last_mut()? += en.re;
            *g.a_im.last_mut()? += en.im;
        }
    }
    g.frag_off.push(g.blk_i.len());
    Some(g)
}

#[derive(Clone, Copy)]
struct Cplx {
    re: f64,
    im: f64,
}

fn cmul(a: Cplx, b: Cplx) -> Cplx {
    Cplx { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }
}

/// Principal square root, the branch chosen to avoid cancellation.
fn csqrt_h(a: Cplx) -> Cplx {
    let r = a.re.hypot(a.im);
    if a.re >= 0.0 {
        let wr = ((r + a.re) * 0.5).sqrt();
        Cplx { re: wr, im: a.im / (2.0 * wr) }
    } else {
        let mut wi = ((r - a.re) * 0.5).sqrt();
        if a.im < 0.0 {
            wi = -wi;
        }
        Cplx { re: a.im / (2.0 * wi), im: wi }
    }
}

/// (c, A, B) for one block: exp(-i theta [[0,a],[b,0]]) =
/// [[c, A], [B, c]], w^2 = ab, series as ab -> 0.
pub fn block_constants(theta: f64, a_re: f64, a_im: f64, b_re: f64,
                       b_im: f64) -> [f64; 6] {
    let a = Cplx { re: a_re, im: a_im };
    let b = Cplx { re: b_re, im: b_im };
    let t2 = theta * theta;
    let ab = cmul(a, b);
    let u = Cplx { re: t2 * ab.re, im: t2 * ab.im };
    let (c, sv);
    if u.re.hypot(u.im) < 1e-6 {
        let u2 = cmul(u, u);
        c = Cplx { re: (1.0 - u.re / 2.0) + u2.re / 24.0,
                   im: (0.0 - u.im / 2.0) + u2.im / 24.0 };
        sv = Cplx { re: theta * ((1.0 - u.re / 6.0) + u2.re / 120.0),
                    im: theta * ((0.0 - u.im / 6.0) + u2.im / 120.0) };
    } else {
        let w = csqrt_h(ab);
        let (xr, xi) = (theta * w.re, theta * w.im);
        let d = w.re * w.re + w.im * w.im;
        c = Cplx { re: xr.cos() * xi.cosh(), im: -(xr.sin() * xi.sinh()) };
        let sn = Cplx { re: xr.sin() * xi.cosh(), im: xr.cos() * xi.sinh() };
        sv = Cplx { re: (sn.re * w.re + sn.im * w.im) / d,
                    im: (sn.im * w.re - sn.re * w.im) / d };
    }
    let (sa, sb) = (cmul(sv, a), cmul(sv, b));
    // -i (x + iy) = y - ix
    [c.re, c.im, sa.im, -sa.re, sb.im, -sb.re]
}

fn apply_diag(re: &mut [f64], im: &mut [f64], g: &Groups, dm: &[f64]) {
    for (d, &i) in g.diag_i.iter().enumerate() {
        let i = i as usize;
        let (mr, mi) = (dm[2 * d], dm[2 * d + 1]);
        let (pr, pi) = (re[i], im[i]);
        re[i] = mr * pr - mi * pi;
        im[i] = mr * pi + mi * pr;
    }
}

fn apply_frag(re: &mut [f64], im: &mut [f64], g: &Groups, f: usize,
              bm: &[f64]) {
    for b in g.frag_off[f]..g.frag_off[f + 1] {
        let k = &bm[6 * b..6 * b + 6];
        let (i, j) = (g.blk_i[b] as usize, g.blk_j[b] as usize);
        let (pr, pi, qr, qi) = (re[i], im[i], re[j], im[j]);
        re[i] = k[0] * pr - k[1] * pi + k[2] * qr - k[3] * qi;
        im[i] = k[0] * pi + k[1] * pr + k[2] * qi + k[3] * qr;
        re[j] = k[4] * pr - k[5] * pi + k[0] * qr - k[1] * qi;
        im[j] = k[4] * pi + k[5] * pr + k[0] * qi + k[1] * qr;
    }
}

/// Evolve (re, im) in place; None on allocation failure.
pub fn evolve(re: &mut [f64], im: &mut [f64], g: &Groups, dt: &[f64],
              n_steps: usize) -> Option<()> {
    let (n_diag, n_blk) = (g.diag_i.len(), g.blk_i.len());
    let n_frag = g.frag_off.len() - 1;
    let mut dm: Vec<f64> = Vec::new();
    let mut bm: Vec<f64> = Vec::new();
    dm.try_reserve_exact(dt.len().checked_mul(2 * n_diag)?).ok()?;
    bm.try_reserve_exact(dt.len().checked_mul(6 * n_blk)?).ok()?;
    for &step in dt {
        let theta = 0.5 * step;
        for d in 0..n_diag {
            let grow = (theta * g.diag_im[d]).exp();
            let ang = theta * g.diag_re[d];
            dm.push(grow * ang.cos());
            dm.push(-grow * ang.sin());
        }
        for b in 0..n_blk {
            bm.extend_from_slice(&block_constants(theta, g.a_re[b],
                                                  g.a_im[b], g.b_re[b],
                                                  g.b_im[b]));
        }
    }
    for _ in 0..n_steps {
        for st in 0..dt.len() {
            let ds = &dm[2 * st * n_diag..2 * (st + 1) * n_diag];
            let bs = &bm[6 * st * n_blk..6 * (st + 1) * n_blk];
            apply_diag(re, im, g, ds);
            for f in 0..n_frag {
                apply_frag(re, im, g, f, bs);
            }
            for f in (0..n_frag).rev() {
                apply_frag(re, im, g, f, bs);
            }
            apply_diag(re, im, g, ds);
        }
    }
    Some(())
}
