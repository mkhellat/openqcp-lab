// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! trotterkit Rust backend: the C ABI of include/trotterkit.h with the
//! C backend's design (groups of 4 amplitudes with AVX2, branch-free
//! signs, tiles, a thread team with one barrier per term or per
//! tile-local run, the same automatic choices). Results equal the C
//! backend's to the bit. No error unwinds across the ABI: every
//! failure is a returned code, and allocation sized by the input is
//! fallible (TK_ENOMEM).

pub mod barrier;
pub mod grouped;
pub mod kernel;
pub mod sha256;
pub mod workload;

use std::alloc::{alloc, dealloc, Layout};
use std::ffi::c_char;

use barrier::SpinBarrier;
use kernel::{apply_range, parity, path_of, term_constants, units_of};

pub const TK_OK: i32 = 0;
pub const TK_EINVAL: i32 = 1;
pub const TK_ENOMEM: i32 = 2;
pub const TK_EQUBITS: i32 = 3;
pub const TK_AUTO: i32 = 0;
pub const TK_ONTHEFLY: i32 = 1;
pub const TK_TABLES: i32 = 2;
const TK_MAX_QUBITS: u32 = 32;

// Automatic choices, the C backend's (measured on an i7-8550U):
// serial below 10 qubits, 2 threads at 10, up to 4 from 11.
const AUTO_MIN_QUBITS: u32 = 10;
const AUTO_MAX_THREADS: usize = 4;

#[repr(C)]
pub struct TkTerms {
    pub x: *const u64,
    pub z: *const u64,
    pub c_re: *const f64,
    pub c_im: *const f64,
    pub n: usize,
}

#[repr(C)]
pub struct TkSchedule {
    pub dt: *const f64,
    pub n_stages: usize,
    pub n_steps: usize,
}

/// 64-byte aligned f64 buffer (the AVX2 path uses aligned loads).
struct AlignedBuf {
    ptr: *mut f64,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize) -> Option<Self> {
        let bytes = len.checked_mul(8)?.max(64);
        let layout = Layout::from_size_align(bytes, 64).ok()?;
        // SAFETY: layout has a non-zero size.
        let ptr = unsafe { alloc(layout) } as *mut f64;
        if ptr.is_null() {
            None
        } else {
            Some(AlignedBuf { ptr, layout })
        }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in new with this layout.
        unsafe { dealloc(self.ptr as *mut u8, self.layout) }
    }
}

fn try_vec<T: Clone>(len: usize, v: T) -> Option<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).ok()?;
    out.resize(len, v);
    Some(out)
}

/// Everything one evolution needs, shared by the thread team.
struct Drive<'a> {
    re: *mut f64,
    im: *mut f64,
    dim: usize,
    x: &'a [u64],
    z: &'a [u64],
    consts: &'a [f64],
    n_st: usize,
    n_steps: usize,
    scalar: bool,
    tile_bits: u32, // 0: no tiles
    local: &'a [bool],
    run_end: &'a [usize],
    run_begin: &'a [usize],
}

// SAFETY: threads write disjoint index ranges of re and im between
// barriers (per term: a split of the term's units; per tile-local
// run: whole tiles), so sharing the pointers is race-free.
unsafe impl Sync for Drive<'_> {}

impl Drive<'_> {
    fn term_share(&self, g: usize, k: &[f64; 4], tid: usize, t: usize) {
        let path = path_of(self.x[g], self.dim, self.scalar);
        let u = units_of(path, self.dim);
        let (lo, hi) = if t > 1 {
            (u * tid / t, u * (tid + 1) / t)
        } else {
            (0, u)
        };
        // SAFETY: [lo, hi) is this thread's share of term g.
        unsafe {
            apply_range(path, self.re, self.im, self.x[g], self.z[g], k,
                        lo, hi)
        }
    }

    /// Tile-local run [b, e): in a tile at off every sign is the
    /// whole-state sign times (-1)^popcount(off & z), i.e. the term
    /// with its off-diagonal constant negated (exact).
    fn run_tiles(&self, b: usize, e: usize, cs: &[f64], reverse: bool,
                 tid: usize, t: usize) {
        let len = 1usize << self.tile_bits;
        let n_tiles = self.dim / len;
        for tile in n_tiles * tid / t..n_tiles * (tid + 1) / t {
            let off = tile * len;
            for i in 0..e - b {
                let g = if reverse { e - 1 - i } else { b + i };
                let mut k = [cs[4 * g], cs[4 * g + 1], cs[4 * g + 2],
                             cs[4 * g + 3]];
                if parity(off as u64 & self.z[g]) == 1 {
                    k[2] = -k[2];
                    k[3] = -k[3];
                }
                let path = path_of(self.x[g], len, self.scalar);
                // SAFETY: the tile [off, off + len) is this thread's.
                unsafe {
                    apply_range(path, self.re.add(off), self.im.add(off),
                                self.x[g], self.z[g], &k, 0,
                                units_of(path, len));
                }
            }
        }
    }

    fn run(&self, tid: usize, t: usize, bar: Option<&SpinBarrier>) {
        let n = self.x.len();
        let wait = || {
            if let Some(b) = bar {
                b.wait();
            }
        };
        let k_of = |cs: &[f64], g: usize| {
            [cs[4 * g], cs[4 * g + 1], cs[4 * g + 2], cs[4 * g + 3]]
        };
        for _ in 0..self.n_steps {
            for st in 0..self.n_st {
                let cs = &self.consts[4 * st * n..4 * (st + 1) * n];
                let mut g = 0;
                while g < n {
                    if self.tile_bits != 0 && self.local[g] {
                        self.run_tiles(g, self.run_end[g], cs, false, tid,
                                       t);
                        g = self.run_end[g];
                    } else {
                        self.term_share(g, &k_of(cs, g), tid, t);
                        g += 1;
                    }
                    wait();
                }
                let mut g = n;
                while g > 0 {
                    let last = g - 1;
                    if self.tile_bits != 0 && self.local[last] {
                        self.run_tiles(self.run_begin[last], g, cs, true,
                                       tid, t);
                        g = self.run_begin[last];
                    } else {
                        self.term_share(last, &k_of(cs, last), tid, t);
                        g = last;
                    }
                    wait();
                }
            }
        }
    }
}

fn auto_threads(n_qubits: u32) -> usize {
    if n_qubits < AUTO_MIN_QUBITS {
        return 1;
    }
    let want = if n_qubits == AUTO_MIN_QUBITS { 2 } else { AUTO_MAX_THREADS };
    let procs = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    procs.min(want)
}

fn force_scalar() -> bool {
    std::env::var("TROTTERKIT_FORCE_SCALAR").is_ok_and(|v| v == "1")
}

/// # Safety
/// The pointers must satisfy include/trotterkit.h's contract.
#[no_mangle]
pub unsafe extern "C" fn tk_evolve_terms(
    psi: *mut f64, n_qubits: u32, terms: *const TkTerms,
    s: *const TkSchedule, strategy: i32, tile_bits: i32, n_threads: i32,
) -> i32 {
    if psi.is_null() || terms.is_null() || s.is_null() {
        return TK_EINVAL;
    }
    if !(1..=TK_MAX_QUBITS).contains(&n_qubits) {
        return TK_EQUBITS;
    }
    if strategy != TK_AUTO && strategy != TK_ONTHEFLY {
        return TK_EINVAL;
    }
    if n_threads < 0 {
        return TK_EINVAL;
    }
    if tile_bits > 0 && (tile_bits < 3 || tile_bits as u32 >= n_qubits) {
        return TK_EINVAL;
    }
    let (t, s) = (&*terms, &*s);
    let (n, n_st) = (t.n, s.n_stages);
    if n_st > 0 && s.dt.is_null() {
        return TK_EINVAL;
    }
    if n > 0 && (t.x.is_null() || t.z.is_null() || t.c_re.is_null()
                 || t.c_im.is_null()) {
        return TK_EINVAL;
    }
    let slice = |p: *const u64| {
        if n == 0 { &[][..] } else { std::slice::from_raw_parts(p, n) }
    };
    let (x, z) = (slice(t.x), slice(t.z));
    if x.iter().zip(z).any(|(&a, &b)| (a | b) >> n_qubits != 0) {
        return TK_EINVAL;
    }
    let (c_re, c_im) = if n == 0 {
        (&[][..], &[][..])
    } else {
        (std::slice::from_raw_parts(t.c_re, n),
         std::slice::from_raw_parts(t.c_im, n))
    };
    let dts = if n_st == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(s.dt, n_st)
    };

    let threads = if n_threads == 0 {
        auto_threads(n_qubits)
    } else {
        n_threads as usize
    };
    let tile_bits = if tile_bits == 0 {
        if threads > 1 && n_qubits >= 5 { n_qubits as i32 - 2 } else { -1 }
    } else {
        tile_bits
    };
    let dim = 1usize << n_qubits;

    let (Some(re), Some(im)) = (AlignedBuf::new(dim), AlignedBuf::new(dim))
    else {
        return TK_ENOMEM;
    };
    let Some(mut consts) = n_st.checked_mul(n).and_then(|v| v.checked_mul(4))
        .and_then(|len| try_vec(len, 0.0f64))
    else {
        return TK_ENOMEM;
    };
    let tiles = tile_bits > 0;
    let len = if tiles { n } else { 0 };
    let (Some(mut local), Some(mut run_end), Some(mut run_begin)) =
        (try_vec(len, false), try_vec(len, 0usize), try_vec(len, 0usize))
    else {
        return TK_ENOMEM;
    };
    if tiles {
        for g in 0..n {
            local[g] = x[g] >> tile_bits == 0;
        }
        for g in (0..n).rev() {
            run_end[g] = if local[g] && g + 1 < n && local[g + 1] {
                run_end[g + 1]
            } else {
                g + 1
            };
        }
        for g in 0..n {
            run_begin[g] = if local[g] && g > 0 && local[g - 1] {
                run_begin[g - 1]
            } else {
                g
            };
        }
    }
    for (st, &dt) in dts.iter().enumerate() {
        for g in 0..n {
            let k = term_constants(0.5 * dt, c_re[g], c_im[g],
                                   (x[g] & z[g]).count_ones() & 3);
            consts[4 * (st * n + g)..4 * (st * n + g) + 4]
                .copy_from_slice(&k);
        }
    }
    for j in 0..dim {
        *re.ptr.add(j) = *psi.add(2 * j);
        *im.ptr.add(j) = *psi.add(2 * j + 1);
    }

    let d = Drive {
        re: re.ptr,
        im: im.ptr,
        dim,
        x,
        z,
        consts: &consts,
        n_st,
        n_steps: s.n_steps,
        scalar: force_scalar(),
        tile_bits: if tiles { tile_bits as u32 } else { 0 },
        local: &local,
        run_end: &run_end,
        run_begin: &run_begin,
    };
    if threads > 1 {
        let bar = SpinBarrier::new(threads);
        std::thread::scope(|sc| {
            for tid in 1..threads {
                let (d, bar) = (&d, &bar);
                sc.spawn(move || d.run(tid, threads, Some(bar)));
            }
            d.run(0, threads, Some(&bar));
        });
    } else {
        d.run(0, 1, None);
    }

    for j in 0..dim {
        *psi.add(2 * j) = *re.ptr.add(j);
        *psi.add(2 * j + 1) = *im.ptr.add(j);
    }
    TK_OK
}

/// Opaque handle type of the ABI.
pub type TkGroups = grouped::Groups;

/// # Safety
/// The pointers must satisfy include/trotterkit.h's contract.
#[no_mangle]
pub unsafe extern "C" fn tk_groups_build(
    n_qubits: u32, row: *const u64, col: *const u64, v_re: *const f64,
    v_im: *const f64, nnz: usize, out: *mut *mut TkGroups,
) -> i32 {
    if out.is_null() {
        return TK_EINVAL;
    }
    *out = std::ptr::null_mut();
    if !(1..=TK_MAX_QUBITS).contains(&n_qubits) {
        return TK_EQUBITS;
    }
    if nnz > 0 && (row.is_null() || col.is_null() || v_re.is_null()
                   || v_im.is_null()) {
        return TK_EINVAL;
    }
    let dim = 1u64 << n_qubits;
    let get = |p: *const u64| {
        if nnz == 0 { &[][..] } else { std::slice::from_raw_parts(p, nnz) }
    };
    let getf = |p: *const f64| {
        if nnz == 0 { &[][..] } else { std::slice::from_raw_parts(p, nnz) }
    };
    let (r, c, vr, vi) = (get(row), get(col), getf(v_re), getf(v_im));
    if r.iter().zip(c).any(|(&a, &b)| a >= dim || b >= dim) {
        return TK_EINVAL;
    }
    let entries = (0..nnz).map(|e| (r[e], c[e], vr[e], vi[e]));
    match grouped::build(n_qubits, entries, nnz) {
        Some(g) => {
            *out = Box::into_raw(Box::new(g));
            TK_OK
        }
        None => TK_ENOMEM,
    }
}

/// # Safety
/// The pointers must satisfy include/trotterkit.h's contract.
#[no_mangle]
pub unsafe extern "C" fn tk_evolve_grouped(
    psi: *mut f64, n_qubits: u32, g: *const TkGroups,
    s: *const TkSchedule, n_threads: i32,
) -> i32 {
    if psi.is_null() || g.is_null() || s.is_null() {
        return TK_EINVAL;
    }
    let (g, s) = (&*g, &*s);
    if n_qubits != g.n_qubits || !(0..=1).contains(&n_threads) {
        return TK_EINVAL;
    }
    if s.n_stages > 0 && s.dt.is_null() {
        return TK_EINVAL;
    }
    let dts = if s.n_stages == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(s.dt, s.n_stages)
    };
    let dim = 1usize << n_qubits;
    let (Some(mut re), Some(mut im)) = (try_vec(dim, 0.0f64),
                                        try_vec(dim, 0.0f64))
    else {
        return TK_ENOMEM;
    };
    for j in 0..dim {
        re[j] = *psi.add(2 * j);
        im[j] = *psi.add(2 * j + 1);
    }
    if grouped::evolve(&mut re, &mut im, g, dts, s.n_steps).is_none() {
        return TK_ENOMEM;
    }
    for j in 0..dim {
        *psi.add(2 * j) = re[j];
        *psi.add(2 * j + 1) = im[j];
    }
    TK_OK
}

/// # Safety
/// g must be NULL or a handle from tk_groups_build, freed once.
#[no_mangle]
pub unsafe extern "C" fn tk_groups_free(g: *mut TkGroups) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}

#[no_mangle]
pub extern "C" fn tk_backend_name() -> *const c_char {
    c"rust".as_ptr()
}

#[no_mangle]
pub extern "C" fn tk_strerror(code: i32) -> *const c_char {
    match code {
        TK_OK => c"success",
        TK_EINVAL => c"invalid argument",
        TK_ENOMEM => c"out of memory",
        TK_EQUBITS => c"qubit count out of range (1..32)",
        _ => c"unknown error",
    }
    .as_ptr()
}
