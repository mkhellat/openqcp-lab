// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! Reader for trotterkit workload files (format: trotterkit/workload.py).

use crate::sha256::sha256;

pub struct Workload {
    pub n_qubits: u32,
    pub x: Vec<u64>,
    pub z: Vec<u64>,
    pub c_re: Vec<f64>,
    pub c_im: Vec<f64>,
    pub row: Vec<u64>,
    pub col: Vec<u64>,
    pub v_re: Vec<f64>,
    pub v_im: Vec<f64>,
    pub psi0: Vec<f64>, // 2 * dim, interleaved
}

fn le(d: &[u8], off: usize, bytes: usize) -> u64 {
    d[off..off + bytes].iter().rev().fold(0u64, |v, &b| v << 8 | b as u64)
}

pub fn read(path: &str) -> Result<Workload, String> {
    let d = std::fs::read(path).map_err(|_| format!("{path}: cannot open"))?;
    if d.len() < 64 || &d[..4] != b"TKWL" {
        return Err(format!("{path}: not a trotterkit workload"));
    }
    if le(&d, 4, 4) != 1 {
        return Err(format!("{path}: unsupported version"));
    }
    let n_qubits = le(&d, 8, 4) as u32;
    let (n, nnz) = (le(&d, 16, 8), le(&d, 24, 8));
    if !(1..=32).contains(&n_qubits) {
        return Err(format!("{path}: qubit count out of range"));
    }
    let (dim, len) = (1u64 << n_qubits, d.len() as u64);
    if n > len / 32 || nnz > len / 32 || dim > len / 16
        || len != 64 + 32 * n + 32 * nnz + 16 * dim
    {
        return Err(format!("{path}: payload length mismatch"));
    }
    if sha256(&d[64..])[..] != d[32..64] {
        return Err(format!("{path}: payload checksum mismatch"));
    }
    let mut off = 64usize;
    let mut words = |count: u64| {
        let v: Vec<[u8; 8]> = d[off..off + 8 * count as usize]
            .chunks_exact(8)
            .map(|c| c.try_into().unwrap_or([0; 8]))
            .collect();
        off += 8 * count as usize;
        v
    };
    let u = |v: Vec<[u8; 8]>| v.into_iter().map(u64::from_le_bytes).collect();
    let f = |v: Vec<[u8; 8]>| v.into_iter().map(f64::from_le_bytes).collect();
    Ok(Workload {
        n_qubits,
        x: u(words(n)),
        z: u(words(n)),
        c_re: f(words(n)),
        c_im: f(words(n)),
        row: u(words(nnz)),
        col: u(words(nnz)),
        v_re: f(words(nnz)),
        v_im: f(words(nnz)),
        psi0: f(words(2 * dim)),
    })
}
