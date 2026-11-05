// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! trotterkit-rs: run one evolution from a workload file and print a
//! JSON summary; only the kernel call is timed. Arguments, output and
//! exit status match trotterkit-c:
//!
//!   trotterkit-rs WORKLOAD --t T --steps S [--mode terms|grouped]
//!       [--order 2|4] [--strategy auto|onthefly|tables]
//!       [--tile-bits K] [--threads K] [--out FILE]
//!
//! Exit status: 0 success, 1 read or kernel error, 2 usage error.

use std::ffi::CStr;
use std::process::ExitCode;
use std::time::Instant;

use trotterkit_rs::{
    tk_backend_name, tk_evolve_grouped, tk_evolve_terms, tk_groups_build,
    tk_groups_free, tk_strerror, workload, TkGroups, TkSchedule, TkTerms,
    TK_AUTO, TK_OK, TK_ONTHEFLY, TK_TABLES,
};

fn usage(msg: Option<&str>) -> ExitCode {
    if let Some(m) = msg {
        eprintln!("trotterkit-rs: {m}");
    }
    eprintln!("usage: trotterkit-rs WORKLOAD --t T --steps S \
               [--mode terms|grouped]\n       [--order 2|4] \
               [--strategy auto|onthefly|tables] [--tile-bits K]\n       \
               [--threads K] [--out FILE]");
    ExitCode::from(2)
}

fn c_str(p: *const std::ffi::c_char) -> String {
    // SAFETY: the backend returns static NUL-terminated strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// The expressions of trotterkit.schedule.stages, in the same order.
fn stages(t: f64, steps: i64, order: i64) -> Vec<f64> {
    let h = t / steps as f64;
    let p4 = 1.0 / (4.0 - 4.0f64.powf(1.0 / 3.0));
    if order == 2 {
        vec![h]
    } else {
        vec![p4 * h, p4 * h, (1.0 - 4.0 * p4) * h, p4 * h, p4 * h]
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut path, mut out_path) = (None::<String>, None::<String>);
    let (mut mode, mut strategy_name) = ("terms".to_string(),
                                         "auto".to_string());
    let (mut t, mut steps, mut order) = (None::<f64>, 0i64, 2i64);
    let (mut tile_bits, mut threads, mut strategy) = (0i64, 0i64, TK_AUTO);
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with('-') {
            if path.is_some() {
                return usage(Some("more than one workload given"));
            }
            path = Some(a.clone());
            i += 1;
            continue;
        }
        let Some(v) = args.get(i + 1) else {
            return usage(Some("option needs a value"));
        };
        i += 2;
        match a.as_str() {
            "--t" => match v.parse() {
                Ok(x) => t = Some(x),
                Err(_) => return usage(Some("bad --t")),
            },
            "--steps" => match v.parse() {
                Ok(x) if x >= 1 => steps = x,
                _ => return usage(Some("bad --steps")),
            },
            "--order" => match v.parse() {
                Ok(x @ (2 | 4)) => order = x,
                _ => return usage(Some("bad --order")),
            },
            "--mode" if v == "terms" || v == "grouped" => mode = v.clone(),
            "--mode" => return usage(Some("bad --mode")),
            "--strategy" => {
                strategy = match v.as_str() {
                    "auto" => TK_AUTO,
                    "onthefly" => TK_ONTHEFLY,
                    "tables" => TK_TABLES,
                    _ => return usage(Some("bad --strategy")),
                };
                strategy_name = v.clone();
            }
            "--tile-bits" => match v.parse() {
                Ok(x) => tile_bits = x,
                Err(_) => return usage(Some("bad --tile-bits")),
            },
            "--threads" => match v.parse() {
                Ok(x) => threads = x,
                Err(_) => return usage(Some("bad --threads")),
            },
            "--out" => out_path = Some(v.clone()),
            _ => return usage(Some("unknown option")),
        }
    }
    let (Some(path), Some(t)) = (path, t) else {
        return usage(None);
    };
    if steps == 0 {
        return usage(None);
    }

    let mut w = match workload::read(&path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("trotterkit-rs: {e}");
            return ExitCode::from(1);
        }
    };
    let dt = stages(t, steps, order);
    let sched = TkSchedule { dt: dt.as_ptr(), n_stages: dt.len(),
                             n_steps: steps as usize };
    let psi = w.psi0.as_mut_ptr();
    let mut secs = 0.0;
    let rc;
    // SAFETY: every array outlives the calls and has the sizes the
    // header requires (the workload reader checked them).
    unsafe {
        if mode == "terms" {
            let terms = TkTerms { x: w.x.as_ptr(), z: w.z.as_ptr(),
                                  c_re: w.c_re.as_ptr(),
                                  c_im: w.c_im.as_ptr(), n: w.x.len() };
            let t0 = Instant::now();
            rc = tk_evolve_terms(psi, w.n_qubits, &terms, &sched, strategy,
                                 tile_bits as i32, threads as i32);
            secs = t0.elapsed().as_secs_f64();
        } else {
            let mut g: *mut TkGroups = std::ptr::null_mut();
            let built = tk_groups_build(w.n_qubits, w.row.as_ptr(),
                                        w.col.as_ptr(), w.v_re.as_ptr(),
                                        w.v_im.as_ptr(), w.row.len(),
                                        &mut g);
            if built == TK_OK {
                let t0 = Instant::now();
                rc = tk_evolve_grouped(psi, w.n_qubits, g, &sched,
                                       threads as i32);
                secs = t0.elapsed().as_secs_f64();
            } else {
                rc = built;
            }
            tk_groups_free(g);
        }
    }
    if rc != TK_OK {
        eprintln!("trotterkit-rs: {}", c_str(tk_strerror(rc)));
        return ExitCode::from(1);
    }
    let norm2: f64 = w.psi0.iter().map(|v| v * v).sum();
    if let Some(out) = out_path {
        let bytes: Vec<u8> = w.psi0.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        if std::fs::write(&out, bytes).is_err() {
            eprintln!("trotterkit-rs: cannot write {out}");
            return ExitCode::from(1);
        }
    }
    println!("{{\"backend\": \"{}\", \"mode\": \"{mode}\", \
              \"n_qubits\": {}, \"n_terms\": {}, \"nnz\": {}, \
              \"order\": {order}, \"steps\": {steps}, \"t\": {t:e}, \
              \"strategy\": \"{strategy_name}\", \"tile_bits\": \
              {tile_bits}, \"threads\": {threads}, \
              \"kernel_seconds\": {secs:.9}, \"norm2\": {norm2:e}}}",
             c_str(tk_backend_name()), w.n_qubits, w.x.len(),
             w.row.len());
    ExitCode::SUCCESS
}
