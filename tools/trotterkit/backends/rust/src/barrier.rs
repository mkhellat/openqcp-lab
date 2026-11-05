// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>

//! Spinning barrier for the per-term thread team.
//!
//! std::sync::Barrier parks waiting threads on a mutex and condition
//! variable. The kernel crosses a barrier after every term (a few
//! microseconds of work), so threads must spin as OpenMP's barrier
//! does in the C and C++ backends; a parking barrier would measure
//! the scheduler, not the language.

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct SpinBarrier {
    n: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    pub fn new(n: usize) -> Self {
        SpinBarrier {
            n,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    /// Returns once all n threads have called wait for this phase.
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            // Reset before releasing: the next phase's arrivals only
            // start after they observe the new generation.
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == gen {
                std::hint::spin_loop();
            }
        }
    }
}
