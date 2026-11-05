/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * The trotterkit C ABI. Every backend (C, C++, Rust) implements
 * exactly these functions. Complex arrays are interleaved doubles
 * (re, im) - numpy complex128 memory. Functions return TK_OK or an
 * error code; they never abort.
 *
 * S2(dt) applies every fragment with angle dt/2 in order, then in
 * reverse order; a schedule repeats its stage list n_steps times,
 * stage k being S2(dt[k]).
 */
#ifndef TROTTERKIT_H
#define TROTTERKIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TK_OK = 0, TK_EINVAL = 1, TK_ENOMEM = 2, TK_EQUBITS = 3 };
enum { TK_AUTO = 0, TK_ONTHEFLY = 1, TK_TABLES = 2 };

#define TK_MAX_QUBITS 32u

typedef struct {
    const uint64_t *x;
    const uint64_t *z;
    const double *c_re;
    const double *c_im;
    size_t n;
} tk_terms;

typedef struct {
    const double *dt;
    size_t n_stages;
    size_t n_steps;
} tk_schedule;

typedef struct tk_groups tk_groups;

int tk_evolve_terms(double *psi, uint32_t n_qubits,
                    const tk_terms *terms, const tk_schedule *s,
                    int strategy, int tile_bits, int n_threads);
int tk_groups_build(uint32_t n_qubits, const uint64_t *row,
                    const uint64_t *col, const double *v_re,
                    const double *v_im, size_t nnz, tk_groups **out);
int tk_evolve_grouped(double *psi, uint32_t n_qubits,
                      const tk_groups *g, const tk_schedule *s,
                      int n_threads);
void tk_groups_free(tk_groups *g);
const char *tk_backend_name(void);
const char *tk_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif
