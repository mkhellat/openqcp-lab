/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * Reader for trotterkit workload files (see trotterkit/workload.py
 * for the format). Used by the trotterkit-c CLI only.
 */
#ifndef TROTTERKIT_WORKLOAD_C_H
#define TROTTERKIT_WORKLOAD_C_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t n_qubits;
    size_t n_terms, nnz;
    uint64_t *x, *z, *row, *col;
    double *c_re, *c_im, *v_re, *v_im;
    double *psi0;               /* 2 * 2^n_qubits, interleaved */
} tk_workload;

/* 0 on success; -1 with a message in err otherwise. On failure w
 * holds no allocations. */
int tk_workload_read(const char *path, tk_workload *w, char *err,
                     size_t err_len);
void tk_workload_free(tk_workload *w);

/* SHA-256 of n bytes into out[32]; exposed for the reader's own
 * check. */
void tk_sha256(const unsigned char *data, size_t n,
               unsigned char out[32]);

#endif
