/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * Workload file reader and the SHA-256 it verifies the payload with
 * (FIPS 180-4; written here so the CLI needs no crypto library).
 */
#include "workload_c.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static uint32_t rotr(uint32_t v, unsigned n)
{
    return (v >> n) | (v << (32u - n));
}

static void sha256_block(uint32_t h[8], const unsigned char *p)
{
    uint32_t w[64], s[8];
    unsigned i;

    for (i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16
               | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18)
                      ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19)
                      ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    memcpy(s, h, sizeof s);
    for (i = 0; i < 64; i++) {
        uint32_t e1 = rotr(s[4], 6) ^ rotr(s[4], 11) ^ rotr(s[4], 25);
        uint32_t ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
        uint32_t t1 = s[7] + e1 + ch + K256[i] + w[i];
        uint32_t e0 = rotr(s[0], 2) ^ rotr(s[0], 13) ^ rotr(s[0], 22);
        uint32_t mj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]);
        uint32_t t2 = e0 + mj;
        s[7] = s[6];
        s[6] = s[5];
        s[5] = s[4];
        s[4] = s[3] + t1;
        s[3] = s[2];
        s[2] = s[1];
        s[1] = s[0];
        s[0] = t1 + t2;
    }
    for (i = 0; i < 8; i++)
        h[i] += s[i];
}

void tk_sha256(const unsigned char *data, size_t n,
               unsigned char out[32])
{
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                     0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                     0x1f83d9abu, 0x5be0cd19u};
    unsigned char tail[128];
    size_t full = n / 64u * 64u, rest = n - full, tail_len, i;
    uint64_t bits = (uint64_t)n * 8u;

    for (i = 0; i < full; i += 64u)
        sha256_block(h, data + i);
    memset(tail, 0, sizeof tail);
    memcpy(tail, data + full, rest);
    tail[rest] = 0x80u;
    tail_len = rest + 1u + 8u <= 64u ? 64u : 128u;
    for (i = 0; i < 8u; i++)
        tail[tail_len - 1u - i] = (unsigned char)(bits >> (8u * i));
    for (i = 0; i < tail_len; i += 64u)
        sha256_block(h, tail + i);
    for (i = 0; i < 8u; i++) {
        out[4 * i] = (unsigned char)(h[i] >> 24);
        out[4 * i + 1] = (unsigned char)(h[i] >> 16);
        out[4 * i + 2] = (unsigned char)(h[i] >> 8);
        out[4 * i + 3] = (unsigned char)h[i];
    }
}

static void fail(char *err, size_t err_len, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, err_len, fmt, ap);
    va_end(ap);
}

static uint32_t le32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16
           | (uint32_t)p[3] << 24;
}

static uint64_t le64(const unsigned char *p)
{
    return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
}

/* Copies count 8-byte little-endian values into a fresh array. The
 * target machines are little-endian; memcpy avoids unaligned reads. */
static void *take(const unsigned char **p, size_t count)
{
    void *a = malloc(count == 0u ? 8u : count * 8u);
    if (a != NULL && count > 0u)
        memcpy(a, *p, count * 8u);
    *p += count * 8u;
    return a;
}

void tk_workload_free(tk_workload *w)
{
    free(w->x);
    free(w->z);
    free(w->c_re);
    free(w->c_im);
    free(w->row);
    free(w->col);
    free(w->v_re);
    free(w->v_im);
    free(w->psi0);
    memset(w, 0, sizeof *w);
}

int tk_workload_read(const char *path, tk_workload *w, char *err,
                     size_t err_len)
{
    FILE *fh;
    unsigned char *data = NULL, digest[32];
    const unsigned char *p;
    long len;
    size_t n, nnz, dim, want;
    uint32_t n_qubits;

    memset(w, 0, sizeof *w);
    fh = fopen(path, "rb");
    if (fh == NULL) {
        fail(err, err_len, "%s: cannot open", path);
        return -1;
    }
    if (fseek(fh, 0, SEEK_END) != 0 || (len = ftell(fh)) < 0
        || fseek(fh, 0, SEEK_SET) != 0) {
        fclose(fh);
        fail(err, err_len, "%s: cannot read", path);
        return -1;
    }
    data = malloc(len > 0 ? (size_t)len : 1u);
    if (data == NULL
        || fread(data, 1, (size_t)len, fh) != (size_t)len) {
        fclose(fh);
        free(data);
        fail(err, err_len, "%s: cannot read", path);
        return -1;
    }
    fclose(fh);
    if (len < 64 || memcmp(data, "TKWL", 4) != 0) {
        free(data);
        fail(err, err_len, "%s: not a trotterkit workload", path);
        return -1;
    }
    if (le32(data + 4) != 1u) {
        free(data);
        fail(err, err_len, "%s: unsupported version", path);
        return -1;
    }
    n_qubits = le32(data + 8);
    n = (size_t)le64(data + 16);
    nnz = (size_t)le64(data + 24);
    if (n_qubits < 1u || n_qubits > 32u) {
        free(data);
        fail(err, err_len, "%s: qubit count out of range", path);
        return -1;
    }
    dim = (size_t)1 << n_qubits;
    /* Bound the counts by the file size first so want cannot wrap. */
    if (n > (size_t)len / 32u || nnz > (size_t)len / 32u
        || dim > (size_t)len / 16u) {
        free(data);
        fail(err, err_len, "%s: payload length mismatch", path);
        return -1;
    }
    want = 64u + 32u * n + 32u * nnz + 16u * dim;
    if ((size_t)len != want) {
        free(data);
        fail(err, err_len, "%s: payload length mismatch", path);
        return -1;
    }
    tk_sha256(data + 64, (size_t)len - 64u, digest);
    if (memcmp(digest, data + 32, 32) != 0) {
        free(data);
        fail(err, err_len, "%s: payload checksum mismatch", path);
        return -1;
    }
    p = data + 64;
    w->n_qubits = n_qubits;
    w->n_terms = n;
    w->nnz = nnz;
    w->x = take(&p, n);
    w->z = take(&p, n);
    w->c_re = take(&p, n);
    w->c_im = take(&p, n);
    w->row = take(&p, nnz);
    w->col = take(&p, nnz);
    w->v_re = take(&p, nnz);
    w->v_im = take(&p, nnz);
    w->psi0 = take(&p, 2u * dim);
    free(data);
    if (w->x == NULL || w->z == NULL || w->c_re == NULL
        || w->c_im == NULL || w->row == NULL || w->col == NULL
        || w->v_re == NULL || w->v_im == NULL || w->psi0 == NULL) {
        tk_workload_free(w);
        fail(err, err_len, "%s: out of memory", path);
        return -1;
    }
    return 0;
}
