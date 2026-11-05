/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
 *
 * trotterkit-c: run one evolution from a workload file and print a
 * JSON summary. Only the kernel call is timed.
 *
 *   trotterkit-c WORKLOAD --t T --steps S [--mode terms|grouped]
 *       [--order 2|4] [--strategy auto|onthefly|tables]
 *       [--tile-bits K] [--threads K] [--out FILE]
 *
 * Exit status: 0 success, 1 read or kernel error, 2 usage error.
 */
#define _POSIX_C_SOURCE 199309L

#include "trotterkit.h"
#include "workload_c.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char USAGE[] =
    "usage: trotterkit-c WORKLOAD --t T --steps S"
    " [--mode terms|grouped]\n"
    "       [--order 2|4] [--strategy auto|onthefly|tables]"
    " [--tile-bits K]\n"
    "       [--threads K] [--out FILE]\n";

static int usage(const char *msg)
{
    if (msg != NULL)
        fprintf(stderr, "trotterkit-c: %s\n", msg);
    fputs(USAGE, stderr);
    return 2;
}

static int parse_long(const char *s, long *out)
{
    char *end;
    *out = strtol(s, &end, 10);
    return *s != '\0' && *end == '\0';
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Same expressions, in the same order, as trotterkit.schedule.stages,
 * so the stage step sizes are bitwise identical. */
static size_t stages(double t, long steps, int order, double dt[5])
{
    double h = t / (double)steps;
    double p4 = 1.0 / (4.0 - pow(4.0, 1.0 / 3.0));
    if (order == 2) {
        dt[0] = h;
        return 1;
    }
    dt[0] = p4 * h;
    dt[1] = p4 * h;
    dt[2] = (1 - 4 * p4) * h;
    dt[3] = p4 * h;
    dt[4] = p4 * h;
    return 5;
}

int main(int argc, char **argv)
{
    const char *path = NULL, *out_path = NULL, *mode = "terms";
    const char *strategy_name = "auto";
    double t = 0.0, dt[5], t0, secs, norm2 = 0.0;
    long steps = 0, order = 2, tile_bits = 0, threads = 0;
    int have_t = 0, strategy = TK_AUTO, rc, i;
    size_t dim, j;
    tk_workload w;
    tk_schedule sched;
    char err[256];
    double *psi;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (a[0] != '-') {
            if (path != NULL)
                return usage("more than one workload given");
            path = a;
            continue;
        }
        if (v == NULL)
            return usage("option needs a value");
        i++;
        if (strcmp(a, "--t") == 0) {
            char *end;
            t = strtod(v, &end);
            have_t = *v != '\0' && *end == '\0';
            if (!have_t)
                return usage("bad --t");
        } else if (strcmp(a, "--steps") == 0) {
            if (!parse_long(v, &steps) || steps < 1)
                return usage("bad --steps");
        } else if (strcmp(a, "--order") == 0) {
            if (!parse_long(v, &order) || (order != 2 && order != 4))
                return usage("bad --order");
        } else if (strcmp(a, "--mode") == 0) {
            if (strcmp(v, "terms") != 0 && strcmp(v, "grouped") != 0)
                return usage("bad --mode");
            mode = v;
        } else if (strcmp(a, "--strategy") == 0) {
            if (strcmp(v, "auto") == 0)
                strategy = TK_AUTO;
            else if (strcmp(v, "onthefly") == 0)
                strategy = TK_ONTHEFLY;
            else if (strcmp(v, "tables") == 0)
                strategy = TK_TABLES;
            else
                return usage("bad --strategy");
            strategy_name = v;
        } else if (strcmp(a, "--tile-bits") == 0) {
            if (!parse_long(v, &tile_bits))
                return usage("bad --tile-bits");
        } else if (strcmp(a, "--threads") == 0) {
            if (!parse_long(v, &threads))
                return usage("bad --threads");
        } else if (strcmp(a, "--out") == 0) {
            out_path = v;
        } else {
            return usage("unknown option");
        }
    }
    if (path == NULL || !have_t || steps == 0)
        return usage(NULL);

    if (tk_workload_read(path, &w, err, sizeof err) != 0) {
        fprintf(stderr, "trotterkit-c: %s\n", err);
        return 1;
    }
    dim = (size_t)1 << w.n_qubits;
    psi = w.psi0;
    sched.dt = dt;
    sched.n_stages = stages(t, steps, (int)order, dt);
    sched.n_steps = (size_t)steps;

    if (strcmp(mode, "terms") == 0) {
        tk_terms terms;
        terms.x = w.x;
        terms.z = w.z;
        terms.c_re = w.c_re;
        terms.c_im = w.c_im;
        terms.n = w.n_terms;
        t0 = now();
        rc = tk_evolve_terms(psi, w.n_qubits, &terms, &sched, strategy,
                             (int)tile_bits, (int)threads);
        secs = now() - t0;
    } else {
        tk_groups *g = NULL;
        rc = tk_groups_build(w.n_qubits, w.row, w.col, w.v_re, w.v_im,
                             w.nnz, &g);
        if (rc == TK_OK) {
            t0 = now();
            rc = tk_evolve_grouped(psi, w.n_qubits, g, &sched,
                                   (int)threads);
            secs = now() - t0;
        } else {
            secs = 0.0;
        }
        tk_groups_free(g);
    }
    if (rc != TK_OK) {
        fprintf(stderr, "trotterkit-c: %s\n", tk_strerror(rc));
        tk_workload_free(&w);
        return 1;
    }
    for (j = 0; j < 2u * dim; j++)
        norm2 += psi[j] * psi[j];
    if (out_path != NULL) {
        FILE *fh = fopen(out_path, "wb");
        if (fh == NULL
            || fwrite(psi, sizeof *psi, 2u * dim, fh) != 2u * dim
            || fclose(fh) != 0) {
            fprintf(stderr, "trotterkit-c: cannot write %s\n", out_path);
            tk_workload_free(&w);
            return 1;
        }
    }
    printf("{\"backend\": \"%s\", \"mode\": \"%s\", \"n_qubits\": %u, "
           "\"n_terms\": %lu, \"nnz\": %lu, \"order\": %ld, "
           "\"steps\": %ld, \"t\": %.17g, \"strategy\": \"%s\", "
           "\"tile_bits\": %ld, \"threads\": %ld, "
           "\"kernel_seconds\": %.9f, \"norm2\": %.17g}\n",
           tk_backend_name(), mode, (unsigned)w.n_qubits,
           (unsigned long)w.n_terms, (unsigned long)w.nnz, order, steps,
           t, strategy_name, tile_bits, threads, secs, norm2);
    tk_workload_free(&w);
    return 0;
}
