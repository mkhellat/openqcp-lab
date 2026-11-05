// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
//
// trotterkit-cpp: run one evolution from a workload file and print a
// JSON summary; only the kernel call is timed. Arguments, output and
// exit status match trotterkit-c:
//
//   trotterkit-cpp WORKLOAD --t T --steps S [--mode terms|grouped]
//       [--order 2|4] [--strategy auto|onthefly|tables]
//       [--tile-bits K] [--threads K] [--out FILE]
//
// Exit status: 0 success, 1 read or kernel error, 2 usage error.

#include "trotterkit.h"
#include "workload_cpp.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

namespace {

int usage(const char *msg)
{
    if (msg != nullptr)
        std::fprintf(stderr, "trotterkit-cpp: %s\n", msg);
    std::fputs("usage: trotterkit-cpp WORKLOAD --t T --steps S"
               " [--mode terms|grouped]\n"
               "       [--order 2|4] [--strategy auto|onthefly|tables]"
               " [--tile-bits K]\n"
               "       [--threads K] [--out FILE]\n",
               stderr);
    return 2;
}

bool to_long(const std::string &s, long &out)
{
    char *end;
    out = std::strtol(s.c_str(), &end, 10);
    return !s.empty() && *end == '\0';
}

// The expressions of trotterkit.schedule.stages, in the same order.
std::size_t stages(double t, long steps, long order, double dt[5])
{
    double h = t / static_cast<double>(steps);
    double p4 = 1.0 / (4.0 - std::pow(4.0, 1.0 / 3.0));
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

}  // namespace

int main(int argc, char **argv)
{
    std::string path, out_path, mode = "terms", strategy_name = "auto";
    double t = 0.0;
    bool have_t = false;
    long steps = 0, order = 2, tile_bits = 0, threads = 0;
    int strategy = TK_AUTO;
    const std::map<std::string, int> strategies = {
        {"auto", TK_AUTO}, {"onthefly", TK_ONTHEFLY}, {"tables", TK_TABLES}};

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.empty() || a[0] != '-') {
            if (!path.empty())
                return usage("more than one workload given");
            path = a;
            continue;
        }
        if (i + 1 >= argc)
            return usage("option needs a value");
        std::string v = argv[++i];
        if (a == "--t") {
            char *end;
            t = std::strtod(v.c_str(), &end);
            have_t = !v.empty() && *end == '\0';
            if (!have_t)
                return usage("bad --t");
        } else if (a == "--steps") {
            if (!to_long(v, steps) || steps < 1)
                return usage("bad --steps");
        } else if (a == "--order") {
            if (!to_long(v, order) || (order != 2 && order != 4))
                return usage("bad --order");
        } else if (a == "--mode") {
            if (v != "terms" && v != "grouped")
                return usage("bad --mode");
            mode = v;
        } else if (a == "--strategy") {
            auto it = strategies.find(v);
            if (it == strategies.end())
                return usage("bad --strategy");
            strategy = it->second;
            strategy_name = v;
        } else if (a == "--tile-bits") {
            if (!to_long(v, tile_bits))
                return usage("bad --tile-bits");
        } else if (a == "--threads") {
            if (!to_long(v, threads))
                return usage("bad --threads");
        } else if (a == "--out") {
            out_path = v;
        } else {
            return usage("unknown option");
        }
    }
    if (path.empty() || !have_t || steps == 0)
        return usage(nullptr);

    tk::Workload w;
    try {
        w = tk::read_workload(path);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "trotterkit-cpp: %s\n", e.what());
        return 1;
    }
    double dt[5];
    tk_schedule sched{dt, stages(t, steps, order, dt),
                      static_cast<std::size_t>(steps)};
    double *psi = w.psi0.data();
    double secs = 0.0;
    int rc;
    using clock = std::chrono::steady_clock;
    if (mode == "terms") {
        tk_terms terms{w.x.data(), w.z.data(), w.c_re.data(),
                       w.c_im.data(), w.x.size()};
        auto t0 = clock::now();
        rc = tk_evolve_terms(psi, w.n_qubits, &terms, &sched, strategy,
                             static_cast<int>(tile_bits),
                             static_cast<int>(threads));
        secs = std::chrono::duration<double>(clock::now() - t0).count();
    } else {
        tk_groups *g = nullptr;
        rc = tk_groups_build(w.n_qubits, w.row.data(), w.col.data(),
                             w.v_re.data(), w.v_im.data(), w.row.size(),
                             &g);
        if (rc == TK_OK) {
            auto t0 = clock::now();
            rc = tk_evolve_grouped(psi, w.n_qubits, g, &sched,
                                   static_cast<int>(threads));
            secs = std::chrono::duration<double>(clock::now() - t0).count();
        }
        tk_groups_free(g);
    }
    if (rc != TK_OK) {
        std::fprintf(stderr, "trotterkit-cpp: %s\n", tk_strerror(rc));
        return 1;
    }
    double norm2 = 0.0;
    for (double v : w.psi0)
        norm2 += v * v;
    if (!out_path.empty()) {
        std::ofstream f(out_path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(w.psi0.data()),
                static_cast<std::streamsize>(w.psi0.size() * 8u));
        if (!f) {
            std::fprintf(stderr, "trotterkit-cpp: cannot write %s\n",
                         out_path.c_str());
            return 1;
        }
    }
    std::printf("{\"backend\": \"%s\", \"mode\": \"%s\", \"n_qubits\": %u, "
                "\"n_terms\": %lu, \"nnz\": %lu, \"order\": %ld, "
                "\"steps\": %ld, \"t\": %.17g, \"strategy\": \"%s\", "
                "\"tile_bits\": %ld, \"threads\": %ld, "
                "\"kernel_seconds\": %.9f, \"norm2\": %.17g}\n",
                tk_backend_name(), mode.c_str(), w.n_qubits,
                static_cast<unsigned long>(w.x.size()),
                static_cast<unsigned long>(w.row.size()), order, steps, t,
                strategy_name.c_str(), tile_bits, threads, secs, norm2);
    return 0;
}
