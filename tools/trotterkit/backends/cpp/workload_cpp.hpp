// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Mohammadreza Khellat <mkhellat@beavernets.com>
//
// Reader for trotterkit workload files (format: trotterkit/workload.py)
// and the SHA-256 (FIPS 180-4) it checks the payload with. Used by the
// trotterkit-cpp CLI only.
#ifndef TROTTERKIT_WORKLOAD_CPP_HPP
#define TROTTERKIT_WORKLOAD_CPP_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace tk {

inline std::array<unsigned char, 32> sha256(const unsigned char *data,
                                            std::size_t n)
{
    static const std::uint32_t k[64] = {
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
    auto rotr = [](std::uint32_t v, unsigned s) {
        return (v >> s) | (v << (32u - s));
    };
    std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                          0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                          0x1f83d9abu, 0x5be0cd19u};
    auto block = [&](const unsigned char *p) {
        std::uint32_t w[64], s[8];
        for (unsigned i = 0; i < 16; i++)
            w[i] = std::uint32_t{p[4 * i]} << 24
                   | std::uint32_t{p[4 * i + 1]} << 16
                   | std::uint32_t{p[4 * i + 2]} << 8 | p[4 * i + 3];
        for (unsigned i = 16; i < 64; i++) {
            std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18)
                               ^ (w[i - 15] >> 3);
            std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19)
                               ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::memcpy(s, h, sizeof s);
        for (unsigned i = 0; i < 64; i++) {
            std::uint32_t e1 = rotr(s[4], 6) ^ rotr(s[4], 11)
                               ^ rotr(s[4], 25);
            std::uint32_t ch = (s[4] & s[5]) ^ (~s[4] & s[6]);
            std::uint32_t t1 = s[7] + e1 + ch + k[i] + w[i];
            std::uint32_t e0 = rotr(s[0], 2) ^ rotr(s[0], 13)
                               ^ rotr(s[0], 22);
            std::uint32_t mj = (s[0] & s[1]) ^ (s[0] & s[2])
                               ^ (s[1] & s[2]);
            for (unsigned r = 7; r > 0; r--)
                s[r] = s[r - 1];
            s[4] += t1;
            s[0] = t1 + e0 + mj;
        }
        for (unsigned i = 0; i < 8; i++)
            h[i] += s[i];
    };
    std::size_t full = n / 64u * 64u, rest = n - full;
    for (std::size_t i = 0; i < full; i += 64u)
        block(data + i);
    unsigned char tail[128] = {};
    std::memcpy(tail, data + full, rest);
    tail[rest] = 0x80u;
    std::size_t tail_len = rest + 9u <= 64u ? 64u : 128u;
    std::uint64_t bits = std::uint64_t{n} * 8u;
    for (unsigned i = 0; i < 8u; i++)
        tail[tail_len - 1u - i] = static_cast<unsigned char>(bits >> (8u * i));
    for (std::size_t i = 0; i < tail_len; i += 64u)
        block(tail + i);
    std::array<unsigned char, 32> out;
    for (unsigned i = 0; i < 8u; i++)
        for (unsigned b = 0; b < 4u; b++)
            out[4 * i + b] = static_cast<unsigned char>(h[i] >> (24 - 8 * b));
    return out;
}

struct Workload {
    std::uint32_t n_qubits = 0;
    std::vector<std::uint64_t> x, z, row, col;
    std::vector<double> c_re, c_im, v_re, v_im, psi0;  // psi0: 2 * dim
};

// Throws std::runtime_error with a message on any problem.
inline Workload read_workload(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(path + ": cannot open");
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
    auto le = [&](std::size_t off, unsigned bytes) {
        std::uint64_t v = 0;
        for (unsigned i = 0; i < bytes; i++)
            v |= std::uint64_t{d[off + i]} << (8u * i);
        return v;
    };
    if (d.size() < 64 || std::memcmp(d.data(), "TKWL", 4) != 0)
        throw std::runtime_error(path + ": not a trotterkit workload");
    if (le(4, 4) != 1u)
        throw std::runtime_error(path + ": unsupported version");
    Workload w;
    w.n_qubits = static_cast<std::uint32_t>(le(8, 4));
    std::uint64_t n = le(16, 8), nnz = le(24, 8);
    if (w.n_qubits < 1u || w.n_qubits > 32u)
        throw std::runtime_error(path + ": qubit count out of range");
    std::uint64_t dim = std::uint64_t{1} << w.n_qubits, len = d.size();
    if (n > len / 32u || nnz > len / 32u || dim > len / 16u
        || len != 64u + 32u * n + 32u * nnz + 16u * dim)
        throw std::runtime_error(path + ": payload length mismatch");
    auto digest = sha256(d.data() + 64, d.size() - 64u);
    if (std::memcmp(digest.data(), d.data() + 32, 32) != 0)
        throw std::runtime_error(path + ": payload checksum mismatch");
    std::size_t off = 64;
    auto take = [&](auto &v, std::uint64_t count) {
        v.resize(count);
        std::memcpy(v.data(), d.data() + off, count * 8u);
        off += count * 8u;
    };
    take(w.x, n);
    take(w.z, n);
    take(w.c_re, n);
    take(w.c_im, n);
    take(w.row, nnz);
    take(w.col, nnz);
    take(w.v_re, nnz);
    take(w.v_im, nnz);
    take(w.psi0, 2u * dim);
    return w;
}

}  // namespace tk

#endif
