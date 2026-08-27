"""Controlled comparison: fwht_pauli_terms (dense/dict, no chunking)
vs fwht_pauli_terms_iter (streaming, chunk_size=256) at the same N,
same warm-up discipline, same process - to see whether streaming's
overhead (generator suspend/resume, per-chunk dict construction,
extra bookkeeping in _iter_chunked_coefficients) costs something real
at N where chunking isn't needed for memory reasons.
"""
import argparse
import time

from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms, fwht_pauli_terms_iter


def time_dense(padded, reps):
    warm = fwht_pauli_terms(padded)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        terms = fwht_pauli_terms(padded)
        times.append(time.perf_counter() - t0)
    assert len(terms) == len(warm)
    return times, len(terms)


def time_streaming(padded, chunk_size, reps):
    def run():
        total = 0
        for chunk in fwht_pauli_terms_iter(padded, chunk_size=chunk_size):
            total += len(chunk)
        return total

    warm = run()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        n = run()
        times.append(time.perf_counter() - t0)
    assert n == warm
    return times, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-oscillators", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    spring_constants = _default_spring_constants(args.n_oscillators)
    masses = _default_masses(args.n_oscillators)
    unpadded = build_hamiltonian(args.n_oscillators, spring_constants, masses, sparse=True)
    padded, n_qubits = pad_to_power_of_two(unpadded, sparse=True)

    dense_times, n_terms_dense = time_dense(padded, args.reps)
    stream_times, n_terms_stream = time_streaming(padded, args.chunk_size, args.reps)
    assert n_terms_dense == n_terms_stream

    mean_dense = sum(dense_times) / len(dense_times)
    mean_stream = sum(stream_times) / len(stream_times)

    print(f"N={args.n_oscillators} terms={n_terms_dense} chunk_size={args.chunk_size}")
    print(f"  dense (fwht_pauli_terms):        mean={mean_dense:.4f}s  individual={[f'{t:.4f}' for t in dense_times]}")
    print(f"  streaming (fwht_pauli_terms_iter): mean={mean_stream:.4f}s  individual={[f'{t:.4f}' for t in stream_times]}")
    print(f"  streaming/dense ratio: {mean_stream/mean_dense:.3f}x")


if __name__ == "__main__":
    main()
