"""In-process, warmed-up sweep of fwht_pauli_terms_iter (Phase 10
streaming path) across N=25/50/100/150 - uniform sparse+streaming
code path at every N (not dense for small N, sparse for large N),
so timings are apples-to-apples across the table. Same warm-up
discipline as cache_locality/steady_state_decompose.py: one untimed
call pays for first-call costs (imports, allocator warm-up, first-touch
page faults), then --reps further calls are timed and averaged.
"""
import argparse
import time

from paulikit.cli import _default_masses, _default_spring_constants
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_terms_iter


def run_once(padded_sparse, chunk_size):
    total_terms = 0
    n_chunks = 0
    for chunk_terms in fwht_pauli_terms_iter(padded_sparse, chunk_size=chunk_size):
        n_chunks += 1
        total_terms += len(chunk_terms)
    return total_terms, n_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-oscillators", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    spring_constants = _default_spring_constants(args.n_oscillators)
    masses = _default_masses(args.n_oscillators)
    unpadded_sparse = build_hamiltonian(args.n_oscillators, spring_constants, masses, sparse=True)
    padded_sparse, n_qubits = pad_to_power_of_two(unpadded_sparse, sparse=True)

    warm_terms, warm_chunks = run_once(padded_sparse, args.chunk_size)

    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        n_terms, n_chunks = run_once(padded_sparse, args.chunk_size)
        times.append(time.perf_counter() - t0)

    if n_terms != warm_terms:
        print(f"ERROR: term count changed ({warm_terms} vs {n_terms})")
        return 1

    mean_time = sum(times) / len(times)
    print(f"N={args.n_oscillators} qubits={n_qubits} dim={padded_sparse.shape[0]} "
          f"terms={n_terms} chunks={n_chunks} "
          f"mean_time={mean_time:.4f}s individual={[f'{t:.4f}' for t in times]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
