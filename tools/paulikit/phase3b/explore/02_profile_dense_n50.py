import sys
from pathlib import Path
import time
import cProfile
import pstats
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from paulikit.hamiltonian import build_hamiltonian, pad_to_power_of_two
from paulikit.algorithms.fwht import fwht_pauli_coefficients

def synth(n):
    sc = {(i, j): 1.0 + 0.1 * (i + j) for i in range(n) for j in range(i, n)}
    masses = [1.0 + 0.05 * i for i in range(n)]
    return pad_to_power_of_two(build_hamiltonian(n, sc, masses))

padded, n_qubits = synth(50)
print("dim", padded.shape, "qubits", n_qubits)

t0 = time.perf_counter()
coeffs = fwht_pauli_coefficients(padded)
t1 = time.perf_counter()
print("fwht_pauli_coefficients wall time:", t1 - t0)

pr = cProfile.Profile()
pr.enable()
coeffs = fwht_pauli_coefficients(padded)
pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(15)
