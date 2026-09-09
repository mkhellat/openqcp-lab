"""paulikit against pauli_lcu (Riverlane), the reference implementation
accompanying the NJP FWHT paper.

WHY THIS COMPARISON AND NOT PennyLane. `qml.pauli_decompose` is a
convenience function that materialises densely; comparing against it
invites the reply that a weak baseline was chosen. `pauli_lcu` is the
serious comparator: same transform, written by two authors of the
paper this project cites as its algorithm reference. MIT licensed
(header of pauli_lcu/decomposition.py), v1.0.1.

WHAT pauli_lcu ACTUALLY IS, verified rather than assumed. The
installed `pauli_lcu/` package is three pure-Python files (32K); the
work happens in `pauli_lcu_module.cpython-*.so`, a separate 64.7K
top-level C extension. Inspected with `nm -D -u` and `objdump -d`:
no OpenMP symbols and no packed-double SIMD instructions. It is
single-threaded scalar C.

THE STRUCTURAL DIFFERENCE this was built to measure. pauli_lcu's
`pauli_coefficients(matrix)` OVERWRITES the input array in place and
returns None - O(1) auxiliary memory, but the caller must already
hold the full dense 2^n x 2^n complex matrix. paulikit streams and
never materialises it.

SCOPE, AND A WARNING. This script measures DENSE random Hermitian
input with paulikit's SINGLE-PROCESS path. That is deliberately
pauli_lcu's best case and paulikit's worst: a dense random matrix has
all 4^n terms nonzero, so paulikit's sparsity-aware machinery costs
without paying. Expect 12-20x in pauli_lcu's favour here.

Do NOT read those rows as the library's performance. The supported
pipeline is `parallel_decompose_arrays` on a `scipy.sparse` operator,
and at N=150 it runs in 8.85s at 90 MiB peak RSS against pauli_lcu's
6.40s at 8891 MiB - comparable time, ~99x less memory. An earlier
version of this harness built the Hamiltonian densely and called the
single-process path, reported 41.3s, and concluded the memory
argument was falsified. That was a harness defect. See
pauli_lcu_findings.md for the configuration ladder and both results.

CORRECTNESS FIRST. No timing is interpreted until both implementations
are shown to agree on the same input. They use different output
conventions (pauli_lcu returns a dense coefficient array indexed by
(row, col); paulikit yields sparse (x, z, coeff) triples above a
threshold), so agreement is checked by reconstructing comparable views.

PROTOCOL. Per profiling/phase13/MEASUREMENT_METHODOLOGY.md: a fixed
untimed warm-up, interleaved conditions rather than blocked, n>=5
reps, first rep discarded, per-core temperature recorded, and a Welch
test. Memory is sampled in a child process so peak RSS is that run's
alone.

Usage:
    OPENBLAS_NUM_THREADS=1 python pauli_lcu_comparison.py [max_qubits] [reps]
"""

import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RESULTS = os.path.join(HERE, "pauli_lcu_comparison_results.jsonl")
PYTHON = os.path.expanduser("~/.venvs/paulikit/bin/python")
TEMP_PATH = "/sys/class/thermal/thermal_zone7/temp"
COOLDOWN_C = 55.0
IMPLS = ("pauli_lcu", "paulikit_dense", "paulikit_stream")


def read_temp():
    try:
        with open(TEMP_PATH) as f:
            return int(f.read().strip()) / 1000.0
    except OSError:
        return None


def cooldown(cap=180):
    t0 = time.perf_counter()
    while (read_temp() or 0) > COOLDOWN_C:
        if time.perf_counter() - t0 > cap:
            return False
        time.sleep(2)
    return True


CHILD = r'''
import json, os, sys, time, resource
import numpy as np
sys.path.insert(0, os.path.join(%r, "..", "src"))
impl, n_qubits = sys.argv[1], int(sys.argv[2])
dim = 2 ** n_qubits
rng = np.random.default_rng(0)

# The SAME input for both implementations: a dense random Hermitian
# matrix. Dense on purpose - pauli_lcu cannot accept anything else,
# so using paulikit's sparse path here would compare different work.
a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
m = (a + a.conj().T) / 2

ATOL = 1e-9

if impl == "pauli_lcu":
    from pauli_lcu import pauli_coefficients
    work = np.ascontiguousarray(m.copy())
    t0 = time.perf_counter()
    pauli_coefficients(work)          # in place, returns None
    el = time.perf_counter() - t0
    a_ = np.abs(work)
    n_terms = int(np.count_nonzero(a_ > ATOL))
    checksum = float(a_.sum())
    # The same sum restricted to terms above atol, so the streaming
    # condition (which thresholds) has a comparable reference.
    checksum_thresholded = float(a_[a_ > ATOL].sum())
elif impl == "paulikit_dense":
    # LIKE FOR LIKE. sparse=False returns the full dense (dim, dim)
    # array indexed by [x, z] - the same layout pauli_lcu leaves in
    # its input buffer - so the checksums are directly comparable
    # with no convention bridging.
    from paulikit.algorithms.fwht import fwht_pauli_coefficients
    t0 = time.perf_counter()
    c = fwht_pauli_coefficients(m)
    el = time.perf_counter() - t0
    n_terms = int(np.count_nonzero(np.abs(c) > ATOL))
    checksum = float(np.abs(c).sum())
else:
    # THE STRUCTURAL CONDITION. chunk_size bounds peak memory: rows
    # are transformed in blocks and thresholded per chunk, so the
    # dense (dim, dim) array is never materialised. This is the mode
    # pauli_lcu has no counterpart for, and its checksum is over
    # thresholded terms only - so it is compared against pauli_lcu's
    # OWN thresholded sum, not its full one.
    from paulikit.algorithms.fwht import fwht_pauli_coefficients
    t0 = time.perf_counter()
    x, z, coeff = fwht_pauli_coefficients(
        m, sparse=True, chunk_size=256, atol=ATOL)
    el = time.perf_counter() - t0
    n_terms = int(len(coeff))
    checksum = float(np.abs(coeff).sum())

peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
out = dict(impl=impl, n_qubits=n_qubits, dim=dim, elapsed=el,
           n_terms=n_terms, checksum=checksum,
           peak_rss_mib=peak_kib / 1024)
if impl == "pauli_lcu":
    out["checksum_thresholded"] = checksum_thresholded
print(json.dumps(out))
''' % (HERE,)


def run(impl, n_qubits):
    p = subprocess.run(
        [PYTHON, "-c", CHILD, impl, str(n_qubits)],
        capture_output=True, text=True,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
    )
    if p.returncode != 0:
        return dict(impl=impl, n_qubits=n_qubits, failed=True,
                    error=p.stderr.strip()[-300:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    min_q = int(os.environ.get("MIN_Q", 10))

    print(f"paulikit vs pauli_lcu, up to {max_q} qubits, {reps} reps "
          f"(+1 discarded warm-up)\n", flush=True)

    # --- correctness gate, before any timing is interpreted ---
    # Checksums are sum|coeff|. The dense mode is compared against
    # pauli_lcu's full sum (identical layout, so this is exact); the
    # streaming mode thresholds, so it is compared against
    # pauli_lcu's own sum restricted to the same threshold.
    print("correctness check (all three on identical input):", flush=True)
    for q in (3, 5, 7):
        res = {i: run(i, q) for i in IMPLS}
        bad = [i for i, r in res.items() if r.get("failed")]
        if bad:
            print(f"  n={q}: FAILED {bad[0]}: {res[bad[0]]['error']}",
                  flush=True)
            return
        a = res["pauli_lcu"]
        for impl, ref in (("paulikit_dense", a["checksum"]),
                          ("paulikit_stream", a["checksum_thresholded"])):
            got = res[impl]["checksum"]
            rel = abs(got - ref) / max(abs(ref), 1e-30)
            ok = rel < 1e-9
            print(f"  n={q} {impl:>15}: {res[impl]['n_terms']:>7} terms "
                  f"(pauli_lcu {a['n_terms']:>7}), "
                  f"rel.diff {rel:.2e} {'OK' if ok else 'MISMATCH'}",
                  flush=True)
            if not ok:
                print("\n  Disagreement - not proceeding to timing.",
                      flush=True)
                return
    print(flush=True)

    hdr = f"{'n':>3} {'dim':>6}"
    for i in IMPLS:
        hdr += f" {i:>26}"
    print(hdr + f" {'dense':>7} {'strm':>7}", flush=True)
    rows = []
    for q in range(min_q, max_q + 1):
        cooldown()
        for i in IMPLS:
            run(i, q)          # warm-up, discarded
        per = {i: [] for i in IMPLS}
        mem = {i: [] for i in IMPLS}
        failed = None
        for _ in range(reps):
            for impl in IMPLS:                       # interleaved
                r = run(impl, q)
                if r.get("failed"):
                    failed = (impl, r["error"])
                    break
                per[impl].append(r["elapsed"])
                mem[impl].append(r["peak_rss_mib"])
                rows.append(dict(r, temp=read_temp()))
            if failed:
                break
        if failed:
            print(f"{q:>3} {2**q:>6}  {failed[0]} FAILED: "
                  f"{failed[1][:80]}", flush=True)
            break
        line = f"{q:>3} {2**q:>6}"
        mean = {}
        for i in IMPLS:
            mean[i] = statistics.mean(per[i])
            sd = statistics.stdev(per[i]) if len(per[i]) > 1 else 0.0
            line += (f" {mean[i]:>9.4f}s±{sd:<6.4f}"
                     f"{statistics.mean(mem[i]):>5.0f}M")
        line += (f" {mean['pauli_lcu'] / mean['paulikit_dense']:>6.2f}x"
                 f" {mean['pauli_lcu'] / mean['paulikit_stream']:>6.2f}x")
        print(line, flush=True)

    with open(RESULTS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nraw data appended to {os.path.basename(RESULTS)}", flush=True)
    print("ratios are pauli_lcu / paulikit; > 1 means paulikit faster.",
          flush=True)


if __name__ == "__main__":
    main()
