"""Step 1: prove (or refute) the R3 mechanism by ADDING ONE VARIABLE
to an existing control that is already known to scale.

THE MECHANISM UNDER TEST.
`traffic_intensity_findings.md` established that `wht_large` - which
pushes a full (2, 16384) complex128 array (~512 KiB, essentially the
same payload size paulikit's own chunks return) through the SAME
ProcessPoolExecutor result queue - scales 2.23x from w2_c1 to w8_c4,
with effective concurrency 5.80. paulikit, at the same payload size
through the same machinery, gets 0.89x (w8 SLOWER) and effP 2.33.

The one structural difference identified by direct code reading: the
control's drain loop is `future.result()` and NOTHING ELSE, whereas
paulikit's drain loop additionally runs ~21.5s of GIL-HELD work per
run (labeling + dict construction, measured in
`dag_v2_falsification_test.py`: d5 dict-build 58.4%, d4 labeling
28.5% of drain time at w8_c4).

Proposed mechanism: the parent process must run CPython bytecode to
service the pool's result pipe. While the parent holds the GIL doing
dict construction, it is NOT draining that pipe. The pipe backs up,
and workers BLOCK inside `multiprocessing.Queue.put()` (one shared
semaphore + lock + feeder thread for ALL workers - verified in
CPython's `concurrent.futures.process` / `multiprocessing.queues`
source). More workers => more of them blocked behind a parent that is
too busy to empty the pipe. This predicts the observed direction:
w8 WORSE than w2, workers holding work but not running it, all CPUs
~70% idle, and extra CPU-seconds burned without extra instructions.

WHAT THIS SCRIPT CHANGES: exactly one thing. It runs the SAME
`wht_large` task body and the SAME pool/drain structure as
`traffic_intensity_target.py`, with a `--drain-work` mode that adds
GIL-held dict construction to the drain loop, sized to paulikit's real
per-chunk term count. Payload, chunk count, pool shape, pinning, and
submission policy are all held identical between modes.

FALSIFIABLE PREDICTION (stated before running):
  If the mechanism is real, `wht_large` WITH drain work must lose its
  scaling - collapsing from ~2.23x toward paulikit's ~0.89x, i.e. w8
  becoming no faster (or slower) than w2.

  If `wht_large` WITH drain work still scales ~2x, the mechanism is
  FALSE: GIL-held drain work is not what breaks paulikit's scaling,
  and R3 as currently stated must be retracted.

NOT A DESIGN COMMITMENT. This is a diagnostic. It deliberately does
NOT propose or prototype any fix - per PLAN.md Phases 9/10, any real
change must preserve the cache-bound chunk working set (chunk_size
tuned so cs*dim*16B fits L2) and the streaming one-chunk-at-a-time
memory contract (peak RSS 208-627 MiB; materializing all 91.6M terms
would be ~2.9 GB of raw triples alone, regressing to the pre-Phase-10
failure).

Deliberately parameterized by n_workers/cpu_list rather than assuming
this machine's 4 physical / 8 logical CPUs - the R3 bottleneck is
expected to worsen with core count, so nothing here may hard-code a
small worker count.

Usage (foreground, one at a time):
    python drain_gil_backpressure_target.py <condition> <mode>
        condition: any key of condition_table.CONDITIONS (e.g. w2_c1)
        mode:      bare | drain_work
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condition_table import CONDITIONS as _CONDITIONS  # noqa: E402

from paulikit.algorithms.fwht import (  # noqa: E402
    _build_real_terms,
    _pauli_label_batch,
    _walsh_hadamard_transform_rows,
)

condition = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "bare"
if mode not in ("bare", "drain_work"):
    raise SystemExit(f"mode must be bare|drain_work, got {mode!r}")

# Matched to the real N=150/chunk_size=2 workload these are modeling.
DIM = 16384
CHUNK_ROWS = 2
N_CHUNKS = 5595
N_QUBITS = 14
# Real surviving terms per chunk: T_x / C = 91,652,096 / 5595.
TERMS_PER_CHUNK = 91_652_096 // 5595
ATOL = 1e-10


def _wht_large(seed: int) -> np.ndarray:
    """Byte-for-byte the same task body as
    traffic_intensity_target.py's `wht_large` - the control already
    measured at 2.23x scaling. Not modified here."""
    rng = np.random.default_rng(seed)
    buf = rng.standard_normal((CHUNK_ROWS, DIM)) + 1j * rng.standard_normal(
        (CHUNK_ROWS, DIM)
    )
    return _walsh_hadamard_transform_rows(buf, overwrite_input=True)


# Drain-side inputs are built ONCE, before timing, and reused for every
# chunk: the point is to reproduce paulikit's GIL-HELD drain COST, not
# to recompute its inputs. Building them per-chunk would add unrelated
# RNG/allocation work and confound the one variable under test.
_rng = np.random.default_rng(12345)
_X = _rng.integers(0, 2**N_QUBITS, TERMS_PER_CHUNK, dtype=np.uint32)
_Z = _rng.integers(0, 2**N_QUBITS, TERMS_PER_CHUNK, dtype=np.uint32)
_LABELS = _pauli_label_batch(_X, _Z, N_QUBITS)
_COEFFS = (_rng.standard_normal(TERMS_PER_CHUNK) + 0j).astype(complex)


def _drain_work() -> int:
    """The GIL-held per-chunk work paulikit's real drain loop does and
    the original control does not: label construction + dict build.

    Calls the REAL shipped `_build_real_terms`, not an imitation, so
    the GIL-holding profile matches production exactly. The returned
    dict is dropped immediately - one chunk's dict is live at a time,
    preserving the streaming memory contract this experiment must not
    violate.
    """
    labels = _pauli_label_batch(_X, _Z, N_QUBITS)
    return len(_build_real_terms(labels, _COEFFS, ATOL))


def _worker_init(cpu_list, next_pin_index):
    if not cpu_list:
        return
    with next_pin_index.get_lock():
        idx = next_pin_index.value
        next_pin_index.value += 1
    if idx < len(cpu_list):
        try:
            os.sched_setaffinity(0, {cpu_list[idx]})
        except (AttributeError, OSError):
            pass


n_workers, cpu_list = _CONDITIONS[condition]
next_pin_index = multiprocessing.Value("i", 0)

t0 = time.perf_counter()
n_done = 0
terms_seen = 0
with ProcessPoolExecutor(
    max_workers=n_workers,
    initializer=_worker_init,
    initargs=(cpu_list, next_pin_index),
) as pool:
    pending = iter(range(N_CHUNKS))
    in_flight: set = set()
    max_in_flight = max(1, 2 * n_workers)

    def _submit_next() -> bool:
        i = next(pending, None)
        if i is None:
            return False
        in_flight.add(pool.submit(_wht_large, i))
        return True

    for _ in range(max_in_flight):
        if not _submit_next():
            break

    while in_flight:
        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
            n_done += 1
            _submit_next()
            if mode == "drain_work":
                terms_seen += _drain_work()

elapsed = time.perf_counter() - t0
print(
    f"condition={condition} mode={mode} n_chunks={n_done} "
    f"terms={terms_seen} elapsed={elapsed:.4f}s"
)
