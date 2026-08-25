#!/usr/bin/env bash
# Compares the serial label-generation kernel (steady_state_decompose.py,
# what fwht_pauli_terms actually calls today) against the TBB-parallel
# one (steady_state_decompose_tbb.py, monkeypatched in-process only -
# see that script's docstring) under the same perf event set, at each
# of N=25/50/100, so a reader can see whether TBB parallelization of
# the label-generation loop changes cache-miss ratio or stall cycles,
# not just wall-clock time (which Phase 3a already measured and found
# "barely helps").
#
# OPENBLAS_NUM_THREADS=1 is set for every run here, per
# stall_floor_mystery_solved.md's finding that OpenBLAS's own
# thread-pool otherwise dominates cycle/stall counters as noise
# unrelated to either kernel being compared.
#
# N=150 is deliberately NOT included: n150_oom_finding.md already
# established the unmodified code OOM-kills at that size regardless of
# which label-generation kernel is used (the OOM happens in
# fwht_pauli_coefficients's dense-array allocation, upstream of label
# generation entirely) - re-attempting it here would just reproduce a
# known crash, not new information.
#
# LINUX-ONLY, NOT POSIX SH - same reasoning as the other scripts in
# this directory: Linux perf_events subsystem + bash syntax. No
# macOS/BSD equivalent is attempted here.
#
# Usage:
#   ./run_tbb_comparison.sh [N_RUNS_PER_N] [REPS_PER_RUN]
# Defaults: N_RUNS_PER_N=3, REPS_PER_RUN=5.
#
# Fails loudly and exits nonzero on any precondition or command
# failure - never leaves a partial/misleading result file.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this script uses Linux 'perf' - not supported on $(uname -s)" >&2
    exit 1
fi

N_RUNS_PER_N="${1:-3}"
REPS_PER_RUN="${2:-5}"
N_VALUES=(25 50 100)

if ! [[ "${N_RUNS_PER_N}" =~ ^[0-9]+$ ]] || [[ "${N_RUNS_PER_N}" -lt 1 ]]; then
    echo "error: N_RUNS_PER_N must be a positive integer, got '${N_RUNS_PER_N}'" >&2
    exit 1
fi
if ! [[ "${REPS_PER_RUN}" =~ ^[0-9]+$ ]] || [[ "${REPS_PER_RUN}" -lt 1 ]]; then
    echo "error: REPS_PER_RUN must be a positive integer, got '${REPS_PER_RUN}'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERIAL_DRIVER="${SCRIPT_DIR}/steady_state_decompose.py"
TBB_DRIVER="${SCRIPT_DIR}/steady_state_decompose_tbb.py"

command -v python >/dev/null 2>&1 || {
    echo "error: 'python' not found on PATH" >&2
    exit 1
}
command -v perf >/dev/null 2>&1 || {
    echo "error: 'perf' not found - install linux-tools/perf (Debian/Ubuntu)" \
        "or 'perf' (Arch/Fedora) for your distro" >&2
    exit 1
}
[[ -f "${SERIAL_DRIVER}" ]] || {
    echo "error: driver script not found: ${SERIAL_DRIVER}" >&2
    exit 1
}
[[ -f "${TBB_DRIVER}" ]] || {
    echo "error: driver script not found: ${TBB_DRIVER}" >&2
    exit 1
}
python -c "import paulikit" 2>/dev/null || {
    echo "error: paulikit is not importable - install it first" \
        "(pip install -e . --no-build-isolation from the paulikit/ directory)" >&2
    exit 1
}
python -c "from paulikit._native import pauli_label_native" 2>/dev/null || {
    echo "error: paulikit's native extension is not importable - this comparison" \
        "requires it (to exercise the TBB path). Rebuild with the native" \
        "extension enabled first." >&2
    exit 1
}

if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
    paranoid_level="$(cat /proc/sys/kernel/perf_event_paranoid)"
    if [[ "${paranoid_level}" -gt 1 ]] && [[ "${EUID}" -ne 0 ]]; then
        echo "error: /proc/sys/kernel/perf_event_paranoid is ${paranoid_level}" \
            "(needs <=1 for unprivileged hardware-counter access, or run as root)." >&2
        echo "  fix: sudo sysctl kernel.perf_event_paranoid=1" >&2
        exit 1
    fi
fi

if [[ ! -w "${SCRIPT_DIR}" ]]; then
    echo "error: output directory is not writable: ${SCRIPT_DIR}" >&2
    exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${SCRIPT_DIR}/tbb_comparison_${STAMP}.txt"
TMP_FILE="$(mktemp "${SCRIPT_DIR}/.tbb_comparison_${STAMP}.XXXXXX")"
trap 'rm -f "${TMP_FILE}"' EXIT

EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
cycle_activity.stalls_total,cycle_activity.stalls_mem_any"

{
    echo "# Machine info"
    lscpu
    echo
    echo "# OPENBLAS_NUM_THREADS=1 set for every run below (see stall_floor_mystery_solved.md)"
    echo
} > "${TMP_FILE}"

for n in "${N_VALUES[@]}"; do
    for i in $(seq 1 "${N_RUNS_PER_N}"); do
        echo "=== N=${n}, run ${i}/${N_RUNS_PER_N}, SERIAL kernel ===" >> "${TMP_FILE}"
        if ! OPENBLAS_NUM_THREADS=1 perf stat -e "${EVENTS}" \
            python "${SERIAL_DRIVER}" --n-oscillators "${n}" --reps "${REPS_PER_RUN}" >> "${TMP_FILE}" 2>&1; then
            echo "error: N=${n} run ${i}/${N_RUNS_PER_N} (serial) failed - see ${TMP_FILE} for partial output" >&2
            trap - EXIT
            exit 1
        fi
        tail -n 22 "${TMP_FILE}"

        echo "=== N=${n}, run ${i}/${N_RUNS_PER_N}, TBB kernel ===" >> "${TMP_FILE}"
        if ! OPENBLAS_NUM_THREADS=1 perf stat -e "${EVENTS}" \
            python "${TBB_DRIVER}" --n-oscillators "${n}" --reps "${REPS_PER_RUN}" >> "${TMP_FILE}" 2>&1; then
            echo "error: N=${n} run ${i}/${N_RUNS_PER_N} (TBB) failed - see ${TMP_FILE} for partial output" >&2
            trap - EXIT
            exit 1
        fi
        tail -n 22 "${TMP_FILE}"
    done
done

mv "${TMP_FILE}" "${OUT_FILE}"
trap - EXIT
echo "Results written to ${OUT_FILE}"
