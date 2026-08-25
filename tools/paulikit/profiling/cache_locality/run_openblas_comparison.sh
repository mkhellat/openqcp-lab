#!/usr/bin/env bash
# Reproduces stall_floor_mystery_solved.md's OPENBLAS_NUM_THREADS
# comparison: runs the steady-state driver once with the environment
# as-is and once with OPENBLAS_NUM_THREADS=1, under the same perf
# event set, so a reader can directly see how much of
# cycle_activity.stalls_total (and total cycles) is OpenBLAS
# thread-pool noise versus paulikit's own work.
#
# LINUX-ONLY, NOT POSIX SH - same reasoning as the other scripts in
# this directory: Linux perf_events subsystem + bash syntax. No
# macOS/BSD equivalent is attempted here.
#
# Usage:
#   ./run_openblas_comparison.sh [N_OSCILLATORS] [REPS]
# Defaults: N_OSCILLATORS=25, REPS=5 (matches the committed findings -
# N=25 was chosen there because the dense coefficients array fits
# comfortably in L3 at that size, ruling it out as an alternative
# explanation for the stall-cycle noise being measured here).
#
# Fails loudly and exits nonzero on any precondition or command
# failure - never leaves a partial/misleading result file.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this script uses Linux 'perf' - not supported on $(uname -s)" >&2
    exit 1
fi

N_OSCILLATORS="${1:-25}"
REPS="${2:-5}"

if ! [[ "${N_OSCILLATORS}" =~ ^[0-9]+$ ]] || [[ "${N_OSCILLATORS}" -lt 1 ]]; then
    echo "error: N_OSCILLATORS must be a positive integer, got '${N_OSCILLATORS}'" >&2
    exit 1
fi
if ! [[ "${REPS}" =~ ^[0-9]+$ ]] || [[ "${REPS}" -lt 1 ]]; then
    echo "error: REPS must be a positive integer, got '${REPS}'" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/steady_state_decompose.py"

command -v python >/dev/null 2>&1 || {
    echo "error: 'python' not found on PATH" >&2
    exit 1
}
command -v perf >/dev/null 2>&1 || {
    echo "error: 'perf' not found - install linux-tools/perf (Debian/Ubuntu)" \
        "or 'perf' (Arch/Fedora) for your distro" >&2
    exit 1
}
[[ -f "${DRIVER}" ]] || {
    echo "error: driver script not found: ${DRIVER}" >&2
    exit 1
}
python -c "import paulikit" 2>/dev/null || {
    echo "error: paulikit is not importable - install it first" \
        "(pip install -e . --no-build-isolation from the paulikit/ directory)" >&2
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
OUT_FILE="${SCRIPT_DIR}/openblas_comparison_n${N_OSCILLATORS}_${STAMP}.txt"
TMP_FILE="$(mktemp "${SCRIPT_DIR}/.openblas_comparison_n${N_OSCILLATORS}_${STAMP}.XXXXXX")"
trap 'rm -f "${TMP_FILE}"' EXIT

EVENTS="cycles,cycle_activity.stalls_total,cycle_activity.stalls_mem_any"

{
    echo "# Machine info"
    lscpu
    echo
} > "${TMP_FILE}"

echo "=== baseline (OPENBLAS_NUM_THREADS unset) ===" >> "${TMP_FILE}"
if ! perf stat -e "${EVENTS}" \
    python "${DRIVER}" --n-oscillators "${N_OSCILLATORS}" --reps "${REPS}" >> "${TMP_FILE}" 2>&1; then
    echo "error: baseline run failed - see ${TMP_FILE} for partial output" >&2
    trap - EXIT
    exit 1
fi
tail -n 15 "${TMP_FILE}"

echo "=== OPENBLAS_NUM_THREADS=1 ===" >> "${TMP_FILE}"
if ! OPENBLAS_NUM_THREADS=1 perf stat -e "${EVENTS}" \
    python "${DRIVER}" --n-oscillators "${N_OSCILLATORS}" --reps "${REPS}" >> "${TMP_FILE}" 2>&1; then
    echo "error: OPENBLAS_NUM_THREADS=1 run failed - see ${TMP_FILE} for partial output" >&2
    trap - EXIT
    exit 1
fi
tail -n 15 "${TMP_FILE}"

mv "${TMP_FILE}" "${OUT_FILE}"
trap - EXIT
echo "Results written to ${OUT_FILE}"
