#!/usr/bin/env bash
# Runs the standard cache-locality perf-stat sweep across N=25/50/100/150
# using steady_state_decompose.py (warmed-up, in-process repeated
# timing) instead of one-shot CLI invocations, to avoid conflating
# process-startup cost with algorithm cost - see that script's
# docstring and n_scaling_findings.md / the N=25 perf-record
# localization that motivated this.
#
# LINUX-ONLY, NOT POSIX SH - same reasoning as run_baseline_perf_stat.sh's
# header: Linux perf_events subsystem + bash syntax. No macOS/BSD
# equivalent is attempted here.
#
# Per project convention (as of 2026-08-25): cache-locality analysis
# should cover N=25, N=50, N=100, N=150 by default.
#
# Usage:
#   ./run_steady_state_sweep.sh [N_RUNS_PER_N] [REPS_PER_RUN]
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
N_VALUES=(25 50 100 150)

if ! [[ "${N_RUNS_PER_N}" =~ ^[0-9]+$ ]] || [[ "${N_RUNS_PER_N}" -lt 1 ]]; then
    echo "error: N_RUNS_PER_N must be a positive integer, got '${N_RUNS_PER_N}'" >&2
    exit 1
fi
if ! [[ "${REPS_PER_RUN}" =~ ^[0-9]+$ ]] || [[ "${REPS_PER_RUN}" -lt 1 ]]; then
    echo "error: REPS_PER_RUN must be a positive integer, got '${REPS_PER_RUN}'" >&2
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
OUT_FILE="${SCRIPT_DIR}/steady_state_sweep_${STAMP}.txt"
TMP_FILE="$(mktemp "${SCRIPT_DIR}/.steady_state_sweep_${STAMP}.XXXXXX")"
trap 'rm -f "${TMP_FILE}"' EXIT

{
    echo "# Machine info"
    lscpu
    echo
    echo "# paulikit native extension check"
    if python -c "from paulikit._native import pauli_label_native; print('native OK:', pauli_label_native.__file__)"; then
        :
    else
        echo "native extension NOT available - results will reflect the pure-Python fallback path"
    fi
    echo
} > "${TMP_FILE}"

for n in "${N_VALUES[@]}"; do
    for i in $(seq 1 "${N_RUNS_PER_N}"); do
        echo "=== N=${n}, run ${i}/${N_RUNS_PER_N} ===" >> "${TMP_FILE}"
        if ! perf stat -e task-clock,cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
cycle_activity.stalls_total,cycle_activity.stalls_mem_any \
            python "${DRIVER}" --n-oscillators "${n}" --reps "${REPS_PER_RUN}" >> "${TMP_FILE}" 2>&1; then
            echo "error: N=${n} run ${i}/${N_RUNS_PER_N} failed - see ${TMP_FILE} for partial output" >&2
            trap - EXIT
            exit 1
        fi
        tail -n 20 "${TMP_FILE}"
    done
done

mv "${TMP_FILE}" "${OUT_FILE}"
trap - EXIT
echo "Results written to ${OUT_FILE}"
