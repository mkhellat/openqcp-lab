#!/usr/bin/env bash
# Reproduces perf_record_n50_findings.md's localization data.
#
# LINUX-ONLY, NOT POSIX SH - same reasoning as run_baseline_perf_stat.sh's
# header: this uses Linux's `perf_events` subsystem specifically, plus
# bash syntax. No macOS/BSD equivalent is attempted here.
#
# Requires the same prerequisites as run_baseline_perf_stat.sh, plus
# enough disk space in the output directory for perf.data (~1-5 MB
# at N=50; scales with N).
#
# Usage:
#   ./run_perf_record_localize.sh [N_OSCILLATORS]
# Default: N_OSCILLATORS=50 (matches the committed findings).
#
# perf.data is NOT committed to the repo (large, machine-specific
# binary format) - this script regenerates it locally and produces a
# human-readable report alongside it.
#
# Fails loudly and exits nonzero on any precondition or command
# failure - never leaves a stale/partial perf.data or report behind
# under the final expected filename.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this script uses Linux 'perf' - not supported on $(uname -s)" >&2
    exit 1
fi

N_OSCILLATORS="${1:-50}"
if ! [[ "${N_OSCILLATORS}" =~ ^[0-9]+$ ]] || [[ "${N_OSCILLATORS}" -lt 1 ]]; then
    echo "error: N_OSCILLATORS must be a positive integer, got '${N_OSCILLATORS}'" >&2
    exit 1
fi

command -v paulikit >/dev/null 2>&1 || {
    echo "error: 'paulikit' not found on PATH - install it first" \
        "(pip install -e . --no-build-isolation from the paulikit/ directory)" >&2
    exit 1
}
command -v perf >/dev/null 2>&1 || {
    echo "error: 'perf' not found - install linux-tools/perf (Debian/Ubuntu)" \
        "or 'perf' (Arch/Fedora) for your distro" >&2
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

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -w "${OUT_DIR}" ]]; then
    echo "error: output directory is not writable: ${OUT_DIR}" >&2
    exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DATA_FILE="${OUT_DIR}/perf_cachemiss_n${N_OSCILLATORS}_${STAMP}.data"
REPORT_FILE="${OUT_DIR}/perf_report_n${N_OSCILLATORS}_${STAMP}.txt"

cleanup_on_failure() {
    rm -f "${DATA_FILE}" "${DATA_FILE}.old" "${REPORT_FILE}"
}
trap cleanup_on_failure ERR

perf record -g -e cache-misses -o "${DATA_FILE}" -- \
    paulikit decompose --n-oscillators "${N_OSCILLATORS}"

if [[ ! -s "${DATA_FILE}" ]]; then
    echo "error: perf record did not produce a nonempty data file at ${DATA_FILE}" >&2
    exit 1
fi

{
    echo "=== Flat self-time-by-symbol report (top cache-miss sources) ==="
    perf report -i "${DATA_FILE}" --stdio --sort=symbol -g none --percent-limit=1
} > "${REPORT_FILE}"

trap - ERR

cat "${REPORT_FILE}"
echo
echo "Raw data: ${DATA_FILE}"
echo "Report:   ${REPORT_FILE}"
echo
echo "For the call-graph view instead of the flat view, run:"
echo "  perf report -i ${DATA_FILE} --stdio --sort=comm,dso,symbol --percent-limit=1"
