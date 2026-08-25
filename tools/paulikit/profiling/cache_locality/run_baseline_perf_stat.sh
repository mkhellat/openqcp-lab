#!/usr/bin/env bash
# Reproduces baseline_perf_stat.md's measurements.
#
# LINUX-ONLY, NOT POSIX SH. This script uses Linux's `perf_events`
# subsystem (the `perf` command, /proc/sys/kernel/perf_event_paranoid,
# /proc/cpuinfo via `lscpu`) which has no equivalent on Darwin/BSD -
# those platforms would need a genuinely different tool (dtrace,
# Instruments) and a different invocation, not just a syntax port. It
# also uses bash-specific syntax ([[ ]], ${BASH_SOURCE[0]}, EUID,
# `set -o pipefail`), so it won't run under a strict POSIX /bin/sh
# either. A macOS/BSD equivalent is tracked as a separate, future item
# - not attempted here.
#
# Requires: paulikit installed (with native extension built - see
# ../../README.md's "Native extension" section) and Linux `perf`
# with access to hardware performance counters (may need
# `sudo sysctl kernel.perf_event_paranoid=1` or running as root,
# depending on your system's default - checked below, not just
# documented).
#
# Usage:
#   ./run_baseline_perf_stat.sh [N_OSCILLATORS] [N_RUNS]
# Defaults: N_OSCILLATORS=50, N_RUNS=3 (matches the committed results).
#
# Fails loudly and exits nonzero on any precondition or command
# failure - never silently produces a partial or misleading result
# file.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this script uses Linux 'perf' and /proc - not supported on $(uname -s)" >&2
    exit 1
fi

N_OSCILLATORS="${1:-50}"
N_RUNS="${2:-3}"

if ! [[ "${N_OSCILLATORS}" =~ ^[0-9]+$ ]] || [[ "${N_OSCILLATORS}" -lt 1 ]]; then
    echo "error: N_OSCILLATORS must be a positive integer, got '${N_OSCILLATORS}'" >&2
    exit 1
fi
if ! [[ "${N_RUNS}" =~ ^[0-9]+$ ]] || [[ "${N_RUNS}" -lt 1 ]]; then
    echo "error: N_RUNS must be a positive integer, got '${N_RUNS}'" >&2
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
command -v python >/dev/null 2>&1 || {
    echo "error: 'python' not found on PATH" >&2
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
OUT_FILE="${OUT_DIR}/perf_stat_n${N_OSCILLATORS}_${STAMP}.txt"
TMP_FILE="$(mktemp "${OUT_DIR}/.perf_stat_n${N_OSCILLATORS}_${STAMP}.XXXXXX")"
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

for i in $(seq 1 "${N_RUNS}"); do
    echo "=== run ${i}/${N_RUNS} ===" >> "${TMP_FILE}"
    # perf stat writes its report to stderr; redirect explicitly rather
    # than piping through tee, so a perf/paulikit failure (nonzero exit)
    # is caught by `set -e` instead of being masked by tee's own exit code.
    if ! perf stat -e task-clock,cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
        paulikit decompose --n-oscillators "${N_OSCILLATORS}" >> "${TMP_FILE}" 2>&1; then
        echo "error: run ${i}/${N_RUNS} failed - see ${TMP_FILE} for partial output" >&2
        echo "  (not renamed to ${OUT_FILE} since the run set is incomplete)" >&2
        trap - EXIT
        exit 1
    fi
    tail -n 20 "${TMP_FILE}"
done

mv "${TMP_FILE}" "${OUT_FILE}"
trap - EXIT
echo "Results written to ${OUT_FILE}"
