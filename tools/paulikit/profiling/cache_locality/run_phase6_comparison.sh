#!/usr/bin/env bash
# Compares the OLD dense-array-plus-re-scan code path
# (steady_state_decompose_dense.py, replicating fwht_pauli_terms's
# pre-Phase-6 body in full) against the NEW sparse path
# (steady_state_decompose.py, what fwht_pauli_terms actually does
# today) under the same perf event set, at each of N=25/50/100/150 -
# the actual A/B measurement PLAN.md's Phase 6 plan (step 2/4) requires
# before claiming the fix helped, per the user's explicit instruction
# not to assume "sparse is obviously better."
#
# N=150 IS included here (unlike run_tbb_comparison.sh, which excluded
# it since the code under test there was unmodified and already known
# to OOM): the whole point of this comparison is to verify whether
# Phase 6's fix actually resolves n150_oom_finding.md's OOM. The dense
# leg is expected to still OOM (or come close) - that's the baseline
# this fix is measured against, not a script bug.
#
# OPENBLAS_NUM_THREADS=1 is set for every run here, per
# stall_floor_mystery_solved.md's finding that OpenBLAS's own
# thread-pool otherwise dominates cycle/stall counters as noise
# unrelated to either code path being compared.
#
# LINUX-ONLY, NOT POSIX SH - same reasoning as the other scripts in
# this directory: Linux perf_events subsystem + bash syntax. No
# macOS/BSD equivalent is attempted here.
#
# Usage:
#   ./run_phase6_comparison.sh [N_RUNS_PER_N] [REPS_PER_RUN] [N_VALUES]
# Defaults: N_RUNS_PER_N=3, REPS_PER_RUN=5, N_VALUES="25 50 100 150".
#
# N_VALUES is a space-separated, quoted list, e.g.:
#   ./run_phase6_comparison.sh 3 5 "25 50 100"
# Given how close N=150's dense leg came to destabilizing the dev
# machine (2026-08-26 - swap exhaustion while a second heavy job ran
# concurrently, not the dense leg alone; see
# n150_oom_finding.md/README.md's swap/resource-exhaustion notes), the
# recommended reproduction is to run N=150 SEPARATELY, alone, with
# nothing else memory-heavy running concurrently, and with a reduced
# REPS_PER_RUN (e.g. 1) for that call specifically -
#   ./run_phase6_comparison.sh 1 1 "150"
# - rather than as part of the same invocation as the smaller, safe
# N values. This override exists specifically to make that split
# reproducible via a real command, not ad hoc inline shell.
#
# Each N/kernel/run's exit code is captured individually rather than
# under `set -e` for the perf invocations themselves - an OOM kill on
# the dense leg at N=150 is an expected, informative outcome (matching
# n150_oom_finding.md), not a script failure, so the sweep continues
# and records it rather than aborting the whole comparison.

set -uo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: this script uses Linux 'perf' - not supported on $(uname -s)" >&2
    exit 1
fi

N_RUNS_PER_N="${1:-3}"
REPS_PER_RUN="${2:-5}"
read -r -a N_VALUES <<< "${3:-25 50 100 150}"

if ! [[ "${N_RUNS_PER_N}" =~ ^[0-9]+$ ]] || [[ "${N_RUNS_PER_N}" -lt 1 ]]; then
    echo "error: N_RUNS_PER_N must be a positive integer, got '${N_RUNS_PER_N}'" >&2
    exit 1
fi
if ! [[ "${REPS_PER_RUN}" =~ ^[0-9]+$ ]] || [[ "${REPS_PER_RUN}" -lt 1 ]]; then
    echo "error: REPS_PER_RUN must be a positive integer, got '${REPS_PER_RUN}'" >&2
    exit 1
fi
if [[ "${#N_VALUES[@]}" -eq 0 ]]; then
    echo "error: N_VALUES (3rd argument) parsed to an empty list" >&2
    exit 1
fi
for n in "${N_VALUES[@]}"; do
    if ! [[ "${n}" =~ ^[0-9]+$ ]] || [[ "${n}" -lt 1 ]]; then
        echo "error: each value in N_VALUES must be a positive integer, got '${n}'" >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENSE_DRIVER="${SCRIPT_DIR}/steady_state_decompose_dense.py"
SPARSE_DRIVER="${SCRIPT_DIR}/steady_state_decompose.py"

command -v python >/dev/null 2>&1 || {
    echo "error: 'python' not found on PATH" >&2
    exit 1
}
command -v perf >/dev/null 2>&1 || {
    echo "error: 'perf' not found - install linux-tools/perf (Debian/Ubuntu)" \
        "or 'perf' (Arch/Fedora) for your distro" >&2
    exit 1
}
[[ -f "${DENSE_DRIVER}" ]] || {
    echo "error: driver script not found: ${DENSE_DRIVER}" >&2
    exit 1
}
[[ -f "${SPARSE_DRIVER}" ]] || {
    echo "error: driver script not found: ${SPARSE_DRIVER}" >&2
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
OUT_FILE="${SCRIPT_DIR}/phase6_comparison_${STAMP}.txt"
TMP_FILE="$(mktemp "${SCRIPT_DIR}/.phase6_comparison_${STAMP}.XXXXXX")"
trap 'rm -f "${TMP_FILE}"' EXIT

EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,\
cycle_activity.stalls_total,cycle_activity.stalls_mem_any"

{
    echo "# Machine info"
    lscpu
    echo
    echo "# OPENBLAS_NUM_THREADS=1 set for every run below (see stall_floor_mystery_solved.md)"
    echo "# N=150 dense leg may OOM-kill (expected, matches n150_oom_finding.md - see comment"
    echo "# in this script) - a nonzero exit there is recorded, not treated as a script error."
    echo
} > "${TMP_FILE}"

for n in "${N_VALUES[@]}"; do
    for i in $(seq 1 "${N_RUNS_PER_N}"); do
        echo "=== N=${n}, run ${i}/${N_RUNS_PER_N}, DENSE (pre-Phase-6) ===" >> "${TMP_FILE}"
        OPENBLAS_NUM_THREADS=1 perf stat -e "${EVENTS}" \
            python "${DENSE_DRIVER}" --n-oscillators "${n}" --reps "${REPS_PER_RUN}" \
            >> "${TMP_FILE}" 2>&1
        dense_status=$?
        if [[ ${dense_status} -ne 0 ]]; then
            echo "[dense leg exited ${dense_status} - see output above; likely OOM at large N]" \
                >> "${TMP_FILE}"
        fi
        tail -n 24 "${TMP_FILE}"

        echo "=== N=${n}, run ${i}/${N_RUNS_PER_N}, SPARSE (Phase 6 fix) ===" >> "${TMP_FILE}"
        OPENBLAS_NUM_THREADS=1 perf stat -e "${EVENTS}" \
            python "${SPARSE_DRIVER}" --n-oscillators "${n}" --reps "${REPS_PER_RUN}" \
            >> "${TMP_FILE}" 2>&1
        sparse_status=$?
        if [[ ${sparse_status} -ne 0 ]]; then
            echo "error: N=${n} run ${i}/${N_RUNS_PER_N} (sparse/Phase-6 fix) failed" \
                "unexpectedly - see ${TMP_FILE} for partial output" >&2
            trap - EXIT
            exit 1
        fi
        tail -n 24 "${TMP_FILE}"
    done
done

mv "${TMP_FILE}" "${OUT_FILE}"
trap - EXIT
echo "Results written to ${OUT_FILE}"
