"""Side-effect-free condition table shared by full_matrix_target.py
(the actual measurement target script, which reads sys.argv[1] at
import time and therefore cannot itself be safely imported by other
scripts) and full_optimum_sweep.py (the sweep driver, which needs the
config table WITHOUT triggering full_matrix_target.py's own
argv-parsing side effect - a real bug caught directly: importing
full_matrix_target from the sweep driver mistook the driver's own
--reps flag for a condition name).

Physical-core topology on this dev machine (checked directly, see
scoping.md): core A=(0,4), B=(1,5), C=(2,6), D=(3,7).
"""

# Each entry is (n_workers, explicit_cpu_list_or_None). None means
# unpinned (_physical_core_representative_cpus -> None, the same code
# path parallel_decompose already uses when pinning is genuinely
# unavailable). A list gives one CPU per worker, in claim order -
# workers 2+ on the SAME physical core as an earlier worker means that
# core is "doubled up" (both hyperthread siblings in use); a core
# never appearing is left completely idle.
CONDITIONS: dict[str, tuple[int, list[int] | None]] = {
    "seq_1": (0, None),  # special-cased by the caller (no pool at all)
    "pinned_2": (2, [0, 1]),
    "unpinned_2": (2, None),
    "pinned_4": (4, [0, 1, 2, 3]),
    "unpinned_4": (4, None),
    "workers_8": (8, [0, 1, 2, 3]),  # 8 logical CPUs, pinned default
    # 4-vs-2-physical-cores comparison (pinned4_4cores_vs_2cores_findings.md):
    "pinned_4_4cores": (4, [0, 1, 2, 3]),
    "pinned_4_2cores": (4, [0, 4, 1, 5]),
    # 2-vs-1-physical-core comparison:
    "pinned_2_2cores": (2, [0, 1]),
    "pinned_2_1core": (2, [0, 4]),
    # 3-vs-2-physical-cores comparison (2+1 packing on the 2-core side):
    "pinned_3_3cores": (3, [0, 1, 2]),
    "pinned_3_2cores": (3, [0, 4, 1]),
    # 5-vs-4-vs-3-physical-cores comparison (2+1+1+1, then 2+2+1 packing):
    "pinned_5_4cores": (5, [0, 4, 1, 2, 3]),
    "pinned_5_3cores": (5, [0, 4, 1, 5, 2]),
}

# Full enumeration for the publication-grade sweep
# (full_optimum_sweep_findings.md): every DISTINCT valid (n_workers,
# n_physical_cores_used) configuration on this machine (4 physical
# cores, 2 hyperthreads each - a core hosts at most 2 workers). Named
# w<n_workers>_c<n_cores> for unambiguous, systematic identification.
_CORE_PAIRS = [(0, 4), (1, 5), (2, 6), (3, 7)]  # physical cores A,B,C,D


def _packed_cpu_list(n_workers: int, n_cores: int) -> list[int]:
    """One representative logical CPU per worker, spread as EVENLY as
    possible across exactly n_cores DISTINCT physical cores (using
    both hyperthread siblings of a core only when n_workers exceeds
    n_cores) - e.g. (4 workers, 3 cores) -> 2+1+1, not 2+2+(0 workers
    on a 3rd core). This is the only assignment that actually uses
    n_cores distinct cores for every (n_workers, n_cores) pair in
    range - a naive "pack cores 0..n_cores-1 to 2 each, ignore the
    rest" would silently collapse several distinct n_cores values onto
    the same CPU list (a real bug caught before running anything: an
    earlier version of this function packed greedily regardless of
    n_cores, producing several duplicate CPU-list assignments under
    different names - fixed here; the correct total enumeration is 14
    distinct configurations, not the 17 first guessed before deriving
    it properly - see full_optimum_sweep_findings.md)."""
    assert 1 <= n_cores <= 4
    assert n_cores <= n_workers <= 2 * n_cores
    base, extra = divmod(n_workers, n_cores)
    cpus = []
    for core_idx in range(n_cores):
        take = base + (1 if core_idx < extra else 0)
        cpus.extend(_CORE_PAIRS[core_idx][:take])
    assert len(cpus) == n_workers
    return cpus


SWEEP_CONFIGS: dict[str, tuple[int, list[int]]] = {}
for _n_workers in range(1, 9):
    _min_cores = (_n_workers + 1) // 2  # ceil(n_workers / 2)
    _max_cores = min(_n_workers, 4)
    for _n_cores in range(_min_cores, _max_cores + 1):
        SWEEP_CONFIGS[f"w{_n_workers}_c{_n_cores}"] = (
            _n_workers, _packed_cpu_list(_n_workers, _n_cores)
        )

CONDITIONS.update(SWEEP_CONFIGS)
