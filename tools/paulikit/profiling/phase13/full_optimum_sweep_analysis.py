"""Joint statistical analysis of the full 14-config x 10-rep optimum
sweep (full_optimum_sweep_results.jsonl) - the analysis this whole
sweep was commissioned to provide: one genuinely joint, properly-
powered comparison across every (n_workers, n_cores) configuration,
not a series of separate pairwise Welch's t-tests (which was the
user's explicit, repeated methodological objection to every earlier
iteration of this investigation).

Method:
  1. One-way ANOVA (scipy.stats.f_oneway) on wall-clock `elapsed`
     across all 14 conditions - tests whether configuration matters
     at all, jointly, with one p-value (not 91 separate pairwise
     tests' worth of multiple-comparisons risk).
  2. Tukey HSD post-hoc (implemented directly via
     scipy.stats.studentized_range - statsmodels is not an existing
     project dependency and was not added for one script) - identifies
     WHICH pairs of the 14 conditions differ significantly, at a
     single family-wise alpha=0.05, correctly controlling for running
     91 pairwise comparisons at once (the exact problem repeated
     individual Welch's t-tests do not control for).
  3. Ranks all 14 conditions by mean wall-clock and reports which are
     statistically indistinguishable from the observed minimum (i.e.
     which configurations the data cannot actually tell apart from the
     best one, honestly, rather than reporting a single point-estimate
     "winner").

Usage:
    python full_optimum_sweep_analysis.py
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "full_optimum_sweep_results.jsonl")

CONFIG_ORDER = [
    "w1_c1", "w2_c1", "w2_c2", "w3_c2", "w3_c3", "w4_c2", "w4_c3",
    "w4_c4", "w5_c3", "w5_c4", "w6_c3", "w6_c4", "w7_c4", "w8_c4",
]


def load() -> dict[str, list[float]]:
    by_condition = defaultdict(list)
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_condition[rec["condition"]].append(rec["elapsed"])
    return by_condition


def tukey_hsd(groups: dict[str, np.ndarray], alpha: float = 0.05):
    """Direct Tukey HSD implementation (no statsmodels dependency).

    Standard textbook formula: for equal-n groups (true here, n=10
    every condition), the HSD critical difference is
        q_crit * sqrt(MSE / n)
    where q_crit is the alpha-level critical value of the studentized
    range distribution with k groups and (N-k) error degrees of
    freedom, and MSE is the pooled within-group variance from the
    one-way ANOVA.
    """
    names = list(groups.keys())
    k = len(names)
    n = len(next(iter(groups.values())))
    assert all(len(v) == n for v in groups.values()), "unequal n breaks the equal-n HSD formula used here"
    N = k * n

    grand_mean = np.mean([v for arr in groups.values() for v in arr])
    ss_within = sum(np.sum((np.asarray(arr) - np.mean(arr)) ** 2) for arr in groups.values())
    df_within = N - k
    mse = ss_within / df_within

    q_crit = stats.studentized_range.ppf(1 - alpha, k, df_within)
    hsd_crit_diff = q_crit * np.sqrt(mse / n)

    rows = []
    for a, b in itertools.combinations(names, 2):
        mean_a, mean_b = np.mean(groups[a]), np.mean(groups[b])
        diff = mean_a - mean_b
        significant = abs(diff) > hsd_crit_diff
        rows.append((a, b, mean_a, mean_b, diff, significant))
    return rows, hsd_crit_diff, mse, df_within, q_crit


def main():
    by_condition = load()
    missing = [c for c in CONFIG_ORDER if c not in by_condition]
    if missing:
        raise SystemExit(f"sweep incomplete, missing configs: {missing}")
    for c in CONFIG_ORDER:
        n = len(by_condition[c])
        if n != 10:
            raise SystemExit(f"{c} has {n} reps, expected 10 - sweep incomplete or corrupted")

    groups = {c: np.array(by_condition[c]) for c in CONFIG_ORDER}

    print("=" * 78)
    print("Per-condition wall-clock summary (n=10 each)")
    print("=" * 78)
    summary = []
    for c in CONFIG_ORDER:
        arr = groups[c]
        n_workers = int(c.split("_")[0][1:])
        n_cores = int(c.split("_")[1][1:])
        summary.append((c, n_workers, n_cores, arr.mean(), arr.std(ddof=1)))
        print(f"{c:8s} n_workers={n_workers} n_cores={n_cores}  "
              f"mean={arr.mean():7.3f}s  sd={arr.std(ddof=1):6.3f}s")

    print()
    print("=" * 78)
    print("One-way ANOVA across all 14 conditions (wall-clock)")
    print("=" * 78)
    f_stat, p_value = stats.f_oneway(*[groups[c] for c in CONFIG_ORDER])
    print(f"F({len(CONFIG_ORDER) - 1}, {14 * 10 - 14}) = {f_stat:.4f}, p = {p_value:.6g}")
    if p_value < 0.05:
        print("=> configuration has a statistically significant joint effect on wall-clock.")
    else:
        print("=> NO statistically significant joint effect detected across configurations.")

    print()
    print("=" * 78)
    print("Tukey HSD post-hoc (family-wise alpha=0.05, all 91 pairs)")
    print("=" * 78)
    rows, hsd_crit_diff, mse, df_within, q_crit = tukey_hsd(groups)
    print(f"MSE={mse:.5f}  df_within={df_within}  q_crit={q_crit:.4f}  "
          f"HSD critical difference = {hsd_crit_diff:.4f}s")
    print()
    sig_rows = [r for r in rows if r[5]]
    print(f"{len(sig_rows)} of {len(rows)} pairs significantly different at alpha=0.05:")
    for a, b, ma, mb, diff, sig in sorted(rows, key=lambda r: -abs(r[4])):
        marker = "*" if sig else " "
        print(f"  {marker} {a:8s} ({ma:7.3f}s) vs {b:8s} ({mb:7.3f}s)  "
              f"diff={diff:+7.3f}s")

    print()
    print("=" * 78)
    print("Ranking and honest 'indistinguishable from the minimum' set")
    print("=" * 78)
    summary.sort(key=lambda r: r[3])
    best_condition = summary[0][0]
    best_mean = summary[0][3]
    print(f"Observed minimum mean wall-clock: {best_condition} ({best_mean:.3f}s)")
    print()
    print("Rank | config   | n_workers | n_cores | mean(s) | sd(s) | "
          "diff-from-best(s) | stat. distinguishable from best?")
    for rank, (c, nw, nc, mean, sd) in enumerate(summary, start=1):
        if c == best_condition:
            distinguishable = "-- (is the best) --"
        else:
            match = next(r for r in rows if {r[0], r[1]} == {c, best_condition})
            distinguishable = "YES (worse)" if match[5] else "no (statistically tied)"
        print(f"{rank:4d} | {c:8s} | {nw:9d} | {nc:7d} | {mean:7.3f} | {sd:5.3f} | "
              f"{mean - best_mean:+7.3f}          | {distinguishable}")


if __name__ == "__main__":
    main()
