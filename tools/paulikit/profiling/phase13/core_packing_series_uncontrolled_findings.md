# Core-packing series (pinned_2/3/5): raw uncontrolled data, superseded by a thermally-controlled re-run

Recorded 2026-09-03. Direct follow-up to `pinned4_4cores_vs_2cores_findings.md`,
extending the same "N workers packed onto fewer physical cores vs.
spread across more" comparison to `n_workers` in {2, 3, 5}, per direct
user request, with core-assignment patterns specified directly:

- **`pinned_2`**: 2 workers on 2 distinct cores (`0,1`) vs. 2 workers
  on 1 core, both hyperthread siblings (`0,4`).
- **`pinned_3`**: 3 workers on 3 distinct cores (`0,1,2`) vs. 3 workers
  on 2 cores, 2+1 packing (`0,4,1`).
- **`pinned_5`**: 5 workers on 4 cores, 2+1+1+1 packing (`0,4,1,2,3`)
  vs. 5 workers on 3 cores, 2+2+1 packing (`0,4,1,5,2`).

All three verified for correct pinning beforehand via
`verify_asymmetric_pinning.py` (self-reported `os.sched_getaffinity`
from each real forked worker) - all exact matches to intent.

**IMPORTANT CAVEAT, found and flagged by the user mid-run**: this run
was performed WITHOUT the cooldown-between-reps protocol
(`thermal_controlled_welch_ttest.py`'s own discipline, built earlier
in this same investigation specifically because uncontrolled thermal
history between back-to-back runs was shown to meaningfully affect
wall-clock results). **This document's numbers should be treated as
provisional** - see `core_packing_series_thermal_controlled_findings.md`
for the corrected re-run. Recorded here anyway per this project's own
"always record raw data" discipline, not silently discarded.

## Raw data and results, AS MEASURED (uncontrolled)

### pinned_2: 2cores vs 1core

pinned_2_2cores: elapsed=[26.2347, 27.9133, 27.4554, 28.2183, 27.5758];
cycles=[67344447206, 64903038354, 66374197190, 67089108446, 66269026231];
cache_miss_ratio=[34.270, 35.334, 34.651, 35.310, 35.052];
llc_miss_ratio=[20.189, 21.263, 20.250, 20.531, 20.541];
peak_rss_mib=[263.2, 261.5, 262.7, 261.8, 262.2]

pinned_2_1core: elapsed=[25.4891, 26.2077, 25.3839, 25.1092, 25.7058];
cycles=[59267612003, 58824544019, 59284919007, 58216041612, 59359568541];
cache_miss_ratio=[36.297, 36.924, 36.442, 36.849, 37.069];
llc_miss_ratio=[23.251, 22.872, 22.799, 23.877, 23.922];
peak_rss_mib=[261.5, 260.0, 261.3, 261.4, 261.2]

| metric | 2cores mean | 1core mean | diff | p | significant? |
|---|---|---|---|---|---|
| wall-clock | 27.48s | 25.58s | -1.90s | 0.0024 | **Yes**, 1core faster |
| cycles | 66.40B | 58.99B | -7.41B | 0.000005 | **Yes**, 1core fewer |
| cache-miss ratio | 34.92% | 36.72% | +1.79pp | 0.00016 | **Yes**, 1core worse |
| LLC-miss ratio | 20.55% | 23.34% | +2.79pp | 0.000023 | **Yes**, 1core worse |
| peak RSS | 262.3 MiB | 261.1 MiB | -1.2 MiB | 0.0195 | **Yes**, 1core lower |

### pinned_3: 3cores vs 2cores

pinned_3_3cores: elapsed=[25.0465, 27.1149, 27.8346, 27.4551, 28.0258];
cycles=[68185337717, 66937372010, 67853607062, 66994132279, 68509727973];
cache_miss_ratio=[34.981, 34.549, 34.988, 35.442, 35.341];
llc_miss_ratio=[20.148, 20.035, 20.258, 20.056, 20.328];
peak_rss_mib=[315.5, 315.0, 314.3, 317.9, 317.2]

pinned_3_2cores: elapsed=[26.4812, 26.8379, 26.4273, 26.8795, 26.7767];
cycles=[62792220552, 63952153818, 62038631423, 63890056239, 63338672770];
cache_miss_ratio=[37.369, 36.233, 36.665, 36.430, 37.000];
llc_miss_ratio=[22.988, 21.113, 22.153, 22.019, 22.422];
peak_rss_mib=[314.5, 314.5, 316.0, 316.3, 315.8]

| metric | 3cores mean | 2cores mean | diff | p | significant? |
|---|---|---|---|---|---|
| wall-clock | 27.10s | 26.68s | -0.41s | 0.486 | No |
| cycles | 67.70B | 63.20B | -4.49B | 0.000015 | **Yes**, 2cores fewer |
| cache-miss ratio | 35.06% | 36.74% | +1.68pp | 0.00023 | **Yes**, 2cores worse |
| LLC-miss ratio | 20.16% | 22.14% | +1.97pp | 0.0025 | **Yes**, 2cores worse |
| peak RSS | 316.0 MiB | 315.4 MiB | -0.56 MiB | 0.498 | No |

### pinned_5: 4cores vs 3cores

pinned_5_4cores: elapsed=[25.9178, 27.3660, 28.2786, 28.2224, 28.3419];
cycles=[69080264298, 68598563182, 68020682126, 67892372730, 67865722344];
cache_miss_ratio=[35.337, 35.664, 35.275, 35.506, 35.327];
llc_miss_ratio=[20.731, 21.304, 20.326, 21.030, 20.734];
peak_rss_mib=[424.3, 423.7, 422.9, 423.6, 423.0]

pinned_5_3cores: elapsed=[27.2930, 27.5089, 27.4498, 27.1728, 26.9188];
cycles=[64134915135, 64651937359, 64345320596, 64101033057, 63512980076];
cache_miss_ratio=[35.743, 35.884, 36.041, 35.958, 35.937];
llc_miss_ratio=[21.483, 21.527, 21.851, 22.069, 21.882];
peak_rss_mib=[425.7, 424.1, 421.7, 421.6, 422.1]

| metric | 4cores mean | 3cores mean | diff | p | significant? |
|---|---|---|---|---|---|
| wall-clock | 27.63s | 27.27s | -0.36s | 0.490 | No |
| cycles | 68.29B | 64.15B | -4.14B | 0.000001 | **Yes**, 3cores fewer |
| cache-miss ratio | 35.42% | 35.91% | +0.49pp | 0.00077 | **Yes**, 3cores worse |
| LLC-miss ratio | 20.83% | 21.76% | +0.94pp | 0.0021 | **Yes**, 3cores worse |
| peak RSS | 423.5 MiB | 423.0 MiB | -0.46 MiB | 0.610 | No |

## Pattern observed (provisional - see thermal-controlled re-run)

At `n_workers=2` and `n_workers=4`
(`pinned4_4cores_vs_2cores_findings.md`), packing onto fewer physical
cores was significantly FASTER despite worse cache-miss ratios. At
`n_workers=3` and `n_workers=5`, the CYCLES and CACHE-MISS results
still show the same pattern (fewer cycles, worse cache misses when
packed), but the WALL-CLOCK difference is NOT statistically
significant either time. This split (2/4 vs. 3/5) is itself
interesting but should not be trusted without ruling out uncontrolled
thermal history first - see the follow-up document.
