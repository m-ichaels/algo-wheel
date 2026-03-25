# Results summary

Generated from `results/*.json`. Calibration (phase A) 2010-01-01 to 2016-12-31; handoff (phase B) 2017-01-01 to 2026-09-17.

## Signal

Out-of-sample Spearman IC of the blended prediction against the 5-day cross-sectional return: mean 0.0178, sd 0.214, t = 5.4 over 4,197 days (ridge 0.0182, gradient boosting 0.0159). Decile 10 minus decile 1: 36 bp per 5 days. Alpha = 0.5 x in-sample IC x sigma x sqrt(5) x z; mean |alpha| 5.3 bp per week.

| year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IC | 0.018 | -0.001 | 0.015 | 0.032 | 0.034 | 0.004 | 0.024 | 0.005 | 0.029 | 0.038 | 0.011 | 0.002 | -0.036 | 0.061 | 0.013 | 0.030 | 0.028 |

## Wheel

Phase A: 52,374 orders over 364 days, brokers dealt by stratified randomisation. True broker parameters: A: eta 0.11 (0.190 illiquid), spread capture 0.75; B: eta 0.142 (0.142 illiquid), spread capture 1.0; C: eta 0.175 (0.085 illiquid), spread capture 1.15.

| target | residual sd (bp) | liquid: B vs A | liquid: C vs A | illiquid: B vs A | illiquid: C vs A | best liquid / illiquid | orders per broker for 1 bp | for 2 bp |
|---|---|---|---|---|---|---|---|---|
| cost vs arrival (all orders) | 63.7 | +0.69 [-0.67, +2.17] | +1.55 [-0.05, +3.02] | +1.85 [-3.06, +6.64] | -6.14 [-11.08, -1.15] | A / C | 63,775 | 15,944 |
| cost vs interval VWAP (schedule algos) | 9.0 | +1.18 [+1.00, +1.38] | +2.20 [+2.02, +2.38] | -1.20 [-2.25, -0.19] | -4.58 [-5.57, -3.57] | A / C | 1,283 | 321 |
| oracle: true impact + spread | 3.2 | +1.23 [+1.19, +1.28] | +2.29 [+2.22, +2.35] | -1.34 [-1.76, -0.92] | -3.54 [-3.92, -3.16] | A / C | 159 | 40 |

Per stratum (interval-VWAP wheel): mean cost by broker and the oracle paired effect.

| stratum | n | mean %ADV | A | B | C | oracle B - A | oracle C - A | oracle best | wheel best |
|---|---|---|---|---|---|---|---|---|---|
| liquid / small | 13,366 | 0.15 | 1.49 | 1.93 | 2.23 | +0.46 | +0.79 | A | A |
| liquid / large | 25,405 | 6.79 | 5.67 | 7.25 | 8.58 | +1.63 | +3.15 | A | A |
| illiquid / small | 456 | 0.24 | 2.80 | 3.32 | 3.64 | +0.50 | +0.66 | A | C |
| illiquid / large | 5,103 | 14.68 | 19.33 | 17.34 | 11.77 | -4.22 | -9.63 | C | C |

Routing map handed over (stratum -> broker; the medium strata are implementation-shortfall orders and inherit their liquidity group): {'0': 'A', '1': 'A', '2': 'A', '3': 'C', '4': 'C', '5': 'C'}; oracle map on the schedule-algo strata: {'0': 'A', '2': 'A', '3': 'A', '5': 'C'}; the wheel agrees with the oracle in 3 of 4 strata.

Adaptive allocation on the same orders over 7.0 years (cost paid, and regret against oracle-best routing; the wheel-for-a-year column is the fixed wheel's regret after its first 52 weeks, what a desk pays to learn the map once):

| reward | fixed randomisation | Thompson sampling | oracle best | regret fixed | wheel for a year | regret Thompson | Thompson picks the best broker (last year) |
|---|---|---|---|---|---|---|---|
| arrival | 7.70 bp | 6.72 bp | 6.34 bp | $90.1m | $13.4m | $17.4m | 62% |
| vwap | 5.89 bp | 4.82 bp | 4.64 bp | $83.8m | $12.5m | $8.1m | 91% |

## Pre-trade cost model

target: observed cost vs arrival; fitted on 36,957 orders, tested on 15,417.

| spec | parameters | test RMSE (bp) | RMSE of the mean | R² | corr | bias (bp) |
|---|---|---|---|---|---|---|
| sqrt | a -1.14, b 0.257, beta 0.50, c 0.79 | 59.7 | 60.3 | 0.014 | 0.12 | -0.92 |
| power | a -0.01, b 0.277, beta 0.61, c 0.62 | 59.7 | 60.3 | 0.014 | 0.12 | -0.96 |
| gbm | 35 trees | 59.8 | 60.3 | 0.010 | 0.11 | -1.40 |

target: oracle (impact + spread); fitted on 36,957 orders, tested on 15,417.

| spec | parameters | test RMSE (bp) | RMSE of the mean | R² | corr | bias (bp) |
|---|---|---|---|---|---|---|
| sqrt | a -0.81, b 0.140, beta 0.50, c 0.41 | 2.2 | 6.0 | 0.823 | 0.91 | -0.25 |
| power | a -2.95, b 0.115, beta 0.30, c 0.61 | 2.2 | 6.0 | 0.827 | 0.91 | -0.23 |
| gbm | 239 trees | 2.0 | 6.0 | 0.862 | 0.94 | -0.20 |

Alpha contamination: observed cost vs arrival 7.69 bp against true impact + spread 7.11 bp, gap 0.58 ± 0.28 bp (buys +1.26, sells -0.19); delay cost (previous close to arrival) 0.25 bp. Square-root fit on observed costs: b = 0.252; on the oracle: b = 0.164.

Optimiser's model: sqrt (validation RMSE sqrt 60.61 vs power 60.65 bp), fitted on 17,458 orders routed to the wheel's best broker: cost_bp = 0.86 + 0.201 sigma_bp (Q/ADV)^0.50 + 0.00 spread_bp.

True out-of-sample test (fitted on all of phase A, applied to the phase B orders of the chosen run):

| target | sqrt RMSE | power RMSE | gbm RMSE | RMSE of the mean | sqrt R² | gbm R² |
|---|---|---|---|---|---|---|
| observed | 88.4 | 88.3 | 88.4 | 88.7 | -0.000 | -0.001 |
| oracle | 1.1 | 2.3 | 0.9 | 7.5 | 0.323 | 0.568 |

Optimiser's model on phase B orders: RMSE 88.3 bp, bias -0.46 bp (notional-weighted -0.52).

## Handoff

gamma chosen on phase A by realised Sharpe: 1.0. Phase A variants (routed to the wheel's best broker):

| variant | paper return %/yr | paper Sharpe | realised %/yr | realised Sharpe | realised vol % | survival | turnover %/wk | gross | orders | IS bp (delay / exec / fees) | cost %/yr of capital | median %ADV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gross | 7.42 | 1.58 | -0.19 | -0.04 | 4.9 | -0.03 | 150 | 1.96 | 52,374 | 9.7 (1.3 / 8.1 / 0.3) | 7.59 | 3.20 |
| naive10bp | 4.56 | 1.05 | 3.46 | 0.79 | 4.4 | 0.76 | 15 | 1.83 | 11,271 | 14.4 (5.0 / 9.1 / 0.3) | 1.10 | 0.32 |
| cost_g0.25 | 6.35 | 1.37 | 2.33 | 0.50 | 4.7 | 0.37 | 83 | 1.91 | 49,945 | 9.3 (1.9 / 7.1 / 0.3) | 4.01 | 1.03 |
| cost_g0.5 | 5.51 | 1.20 | 3.12 | 0.67 | 4.7 | 0.57 | 51 | 1.85 | 47,353 | 9.0 (2.5 / 6.2 / 0.3) | 2.37 | 0.56 |
| cost_g1.0 | 4.58 | 1.02 | 3.34 | 0.74 | 4.5 | 0.73 | 25 | 1.71 | 42,407 | 9.4 (3.1 / 6.1 / 0.3) | 1.23 | 0.28 |
| cost_g2.0 | 2.99 | 0.73 | 2.67 | 0.65 | 4.1 | 0.89 | 10 | 1.49 | 34,440 | 6.1 (3.4 / 2.4 / 0.3) | 0.32 | 0.12 |

Phase B, out of sample (2017-01-01 to 2026-09-17):

| variant | paper return %/yr | paper Sharpe | realised %/yr | realised Sharpe | realised vol % | survival | turnover %/wk | gross | orders | IS bp (delay / exec / fees) | cost %/yr of capital | median %ADV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gross | 4.77 | 0.79 | 0.38 | 0.06 | 6.0 | 0.08 | 139 | 1.80 | 76,995 | 6.1 (-0.9 / 6.7 / 0.3) | 4.42 | 1.06 |
| naive10bp | 1.56 | 0.31 | 1.54 | 0.31 | 4.9 | 0.99 | 6 | 1.43 | 8,349 | 0.6 (-7.0 / 7.3 / 0.3) | 0.02 | 0.15 |
| cost_g0.25 | 3.66 | 0.64 | 2.45 | 0.43 | 5.7 | 0.67 | 56 | 1.66 | 70,183 | 4.2 (-1.8 / 5.7 / 0.3) | 1.24 | 0.33 |
| cost_g0.5 | 2.81 | 0.50 | 2.53 | 0.45 | 5.6 | 0.90 | 32 | 1.56 | 63,928 | 1.8 (-3.3 / 4.8 / 0.3) | 0.30 | 0.18 |
| cost_g1.0 | 1.73 | 0.32 | 1.92 | 0.35 | 5.4 | 1.11 | 15 | 1.38 | 53,794 | -2.3 (-6.6 / 4.1 / 0.3) | -0.18 | 0.09 |
| cost_g2.0 | 1.33 | 0.26 | 1.71 | 0.33 | 5.1 | 1.28 | 6 | 1.10 | 40,725 | -11.5 (-15.0 / 3.2 / 0.3) | -0.37 | 0.04 |
| cost_g1.0_random | 1.73 | 0.32 | 1.86 | 0.34 | 5.4 | 1.08 | 15 | 1.38 | 53,794 | -1.5 (-6.3 / 4.4 / 0.3) | -0.12 | 0.09 |
| cost_g1.0_worst | 1.73 | 0.32 | 1.81 | 0.33 | 5.4 | 1.05 | 15 | 1.38 | 53,794 | -0.8 (-6.6 / 5.5 / 0.3) | -0.07 | 0.09 |
| cost_g1.0_vwap | 1.73 | 0.32 | 2.01 | 0.37 | 5.4 | 1.17 | 15 | 1.38 | 53,794 | -3.4 (-6.6 / 3.0 / 0.3) | -0.27 | 0.09 |
| cost_g1.0_urgent | 1.73 | 0.32 | 1.89 | 0.35 | 5.4 | 1.09 | 15 | 1.38 | 53,794 | -1.8 (-6.6 / 4.5 / 0.3) | -0.14 | 0.09 |
| gross_random | 4.77 | 0.79 | -0.66 | -0.11 | 6.0 | -0.14 | 139 | 1.80 | 76,995 | 7.6 (-1.0 / 8.2 / 0.3) | 5.47 | 1.06 |
| naive3bp | 3.34 | 0.58 | 2.05 | 0.36 | 5.7 | 0.62 | 30 | 1.64 | 26,370 | 8.5 (-1.7 / 9.8 / 0.3) | 1.31 | 0.34 |
| naive30bp | 1.92 | 0.51 | 1.97 | 0.52 | 3.8 | 1.02 | 1 | 0.72 | 2,826 | -7.3 (-18.6 / 11.0 / 0.3) | -0.04 | 0.08 |
