#!/usr/bin/env python3
"""results/*.json -> results/summary.md (the tables the README and report quote)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xcost.market import BROKER_NAMES, BROKERS  # noqa: E402
from xcost.wheel import STRATA  # noqa: E402

RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def f1(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    sig = load("signal.json"); w = load("wheel.json"); pt = load("pretrade.json"); h = load("handoff.json"); pa = load("phaseA_variants.json"); m = load("models.json")
    L = []
    L.append("# Results summary\n")
    L.append(f"Generated from `results/*.json`. Calibration (phase A) {m['period'][0]} to {m['period'][1]}; handoff (phase B) {h['period'][0]} to {h['period'][1]}.\n")
    # signal
    L.append("## Signal\n")
    z = sig["ic_z"]
    L.append(f"Out-of-sample Spearman IC of the blended prediction against the 5-day cross-sectional return: mean {z['mean']:.4f}, sd {z['sd']:.3f}, t = {z['t']:.1f} over {z['n_days']:,} days (ridge {sig['ic_pred_ridge']['mean']:.4f}, gradient boosting {sig['ic_pred_gbm']['mean']:.4f}). Decile 10 minus decile 1: {sig['decile_5d_bp']['9'] - sig['decile_5d_bp']['0']:.0f} bp per 5 days. Alpha = {sig['ic_shrink']} x in-sample IC x sigma x sqrt(5) x z; mean |alpha| {sig['alpha_abs_mean_bp']:.1f} bp per week.\n")
    L.append("| year | " + " | ".join(str(y) for y in z["by_year"]) + " |")
    L.append("|---|" + "---|" * len(z["by_year"]))
    L.append("| IC | " + " | ".join(f"{v:.3f}" for v in z["by_year"].values()) + " |\n")
    # wheel
    L.append("## Wheel\n")
    arr, vw, tru = w["arrival"], w["vwap_schedule_algos"], w["oracle_target"]
    L.append(f"Phase A: {arr['n_orders']:,} orders over {arr['n_days']:,} days, brokers dealt by stratified randomisation. True broker parameters: " + "; ".join(f"{b}: eta {BROKERS[b].eta} ({BROKERS[b].eta + BROKERS[b].eta_illiquid:.3f} illiquid), spread capture {BROKERS[b].spread_capture}" for b in BROKER_NAMES) + ".\n")
    L.append("| target | residual sd (bp) | liquid: B vs A | liquid: C vs A | illiquid: B vs A | illiquid: C vs A | best liquid / illiquid | orders per broker for 1 bp | for 2 bp |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in (("cost vs arrival (all orders)", arr), ("cost vs interval VWAP (schedule algos)", vw), ("oracle: true impact + spread", tru)):
        def cell(g, b):
            e = r["by_liquidity"][g][b]; return f"{e['bp']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]"
        L.append(f"| {name} | {r['residual_sd_interaction_bp']:.1f} | {cell('liquid', 'B')} | {cell('liquid', 'C')} | {cell('illiquid', 'B')} | {cell('illiquid', 'C')} | {r['by_liquidity']['liquid']['best']} / {r['by_liquidity']['illiquid']['best']} | {r['power']['1.0bp']:,} | {r['power']['2.0bp']:,} |")
    L.append("")
    L.append("Per stratum (interval-VWAP wheel): mean cost by broker and the oracle paired effect.\n")
    L.append("| stratum | n | mean %ADV | A | B | C | oracle B - A | oracle C - A | oracle best | wheel best |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, row in vw["strata"].items():
        rb = row["raw_bp"]; oe = row["oracle_effect_vs_A"]
        L.append(f"| {row['name']} | {row['n']:,} | {row['mean_pct_adv'] * 100:.2f} | {f1(rb['A'], 2)} | {f1(rb['B'], 2)} | {f1(rb['C'], 2)} | {oe['B']:+.2f} | {oe['C']:+.2f} | {row['oracle_best']} | {vw['best_map'][str(s)]} |")
    match = vw.get("best_map_matches_oracle", {})
    L.append(f"\nRouting map handed over (stratum -> broker; the medium strata are implementation-shortfall orders and inherit their liquidity group): {m['best_map']}; oracle map on the schedule-algo strata: {m['oracle_best_map']}; the wheel agrees with the oracle in {sum(match.values())} of {len(match)} strata.\n")
    ad = w["adaptive"]
    year1 = {}
    try:
        import pandas as pd
        from xcost import wheel as wh
        orders = pd.read_parquet(os.path.join(RES, "phaseA_gross_wheel_orders.parquet"))
        for k, prefix in (("arrival", "exec_bp_"), ("vwap", "vwap_bp_")):
            c = wh.adaptive_allocation(orders, prefix, curves=True)["regret_curve"]
            i = max(j for j, wk in enumerate(c["week"]) if wk <= 52)
            year1[k] = c["fixed_usd"][i]
    except Exception:  # noqa: BLE001
        pass
    L.append(f"Adaptive allocation on the same orders over {list(ad.values())[0]['years']:.1f} years (cost paid, and regret against oracle-best routing; the wheel-for-a-year column is the fixed wheel's regret after its first 52 weeks, what a desk pays to learn the map once):\n")
    L.append("| reward | fixed randomisation | Thompson sampling | oracle best | regret fixed | wheel for a year | regret Thompson | Thompson picks the best broker (last year) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, r in ad.items():
        y1 = f"${year1[k] / 1e6:.1f}m" if k in year1 else "-"
        L.append(f"| {k} | {r['fixed_randomisation']['bp']:.2f} bp | {r['thompson']['bp']:.2f} bp | {r['oracle_best']['bp']:.2f} bp | ${r['regret_fixed_usd'] / 1e6:.1f}m | {y1} | ${r['regret_thompson_usd'] / 1e6:.1f}m | {r['thompson_share_best_last_year']:.0%} |")
    L.append("")
    # pretrade
    L.append("## Pre-trade cost model\n")
    for key, title in (("observed", "target: observed cost vs arrival"), ("oracle", "target: oracle (impact + spread)")):
        r = pt[key]
        L.append(f"{title}; fitted on {r['n_train']:,} orders, tested on {r['n_test']:,}.\n")
        L.append("| spec | parameters | test RMSE (bp) | RMSE of the mean | R² | corr | bias (bp) |")
        L.append("|---|---|---|---|---|---|---|")
        for sp, v in r["specs"].items():
            p = v["params"]; t = v["test"]
            ptxt = f"a {p['a']:.2f}, b {p['b']:.3f}, beta {p['beta']:.2f}, c {p['c']:.2f}" if "b" in p else f"{int(p['n_iter'])} trees"
            L.append(f"| {sp} | {ptxt} | {t['rmse_bp']:.1f} | {t['rmse_mean_baseline_bp']:.1f} | {t['r2']:.3f} | {f1(t['corr'], 2)} | {t['bias_bp']:+.2f} |")
        L.append("")
    c = pt["contamination"]
    L.append(f"Alpha contamination: observed cost vs arrival {c['observed_bp']:.2f} bp against true impact + spread {c['true_bp']:.2f} bp, gap {c['gap_bp']:.2f} ± {c['gap_se_bp']:.2f} bp (buys {c['gap_buys_bp']:+.2f}, sells {c['gap_sells_bp']:+.2f}); delay cost (previous close to arrival) {c['delay_bp']:.2f} bp. Square-root fit on observed costs: b = {c['sqrt_fit_observed']['b']:.3f}; on the oracle: b = {c['sqrt_fit_true']['b']:.3f}.\n")
    om = pt["optimiser_model"]
    L.append(f"Optimiser's model: {om['chosen']} (validation RMSE sqrt {om['validation_rmse_bp']['sqrt']:.2f} vs power {om['validation_rmse_bp']['power']:.2f} bp), fitted on {om['n_orders']:,} orders routed to the wheel's best broker: cost_bp = {om['params']['a']:.2f} + {om['params']['b']:.3f} sigma_bp (Q/ADV)^{om['params']['beta']:.2f} + {om['params']['c']:.2f} spread_bp.\n")
    oos = h["pretrade_oos"]
    L.append("True out-of-sample test (fitted on all of phase A, applied to the phase B orders of the chosen run):\n")
    L.append("| target | sqrt RMSE | power RMSE | gbm RMSE | RMSE of the mean | sqrt R² | gbm R² |")
    L.append("|---|---|---|---|---|---|---|")
    for key in ("observed", "oracle"):
        r = oos[key]["specs"]
        L.append(f"| {key} | {r['sqrt']['test']['rmse_bp']:.1f} | {r['power']['test']['rmse_bp']:.1f} | {r['gbm']['test']['rmse_bp']:.1f} | {r['sqrt']['test']['rmse_mean_baseline_bp']:.1f} | {r['sqrt']['test']['r2']:.3f} | {r['gbm']['test']['r2']:.3f} |")
    L.append(f"\nOptimiser's model on phase B orders: RMSE {oos['optimiser_model_on_B']['rmse_bp']:.1f} bp, bias {oos['optimiser_model_on_B']['bias_bp']:+.2f} bp (notional-weighted {oos['optimiser_model_on_B']['bias_notional_weighted_bp']:+.2f}).\n")
    # handoff
    L.append("## Handoff\n")
    L.append(f"gamma chosen on phase A by realised Sharpe: {h['gamma_star']}. Phase A variants (routed to the wheel's best broker):\n")
    L.append(table(pa))
    L.append(f"\nPhase B, out of sample ({h['period'][0]} to {h['period'][1]}):\n")
    L.append(table(h["runs"]))
    L.append("")
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("summary ->", os.path.join(RES, "summary.md"))


def table(runs):
    L = ["| variant | paper return %/yr | paper Sharpe | realised %/yr | realised Sharpe | realised vol % | survival | turnover %/wk | gross | orders | IS bp (delay / exec / fees) | cost %/yr of capital | median %ADV |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in runs.items():
        o = r["orders"]; surv = r.get("alpha_survival")
        L.append(f"| {k} | {r['paper']['ann_return_bp'] / 100:.2f} | {r['paper']['sharpe']:.2f} | {r['real']['ann_return_bp'] / 100:.2f} | {r['real']['sharpe']:.2f} | {r['real']['ann_vol_bp'] / 100:.1f} | {f1(surv, 2) if surv is not None else '-'} | {r["turnover_per_week"] * 100:.0f} | {r["gross"]:.2f} | {o["n"]:,} | {o['is_bp']:.1f} ({o['delay_bp']:.1f} / {o['exec_bp']:.1f} / {o['fees_bp']:.1f}) | {o['is_bp_of_capital_per_year'] / 100:.2f} | {o['median_pct_adv'] * 100:.2f} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
