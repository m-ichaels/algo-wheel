#!/usr/bin/env python3
"""Figures from results/*.json and the run parquets -> results/figures/*.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xcost import wheel  # noqa: E402
from xcost.market import BROKER_NAMES  # noqa: E402

RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
COL = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def fig_signal(sig):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    by = sig["ic_z"]["by_year"]; years = [int(k) for k in by]
    ax[0].bar(years, [by[str(y)] if str(y) in by else by[y] for y in years], color="#555")
    ax[0].axhline(sig["ic_z"]["mean"], color="crimson", lw=1, label=f"mean {sig['ic_z']['mean']:.3f} (t = {sig['ic_z']['t']:.1f})")
    ax[0].set_title("out-of-sample IC by year (Spearman, 5-day)"); ax[0].legend(); ax[0].set_xticks(years[::2]); ax[0].tick_params(axis="x", rotation=45)
    dec = sig["decile_5d_bp"]
    ax[1].bar([int(k) + 1 for k in dec], list(dec.values()), color="#555"); ax[1].set_title("5-day excess return by decile (bp)"); ax[1].set_xlabel("decile (10 = highest prediction)")
    tr = sig["ic_train_by_year"]
    ax[2].plot([int(k) for k in tr], list(tr.values()), "o-", color="#555", label="in-sample IC of the training window")
    ax[2].plot(years, [by[str(y)] if str(y) in by else by[y] for y in years], "s--", color="crimson", label="realised out of sample")
    ax[2].set_title("in-sample vs realised IC"); ax[2].legend(fontsize=7); ax[2].set_xticks(years[::2]); ax[2].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "signal.png")); plt.close(fig)


def fig_wheel(w):
    arr, vw, tru = w["arrival"], w["vwap_schedule_algos"], w["oracle_target"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    # broker effects vs A by liquidity, for the three targets
    groups = ["liquid", "illiquid"]; x = np.arange(len(groups)); width = 0.12
    for j, (name, res, hatch) in enumerate((("arrival", arr, ""), ("interval VWAP", vw, "//"), ("oracle (true impact)", tru, ".."))):
        for i, b in enumerate(BROKER_NAMES[1:]):
            vals = [res["by_liquidity"][g][b]["bp"] for g in groups]; ci = np.array([res["by_liquidity"][g][b]["ci95"] for g in groups])
            pos = x + (j * 2 + i - 2.5) * width
            ax[0].bar(pos, vals, width, color=COL[b], alpha=0.5 if j == 0 else (0.8 if j == 1 else 1.0), hatch=hatch, edgecolor="k", lw=0.4, label=f"{b} vs A, {name}" if True else None)
            ax[0].errorbar(pos, vals, yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], fmt="none", ecolor="k", lw=0.8, capsize=2)
    for g_i, g in enumerate(groups):
        for b in BROKER_NAMES[1:]:
            orc = np.mean([s["oracle_effect_vs_A"][b] for s in vw["strata"].values() if (s["name"].startswith("illiquid") == (g == "illiquid"))])
            ax[0].plot(g_i + 0.6 * width * 4 * 0, orc, marker="_", color=COL[b], ms=0)
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups); ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylabel("bp vs broker A")
    ax[0].set_title("broker effects, 95 % cluster-bootstrap CIs, by benchmark"); ax[0].legend(fontsize=6.5, ncol=2)
    ax[0].set_ylim(min(-12, ax[0].get_ylim()[0]), max(8, ax[0].get_ylim()[1]))
    # learning curves
    for res, name, ls in ((arr, "arrival", ":"), (vw, "interval VWAP", "-")):
        lc = pd.DataFrame(res["learning_curve"])
        if lc.empty:
            continue
        for key, cikey, col, lab in (("C_minus_A_liquid_bp", "ci_liquid", COL["C"], "C - A, liquid"), ("C_minus_A_illiquid_bp", "ci_illiquid", "#8c564b", "C - A, illiquid")):
            ci = np.array(lc[cikey].tolist())
            ax[1].plot(lc["weeks"], lc[key], ls, color=col, label=f"{lab} ({name})")
            if name == "interval VWAP":
                ax[1].fill_between(lc["weeks"], ci[:, 0], ci[:, 1], color=col, alpha=0.15)
    oracle_liq = np.mean([s["oracle_effect_vs_A"]["C"] for s in vw["strata"].values() if s["name"].startswith("liquid")]); oracle_ill = np.mean([s["oracle_effect_vs_A"]["C"] for s in vw["strata"].values() if s["name"].startswith("illiquid")])
    ax[1].axhline(oracle_liq, color=COL["C"], lw=0.8, ls="--"); ax[1].axhline(oracle_ill, color="#8c564b", lw=0.8, ls="--")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("weeks of wheel data"); ax[1].set_ylabel("bp"); ax[1].set_title("convergence of the C - A estimate (dashed: oracle)"); ax[1].legend(fontsize=6.5)
    ax[1].set_ylim(-25, 15)
    # power
    deltas = [0.5, 1, 2, 5]
    for res, name, m in ((arr, "arrival", "o"), (vw, "interval VWAP", "s"), (tru, "oracle", "^")):
        ax[2].plot(deltas, [res["power"][f"{float(d)}bp"] for d in deltas], m + "-", label=f"{name} (residual sd {res['residual_sd_interaction_bp']:.1f} bp)")
    ax[2].set_yscale("log"); ax[2].set_xlabel("difference to resolve (bp)"); ax[2].set_ylabel("orders per broker, 80 % power"); ax[2].set_title("power: the benchmark sets the sample size"); ax[2].legend(fontsize=7)
    ax[2].axhline(arr["n_orders"] / 3, color="k", lw=0.6, ls=":"); ax[2].text(0.55, arr["n_orders"] / 3 * 1.15, f"orders per broker in the wheel: {arr['n_orders'] // 3:,}", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "wheel.png")); plt.close(fig)


def fig_adaptive(w, orders):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (prefix, name) in enumerate((("exec_bp_", "reward: cost vs arrival"), ("vwap_bp_", "reward: cost vs interval VWAP"))):
        r = wheel.adaptive_allocation(orders, prefix, seed=7, curves=True)
        c = r["regret_curve"]
        ax[k].plot(c["week"], np.array(c["fixed_usd"]) / 1e6, label=f"fixed randomisation (regret ${r['regret_fixed_usd'] / 1e6:.1f}m)", color="#555")
        ax[k].plot(c["week"], np.array(c["thompson_usd"]) / 1e6, label=f"Thompson sampling per stratum (${r['regret_thompson_usd'] / 1e6:.1f}m; best broker {r['thompson_share_best_last_year']:.0%} of orders in the last year)", color="crimson")
        ax[k].set_title(name); ax[k].set_xlabel("week"); ax[k].set_ylabel("cumulative cost above oracle-best routing ($m)"); ax[k].legend(fontsize=6.5, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "adaptive.png")); plt.close(fig)


def fig_pretrade(pt, orders):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for res, name, ls in ((pt["observed"], "observed (vs arrival)", "-"), (pt["oracle"], "oracle (true impact)", "--")):
        for spec, col in (("sqrt", "#1f77b4"), ("power", "#ff7f0e"), ("gbm", "#2ca02c")):
            c = pd.DataFrame(res["specs"][spec]["calibration"])
            ax[0].errorbar(c["pred_bp"], c["realised_bp"], yerr=1.96 * c["realised_se_bp"], fmt="o" + ls, ms=3, color=col, lw=1, label=f"{spec}, {name}: RMSE {res['specs'][spec]['test']['rmse_bp']:.1f} bp, R² {res['specs'][spec]['test']['r2']:.2f}")
    lim = [0, max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])]
    ax[0].plot(lim, lim, "k:", lw=0.8); ax[0].set_xlabel("predicted cost (bp), deciles"); ax[0].set_ylabel("realised (bp)"); ax[0].set_title("calibration on the validation years"); ax[0].legend(fontsize=6)
    # cost vs size, binned, with fitted curve per broker
    o = orders[orders["filled"] > 0].copy()
    o["bin"] = pd.qcut(o["pct_adv"], 12, duplicates="drop")
    for b in BROKER_NAMES:
        g = o[o["broker"] == b].groupby("bin", observed=True).agg(x=("pct_adv", "median"), y=("cost_true_bp", "mean"), yo=("exec_arrival_bp", "mean"), n=("pct_adv", "size"))
        ax[1].plot(g["x"] * 100, g["y"], "o-", color=COL[b], ms=3, label=f"broker {b}: true impact + spread")
        ax[1].plot(g["x"] * 100, g["yo"], "x:", color=COL[b], ms=4, alpha=0.7, label=f"broker {b}: observed vs arrival")
    p = pt["optimiser_model"]["params"]; xs = np.logspace(-3.2, -1, 50)
    sig_med = float(o["sigma_bp"].median()); sp_med = float(o["spread_bp"].median())
    ax[1].plot(xs * 100, p["a"] + p["b"] * sig_med * xs ** p["beta"] + p["c"] * sp_med, "k-", lw=1.5, label=f"optimiser's model ({pt['optimiser_model']['chosen']}, median σ and spread)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("order size, % of ADV"); ax[1].set_ylabel("bp"); ax[1].set_title("cost against size by broker (phase A)"); ax[1].legend(fontsize=6)
    # contamination by stratum
    c = pt["contamination"]["by_stratum"]; names = [wheel.STRATA[int(k)] for k in c]; x = np.arange(len(c))
    ax[2].bar(x - 0.2, [v["observed_bp"] for v in c.values()], 0.4, label="observed vs arrival", color="#999")
    ax[2].bar(x + 0.2, [v["true_bp"] for v in c.values()], 0.4, label="oracle: impact + spread", color="#333")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7); ax[2].set_ylabel("bp")
    ax[2].set_title(f"what the desk sees vs what impact cost: gap {pt['contamination']['gap_bp']:.1f} ± {pt['contamination']['gap_se_bp']:.1f} bp"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pretrade.png")); plt.close(fig)


def fig_handoff(h, ph_a):
    runs = h["runs"]; g = h["gamma_star"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    # NAV paths
    for name, col, ls in (("gross", "#555", "-"), (f"cost_g{g}", "crimson", "-"), ("naive10bp", "#1f77b4", "-")):
        nav = pd.read_parquet(os.path.join(RES, f"phaseB_{name}_nav.parquet"))
        if name == "gross":
            ax[0, 0].plot(nav.index, (nav["nav_paper"] / 1e9 - 1) * 100, color=col, ls=":", label="gross, paper (trades at the decision price)")
        ax[0, 0].plot(nav.index, (nav["nav_real"] / 1e9 - 1) * 100, color=col, ls=ls, label=f"{name}, real (through the brokers)")
        if name == f"cost_g{g}":
            ax[0, 0].plot(nav.index, (nav["nav_paper"] / 1e9 - 1) * 100, color=col, ls=":", label=f"{name}, paper")
    ax[0, 0].set_ylabel("cumulative P&L, % of capital"); ax[0, 0].set_title("phase B (out of sample): paper vs realised"); ax[0, 0].legend(fontsize=7)
    # bars: paper vs real annual return per variant
    order = ["gross"] + sorted([k for k in runs if k.startswith("naive")], key=lambda k: float(k[5:-2])) + [k for k in runs if k.startswith("cost_g") and k.count("_") == 1]
    x = np.arange(len(order))
    ax[0, 1].bar(x - 0.2, [runs[k]["paper"]["ann_return_bp"] / 100 for k in order], 0.4, color="#bbb", label="paper (gross backtest)")
    ax[0, 1].bar(x + 0.2, [runs[k]["real"]["ann_return_bp"] / 100 for k in order], 0.4, color="crimson", label="realised net")
    for i, k in enumerate(order):
        ax[0, 1].text(i + 0.2, runs[k]["real"]["ann_return_bp"] / 100, f"SR {runs[k]['real']['sharpe']:.2f}", ha="center", va="bottom", fontsize=7)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels([k.replace("cost_g", "γ = ").replace("naive", "flat ") for k in order], fontsize=7); ax[0, 1].set_ylabel("% per year"); ax[0, 1].axhline(0, color="k", lw=0.6)
    ax[0, 1].set_title("annual return: what research saw vs what the fund kept"); ax[0, 1].legend(fontsize=7)
    # turnover vs net return frontier
    for k in order:
        r = runs[k]; ax[1, 0].scatter(r["turnover_per_week"] * 100, r["real"]["ann_return_bp"] / 100, s=40, color="crimson" if k.startswith("cost") else ("#555" if k == "gross" else "#1f77b4"))
        ax[1, 0].annotate(k.replace("cost_g", "γ=").replace("naive", "flat "), (r["turnover_per_week"] * 100, r["real"]["ann_return_bp"] / 100), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax[1, 0].scatter(r["turnover_per_week"] * 100, r["paper"]["ann_return_bp"] / 100, s=25, facecolors="none", edgecolors="#555")
    ax[1, 0].set_xlabel("one-way turnover, % of capital per week"); ax[1, 0].set_ylabel("% per year (filled: realised, hollow: paper)"); ax[1, 0].set_title("turnover against return: the cost penalty picks the point"); ax[1, 0].axhline(0, color="k", lw=0.6)
    # IS decomposition and the routing / policy variants
    var = order + [f"cost_g{g}_random", f"cost_g{g}_worst", f"cost_g{g}_vwap", f"cost_g{g}_urgent"]
    var = [v for v in var if v in runs]
    x = np.arange(len(var))
    d = np.array([runs[k]["orders"]["delay_bp"] for k in var]); e = np.array([runs[k]["orders"]["exec_bp"] for k in var]); f = np.array([runs[k]["orders"]["fees_bp"] for k in var]); tot = d + e + f
    ax[1, 1].bar(x - 0.27, d, 0.27, color="#9ecae1", label="delay (decision to arrival)"); ax[1, 1].bar(x, e + f, 0.27, color="#3182bd", label="execution (arrival to fills) + fees"); ax[1, 1].bar(x + 0.27, tot, 0.27, color="#333", label="implementation shortfall")
    for i, k in enumerate(var):
        ax[1, 1].text(i + 0.27, max(tot[i], 0) + 0.2, f"{runs[k]['orders']['is_bp_of_capital_per_year'] / 100:+.1f}%/yr", ha="center", va="bottom", fontsize=6)
    ax[1, 1].axhline(0, color="k", lw=0.6)
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([k.replace("cost_g", "γ=").replace("naive", "flat ") for k in var], rotation=30, ha="right", fontsize=7); ax[1, 1].set_ylabel("bp of traded notional (notional-weighted)")
    ax[1, 1].set_title("shortfall per order; label: shortfall as % of capital per year"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "handoff.png")); plt.close(fig)


def main():
    sig = load("signal.json"); w = load("wheel.json"); pt = load("pretrade.json"); h = load("handoff.json"); pa = load("phaseA_variants.json")
    orders = pd.read_parquet(os.path.join(RES, "phaseA_gross_wheel_orders.parquet"))
    fig_signal(sig); fig_wheel(w); fig_adaptive(w, orders); fig_pretrade(pt, orders); fig_handoff(h, pa)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
