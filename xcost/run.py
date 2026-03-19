"""Pipeline steps.  python -m xcost signal | calibrate | handoff | all  [--quick] [--config configs/pipeline.json]

signal     features, walk-forward ML, Grinold alpha                        -> results/pred.parquet, results/signal.json
calibrate  phase A (2010-16): gross book routed through the wheel with the oracle; wheel analysis; adaptive allocation;
           pre-trade specs on 2010-14 tested on 2015-16; alpha contamination; the optimiser's cost model; gamma chosen
           on phase A                                                        -> results/phaseA_*.parquet, wheel.json, pretrade.json, models.json, phaseA_variants.json
handoff    phase B (2017-26): gross, naive linear, cost-aware for each gamma, routing and algo policy variants, all
           executed through the simulator; the pre-trade specs' true out-of-sample test      -> results/phaseB_*.parquet, handoff.json, pretrade_oos.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import pretrade, signal, wheel
from .backtest import RunConfig, RunResult, run
from .data import RES, Panel, load_prices
from .optimizer import CostModel, OptConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None) -> dict:
    with open(path or os.path.join(ROOT, "configs", "pipeline.json"), encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, name: str):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def get_panel() -> Panel:
    return Panel(load_prices())


def opt_config(cfg: dict, **kw) -> OptConfig:
    o = dict(cfg["optimizer"]); o.update(kw)
    return OptConfig(**o)


# ---- signal ---------------------------------------------------------------------------------------------------------
def step_signal(cfg: dict, panel: Panel) -> pd.DataFrame:
    t = time.time()
    f = signal.features(panel)
    s = cfg["signal"]
    pred = signal.walk_forward(f, s["first_year"], s["last_year"])
    pred["alpha"] = s["ic_shrink"] * pred["ic_train"] * pred["sigma"] * np.sqrt(signal.HORIZON) * pred["z"]
    pred.to_parquet(os.path.join(RES, "pred.parquet"))
    out = {"n_rows": int(len(pred)), "n_days": int(pred["date"].nunique()), "features": signal.FEATURES, "horizon_days": signal.HORIZON, "ic_shrink": s["ic_shrink"]}
    for col in ("z", "pred_ridge", "pred_gbm"):
        ic = signal.ic_series(pred, col)
        out[f"ic_{col}"] = {"mean": float(ic.mean()), "sd": float(ic.std()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic))), "n_days": int(len(ic)), "by_year": {int(k): float(v) for k, v in ic.groupby(ic.index.year).mean().items()}}
    out["decile_5d_bp"] = {int(k): float(v * 1e4) for k, v in signal.decile_returns(pred).items()}
    out["ic_train_by_year"] = {int(k): float(v) for k, v in pred.groupby(pred["date"].dt.year)["ic_train"].first().items()}
    out["alpha_abs_mean_bp"] = float(pred["alpha"].abs().mean() * 1e4)
    out["seconds"] = round(time.time() - t, 1)
    save_json(out, "signal.json")
    print(f"signal: IC {out['ic_z']['mean']:.4f} (t {out['ic_z']['t']:.1f}) over {out['ic_z']['n_days']} days, {out['seconds']} s")
    return pred


def load_pred() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RES, "pred.parquet"))


def save_run(r: RunResult, prefix: str):
    r.orders.to_parquet(os.path.join(RES, f"{prefix}_orders.parquet"))
    r.weeks.to_parquet(os.path.join(RES, f"{prefix}_weeks.parquet"))
    r.nav.to_parquet(os.path.join(RES, f"{prefix}_nav.parquet"))


# ---- calibration ----------------------------------------------------------------------------------------------------
def step_calibrate(cfg: dict, panel: Panel, pred: pd.DataFrame, quick: bool = False) -> dict:
    c = cfg["calibration"]; seed = cfg["seed"]; B = 200 if quick else cfg["bootstrap"]
    start, end = c["start"], c["end"]
    if quick:
        start = "2015-01-01"
    t = time.time()
    base = run(panel, pred, RunConfig("A_gross_wheel", start, end, model=None, gamma=0.0, routing="wheel", oracle=True, seed=seed, opt=opt_config(cfg)))
    save_run(base, "phaseA_gross_wheel")
    print(f"phase A gross/wheel: {len(base.orders)} orders, IS {base.stats['orders']['is_bp']:.1f} bp, {base.stats['seconds']} s")
    o = base.orders
    # the wheel
    w_arr = wheel.analyse(o, "exec_arrival_bp", B=B, seed=seed, oracle_prefix="true_bp_")
    w_vwap = wheel.analyse(o[o["algo"] != "IS"], "vwap_slip_bp", B=B, seed=seed, oracle_prefix="vwap_bp_")
    w_true = wheel.analyse(o, "cost_true_bp", B=B, seed=seed, oracle_prefix="true_bp_")
    adaptive = {"arrival": wheel.adaptive_allocation(o, "exec_bp_", seed=seed), "vwap": wheel.adaptive_allocation(o, "vwap_bp_", seed=seed)}
    save_json({"arrival": w_arr, "vwap_schedule_algos": w_vwap, "oracle_target": w_true, "adaptive": adaptive}, "wheel.json")
    print("wheel (VWAP benchmark): pooled ranking", w_vwap["pooled"]["ranking"], "| liquid best", w_vwap["by_liquidity"]["liquid"]["best"], "| illiquid best", w_vwap["by_liquidity"]["illiquid"]["best"], "| oracle map", w_vwap.get("oracle_best_map"))
    # pre-trade specs: train to pretrade_train_end, validate on the rest of phase A
    tr, te = o[o["date"] <= c["pretrade_train_end"]], o[o["date"] > c["pretrade_train_end"]]
    if quick:
        cut = o["date"].quantile(0.6); tr, te = o[o["date"] <= cut], o[o["date"] > cut]
    pt = {"observed": pretrade.fit_and_test(tr, te, "exec_arrival_bp"), "oracle": pretrade.fit_and_test(tr, te, "cost_true_bp"), "contamination": pretrade.alpha_contamination(o)}
    # the optimiser's reduced form: no algo dummies, on orders routed to the broker the policy will use, chosen by validation RMSE among the convex specs
    best_map = {int(k): v for k, v in w_vwap["best_map"].items()}          # the VWAP-benchmark wheel is the one that resolves the brokers
    routed = o[[best_map[int(s)] == b for s, b in zip(o["stratum"], o["broker"])]]
    rtr, rte = routed[routed["date"].isin(tr["date"])], routed[routed["date"].isin(te["date"])]
    y_tr, y_te = wheel.winsorise(rtr["exec_arrival_bp"]).values, wheel.winsorise(rte["exec_arrival_bp"]).values
    cand = {}
    for kind in ("sqrt", "power"):
        sp = pretrade.Spec(kind, algo_dummies=False, broker_dummies=False).fit(rtr, y_tr)
        cand[kind] = {"spec": sp, "rmse": pretrade.metrics(y_te, sp.predict(rte), rte["notional"].values, float(y_tr.mean()))["rmse_bp"]}
    chosen = min(cand, key=lambda k: cand[k]["rmse"])
    full = pretrade.Spec(chosen, algo_dummies=False, broker_dummies=False).fit(routed, wheel.winsorise(routed["exec_arrival_bp"]).values)
    model = full.to_cost_model(name=f"{chosen}_routed")
    pt["optimiser_model"] = {"chosen": chosen, "validation_rmse_bp": {k: v["rmse"] for k, v in cand.items()}, "params": full.params, "n_orders": int(len(routed))}
    save_json(pt, "pretrade.json")
    print(f"pre-trade: chosen {chosen}, params {full.params}")
    # gamma on phase A: cost-aware variants against gross and naive, all routed best
    variants = {"gross": (None, 0.0), f"naive{int(cfg['naive_linear_bp'])}bp": (CostModel(linear_bp=cfg["naive_linear_bp"], name="naive"), 1.0)}
    for g in cfg["gamma_grid"]:
        variants[f"cost_g{g}"] = (model, g)
    va = {}
    for name, (m, g) in variants.items():
        r = run(panel, pred, RunConfig(f"A_{name}", start, end, model=m, gamma=g, routing="best", best_map={"best": best_map, "worst": {int(k): v for k, v in w_vwap["worst_map"].items()}}, seed=seed, opt=opt_config(cfg, gamma=g)))
        save_run(r, f"phaseA_{name}")
        va[name] = r.stats
        print(f"phase A {name}: paper {r.stats['paper']['ann_return_bp']:.0f} bp, real {r.stats['real']['ann_return_bp']:.0f} bp (Sharpe {r.stats['real']['sharpe']:.2f}), turnover {r.stats['turnover_per_week']:.2f}/wk, {r.stats['seconds']} s")
    cost_names = [k for k in va if k.startswith("cost_g")]
    gamma_star = float(max(cost_names, key=lambda k: va[k]["real"]["sharpe"]).split("cost_g")[1])
    models = {"cost_model": model.__dict__, "gamma_star": gamma_star, "best_map": best_map, "worst_map": {int(k): v for k, v in w_vwap["worst_map"].items()}, "oracle_best_map": w_vwap.get("oracle_best_map"), "period": [start, end]}
    save_json(models, "models.json")
    save_json(va, "phaseA_variants.json")
    print(f"calibrate done: gamma* = {gamma_star}, {round(time.time() - t)} s")
    return models


# ---- handoff --------------------------------------------------------------------------------------------------------
def step_handoff(cfg: dict, panel: Panel, pred: pd.DataFrame, quick: bool = False, resume: bool = False) -> dict:
    h = cfg["handoff"]; seed = cfg["seed"]
    with open(os.path.join(RES, "models.json"), encoding="utf-8") as f:
        models = json.load(f)
    model = CostModel(**models["cost_model"]); g_star = models["gamma_star"]
    maps = {"best": {int(k): v for k, v in models["best_map"].items()}, "worst": {int(k): v for k, v in models["worst_map"].items()}}
    start, end = h["start"], h["end"]
    if quick:
        start = "2024-01-01"
    t = time.time()
    plan = [("gross", None, 0.0, "best", "size")] + [(f"naive{int(nb)}bp", CostModel(linear_bp=nb, name="naive"), 1.0, "best", "size") for nb in cfg.get("naive_grid", [cfg["naive_linear_bp"]])]
    plan += [(f"cost_g{g}", model, g, "best", "size") for g in cfg["gamma_grid"]]
    plan += [(f"cost_g{g_star}_random", model, g_star, "random", "size"), (f"cost_g{g_star}_worst", model, g_star, "worst", "size"),
             (f"cost_g{g_star}_vwap", model, g_star, "best", "vwap"), (f"cost_g{g_star}_urgent", model, g_star, "best", "urgent"),
             ("gross_random", None, 0.0, "random", "size")]
    out = {"period": [start, end], "gamma_star": g_star, "runs": {}}
    prev = os.path.join(RES, "handoff.json")
    if resume and os.path.exists(prev):
        with open(prev, encoding="utf-8") as f:
            out["runs"] = json.load(f).get("runs", {})
    for name, m, g, routing, policy in plan:
        if resume and name in out["runs"] and os.path.exists(os.path.join(RES, f"phaseB_{name}_nav.parquet")):
            continue
        r = run(panel, pred, RunConfig(f"B_{name}", start, end, model=m, gamma=g, routing=routing, best_map=maps, policy=policy, seed=seed, opt=opt_config(cfg, gamma=g)))
        save_run(r, f"phaseB_{name}")
        out["runs"][name] = r.stats | {"routing": routing, "policy": policy, "gamma": g, "model": (m.name if m else None)}
        print(f"phase B {name}: paper {r.stats['paper']['ann_return_bp']:.0f} bp, real {r.stats['real']['ann_return_bp']:.0f} bp (Sharpe {r.stats['real']['sharpe']:.2f}), IS {r.stats['orders']['is_bp']:.1f} bp, turnover {r.stats['turnover_per_week']:.2f}/wk, {r.stats['seconds']} s")
    # the pre-trade specs' true out-of-sample test: fitted on all of phase A, applied to phase B orders of the chosen run
    A = pd.read_parquet(os.path.join(RES, "phaseA_gross_wheel_orders.parquet"))
    Bo = pd.read_parquet(os.path.join(RES, f"phaseB_cost_g{g_star}_orders.parquet"))
    out["pretrade_oos"] = {"observed": pretrade.fit_and_test(A, Bo, "exec_arrival_bp"), "oracle": pretrade.fit_and_test(A, Bo, "cost_true_bp"),
                           "optimiser_model_on_B": pretrade.metrics(wheel.winsorise(Bo["exec_arrival_bp"]).values, model.cost_bp(Bo["sigma_bp"].values, Bo["spread_bp"].values, Bo["pct_adv"].values), Bo["notional"].values, float(wheel.winsorise(A["exec_arrival_bp"]).mean()))}
    out["seconds"] = round(time.time() - t)
    save_json(out, "handoff.json")
    print(f"handoff done, {out['seconds']} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["signal", "calibrate", "handoff", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--resume", action="store_true", help="handoff: keep variants already in results/handoff.json and run only the missing ones")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    os.makedirs(RES, exist_ok=True)
    panel = get_panel()
    if a.step in ("signal", "all"):
        pred = step_signal(cfg, panel)
    else:
        pred = load_pred()
    if a.step in ("calibrate", "all"):
        step_calibrate(cfg, panel, pred, a.quick)
    if a.step in ("handoff", "all"):
        step_handoff(cfg, panel, pred, a.quick, a.resume)


if __name__ == "__main__":
    main()
