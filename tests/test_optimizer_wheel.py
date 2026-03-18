"""Optimiser constraints and the cost penalty's effect on turnover; the wheel estimator on a synthetic experiment with
known broker effects; the pre-trade specs recovering a known square-root law; the Perold identity on a toy book."""
import numpy as np
import pandas as pd

from xcost import pretrade, wheel
from xcost.backtest import WheelAssigner, stratum
from xcost.market import BROKER_NAMES
from xcost.optimizer import CostModel, OptConfig, optimise


def toy(n=30, seed=0):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0, 0.002, n); A = rng.normal(0, 0.01, (n, n)); sigma = A @ A.T / n + np.eye(n) * 1e-4
    w0 = np.zeros(n); sigma_bp = np.full(n, 150.0); spread = rng.uniform(1, 6, n); adv_usd = rng.uniform(5e7, 5e9, n)
    return alpha, sigma, w0, sigma_bp, spread, adv_usd


def test_optimiser_respects_constraints():
    alpha, sigma, w0, sb, sp, adv = toy()
    cfg = OptConfig(capital=1e9, risk_aversion=10, w_max=0.03, gross=1.0, max_pct_adv=0.1)
    w, info = optimise(alpha, sigma, w0, sb, sp, adv, None, cfg)
    assert info["status"] != "failed"
    assert abs(w.sum()) < 1e-6 and np.abs(w).max() <= 0.03 + 1e-6 and np.abs(w).sum() <= 1.0 + 1e-6
    assert (np.abs(w - w0) <= 0.1 * adv / 1e9 + 1e-6).all()


def test_cost_penalty_reduces_turnover_monotonically():
    alpha, sigma, w0, sb, sp, adv = toy()
    m = CostModel(a=0.5, b=0.3, beta=0.5, c=0.5)
    tos = []
    for g in (0.0, 0.5, 1.0, 4.0):
        cfg = OptConfig(capital=1e9, risk_aversion=10, w_max=0.03, gross=1.0, gamma=g)
        w, info = optimise(alpha, sigma, w0, sb, sp, adv, m if g > 0 else None, cfg)
        tos.append(np.abs(w - w0).sum())
    assert all(tos[i] >= tos[i + 1] - 1e-6 for i in range(len(tos) - 1)) and tos[-1] < tos[0]


def test_linear_penalty_zeroes_small_alpha_trades():
    alpha, sigma, w0, sb, sp, adv = toy()
    cfg = OptConfig(capital=1e9, risk_aversion=10, w_max=0.03, gross=1.0, gamma=1.0)
    w, _ = optimise(alpha, sigma, w0, sb, sp, adv, CostModel(linear_bp=1000.0), cfg)
    assert np.abs(w).sum() < 1e-6          # 10 % per unit turnover: nothing is worth trading


def synthetic_orders(n_days=120, per_day=60, seed=1):
    """orders with a known broker effect: B costs +1 bp, C +3 bp in liquid names and -4 bp in illiquid ones"""
    rng = np.random.default_rng(seed)
    rows = []
    wa = WheelAssigner(seed)
    for d in range(n_days):
        pct = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), per_day)); spread = np.where(rng.uniform(size=per_day) < 0.2, 8.0, 2.0)
        sig = rng.uniform(100, 250, per_day); strata = stratum(pct, spread); brokers = wa.assign(strata)
        base = 0.5 * spread + 0.15 * sig * pct ** 0.5
        eff = np.array([{"A": 0.0, "B": 1.0, "C": 3.0}[b] for b in brokers]) + np.where(spread > 5, np.array([{"A": 0.0, "B": 0.0, "C": -7.0}[b] for b in brokers]), 0.0)
        day_noise = rng.normal(0, 3)
        y = base + eff + day_noise + rng.normal(0, 6, per_day)
        for i in range(per_day):
            rows.append({"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d), "week": d // 5, "symbol": f"S{i}", "pct_adv": pct[i], "spread_bp": spread[i], "sigma_bp": sig[i], "stratum": strata[i],
                         "broker": brokers[i], "algo": "VWAP", "notional": 1e6, "filled": 1.0, "y": y[i], "cost_true_bp": base[i] + eff[i], "exec_arrival_bp": y[i], "alpha": 0.0, "side": 1.0, "delay_bp": 0.0})
    return pd.DataFrame(rows)


def test_wheel_recovers_known_effects():
    o = synthetic_orders()
    r = wheel.analyse(o, "y", B=150, seed=3)
    liq, ill = r["by_liquidity"]["liquid"], r["by_liquidity"]["illiquid"]
    assert abs(liq["B"]["bp"] - 1.0) < 0.6 and abs(liq["C"]["bp"] - 3.0) < 0.6
    assert abs(ill["C"]["bp"] - (-4.0)) < 1.5
    assert liq["best"] == "A" and ill["best"] == "C"
    assert liq["C"]["ci95"][0] > 0 and ill["C"]["ci95"][1] < 0
    assert r["power"]["1.0bp"] > r["power"]["2.0bp"] > r["power"]["5.0bp"]


def test_stratified_assignment_is_balanced():
    wa = WheelAssigner(0)
    s = np.repeat(np.arange(6), 30)
    b = wa.assign(s)
    counts = pd.crosstab(s, b)
    assert (counts.max(axis=1) - counts.min(axis=1) <= 1).all()


def test_pretrade_sqrt_recovers_coefficients():
    rng = np.random.default_rng(4)
    n = 4000
    df = pd.DataFrame({"sigma_bp": rng.uniform(80, 300, n), "pct_adv": np.exp(rng.uniform(np.log(1e-3), np.log(0.2), n)), "spread_bp": rng.uniform(1, 10, n),
                       "algo": rng.choice(["VWAP", "IS", "POV"], n), "broker": rng.choice(BROKER_NAMES, n), "notional": 1e6, "filled": 1.0})
    df["y"] = 1.0 + 0.2 * df["sigma_bp"] * np.sqrt(df["pct_adv"]) + 0.4 * df["spread_bp"] + rng.normal(0, 1.0, n)
    sp = pretrade.Spec("sqrt", algo_dummies=False, broker_dummies=False).fit(df, df["y"].values)
    assert abs(sp.params["b"] - 0.2) < 0.02 and abs(sp.params["c"] - 0.4) < 0.05
    pw = pretrade.Spec("power", algo_dummies=False, broker_dummies=False).fit(df, df["y"].values)
    assert abs(pw.params["beta"] - 0.5) < 0.05
    m = sp.to_cost_model()
    assert abs(m.cost_bp(200.0, 4.0, 0.01) - (1 + 0.2 * 200 * 0.1 + 0.4 * 4)) < 0.5


def test_perold_identity_toy():
    """paper book at the decision price minus real book at the fills equals the summed implementation shortfall"""
    dec, arr, fill, qty, side = 100.0, 100.2, 100.5, 1000.0, 1.0
    delay = side * (arr - dec) / dec * 1e4; execu = side * (fill - arr) / dec * 1e4; fees = 0.3 * fill / dec
    is_usd = (delay + execu + fees) * 1e-4 * qty * dec
    cash_gap = side * (fill - dec) * qty + 0.3e-4 * qty * fill
    assert abs(is_usd - cash_gap) < 1e-6
