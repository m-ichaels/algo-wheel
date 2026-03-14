"""The market simulator: bridge endpoints, volume conservation, complete fills, oracle identity, impact monotone in
size and in broker quality, schedule algorithms sum to one, VWAP benchmark inside the day's range."""
import numpy as np
import pandas as pd
import pytest

from xcost import market
from xcost.market import BROKERS, DayMarket, schedule


def bars(n=6, seed=0):
    rng = np.random.default_rng(seed)
    syms = [f"S{i}" for i in range(n)]
    o = 100 * np.exp(rng.normal(0, 0.2, n)); c = o * np.exp(rng.normal(0, 0.01, n))
    return pd.DataFrame({"open": o, "close": c, "volume": rng.uniform(1e6, 2e7, n), "vol": rng.uniform(0.008, 0.03, n), "spread_bp": rng.uniform(1, 8, n), "adv": rng.uniform(1e6, 2e7, n)}, index=syms)


def orders(b, qty_frac=0.01, algo="VWAP"):
    return pd.DataFrame({"symbol": b.index, "side": np.where(np.arange(len(b)) % 2 == 0, 1.0, -1.0), "qty": np.floor(b["adv"].values * qty_frac), "algo": algo, "urgency": "low", "pov": 0.10 if algo == "POV" else 0.0})


def test_bridge_pins_open_and_close():
    b = bars(); m = DayMarket(pd.Timestamp("2020-01-06"), b, seed=1)
    assert np.allclose(m.mid[:, 0], b["open"].values) and np.allclose(m.mid[:, -1], b["close"].values)
    assert np.allclose(m.vol_curve.sum(axis=1), b["volume"].values)


def test_schedules_sum_to_one():
    rng = np.random.default_rng(0)
    for algo, urg in (("VWAP", "low"), ("TWAP", "low"), ("IS", "high"), ("IS", "medium")):
        s, _ = schedule(algo, urg, 390, rng, 0.1)
        assert abs(s.sum() - 1) < 1e-9 and s[:5].sum() == 0
    s, _ = schedule("CLOSE", "low", 390, rng, 0.0)
    assert abs(s.sum() - 0.3) < 1e-9


def test_fills_complete_and_oracle_identity():
    b = bars(); m = DayMarket(pd.Timestamp("2020-01-06"), b, seed=2)
    f = m.execute(orders(b), "B", seed=3)
    assert np.allclose(f["filled"], f["qty"])
    # the true cost is the fill against the impact-free path on the same minutes: for a buy the fill is above it
    assert (f["cost_true_bp"] > 0).all()
    assert (f["spread_paid_bp"] > 0).all() and (f["spread_paid_bp"] <= f["cost_true_bp"] + 1e-9).all()
    lo = np.minimum(b["open"], b["close"]).values * 0.9; hi = np.maximum(b["open"], b["close"]).values * 1.1
    assert ((f["vwap_px"] > lo) & (f["vwap_px"] < hi)).all()


def test_impact_increases_with_size_and_worse_broker():
    b = bars(n=40); m = DayMarket(pd.Timestamp("2020-01-06"), b, seed=4)
    small = m.execute(orders(b, 0.002), "B", seed=5)["cost_true_bp"].mean()
    large = m.execute(orders(b, 0.05), "B", seed=5)["cost_true_bp"].mean()
    assert large > small
    liquid = b[b["spread_bp"] <= market.ILLIQUID_BP]
    if len(liquid) >= 5:
        a = m.execute(orders(liquid, 0.02), "A", seed=6)["cost_true_bp"].mean()
        c = m.execute(orders(liquid, 0.02), "C", seed=6)["cost_true_bp"].mean()
        assert a < c


def test_pov_and_is_and_close_run():
    b = bars(); m = DayMarket(pd.Timestamp("2020-01-06"), b, seed=7)
    for algo in ("POV", "IS", "CLOSE", "TWAP"):
        f = m.execute(orders(b, 0.01, algo), "C", seed=8)
        assert np.allclose(f["filled"], f["qty"]) and np.isfinite(f["avg_px"]).all()
    f = m.execute(orders(b, 0.01, "CLOSE"), "C", seed=8)
    assert (f["auction_qty"] > 0).all()          # CLOSE leaves 70 % for the auction


def test_execute_is_deterministic_given_seed():
    b = bars(); m = DayMarket(pd.Timestamp("2020-01-06"), b, seed=9)
    f1 = m.execute(orders(b), "A", seed=10); f2 = m.execute(orders(b), "A", seed=10)
    assert np.allclose(f1["avg_px"], f2["avg_px"])
