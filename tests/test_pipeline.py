"""One short end-to-end pass on the committed data: features, a two-year walk-forward, a few weeks of backtest with the
oracle, and the books reconcile (paper minus real equals the summed implementation shortfall)."""
import numpy as np
import pandas as pd
import pytest

from xcost.backtest import RunConfig, run
from xcost.data import Panel, load_prices, week_starts
from xcost.optimizer import CostModel, OptConfig
from xcost.signal import HORIZON, features, ic_series, walk_forward


@pytest.fixture(scope="module")
def panel():
    return Panel(load_prices())


def test_panel_and_week_starts(panel):
    assert panel.close.shape[1] > 100 and panel.dates.is_monotonic_increasing
    ws = week_starts(panel.dates, "2019-01-01", "2019-03-31")
    assert 12 <= len(ws) <= 14 and all(ws[i] < ws[i + 1] for i in range(len(ws) - 1))
    st = panel.stats_at(pd.Timestamp("2019-06-03"))
    assert (st["adv"] > 0).all() and (st["spread_bp"] >= 0.8).all()


def test_features_and_walk_forward(panel):
    f = features(panel)
    for c in ("mom_12_1", "rev_1w", "vol_20"):
        day = f[f["date"] == f["date"].iloc[-3000]]
        assert abs(day[c].mean()) < 0.05 and 0.8 < day[c].std() < 1.2       # rank-normalised within the day
    pred = walk_forward(f, 2012, 2012)
    assert pred["date"].min().year == 2012 and pred["date"].max().year == 2012
    ic = ic_series(pred)
    assert len(ic) > 200 and np.isfinite(ic.mean())
    # no target leakage: the last HORIZON days of the sample have no target
    assert f[f["date"] >= f["date"].max() - pd.Timedelta(days=HORIZON)]["target"].isna().all()


def test_backtest_books_reconcile(panel):
    f = features(panel)
    pred = walk_forward(f, 2012, 2012)
    pred["alpha"] = 0.5 * pred["ic_train"] * pred["sigma"] * np.sqrt(HORIZON) * pred["z"]
    cfg = OptConfig(capital=1e9, risk_aversion=20, w_max=0.02, gross=2.0)
    r = run(panel, pred, RunConfig("t", "2012-02-01", "2012-03-31", model=None, gamma=0.0, routing="wheel", oracle=True, seed=1, opt=cfg))
    assert len(r.weeks) >= 7 and len(r.orders) > 200
    assert abs(r.stats["identity_gap_usd"]) < 1.0
    assert r.stats["solver_failures"] == 0
    o = r.orders
    assert np.allclose(o["filled"], o["qty"]) and (o["pct_adv"] <= 0.12).mean() > 0.97      # the cap is in dollars of 20-day ADV, so shares can sit a little above 10 %
    for b in ("A", "B", "C"):
        assert f"true_bp_{b}" in o.columns
    # cost-aware run trades less
    m = CostModel(a=0.5, b=0.25, beta=0.5, c=0.0)
    r2 = run(panel, pred, RunConfig("t2", "2012-02-01", "2012-03-31", model=m, gamma=1.0, routing="best", best_map={"best": {s: "A" for s in range(6)}, "worst": {s: "C" for s in range(6)}}, seed=1, opt=OptConfig(capital=1e9, risk_aversion=20, w_max=0.02, gross=2.0, gamma=1.0)))
    assert r2.stats["turnover_per_week"] < r.stats["turnover_per_week"]
    assert (r2.orders["broker"] == "A").all()
