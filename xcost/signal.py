"""A deliberately ordinary ML equity signal: eight price-and-volume features rank-normalised across the universe each
day, a five-day forward cross-sectional return target, ridge and gradient boosting fitted walk-forward (refit every
January on all history to date with a five-day embargo), averaged.  Expected returns follow Grinold's rule
alpha_i = IC * sigma_i * z_i with the IC measured on the training window, so the optimiser sees returns in the right
units and the same alpha feeds every portfolio variant.  The point is not the signal's quality: it is what happens to
it once it has to be traded."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

HORIZON = 5
EMBARGO = 5
FEATURES = ["mom_12_1", "rev_1m", "rev_1w", "vol_20", "vol_ratio", "volume_trend", "range_20", "hi_52w"]


def features(panel) -> pd.DataFrame:
    """long table date, symbol, features (cross-sectionally rank-normalised to N(0,1) each day), target"""
    adj, vol, hi, lo = panel.adj, panel.volume, panel.high, panel.low
    r = np.log(adj).diff()
    f = {
        "mom_12_1": adj.shift(21) / adj.shift(252) - 1.0,
        "rev_1m": adj / adj.shift(21) - 1.0,
        "rev_1w": adj / adj.shift(5) - 1.0,
        "vol_20": r.rolling(20).std(),
        "vol_ratio": r.rolling(20).std() / r.rolling(120).std(),
        "volume_trend": np.log(vol.rolling(5).mean() / vol.rolling(60).mean()),
        "range_20": (np.log(hi / lo)).rolling(20).mean(),
        "hi_52w": adj / adj.rolling(252).max() - 1.0,
    }
    fwd = adj.shift(-HORIZON) / adj - 1.0
    target = fwd.sub(fwd.mean(axis=1), axis=0)
    rows = []
    for name, x in f.items():
        rows.append(x.stack(future_stack=True).rename(name))
    df = pd.concat(rows + [target.stack(future_stack=True).rename("target"), panel.vol.stack(future_stack=True).rename("sigma")], axis=1)
    df.index.names = ["date", "symbol"]
    df = df.dropna(subset=FEATURES)
    # rank-normalise features within each day
    g = df.groupby(level="date")
    for c in FEATURES:
        rk = g[c].rank(pct=True)
        n_day = g[c].transform("size")
        df[c] = norm.ppf(((rk * n_day - 0.5) / n_day).clip(0.001, 0.999))
    return df.reset_index()


def walk_forward(df: pd.DataFrame, first_year: int, last_year: int, min_train_years: int = 2, seed: int = 7) -> pd.DataFrame:
    """predictions for each date in [first_year, last_year] from models fitted on data ending EMBARGO+HORIZON days before
    1 January of the prediction year; ridge and gradient boosting averaged after z-scoring; per-day IC and Grinold alpha."""
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    out = []
    ic_hist = {}
    for year in range(first_year, last_year + 1):
        cut = pd.Timestamp(f"{year}-01-01") - pd.tseries.offsets.BDay(EMBARGO + HORIZON)
        tr = df[(df["date"] <= cut) & df["target"].notna()]
        te = df[(df["date"] >= f"{year}-01-01") & (df["date"] <= f"{year}-12-31")]
        if tr["date"].nunique() < 250 * min_train_years or te.empty:
            continue
        X, y = tr[FEATURES].values, tr["target"].values
        ridge = Ridge(alpha=10.0).fit(X, y)
        gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0, random_state=seed).fit(X, y)
        te = te.copy()
        p1, p2 = ridge.predict(te[FEATURES].values), gbm.predict(te[FEATURES].values)
        te["pred_ridge"], te["pred_gbm"] = p1, p2
        # in-sample IC of the blend on the training window, the scale for alpha
        tr_p = 0.5 * zscore_by_day(tr, ridge.predict(X)) + 0.5 * zscore_by_day(tr, gbm.predict(X))
        ic_tr = float(pd.DataFrame({"d": tr["date"].values, "p": tr_p, "y": y}).groupby("d").apply(lambda g: spearmanr(g["p"], g["y"])[0] if len(g) > 10 else np.nan, include_groups=False).mean())
        ic_hist[year] = ic_tr
        te["z"] = 0.5 * zscore_by_day(te, p1) + 0.5 * zscore_by_day(te, p2)
        te["ic_train"] = ic_tr
        te["alpha"] = ic_tr * te["sigma"] * np.sqrt(HORIZON) * te["z"]      # expected 5-day return, Grinold
        out.append(te)
    res = pd.concat(out, ignore_index=True)
    return res


def zscore_by_day(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    s = pd.Series(p, index=df.index)
    g = s.groupby(df["date"].values)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).values


def ic_series(pred: pd.DataFrame, col: str = "z") -> pd.Series:
    """Spearman IC per date between the prediction and the realised 5-day cross-sectional return"""
    p = pred.dropna(subset=["target"])
    return p.groupby("date").apply(lambda g: spearmanr(g[col], g["target"])[0] if len(g) > 10 else np.nan, include_groups=False).dropna()


def decile_returns(pred: pd.DataFrame, col: str = "z", n: int = 10) -> pd.Series:
    p = pred.dropna(subset=["target"]).copy()
    p["dec"] = p.groupby("date")[col].transform(lambda s: pd.qcut(s.rank(method="first"), n, labels=False))
    return p.groupby("dec")["target"].mean()
