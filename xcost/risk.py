"""Risk model: Ledoit-Wolf shrunk covariance of daily returns over the trailing window, scaled to the rebalance horizon."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def covariance(ret: pd.DataFrame, d: pd.Timestamp, symbols: list[str], window: int = 250, horizon: int = 5) -> tuple[np.ndarray, list[str]]:
    """covariance over `horizon` days of the symbols with enough history before d (others are dropped)"""
    hist = ret.loc[:d - pd.Timedelta(days=1), symbols].tail(window)
    ok = [s for s in symbols if hist[s].notna().sum() >= int(0.8 * window)]
    x = hist[ok].fillna(0.0).values
    lw = LedoitWolf().fit(x)
    return lw.covariance_ * horizon, ok
