"""Daily bars (Yahoo chart API, committed as parquet), the US universe, and the per-symbol liquidity statistics the
market simulator and the cost model share: 20-day ADV in shares and dollars, daily volatility, a spread rule."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")


def load_universe(exchange: str = "XNYS") -> pd.DataFrame:
    u = pd.read_csv(os.path.join(ROOT, "data", "universe.csv"))
    return u[u["exchange"] == exchange].reset_index(drop=True)


def load_prices(exchange: str = "XNYS") -> pd.DataFrame:
    """long table symbol, date, open, high, low, close, adjclose, volume for the exchange's universe"""
    p = pd.read_parquet(os.path.join(DER, "prices.parquet"))
    syms = set(load_universe(exchange)["symbol"])
    p = p[p["symbol"].isin(syms)].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["close"] > 0) & (p["open"] > 0) & (p["volume"] > 0)]
    return p.sort_values(["symbol", "date"]).reset_index(drop=True)


class Panel:
    """wide daily panels indexed by date with one column per symbol, plus the rolling liquidity statistics"""

    def __init__(self, prices: pd.DataFrame, lookback: int = 20):
        pv = lambda f: prices.pivot(index="date", columns="symbol", values=f).sort_index()
        self.open, self.close, self.adj, self.volume = pv("open"), pv("close"), pv("adjclose"), pv("volume")
        self.high, self.low = pv("high"), pv("low")
        self.ret = self.adj.pct_change(fill_method=None)
        self.dates = self.close.index
        # statistics known at the close of day t (used for orders on t+1): trailing windows ending at t
        self.adv = self.volume.rolling(lookback, min_periods=10).mean()
        self.adv_usd = (self.volume * self.close).rolling(lookback, min_periods=10).mean()
        self.vol = np.log(self.adj).diff().rolling(lookback, min_periods=10).std().clip(lower=0.004)
        self.spread_bp = spread_rule(self.adv_usd)

    def stats_at(self, d: pd.Timestamp) -> pd.DataFrame:
        """ADV, dollar ADV, daily vol, spread and close as of the close of the last day before d"""
        prev = self.dates[self.dates < d]
        if len(prev) == 0:
            return pd.DataFrame()
        t = prev[-1]
        df = pd.DataFrame({"adv": self.adv.loc[t], "adv_usd": self.adv_usd.loc[t], "vol": self.vol.loc[t], "spread_bp": self.spread_bp.loc[t], "close": self.close.loc[t]})
        return df.dropna()


def spread_rule(adv_usd: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """quoted spread in bp from dollar ADV: about 1.2 bp for a $10bn name, 5 bp at $100m, 13 bp at $10m"""
    return (0.8 + 40.0 / np.sqrt((adv_usd / 1e6).clip(lower=1.0))).clip(0.8, 30.0)


def week_starts(dates: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """first trading day of each ISO week inside [start, end]"""
    d = dates[(dates >= start) & (dates <= end)]
    iso = pd.Series(d.isocalendar().week.values * 100 + d.isocalendar().year.values, index=d)
    return [g.index[0] for _, g in iso.groupby(iso.values, sort=False)]
