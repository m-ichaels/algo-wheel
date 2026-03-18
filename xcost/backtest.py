"""The weekly rebalancing backtest that turns the signal into orders, executes them through the simulated brokers and
keeps two books: the paper book that trades at the decision price (the previous close, what a gross backtest assumes)
and the real book that trades at the fills.  The difference is exactly the implementation shortfall of the orders
(Perold 1988), decomposed per order into delay (decision to arrival), execution (arrival to average fill) and fees."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Panel, week_starts
from .market import BROKER_NAMES, FEE_BP, ILLIQUID_BP, DayMarket
from .optimizer import CostModel, OptConfig, optimise
from .risk import covariance

MIN_NOTIONAL = 5_000.0
MIN_ADV_USD = 50e6             # liquidity screen: names below this dollar ADV get no alpha (held positions are still closed)
SIZE_EDGES = (0.005, 0.02)     # pct of ADV: small / medium / large


def size_bucket(pct_adv: np.ndarray) -> np.ndarray:
    return np.digitize(pct_adv, SIZE_EDGES)


def stratum(pct_adv: np.ndarray, spread_bp: np.ndarray) -> np.ndarray:
    """six strata: liquidity (spread above / below ILLIQUID_BP) x size bucket"""
    return (spread_bp > ILLIQUID_BP).astype(int) * 3 + size_bucket(pct_adv)


def algo_policy(pct_adv: np.ndarray, policy: str) -> pd.DataFrame:
    n = len(pct_adv)
    if policy == "vwap":
        return pd.DataFrame({"algo": ["VWAP"] * n, "urgency": ["low"] * n, "pov": np.zeros(n)})
    if policy == "urgent":
        return pd.DataFrame({"algo": ["IS"] * n, "urgency": ["high"] * n, "pov": np.zeros(n)})
    # by size: small orders VWAP, medium implementation shortfall, large POV at 10 %
    b = size_bucket(pct_adv)
    algo = np.where(b == 0, "VWAP", np.where(b == 1, "IS", "POV"))
    urg = np.where(b == 1, "medium", "low")
    return pd.DataFrame({"algo": algo, "urgency": urg, "pov": np.where(b == 2, 0.10, 0.0)})


class WheelAssigner:
    """stratified randomisation: within each stratum brokers are dealt in shuffled blocks so counts stay balanced"""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed); self.queues: dict[int, list] = {}

    def assign(self, strata: np.ndarray) -> np.ndarray:
        out = np.empty(len(strata), dtype=object)
        for i, s in enumerate(strata):
            q = self.queues.setdefault(int(s), [])
            if not q:
                q.extend(self.rng.permutation(BROKER_NAMES).tolist())
            out[i] = q.pop()
        return out


@dataclass
class RunConfig:
    name: str
    start: str
    end: str
    model: CostModel | None = None
    gamma: float = 0.0
    routing: str = "wheel"           # wheel | random | A | B | C | best | worst
    best_map: dict | None = None     # stratum -> broker, for routing best / worst
    policy: str = "size"             # size | vwap | urgent
    oracle: bool = False             # execute every order with every broker as well
    seed: int = 7
    opt: OptConfig = field(default_factory=OptConfig)


@dataclass
class RunResult:
    name: str
    weeks: pd.DataFrame
    orders: pd.DataFrame
    nav: pd.DataFrame        # daily: nav_real, nav_paper
    stats: dict


def run(panel: Panel, pred: pd.DataFrame, cfg: RunConfig) -> RunResult:
    t_start = time.time()
    C = cfg.opt.capital
    alpha_by_date = {d: g.set_index("symbol")["alpha"] for d, g in pred.groupby("date")}
    dates = panel.dates
    rebal = week_starts(dates, cfg.start, cfg.end)
    shares: dict[str, float] = {}
    cash_real = C; cash_paper = C
    wheel = WheelAssigner(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    weeks, orders_all = [], []
    holdings_rows = []
    for wi, d in enumerate(rebal):
        t0 = dates[dates < d][-1]
        if t0 not in alpha_by_date:
            continue
        st = panel.stats_at(d)
        alpha = alpha_by_date[t0]
        syms = sorted(set(alpha.index) & set(st.index) | ({s for s in shares if shares[s] != 0} & set(st.index)))
        sigma, syms = covariance(panel.ret, d, syms)
        if len(syms) < 20:
            continue
        st = st.loc[syms]
        px0 = st["close"].values
        a = alpha.reindex(syms).fillna(0.0).values.copy()
        a[st["adv_usd"].values < MIN_ADV_USD] = 0.0
        w0 = np.array([shares.get(s, 0.0) for s in syms]) * px0 / C
        w, info = optimise(a, sigma, w0, st["vol"].values * 1e4, st["spread_bp"].values, st["adv_usd"].values, cfg.model, cfg.opt)
        target_shares = np.round(w * C / px0)
        delta = target_shares - np.array([shares.get(s, 0.0) for s in syms])
        # bars for the trading day
        has_bar = panel.close.loc[d].reindex(syms).notna().values & panel.open.loc[d].reindex(syms).notna().values & panel.volume.loc[d].reindex(syms).notna().values
        trade = (np.abs(delta) * px0 >= MIN_NOTIONAL) & has_bar
        od = pd.DataFrame({"symbol": np.array(syms)[trade], "side": np.sign(delta[trade]), "qty": np.abs(delta[trade]), "decision_px": px0[trade], "alpha": a[trade], "w0": w0[trade], "w_target": w[trade]})
        if len(od):
            bars = pd.DataFrame({"open": panel.open.loc[d, od["symbol"]].values, "close": panel.close.loc[d, od["symbol"]].values, "volume": panel.volume.loc[d, od["symbol"]].values,
                                 "vol": st.loc[od["symbol"], "vol"].values, "spread_bp": st.loc[od["symbol"], "spread_bp"].values, "adv": st.loc[od["symbol"], "adv"].values}, index=od["symbol"].values)
            pct = od["qty"].values / bars["adv"].values
            od = pd.concat([od.reset_index(drop=True), algo_policy(pct, cfg.policy)], axis=1)
            od["pct_adv"] = pct; od["stratum"] = stratum(pct, bars["spread_bp"].values)
            od["spread_bp"] = bars["spread_bp"].values; od["adv_usd"] = st.loc[od["symbol"], "adv_usd"].values
            od["broker"] = route(od, cfg, wheel, rng)
            mkt = DayMarket(d, bars, seed=int(rng.integers(1 << 31)))
            fills = []
            for b in BROKER_NAMES:
                sub = od[od["broker"] == b]
                if len(sub):
                    f = mkt.execute(sub, b, seed=int(rng.integers(1 << 31)))
                    fills.append(f)
            f = pd.concat(fills).set_index("symbol").loc[od["symbol"].values].reset_index()
            for col in ("filled", "avg_px", "arrival_px", "vwap_px", "close_px", "auction_qty", "minutes", "cost_true_bp", "spread_paid_bp", "sigma_bp"):
                od[col] = f[col].values
            if cfg.oracle:
                for b in BROKER_NAMES:
                    fb = mkt.execute(od, b, seed=int(rng.integers(1 << 31))).set_index("symbol").loc[od["symbol"].values]
                    od[f"exec_bp_{b}"] = (fb["side"] * (fb["avg_px"] / fb["arrival_px"] - 1) * 1e4).values
                    od[f"true_bp_{b}"] = fb["cost_true_bp"].values
                    od[f"vwap_bp_{b}"] = (fb["side"] * (fb["avg_px"] / fb["vwap_px"] - 1) * 1e4).values
            # Perold decomposition in bp of the decision price, and dollars
            s, dec = od["side"], od["decision_px"]
            od["delay_bp"] = s * (od["arrival_px"] - dec) / dec * 1e4
            od["exec_bp"] = s * (od["avg_px"] - od["arrival_px"]) / dec * 1e4
            od["exec_arrival_bp"] = s * (od["avg_px"] / od["arrival_px"] - 1) * 1e4
            od["vwap_slip_bp"] = s * (od["avg_px"] / od["vwap_px"] - 1) * 1e4
            od["fees_bp"] = FEE_BP * od["avg_px"] / dec
            od["is_bp"] = od["delay_bp"] + od["exec_bp"] + od["fees_bp"]
            od["notional"] = od["filled"] * od["decision_px"]
            od["is_usd"] = od["is_bp"] * 1e-4 * od["notional"]        # = side * (avg - decision) * filled + fees on the fill notional
            od["date"] = d; od["decision_date"] = t0; od["week"] = wi
            # books
            for r in od.itertuples(index=False):
                shares[r.symbol] = shares.get(r.symbol, 0.0) + r.side * r.filled
                cash_real -= r.side * r.filled * r.avg_px + FEE_BP * 1e-4 * r.filled * r.avg_px
                cash_paper -= r.side * r.filled * r.decision_px
            orders_all.append(od)
        holdings_rows.append((d, dict(shares)))
        weeks.append({"date": d, "decision_date": t0, "n_names": len(syms), "n_orders": int(len(od)), "turnover": float(np.abs(w - w0).sum()), "gross": float(np.abs(w).sum()),
                      "exp_alpha": info.get("exp_alpha", np.nan), "exp_risk": info.get("exp_risk", np.nan), "status": info.get("status"), "traded_usd": float((od["qty"] * od["decision_px"]).sum()) if len(od) else 0.0,
                      "is_usd": float(od["is_usd"].sum()) if len(od) else 0.0, "cash_real": cash_real, "cash_paper": cash_paper})
    weeks = pd.DataFrame(weeks)
    orders = pd.concat(orders_all, ignore_index=True) if orders_all else pd.DataFrame()
    nav = mark(panel, weeks, holdings_rows, cfg.end, C)
    stats = summarise(nav, orders, weeks, C)
    stats["seconds"] = round(time.time() - t_start, 1)
    return RunResult(cfg.name, weeks, orders, nav, stats)


def route(od: pd.DataFrame, cfg: RunConfig, wheel: WheelAssigner, rng: np.random.Generator) -> np.ndarray:
    if cfg.routing == "wheel":
        return wheel.assign(od["stratum"].values)
    if cfg.routing == "random":
        return rng.choice(BROKER_NAMES, len(od))
    if cfg.routing in BROKER_NAMES:
        return np.full(len(od), cfg.routing, dtype=object)
    if cfg.routing in ("best", "worst"):
        m = cfg.best_map[cfg.routing]
        return np.array([m[int(s)] for s in od["stratum"].values], dtype=object)
    raise ValueError(cfg.routing)


def mark(panel: Panel, weeks: pd.DataFrame, holdings_rows: list, end: str, C: float) -> pd.DataFrame:
    """daily NAV of the real and paper books: cash after the week's trades plus holdings at the real close"""
    if weeks.empty:
        return pd.DataFrame()
    days = panel.dates[(panel.dates >= weeks["date"].iloc[0]) & (panel.dates <= end)]
    close = panel.close.reindex(days).ffill()
    hold = pd.DataFrame([sh for _, sh in holdings_rows], index=[d for d, _ in holdings_rows]).reindex(columns=close.columns).fillna(0.0)
    hold = hold.reindex(days).ffill().fillna(0.0)
    cash_r = weeks.set_index("date")["cash_real"].reindex(days).ffill()
    cash_p = weeks.set_index("date")["cash_paper"].reindex(days).ffill()
    mv = (hold * close).sum(axis=1)
    out = pd.DataFrame({"nav_real": cash_r + mv, "nav_paper": cash_p + mv, "gross_mv": (hold * close).abs().sum(axis=1)})
    out["ret_real"] = out["nav_real"].diff() / C
    out["ret_paper"] = out["nav_paper"].diff() / C
    return out


def summarise(nav: pd.DataFrame, orders: pd.DataFrame, weeks: pd.DataFrame, C: float) -> dict:
    if nav.empty:
        return {}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    def ann(kind):
        r = nav[f"ret_{kind}"].dropna()
        return {"ann_return_bp": float(r.mean() * 252 * 1e4), "ann_vol_bp": float(r.std() * np.sqrt(252) * 1e4), "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
                "total_pnl_usd": float(nav[f"nav_{kind}"].iloc[-1] - C), "max_drawdown_usd": float((nav[f"nav_{kind}"].cummax() - nav[f"nav_{kind}"]).max())}
    out = {"years": round(years, 2), "real": ann("real"), "paper": ann("paper")}
    if len(orders):
        w = orders["notional"]
        out["orders"] = {"n": int(len(orders)), "traded_usd": float(w.sum()), "traded_per_year_usd": float(w.sum() / years),
                         "is_bp": float(np.average(orders["is_bp"], weights=w)), "delay_bp": float(np.average(orders["delay_bp"], weights=w)),
                         "exec_bp": float(np.average(orders["exec_bp"], weights=w)), "fees_bp": float(np.average(orders["fees_bp"], weights=w)),
                         "true_cost_bp": float(np.average(orders["cost_true_bp"], weights=w)), "spread_paid_bp": float(np.average(orders["spread_paid_bp"], weights=w)),
                         "is_usd_per_year": float(orders["is_usd"].sum() / years), "is_bp_of_capital_per_year": float(orders["is_usd"].sum() / years / C * 1e4),
                         "median_pct_adv": float(orders["pct_adv"].median()), "p90_pct_adv": float(orders["pct_adv"].quantile(0.9)), "auction_share": float(orders["auction_qty"].sum() / orders["filled"].sum())}
        out["identity_gap_usd"] = float((nav["nav_paper"].iloc[-1] - nav["nav_real"].iloc[-1]) - orders["is_usd"].sum())
    out["turnover_per_week"] = float(weeks["turnover"].mean()); out["gross"] = float(weeks["gross"].mean())
    out["solver_failures"] = int((weeks["status"] == "failed").sum())
    if out["paper"]["ann_return_bp"] > 0:
        out["alpha_survival"] = out["real"]["ann_return_bp"] / out["paper"]["ann_return_bp"]
    return out
