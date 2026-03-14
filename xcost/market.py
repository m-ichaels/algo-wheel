"""Reduced-form intraday market and broker algorithms, vectorised over a day's orders.

For each symbol traded on a day: a one-minute mid path bridged in log price from the real open to the real close with
the trailing daily volatility, the real daily volume on a U-shaped curve with noise, and a quoted spread from the
liquidity rule.  A child order of q shares in a minute with market volume v pays

    cost_bp = f_b * s/2 + eta_b(x) * sigma_bp * (q / v) ** BETA * exp(noise)          (temporary, per child)

and moves the mid permanently by  psi * sigma_bp * (executed so far, signed) / V_day  (linear, Almgren-Chriss).
The generator's exponent BETA = 0.6 and the broker interaction eta_b(x) are deliberately not the square-root model the
desk fits in pretrade.py, so the fitted model is misspecified as it would be in practice.

Brokers differ in impact (eta), in how much of the half spread they pay (f_b, a proxy for routing and passive fills),
in schedule tracking noise, and in an interaction with liquidity: broker A is cheapest in liquid names and worst in
illiquid ones, broker C the reverse.  A wheel that pools over difficulty finds A; a stratified one finds the switch.

Because the simulator is ours, every order also carries its oracle: the impact-free fill price along the same
schedule, so the true cost of impact and spread (excluding the day's drift and the order's own alpha) is known."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_MIN = 390
BETA = 0.6            # exponent of participation in the temporary impact
PSI = 0.314           # permanent impact per unit of day-volume participation, in units of sigma (Almgren et al. 2005)
NOISE_SD = 0.35       # lognormal noise on each child's temporary impact
FILL_CAP = 0.35       # a child may take at most this share of the minute's volume
AUCTION_DEPTH = 0.03  # closing auction depth as a share of the day's volume
AUCTION_IMPACT = 0.5  # impact multiplier in the auction
ILLIQUID_BP = 5.0     # spread above which a name counts as illiquid for the broker interaction
FEE_BP = 0.3          # commissions and fees per filled notional


@dataclass(frozen=True)
class Broker:
    name: str
    eta: float             # impact multiplier in liquid names
    eta_illiquid: float    # additive change to eta when spread_bp > ILLIQUID_BP
    spread_capture: float  # fraction of the half spread paid
    sched_noise: float     # sd of the multiplicative noise on the schedule, per minute


# eta around the Almgren-Thum-Hauptmann-Li (2005) temporary-impact coefficient 0.142 (cost in units of daily sigma at
# participation rate x, exponent 0.6); PSI near their permanent coefficient 0.314
BROKERS = {
    "A": Broker("A", 0.110, +0.080, 0.75, 0.20),
    "B": Broker("B", 0.142, 0.000, 1.00, 0.12),
    "C": Broker("C", 0.175, -0.090, 1.15, 0.10),
}
BROKER_NAMES = tuple(BROKERS)


def u_shape(n: int = N_MIN) -> np.ndarray:
    x = np.linspace(0, 1, n)
    w = 1.0 + 2.5 * np.exp(-x / 0.08) + 3.5 * np.exp(-(1 - x) / 0.06) + 0.3 * np.cos(2 * np.pi * x) ** 2
    return w / w.sum()


U_SHAPE = u_shape()
IS_TAU = {"low": 0.6, "medium": 0.3, "high": 0.12}     # implementation-shortfall schedules: exp(-k / (tau n))


def schedule(algo: str, urgency: str, n: int, rng: np.random.Generator, noise: float) -> tuple[np.ndarray, int]:
    """target fraction of the parent per minute and the minute the schedule ends (exclusive); POV is handled separately"""
    start = 5                                             # skip the opening auction
    if algo == "VWAP":
        s = U_SHAPE.copy()
    elif algo == "TWAP":
        s = np.ones(n)
    elif algo == "IS":
        k = np.arange(n); s = np.exp(-k / (IS_TAU[urgency] * n))
    elif algo == "CLOSE":
        s = np.zeros(n); s[n - 60:] = 1.0                 # 30 % over the last hour, the rest in the auction (handled by leftover)
        s = s / s.sum() * 0.3
        s[:start] = 0.0
        return s, n
    else:
        raise ValueError(algo)
    s[:start] = 0.0
    s = s * np.exp(rng.normal(0, noise, n))
    return s / s.sum(), n


class DayMarket:
    """the market for one day: symbols with their open, close, volume, vol and spread known for the day"""

    def __init__(self, d: pd.Timestamp, bars: pd.DataFrame, seed: int, n: int = N_MIN):
        """bars: index symbol; columns open, close, volume (real day), vol (trailing daily sigma), spread_bp"""
        self.d, self.n = d, n
        self.bars = bars
        self.rng = np.random.default_rng(seed)
        S = len(bars)
        sig = bars["vol"].values[:, None]
        steps = self.rng.normal(0, 1, (S, n - 1)) * sig / np.sqrt(n)
        w = np.concatenate([np.zeros((S, 1)), np.cumsum(steps, axis=1)], axis=1)
        x = np.arange(n) / (n - 1)
        bridge = w - x[None, :] * w[:, -1:]
        logp = np.log(bars["open"].values)[:, None] + x[None, :] * (np.log(bars["close"].values) - np.log(bars["open"].values))[:, None] + bridge
        self.mid = np.exp(logp)                                                        # S x n, impact free
        curve = U_SHAPE[None, :] * np.exp(self.rng.normal(0, 0.15, (S, n)) - 0.5 * 0.15 ** 2)
        self.vol_curve = curve / curve.sum(axis=1, keepdims=True) * bars["volume"].values[:, None]   # S x n shares
        self.index = {s: i for i, s in enumerate(bars.index)}

    def execute(self, orders: pd.DataFrame, broker: str | Broker, seed: int | None = None) -> pd.DataFrame:
        """orders: symbol, side (+1 buy / -1 sell), qty (shares), algo, urgency, pov (participation for POV).
        One order per symbol.  Returns per order: filled, avg_px, arrival_px, vwap_px (interval VWAP incl. our fills),
        close_px, auction_qty, cost_true_bp (fills vs the impact-free mid on the same minutes), spread_paid_bp, minutes."""
        b = BROKERS[broker] if isinstance(broker, str) else broker
        rng = np.random.default_rng(seed if seed is not None else self.rng.integers(1 << 31))
        n = self.n
        idx = np.array([self.index[s] for s in orders["symbol"]])
        m = len(idx)
        side = orders["side"].values.astype(float); Q = orders["qty"].values.astype(float)
        mid = self.mid[idx]; vcur = self.vol_curve[idx]
        bars = self.bars.iloc[idx]
        sigma_bp = bars["vol"].values * 1e4; spread = bars["spread_bp"].values; V = bars["volume"].values
        eta = b.eta + np.where(spread > ILLIQUID_BP, b.eta_illiquid, 0.0)
        # schedules
        sched = np.zeros((m, n)); pov_rate = np.zeros(m)
        for i, (algo, urg, pov) in enumerate(zip(orders["algo"].values, orders["urgency"].values, orders["pov"].values)):
            if algo == "POV":
                pov_rate[i] = pov
            else:
                sched[i], _ = schedule(algo, urg, n, rng, b.sched_noise)
        is_pov = pov_rate > 0
        filled = np.zeros(m); notional = np.zeros(m); notional_free = np.zeros(m); perm_bp = np.zeros(m)
        first_min = np.full(m, -1); last_min = np.zeros(m, dtype=int); minutes = np.zeros(m)
        spread_paid = np.zeros(m)
        backlog = np.zeros(m)
        mkt_pv = np.zeros(m); mkt_v = np.zeros(m)          # interval VWAP accumulators (market + us)
        for k in range(5, n):
            v = vcur[:, k]
            open_at_start = filled < Q
            want = np.where(is_pov, pov_rate * v, sched[:, k] * Q) + backlog
            want = np.minimum(want, Q - filled)
            q = np.floor(np.minimum(want, FILL_CAP * v))
            backlog = np.where(is_pov, 0.0, want - q)
            trade = q > 0
            part = q / np.maximum(v, 1.0)
            temp = b.spread_capture * 0.5 * spread + eta * sigma_bp * part ** BETA * np.exp(rng.normal(0, NOISE_SD, m) - 0.5 * NOISE_SD ** 2)
            px_free = mid[:, k]
            px = px_free * (1 + side * (perm_bp + 0.5 * PSI * sigma_bp * q / V + temp) * 1e-4)
            filled += q; notional += q * px; notional_free += q * px_free
            spread_paid += q * px_free * b.spread_capture * 0.5 * spread * 1e-4
            perm_bp += PSI * sigma_bp * q / V
            newly = trade & (first_min < 0); first_min[newly] = k
            last_min = np.where(trade, k, last_min); minutes += trade
            # interval VWAP: market volume at the (impacted) mid plus our fills, over the minutes the order was open
            mkt_pv += np.where(open_at_start, v * px_free * (1 + side * perm_bp * 1e-4) + q * px, 0.0)
            mkt_v += np.where(open_at_start, v + q, 0.0)
        # leftover into the closing auction
        left = Q - filled
        auction = left > 0
        close_free = mid[:, -1]
        if auction.any():
            part = left / np.maximum(AUCTION_DEPTH * V, 1.0)
            temp = AUCTION_IMPACT * (b.spread_capture * 0.5 * spread + eta * sigma_bp * part ** BETA)
            px = close_free * (1 + side * (perm_bp + temp) * 1e-4)
            notional += left * px * auction; notional_free += left * close_free * auction; filled += left * auction
            spread_paid += left * close_free * AUCTION_IMPACT * b.spread_capture * 0.5 * spread * 1e-4 * auction
        avg_px = notional / np.maximum(filled, 1); avg_free = notional_free / np.maximum(filled, 1)
        arrival = mid[:, 5]
        out = pd.DataFrame({
            "symbol": orders["symbol"].values, "side": side, "qty": Q, "filled": filled, "avg_px": avg_px,
            "arrival_px": arrival, "vwap_px": np.where(mkt_v > 0, mkt_pv / np.maximum(mkt_v, 1), arrival), "close_px": close_free,
            "auction_qty": left * auction, "minutes": minutes,
            "cost_true_bp": side * (avg_px - avg_free) / arrival * 1e4,
            "spread_paid_bp": spread_paid / np.maximum(notional_free, 1) * 1e4,
            "sigma_bp": sigma_bp, "spread_bp": spread, "adv": bars["adv"].values, "pct_adv": Q / bars["adv"].values,
            "broker": b.name, "date": self.d,
        })
        return out
