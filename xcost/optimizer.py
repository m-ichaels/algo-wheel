"""Single-period mean-variance optimiser with a transaction-cost penalty from the desk's pre-trade model.

    maximise  alpha' w - (lambda / 2) w' Sigma w - gamma * sum_i c_i(w_i - w0_i)
    s.t.      sum w = 0,  |w_i| <= w_max,  sum |w| <= gross

with, for a trade of d (fraction of capital) in name i, participation x_i = |d| * C / ADV$_i and

    c_i(d) = 1e-4 * [ (a + c * spread_i) |d| + b * sigma_i * (C / ADV$_i)^beta * |d|^(1 + beta) ],

the pre-trade model integrated over the trade (convex for beta > 0).  gamma scales the penalty: 1 is the model as
fitted, below 1 approximates the multi-period amortisation of Garleanu-Pedersen, 0 is the gross backtest.  A
"naive" model is a flat linear cost per unit turnover, the number a researcher writes in a spreadsheet."""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class CostModel:
    """cost_bp(x) = a + b * sigma_bp * x**beta + c * spread_bp, x = shares / ADV; per-name inputs sigma_bp, spread_bp, adv_usd"""
    a: float = 0.0
    b: float = 0.0
    beta: float = 0.5
    c: float = 0.0
    linear_bp: float | None = None    # if set: flat cost per unit turnover, ignoring everything else
    name: str = "sqrt"

    def cost_bp(self, sigma_bp, spread_bp, pct_adv):
        if self.linear_bp is not None:
            return np.full_like(np.asarray(pct_adv, dtype=float), self.linear_bp)
        return self.a + self.b * sigma_bp * np.power(np.maximum(pct_adv, 0.0), self.beta) + self.c * spread_bp


@dataclass
class OptConfig:
    capital: float = 1e9
    risk_aversion: float = 8.0
    w_max: float = 0.04
    gross: float = 2.0
    gamma: float = 1.0
    max_pct_adv: float = 0.10       # desk limit: no order above this share of dollar ADV (a name leaving the universe may always be closed)
    solver: str = "CLARABEL"


def optimise(alpha: np.ndarray, sigma: np.ndarray, w0: np.ndarray, sigma_bp: np.ndarray, spread_bp: np.ndarray, adv_usd: np.ndarray, model: CostModel | None, cfg: OptConfig) -> tuple[np.ndarray, dict]:
    n = len(alpha)
    w = cp.Variable(n)
    d = w - w0
    obj = alpha @ w - 0.5 * cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
    if model is not None and cfg.gamma > 0:
        if model.linear_bp is not None:
            obj -= cfg.gamma * 1e-4 * model.linear_bp * cp.sum(cp.abs(d))
        else:
            lin = 1e-4 * np.maximum(model.a + model.c * spread_bp, 0.0)
            k = 1e-4 * max(model.b, 0.0) * sigma_bp * np.power(cfg.capital / np.maximum(adv_usd, 1.0), model.beta)
            obj -= cfg.gamma * (lin @ cp.abs(d) + k @ cp.power(cp.abs(d), 1.0 + model.beta))
    cap = cfg.max_pct_adv * adv_usd / cfg.capital
    cap = np.where(alpha == 0.0, np.maximum(cap, np.abs(w0)), cap)     # a name that has left the universe may always be closed
    cons = [cp.sum(w) == 0, cp.abs(w) <= cfg.w_max, cp.sum(cp.abs(w)) <= cfg.gross, cp.abs(d) <= cap]
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cfg.solver)
        ok = prob.status in ("optimal", "optimal_inaccurate") and w.value is not None
    except cp.error.SolverError:
        ok = False
    if not ok:
        try:
            prob.solve(solver="SCS", max_iters=20000)
            ok = prob.status in ("optimal", "optimal_inaccurate") and w.value is not None
        except cp.error.SolverError:
            ok = False
    if not ok:
        return w0.copy(), {"status": "failed"}
    wv = np.asarray(w.value).ravel()
    wv[np.abs(wv) < 1e-6] = 0.0
    return wv, {"status": prob.status, "turnover": float(np.abs(wv - w0).sum()), "gross": float(np.abs(wv).sum()), "exp_alpha": float(alpha @ wv), "exp_risk": float(np.sqrt(wv @ sigma @ wv))}
