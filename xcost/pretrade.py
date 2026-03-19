"""Pre-trade cost models fitted on the wheel's executions and tested out of sample.

Three specifications of the cost in bp of an order of Q shares in a name with ADV, daily volatility sigma and spread s:

    sqrt   c = a + b * sigma_bp * sqrt(Q/ADV) + c * s               (the square-root law; Grinold-Kahn, Almgren)
    power  c = a + b * sigma_bp * (Q/ADV)^beta + c * s              (beta free; Almgren et al. 2005 found 0.6)
    gbm    gradient boosting on (sigma_bp, Q/ADV, s, algo, broker)  (the nonparametric check)

each with algorithm and broker dummies where the spec is linear in them.  Two targets: what the desk observes (the
average fill against the arrival mid, which includes the intraday drift and the order's own alpha) and the oracle
(fills against the impact-free path).  The gap between the two fits is the alpha contamination of the desk's model.

The optimiser gets a reduced form without algorithm dummies, fitted on orders routed to the broker the policy will use,
so that the cost is a function of the trade size alone."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor

from .market import BROKER_NAMES
from .optimizer import CostModel
from .wheel import winsorise

ALGOS = ("VWAP", "IS", "POV")


def _dummies(df: pd.DataFrame, algo: bool, broker: bool) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    if algo:
        for a in ALGOS[1:]:
            cols.append((df["algo"] == a).astype(float).values); names.append(f"algo_{a}")
    if broker:
        for b in BROKER_NAMES[1:]:
            cols.append((df["broker"] == b).astype(float).values); names.append(f"broker_{b}")
    return (np.column_stack(cols) if cols else np.zeros((len(df), 0))), names


class Spec:
    def __init__(self, kind: str, algo_dummies: bool = True, broker_dummies: bool = True):
        self.kind, self.algo_d, self.broker_d = kind, algo_dummies, broker_dummies
        self.params: dict = {}

    def _x(self, df):
        return df["sigma_bp"].values, df["pct_adv"].values.clip(min=1e-6), df["spread_bp"].values

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        sig, x, s = self._x(df)
        D, dn = _dummies(df, self.algo_d, self.broker_d)
        if self.kind == "sqrt":
            X = np.column_stack([np.ones(len(df)), sig * np.sqrt(x), s, D])
            lo = np.concatenate([[-np.inf, 0.0, 0.0], np.full(D.shape[1], -np.inf)]); hi = np.full(X.shape[1], np.inf)
            beta = lsq_linear(X, y, bounds=(lo, hi)).x                       # impact and spread coefficients non-negative
            self.params = {"a": beta[0], "b": beta[1], "c": beta[2], "beta": 0.5} | dict(zip(dn, beta[3:]))
        elif self.kind == "power":
            def resid(p):
                a, b, c, bt = p[:4]; d = p[4:]
                return a + b * sig * x ** bt + c * s + D @ d - y
            p0 = np.concatenate([[1.0, 0.15, 0.5, 0.5], np.zeros(D.shape[1])])
            lo = np.concatenate([[-np.inf, 0.0, 0.0, 0.2], np.full(D.shape[1], -np.inf)]); hi = np.concatenate([[np.inf, np.inf, np.inf, 1.0], np.full(D.shape[1], np.inf)])
            r = least_squares(resid, p0, bounds=(lo, hi), loss="linear", max_nfev=2000)
            self.params = {"a": r.x[0], "b": r.x[1], "c": r.x[2], "beta": r.x[3]} | dict(zip(dn, r.x[4:]))
        elif self.kind == "gbm":
            F = self._frame(df)
            self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=100, l2_regularization=1.0, categorical_features=[3, 4], random_state=7).fit(F, y)
            self.params = {"n_iter": int(self.model.n_iter_)}
        else:
            raise ValueError(self.kind)
        return self

    def _frame(self, df):
        return np.column_stack([df["sigma_bp"].values, df["pct_adv"].values, df["spread_bp"].values, pd.Categorical(df["algo"], categories=ALGOS).codes, pd.Categorical(df["broker"], categories=BROKER_NAMES).codes])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        sig, x, s = self._x(df)
        if self.kind == "gbm":
            return self.model.predict(self._frame(df))
        D, dn = _dummies(df, self.algo_d, self.broker_d)
        p = self.params
        return p["a"] + p["b"] * sig * x ** p["beta"] + p["c"] * s + D @ np.array([p[k] for k in dn])

    def to_cost_model(self, broker: str | None = None, name: str = "") -> CostModel:
        p = self.params
        a = p["a"] + (p.get(f"broker_{broker}", 0.0) if broker else 0.0)
        return CostModel(a=float(a), b=float(p["b"]), beta=float(p["beta"]), c=float(p["c"]), name=name or self.kind)


def metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray, y_train_mean: float) -> dict:
    """per-order metrics (equal weights) plus the notional-weighted bias, which is what the fund pays or saves"""
    err = y - pred
    return {"rmse_bp": float(np.sqrt(np.mean(err ** 2))), "mae_bp": float(np.mean(np.abs(err))),
            "rmse_mean_baseline_bp": float(np.sqrt(np.mean((y - y_train_mean) ** 2))),
            "corr": float(np.corrcoef(y, pred)[0, 1]) if len(y) > 3 else None, "bias_bp": float(np.mean(err)), "bias_notional_weighted_bp": float(np.average(err, weights=w)),
            "r2": float(1 - np.mean(err ** 2) / np.var(y))}


def calibration(y: np.ndarray, pred: np.ndarray, w: np.ndarray, n: int = 10) -> list[dict]:
    q = pd.qcut(pd.Series(pred).rank(method="first"), n, labels=False).values
    return [{"decile": int(k), "pred_bp": float(np.mean(pred[q == k])), "realised_bp": float(np.mean(y[q == k])), "realised_se_bp": float(np.std(y[q == k]) / np.sqrt((q == k).sum())), "n": int((q == k).sum())} for k in range(n)]


def fit_and_test(train: pd.DataFrame, test: pd.DataFrame, y_col: str, kinds=("sqrt", "power", "gbm")) -> dict:
    tr, te = train[train["filled"] > 0], test[test["filled"] > 0]
    ytr, yte = winsorise(tr[y_col]).values, winsorise(te[y_col]).values
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "y": y_col, "specs": {}}
    for k in kinds:
        sp = Spec(k).fit(tr, ytr)
        pred = sp.predict(te)
        out["specs"][k] = {"params": {kk: float(v) for kk, v in sp.params.items()}, "test": metrics(yte, pred, te["notional"].values, float(np.average(ytr, weights=tr["notional"].values))),
                           "train": metrics(ytr, sp.predict(tr), tr["notional"].values, float(np.average(ytr, weights=tr["notional"].values))), "calibration": calibration(yte, pred, te["notional"].values)}
    return out


def alpha_contamination(orders: pd.DataFrame) -> dict:
    """the desk's observed cost against the oracle: mean gap overall and by size bucket, and how the sqrt fit differs"""
    o = orders[orders["filled"] > 0]
    w = o["notional"].values
    obs, tru = winsorise(o["exec_arrival_bp"]).values, winsorise(o["cost_true_bp"]).values
    out = {"observed_bp": float(np.mean(obs)), "true_bp": float(np.mean(tru)), "gap_bp": float(np.mean(obs - tru)), "gap_se_bp": float(np.std(obs - tru) / np.sqrt(len(o))),
           "observed_notional_weighted_bp": float(np.average(obs, weights=w)), "true_notional_weighted_bp": float(np.average(tru, weights=w)), "gap_notional_weighted_bp": float(np.average(obs - tru, weights=w))}
    # buys against sells: alpha decay shows up with the same sign on both, a market tilt with opposite signs
    out["gap_buys_bp"] = float(np.mean((obs - tru)[o["side"].values > 0])); out["gap_sells_bp"] = float(np.mean((obs - tru)[o["side"].values < 0]))
    out["delay_bp"] = float(np.mean(winsorise(o["delay_bp"])))
    # the gap is the order's own alpha realised over the execution window plus noise; relate it to the signal
    out["corr_gap_alpha"] = float(np.corrcoef(obs - tru, o["alpha"].values * 1e4)[0, 1])
    by = {}
    for s, g in o.groupby("stratum"):
        ww = g["notional"].values
        by[int(s)] = {"observed_bp": float(np.mean(winsorise(g["exec_arrival_bp"]))), "true_bp": float(np.mean(winsorise(g["cost_true_bp"]))), "n": int(len(g))}
    out["by_stratum"] = by
    s_obs = Spec("sqrt", algo_dummies=False, broker_dummies=True).fit(o, obs); s_tru = Spec("sqrt", algo_dummies=False, broker_dummies=True).fit(o, tru)
    out["sqrt_fit_observed"] = {k: float(v) for k, v in s_obs.params.items()}; out["sqrt_fit_true"] = {k: float(v) for k, v in s_tru.params.items()}
    return out
