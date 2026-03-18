"""The broker algo wheel as an experiment.

Orders are assigned to brokers by stratified randomisation (backtest.WheelAssigner), so the broker effect is identified
without a selection story.  Estimation is least squares of the order's cost on its difficulty
(sigma * sqrt(Q/ADV), spread, algorithm) with broker fixed effects, pooled and with a broker x liquidity interaction;
uncertainty from a cluster bootstrap over trading days (orders on one day share the market).  Then the things a
research team asks for: the probability each broker is best, the orders per broker needed to resolve a given
difference at 80 % power, how the estimate converged as weeks accrued, and - because this is a simulator - the oracle:
each order was also executed with every other broker, so the true paired effect is known and the estimator can be
checked against it.  Finally the cost of learning: Thompson sampling per stratum against fixed randomisation and the
oracle-best routing, on the same orders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market import BROKER_NAMES, ILLIQUID_BP

STRATA = {0: "liquid / small", 1: "liquid / medium", 2: "liquid / large", 3: "illiquid / small", 4: "illiquid / medium", 5: "illiquid / large"}
Z80 = 1.96 + 0.8416


def winsorise(y: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = y.quantile(p), y.quantile(1 - p)
    return y.clip(lo, hi)


def design(df: pd.DataFrame, interaction: bool) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(len(df)), (df["sigma_bp"] * np.sqrt(df["pct_adv"])).values, df["spread_bp"].values]
    names = ["const", "impact", "spread"]
    ill = (df["spread_bp"] > ILLIQUID_BP).astype(float).values
    for a in ("IS", "POV"):
        cols.append((df["algo"] == a).astype(float).values); names.append(f"algo_{a}")
    for b in BROKER_NAMES[1:]:
        cols.append((df["broker"] == b).astype(float).values); names.append(f"broker_{b}")
    if interaction:
        cols.append(ill); names.append("illiquid")
        for b in BROKER_NAMES[1:]:
            cols.append((df["broker"] == b).astype(float).values * ill); names.append(f"broker_{b}_x_illiquid")
    return np.column_stack(cols), names


def wls(df: pd.DataFrame, y: np.ndarray, interaction: bool) -> tuple[dict, np.ndarray]:
    """ordinary least squares: the noise in an order's cost is roughly homoskedastic in bp, so equal weights are the
    efficient choice; notional-weighted raw means are reported alongside because that is what the fund pays"""
    X, names = design(df, interaction)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(zip(names, beta)), y - X @ beta


def effects_from(coef: dict, illiquid: bool) -> dict:
    out = {BROKER_NAMES[0]: 0.0}
    for b in BROKER_NAMES[1:]:
        e = coef[f"broker_{b}"]
        if illiquid and f"broker_{b}_x_illiquid" in coef:
            e += coef[f"broker_{b}_x_illiquid"]
        out[b] = float(e)
    return out


def cluster_bootstrap(df: pd.DataFrame, y: np.ndarray, interaction: bool, B: int, rng: np.random.Generator) -> list[dict]:
    """resample trading days with replacement (orders on one day share the market), refit on the stacked design"""
    X, names = design(df, interaction)
    days = df["date"].values; uniq, inv = np.unique(days, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    groups = [order[starts[i]:starts[i + 1]] for i in range(len(uniq))]
    draws = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[i] for i in pick])
        try:
            beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        draws.append(dict(zip(names, beta)))
    return draws


def analyse(orders: pd.DataFrame, y_col: str = "exec_arrival_bp", B: int = 1000, seed: int = 7, oracle_prefix: str = "true_bp_") -> dict:
    """oracle_prefix: the per-broker oracle columns the estimates are checked against (true_bp_ for arrival-type costs,
    vwap_bp_ for costs against the interval VWAP, which our own fills sit inside)"""
    o = orders[orders["filled"] > 0].copy()
    y = winsorise(o[y_col]).values
    rng = np.random.default_rng(seed)
    out = {"n_orders": int(len(o)), "n_days": int(o["date"].nunique()), "y": y_col, "oracle": oracle_prefix,
           "counts": o.groupby(["stratum", "broker"]).size().unstack().fillna(0).astype(int).to_dict()}
    # pooled
    coef, resid = wls(o, y, interaction=False)
    draws = cluster_bootstrap(o, y, False, B, rng)
    eff = effects_from(coef, False)
    boots = {b: np.array([effects_from(c, False)[b] for c in draws]) for b in BROKER_NAMES}
    best = pd.Series([min(BROKER_NAMES, key=lambda b: effects_from(c, False)[b]) for c in draws]).value_counts(normalize=True)
    out["pooled"] = {"coefficients": {k: float(v) for k, v in coef.items()}, "residual_sd_bp": float(np.std(resid)),
                     "effects_vs_A": {b: {"bp": eff[b], "ci95": [float(np.quantile(boots[b], 0.025)), float(np.quantile(boots[b], 0.975))], "p_best": float(best.get(b, 0.0)),
                                          "raw_bp": float(np.mean(y[o["broker"].values == b])), "raw_notional_weighted_bp": float(np.average(y[o["broker"].values == b], weights=o.loc[o["broker"] == b, "notional"]))} for b in BROKER_NAMES},
                     "ranking": sorted(BROKER_NAMES, key=lambda b: eff[b])}
    # interaction: liquid and illiquid effects
    coef_i, resid_i = wls(o, y, interaction=True)
    draws_i = cluster_bootstrap(o, y, True, B, rng)
    out["by_liquidity"] = {}
    for name, ill in (("liquid", False), ("illiquid", True)):
        e = effects_from(coef_i, ill)
        bs = {b: np.array([effects_from(c, ill)[b] for c in draws_i]) for b in BROKER_NAMES}
        bst = pd.Series([min(BROKER_NAMES, key=lambda b: effects_from(c, ill)[b]) for c in draws_i]).value_counts(normalize=True)
        out["by_liquidity"][name] = {b: {"bp": e[b], "ci95": [float(np.quantile(bs[b], 0.025)), float(np.quantile(bs[b], 0.975))], "p_best": float(bst.get(b, 0.0))} for b in BROKER_NAMES}
        out["by_liquidity"][name]["best"] = min(BROKER_NAMES, key=lambda b: e[b])
    out["interaction_coefficients"] = {k: float(v) for k, v in coef_i.items()}
    out["residual_sd_interaction_bp"] = float(np.std(resid_i))
    # per stratum: raw notional-weighted means and the best broker; oracle comparison when available
    strata = {}
    oracle_cols = [f"{oracle_prefix}{b}" for b in BROKER_NAMES]
    has_oracle = all(c in o.columns for c in oracle_cols)
    for s, g in o.groupby("stratum"):
        row = {"name": STRATA[int(s)], "n": int(len(g)), "mean_pct_adv": float(g["pct_adv"].mean()),
               "raw_bp": {b: float(np.mean(winsorise(g[y_col])[g["broker"] == b])) if (g["broker"] == b).sum() > 5 else None for b in BROKER_NAMES},
               "raw_notional_weighted_bp": {b: float(np.average(winsorise(g[y_col])[g["broker"] == b], weights=g.loc[g["broker"] == b, "notional"])) if (g["broker"] == b).sum() > 5 else None for b in BROKER_NAMES}}
        # regression effect for this stratum's liquidity group
        row["estimated_effect_vs_A"] = out["by_liquidity"]["illiquid" if s >= 3 else "liquid"]
        if has_oracle:
            row["oracle_effect_vs_A"] = {b: float(np.mean(g[f"{oracle_prefix}{b}"] - g[f"{oracle_prefix}A"])) for b in BROKER_NAMES}
            row["oracle_best"] = min(BROKER_NAMES, key=lambda b: row["oracle_effect_vs_A"][b])
        strata[int(s)] = row
    out["strata"] = strata
    # best and worst routing maps from the estimates (what the desk would hand over)
    out["best_map"] = {s: out["by_liquidity"]["illiquid" if s >= 3 else "liquid"]["best"] for s in STRATA}
    out["worst_map"] = {s: max(BROKER_NAMES, key=lambda b: out["by_liquidity"]["illiquid" if s >= 3 else "liquid"][b]["bp"]) for s in STRATA}
    if has_oracle:
        out["oracle_best_map"] = {s: strata[s]["oracle_best"] for s in strata}
        out["best_map_matches_oracle"] = {s: out["best_map"][s] == strata[s]["oracle_best"] for s in strata}
        # coverage: does the estimated CI (liquidity group) contain the oracle effect of each stratum?
        cov = []
        for s, row in strata.items():
            grp = out["by_liquidity"]["illiquid" if s >= 3 else "liquid"]
            for b in BROKER_NAMES[1:]:
                lo, hi = grp[b]["ci95"]; cov.append(lo <= row["oracle_effect_vs_A"][b] <= hi)
        out["oracle_within_ci_share"] = float(np.mean(cov))
    # power: orders per broker to resolve delta at 80 % power, 5 % size, two-sample, at the residual sd
    sd = float(np.std(resid_i))
    out["power"] = {f"{d}bp": int(np.ceil(2 * (Z80 * sd / d) ** 2)) for d in (0.5, 1.0, 2.0, 5.0)}
    out["power_by_liquidity"] = {}
    for name, ill in (("liquid", False), ("illiquid", True)):
        m = (o["spread_bp"] > ILLIQUID_BP).values == ill
        sd_g = float(np.std(resid_i[m])) if m.sum() > 10 else sd
        out["power_by_liquidity"][name] = {"residual_sd_bp": sd_g, "n_per_broker": {f"{d}bp": int(np.ceil(2 * (Z80 * sd_g / d) ** 2)) for d in (1.0, 2.0, 5.0)}}
    # learning curve: the C - A effect in illiquid names and A - C in liquid names as weeks accrue
    out["learning_curve"] = learning_curve(o, y, rng)
    return out


def learning_curve(o: pd.DataFrame, y: np.ndarray, rng: np.random.Generator, step_weeks: int = 8, B: int = 100) -> list[dict]:
    weeks = sorted(o["week"].unique())
    rows = []
    for k in range(step_weeks, len(weeks) + 1, step_weeks):
        m = o["week"].isin(weeks[:k]).values
        sub, ys = o[m], y[m]
        if sub["date"].nunique() < 8:
            continue
        try:
            c, _ = wls(sub, ys, True)
        except np.linalg.LinAlgError:
            continue
        draws = cluster_bootstrap(sub, ys, True, B, rng)
        liq = np.array([effects_from(d, False)["C"] for d in draws]); ill = np.array([effects_from(d, True)["C"] for d in draws])
        rows.append({"weeks": int(k), "n_orders": int(m.sum()), "C_minus_A_liquid_bp": effects_from(c, False)["C"], "ci_liquid": [float(np.quantile(liq, 0.025)), float(np.quantile(liq, 0.975))],
                     "C_minus_A_illiquid_bp": effects_from(c, True)["C"], "ci_illiquid": [float(np.quantile(ill, 0.025)), float(np.quantile(ill, 0.975))]})
    return rows


def adaptive_allocation(orders: pd.DataFrame, y_prefix: str = "exec_bp_", seed: int = 7, prior_sd: float = 30.0, curves: bool = False) -> dict:
    """On the same orders, compare the cost paid under three routing policies using the oracle executions of every
    broker: fixed stratified randomisation, Thompson sampling per stratum (normal-normal, known noise), and the oracle
    best broker per stratum.  Cost in notional-weighted bp and in dollars; regret against the oracle."""
    o = orders[orders["filled"] > 0].sort_values(["date", "symbol"]).reset_index(drop=True)
    cols = {b: o[f"{y_prefix}{b}"].values for b in BROKER_NAMES}
    for b in BROKER_NAMES:
        cols[b] = winsorise(pd.Series(cols[b])).values
    notional = o["notional"].values; strata = o["stratum"].values.astype(int)
    rng = np.random.default_rng(seed)
    n = len(o)
    # fixed randomisation: the broker actually assigned
    assigned = o["broker"].values
    cost_fixed = np.array([cols[b][i] for i, b in enumerate(assigned)])
    # oracle best per stratum
    means = {s: {b: float(np.mean(cols[b][strata == s])) for b in BROKER_NAMES} for s in np.unique(strata)}
    best = {s: min(BROKER_NAMES, key=lambda b: means[s][b]) for s in means}
    cost_oracle = np.array([cols[best[s]][i] for i, s in enumerate(strata)])
    # Thompson sampling, one posterior per (stratum, broker), updated weekly (orders inside a week are routed before fills arrive)
    noise_sd = float(np.std(np.concatenate([cols[b] for b in BROKER_NAMES])))
    post_mu = {(s, b): 0.0 for s in means for b in BROKER_NAMES}; post_var = {(s, b): prior_sd ** 2 for s in means for b in BROKER_NAMES}
    cost_ts = np.zeros(n); chosen = np.empty(n, dtype=object)
    weeks = o["week"].values
    for wk in np.unique(weeks):
        idx = np.where(weeks == wk)[0]
        sample = {(s, b): rng.normal(post_mu[(s, b)], np.sqrt(post_var[(s, b)])) for s in means for b in BROKER_NAMES}
        for i in idx:
            s = strata[i]; b = min(BROKER_NAMES, key=lambda bb: sample[(s, bb)])
            chosen[i] = b; cost_ts[i] = cols[b][i]
        for i in idx:
            s, b = strata[i], chosen[i]
            prec = 1.0 / post_var[(s, b)] + 1.0 / noise_sd ** 2
            post_mu[(s, b)] = (post_mu[(s, b)] / post_var[(s, b)] + cols[b][i] / noise_sd ** 2) / prec
            post_var[(s, b)] = 1.0 / prec
    def summ(c):
        return {"bp": float(np.mean(c)), "bp_notional_weighted": float(np.average(c, weights=notional)), "usd": float(np.sum(c * 1e-4 * notional))}
    out = {"n_orders": int(n), "reward": y_prefix, "fixed_randomisation": summ(cost_fixed), "thompson": summ(cost_ts), "oracle_best": summ(cost_oracle),
           "regret_fixed_usd": float(np.sum((cost_fixed - cost_oracle) * 1e-4 * notional)), "regret_thompson_usd": float(np.sum((cost_ts - cost_oracle) * 1e-4 * notional)),
           "thompson_share_best": float(np.mean([chosen[i] == best[strata[i]] for i in range(n)])),
           "thompson_share_best_last_year": float(np.mean([chosen[i] == best[strata[i]] for i in range(n) if weeks[i] >= weeks.max() - 52])),
           "years": float((o["date"].max() - o["date"].min()).days / 365.25)}
    if curves:
        wk = pd.DataFrame({"week": weeks, "fixed": (cost_fixed - cost_oracle) * 1e-4 * notional, "thompson": (cost_ts - cost_oracle) * 1e-4 * notional}).groupby("week").sum().cumsum()
        out["regret_curve"] = {"week": wk.index.tolist(), "fixed_usd": wk["fixed"].tolist(), "thompson_usd": wk["thompson"].tolist()}
    return out
