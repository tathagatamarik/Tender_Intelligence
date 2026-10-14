"""Price-sensitivity estimation and discount scenario simulation.

Elasticities are estimated with a log-log OLS regression of volume on relative net price
(net price / competitor price indicator), controlling for market demand, micro-market and
product fixed effects. These are CORRELATIONAL estimates from observational (simulated) data,
not causal evidence. Scenario results apply the chosen elasticity as an explicit assumption.

Competitor prices are used only as a simulated market indicator. The module never recommends
coordinated or collusive pricing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.config import MAX_DISCOUNT_PER_T_POLICY, MIN_MARGIN_PCT_POLICY


def estimate_elasticities(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, g in panel.groupby("region"):
        g = g.assign(log_vol=np.log(g["volume_t"].clip(lower=0.1)),
                     log_rel_price=np.log(g["net_price_per_t"] / g["competitor_price_per_t"]),
                     log_demand=np.log(g["market_demand_index"].clip(lower=0.05)))
        try:
            fit = smf.ols("log_vol ~ log_rel_price + log_demand + C(micro_market) + C(product_id)", data=g).fit()
            est = fit.params["log_rel_price"]
            lo, hi = fit.conf_int().loc["log_rel_price"]
            rows.append({"region": region, "elasticity": est, "ci_low": lo, "ci_high": hi,
                         "p_value": fit.pvalues["log_rel_price"], "r_squared": fit.rsquared, "n_obs": int(fit.nobs)})
        except Exception:  # pragma: no cover
            rows.append({"region": region, "elasticity": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                         "p_value": np.nan, "r_squared": np.nan, "n_obs": len(g)})
    return pd.DataFrame(rows)


def baseline_position(panel: pd.DataFrame, region: str, product_id: str, months: int = 3) -> dict:
    g = panel[(panel["region"] == region) & (panel["product_id"] == product_id)]
    recent = g[g["month"] >= sorted(g["month"].unique())[-months]]
    vol = recent.groupby("month")["volume_t"].sum().mean()
    return {
        "list_price": float(recent["list_price_per_t"].mean()),
        "discount": float(recent["discount_per_t"].mean()),
        "net_price": float(recent["net_price_per_t"].mean()),
        "competitor_price": float(recent["competitor_price_per_t"].mean()),
        "cost": float(recent["production_cost_per_t"].mean()),
        "freight": float(recent["freight_per_t"].mean()),
        "monthly_volume_t": float(vol),
    }


def simulate_scenario(base: dict, elasticity: float, price_change_pct: float = 0.0,
                      discount_change_per_t: float = 0.0) -> dict:
    """Constant-elasticity volume response to a change in net price (scenario assumption)."""
    new_list = base["list_price"] * (1 + price_change_pct)
    new_disc = max(0.0, base["discount"] + discount_change_per_t)
    new_net = new_list - new_disc
    ratio = new_net / base["net_price"] if base["net_price"] > 0 else 1.0
    vol = base["monthly_volume_t"] * ratio ** elasticity
    cpt = new_net - base["cost"] - base["freight"]
    base_cpt = base["net_price"] - base["cost"] - base["freight"]
    return {
        "list_price": new_list, "discount": new_disc, "net_price": new_net, "monthly_volume_t": vol,
        "revenue": vol * new_net, "contribution_per_t": cpt, "contribution": vol * cpt,
        "margin_pct": cpt / new_net if new_net else 0.0,
        "volume_change_pct": vol / base["monthly_volume_t"] - 1 if base["monthly_volume_t"] else 0.0,
        "contribution_change": vol * cpt - base["monthly_volume_t"] * base_cpt,
    }


def discount_frontier(base: dict, elasticity: float, max_discount: float = MAX_DISCOUNT_PER_T_POLICY,
                      min_margin: float = MIN_MARGIN_PCT_POLICY, step: float = 10.0) -> pd.DataFrame:
    rows = []
    for d in np.arange(0, max_discount + step, step):
        r = simulate_scenario(base, elasticity, 0.0, d - base["discount"])
        r["discount_per_t"] = d
        r["within_policy"] = (d <= max_discount) and (r["margin_pct"] >= min_margin)
        rows.append(r)
    return pd.DataFrame(rows)


def suggest_discount_range(frontier: pd.DataFrame, tolerance: float = 0.02) -> dict | None:
    ok = frontier[frontier["within_policy"]]
    if ok.empty:
        return None
    best = ok.loc[ok["contribution"].idxmax()]
    near = ok[ok["contribution"] >= best["contribution"] * (1 - tolerance)]
    return {"best_discount": float(best["discount_per_t"]), "low": float(near["discount_per_t"].min()),
            "high": float(near["discount_per_t"].max()), "contribution": float(best["contribution"]),
            "margin_pct": float(best["margin_pct"])}


def scheme_effectiveness(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["volume_index"] = p["volume_t"] / p.groupby(["micro_market", "product_id"])["volume_t"].transform("mean")
    out = p.groupby("scheme").agg(observations=("volume_t", "size"), avg_discount_per_t=("discount_per_t", "mean"),
                                  avg_volume_index=("volume_index", "mean"),
                                  avg_conversion=("quote_conversion_rate", "mean"),
                                  avg_contribution_per_t=("contribution_per_t", "mean"),
                                  avg_margin_pct=("contribution_margin_pct", "mean")).reset_index()
    base = out.loc[out["scheme"] == "No Scheme", "avg_volume_index"]
    b = float(base.iloc[0]) if len(base) else 1.0
    out["volume_uplift_vs_no_scheme"] = out["avg_volume_index"] / b - 1
    return out.sort_values("avg_contribution_per_t", ascending=False)


def market_opportunities(panel: pd.DataFrame, elasticities: pd.DataFrame,
                         min_margin: float = MIN_MARGIN_PCT_POLICY,
                         max_discount: float = MAX_DISCOUNT_PER_T_POLICY) -> pd.DataFrame:
    """Rank micro-market x product combinations by contribution gain at the policy-best discount."""
    el = elasticities.set_index("region")["elasticity"]
    last = sorted(panel["month"].unique())[-3:]
    recent = panel[panel["month"].isin(last)]
    rows = []
    for (region, mm, pid), g in recent.groupby(["region", "micro_market", "product_id"]):
        base = {"list_price": g["list_price_per_t"].mean(), "discount": g["discount_per_t"].mean(),
                "net_price": g["net_price_per_t"].mean(), "competitor_price": g["competitor_price_per_t"].mean(),
                "cost": g["production_cost_per_t"].mean(), "freight": g["freight_per_t"].mean(),
                "monthly_volume_t": g["volume_t"].mean()}
        e = float(el.get(region, -1.5))
        sug = suggest_discount_range(discount_frontier(base, e, max_discount, min_margin))
        if sug is None:
            continue
        cur = simulate_scenario(base, e)
        rows.append({"region": region, "micro_market": mm, "product_id": pid, "product_name": g["product_name"].iloc[0],
                     "current_discount_per_t": base["discount"], "suggested_discount_per_t": sug["best_discount"],
                     "suggested_range": f"{sug['low']:.0f} - {sug['high']:.0f}",
                     "current_margin_pct": cur["margin_pct"], "monthly_contribution_now_inr": cur["contribution"],
                     "monthly_contribution_gain_inr": sug["contribution"] - cur["contribution"],
                     "elasticity_used": e})
    return pd.DataFrame(rows).sort_values("monthly_contribution_gain_inr", ascending=False)
