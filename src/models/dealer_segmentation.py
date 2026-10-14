"""Dealer segmentation, decline detection and visit prioritisation.

Segmentation is RULES-BASED (transparent thresholds over computed behavioural metrics); it is
not a trained ML model. Decline detection is evaluated against the simulation's known
ground-truth dealer patterns, which is only possible because the data is synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.metrics import detection_metrics

SEGMENTS = ["High-value", "Growing", "Stable", "Declining", "At-risk"]


def receivables_position(env) -> pd.DataFrame:
    """Current outstanding / overdue receivables per customer as of the simulation date."""
    inv = env.invoices
    open_inv = inv[inv["status"] != "Paid"]
    agg = open_inv.groupby("customer_id").agg(
        outstanding_inr=("amount_inr", "sum"),
        overdue_inr=("amount_inr", lambda s: s[open_inv.loc[s.index, "status"] == "Overdue"].sum()),
        max_days_past_due=("days_past_due", "max"),
        overdue_invoices=("status", lambda s: int((s == "Overdue").sum())),
    )
    out = env.customers[["customer_id", "credit_limit_inr"]].merge(agg, on="customer_id", how="left").fillna(0)
    out["credit_utilisation"] = np.where(out["credit_limit_inr"] > 0,
                                         out["outstanding_inr"] / out["credit_limit_inr"], 0.0)
    out["credit_status"] = np.select(
        [out["max_days_past_due"] > 45, out["credit_utilisation"] > 1.0, out["max_days_past_due"] > 0],
        ["ON HOLD - OVERDUE > 45 DAYS", "OVER LIMIT", "OVERDUE"], "WITHIN LIMIT")
    return out


def dealer_metrics(env, demand_result) -> pd.DataFrame:
    """Behavioural, forecast and credit metrics per dealer."""
    dw = env.dealer_weekly
    last = dw["week_start"].max()
    w = dw.assign(age=((last - dw["week_start"]).dt.days // 7))
    recent13 = w[w["age"] < 13].groupby("dealer_id")["qty_t"].sum()
    prior13 = w[(w["age"] >= 13) & (w["age"] < 26)].groupby("dealer_id")["qty_t"].sum()
    last52 = w[w["age"] < 52].groupby("dealer_id")["qty_t"].sum()
    ly13 = w[(w["age"] >= 52) & (w["age"] < 65)].groupby("dealer_id")["qty_t"].sum()
    freq_recent = w[(w["age"] < 13)].groupby("dealer_id")["qty_t"].apply(lambda s: (s > 0).mean())
    freq_prior = w[(w["age"] >= 13) & (w["age"] < 26)].groupby("dealer_id")["qty_t"].apply(lambda s: (s > 0).mean())
    last_order = w[w["qty_t"] > 0].groupby("dealer_id")["week_start"].max()

    fc = demand_result.forecast.merge(demand_result.series[["series_id", "dealer_id"]], on="series_id")
    fc4 = fc[fc["horizon"] <= 4].groupby("dealer_id")["forecast"].sum()

    d = env.dealers[["dealer_id", "dealer_name", "region", "district", "dealer_segment", "sales_rep_id",
                     "depot_id", "credit_limit_inr", "base_discount_per_t"]].copy()
    d = d.rename(columns={"dealer_segment": "dealer_tier"})
    d["vol_last13_t"] = d["dealer_id"].map(recent13).fillna(0)
    d["vol_prior13_t"] = d["dealer_id"].map(prior13).fillna(0)
    d["vol_last52_t"] = d["dealer_id"].map(last52).fillna(0)
    d["vol_same13_last_year_t"] = d["dealer_id"].map(ly13).fillna(0)
    d["growth_qoq"] = np.where(d["vol_prior13_t"] > 0, d["vol_last13_t"] / d["vol_prior13_t"] - 1, 0.0)
    d["growth_yoy"] = np.where(d["vol_same13_last_year_t"] > 0,
                               d["vol_last13_t"] / d["vol_same13_last_year_t"] - 1, 0.0)
    d["order_freq_recent"] = d["dealer_id"].map(freq_recent).fillna(0)
    d["order_freq_prior"] = d["dealer_id"].map(freq_prior).fillna(0)
    d["weeks_since_last_order"] = ((last - d["dealer_id"].map(last_order)).dt.days // 7).fillna(99).astype(int)
    d["forecast_next4_t"] = d["dealer_id"].map(fc4).fillna(0)
    d["run_rate_4w_t"] = d["vol_last13_t"] / 13 * 4

    rec = receivables_position(env).rename(columns={"customer_id": "dealer_id"})
    d = d.merge(rec[["dealer_id", "outstanding_inr", "overdue_inr", "max_days_past_due", "credit_utilisation",
                     "credit_status"]], on="dealer_id", how="left")
    return d


def segment_dealers(env, demand_result) -> pd.DataFrame:
    """Assign one of five segments with an explicit reason, plus a visit-priority score."""
    d = dealer_metrics(env, demand_result)
    hv_threshold = d["vol_last52_t"].quantile(0.85)
    # Decline uses both sequential and year-on-year comparison so the monsoon dip is not misread.
    declining = ((d["growth_qoq"] < -0.25) & (d["growth_yoy"] < -0.20)) | \
                ((d["order_freq_recent"] < d["order_freq_prior"] - 0.3) & (d["growth_yoy"] < -0.15))
    at_risk = (d["max_days_past_due"] > 45) | (d["credit_utilisation"] > 1.1) | \
              (declining & (d["weeks_since_last_order"] >= 4))
    growing = (d["growth_yoy"] > 0.15) & (d["growth_qoq"] > -0.10)
    high_value = d["vol_last52_t"] >= hv_threshold

    seg = np.select([at_risk, declining, growing, high_value], ["At-risk", "Declining", "Growing", "High-value"],
                    "Stable")
    reasons = []
    for i, r in d.iterrows():
        parts = []
        if r["max_days_past_due"] > 45:
            parts.append(f"overdue {int(r['max_days_past_due'])} days")
        if r["credit_utilisation"] > 1.1:
            parts.append(f"credit utilisation {r['credit_utilisation']:.0%}")
        if declining.iloc[i]:
            parts.append(f"volume {r['growth_yoy']:+.0%} year-on-year, {r['growth_qoq']:+.0%} vs prior quarter")
        if r["weeks_since_last_order"] >= 4:
            parts.append(f"no order for {r['weeks_since_last_order']} weeks")
        if growing.iloc[i] and not parts:
            parts.append(f"volume {r['growth_yoy']:+.0%} year-on-year")
        if high_value.iloc[i] and not parts:
            parts.append(f"top 15% by annual volume ({r['vol_last52_t']:,.0f} t)")
        reasons.append("; ".join(parts) if parts else "volume and payments within normal range")
    d["segment"] = seg
    d["segment_reason"] = reasons

    # Lost-volume opportunity: shortfall of current run rate vs same quarter last year (if positive).
    d["volume_gap_13w_t"] = (d["vol_same13_last_year_t"] - d["vol_last13_t"]).clip(lower=0)
    return d


def visit_priority(segmented: pd.DataFrame, replenishment: pd.DataFrame | None,
                   contribution_per_t: float) -> pd.DataFrame:
    """Rank dealers for sales visits by estimated value at stake (illustrative)."""
    d = segmented.copy()
    stockout_value = pd.Series(0.0, index=d["dealer_id"])
    if replenishment is not None and len(replenishment):
        stockout_value = replenishment.groupby("dealer_id")["protected_contribution_inr"].sum()
    d["stockout_contribution_at_risk_inr"] = d["dealer_id"].map(stockout_value).fillna(0.0)
    # Recoverable share of the lost-volume gap over the next quarter is a scenario assumption (30%).
    d["recovery_opportunity_inr"] = d["volume_gap_13w_t"] * 0.30 * contribution_per_t
    weight = d["segment"].map({"At-risk": 1.2, "Declining": 1.3, "Growing": 1.1, "High-value": 1.0, "Stable": 0.8})
    d["incremental_opportunity_inr"] = d["recovery_opportunity_inr"] + d["stockout_contribution_at_risk_inr"]
    d["visit_priority_score"] = (d["incremental_opportunity_inr"] * weight).round(0)
    d = d.sort_values("visit_priority_score", ascending=False)
    d["visit_rank"] = np.arange(1, len(d) + 1)
    d["priority"] = np.where(d["visit_rank"] <= max(10, int(len(d) * 0.05)), "HIGH PRIORITY",
                             np.where(d["visit_rank"] <= int(len(d) * 0.2), "MEDIUM PRIORITY", "ROUTINE"))
    return d


def decline_detection_quality(env, segmented: pd.DataFrame) -> dict:
    """Precision/recall of the decline flag against simulated ground truth (synthetic data only)."""
    truth = env.dealer_series.groupby("dealer_id")["true_pattern"].apply(lambda s: (s == "declining").mean() >= 0.5)
    flag = segmented.set_index("dealer_id")["segment"].isin(["Declining", "At-risk"]) & \
        (segmented.set_index("dealer_id")["growth_yoy"] < 0)
    y = truth.reindex(flag.index).fillna(False)
    return detection_metrics(y, flag)


def segment_population(pop: pd.DataFrame) -> pd.DataFrame:
    """Vectorised rules-based segmentation of the full ~49,300 dealer network simulation."""
    hv = pop["annual_volume_t"].quantile(0.85)
    seg = np.select(
        [(pop["payment_risk_index"] > 0.55) | (pop["days_since_last_order"] > 45),
         pop["yoy_growth"] < -0.20, pop["yoy_growth"] > 0.15, pop["annual_volume_t"] >= hv],
        ["At-risk", "Declining", "Growing", "High-value"], "Stable")
    return pop.assign(segment=seg)
