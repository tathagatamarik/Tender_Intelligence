"""Dealer replenishment recommendations driven by the demand forecast.

Recommended order = max(0, forecast over (lead time + review period) + safety stock
                        - current stock - open orders), rounded up to the lot size and
                        raised to the minimum order quantity, then capped by credit headroom.
Safety stock = z(service level) x weekly forecast-error sigma x sqrt(lead time + review weeks).

Stock-out probability assumes normally distributed lead-time demand (a simplification).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.models.demand_forecasting import DemandModelResult, horizon_weights


@dataclass(frozen=True)
class ReplenishmentSettings:
    service_level: float = 0.95
    lead_time_days: int = 7
    review_days: int = 7
    lot_size_t: float = 5.0
    min_order_t: float = 5.0
    respect_credit_limit: bool = True


def _dealer_state(env) -> pd.DataFrame:
    """Current dealer stock and open orders per series (simulated snapshot)."""
    rng = np.random.default_rng(env.settings.seed + 11)
    s = env.dealer_series[["series_id", "dealer_id", "product_id", "expected_weekly_t"]].copy()
    recent = env.dealer_weekly[env.dealer_weekly["week_start"] >= env.dealer_weekly["week_start"].max()
                               - pd.Timedelta(weeks=3)].groupby("series_id")["qty_t"].mean()
    base = s["series_id"].map(recent).fillna(0).clip(lower=0.5)
    s["current_stock_t"] = np.round(base * rng.uniform(0.5, 3.2, len(s)) * 2) / 2
    has_open = rng.random(len(s)) < 0.3
    s["open_orders_t"] = np.where(has_open, np.round(base * rng.uniform(0.5, 1.5, len(s)) * 2) / 2, 0.0)
    return s


def recommend_replenishment(env, demand: DemandModelResult, settings: ReplenishmentSettings,
                            receivables: pd.DataFrame, net_price_per_t: float,
                            contribution_per_t: float) -> pd.DataFrame:
    state = _dealer_state(env)
    fc = demand.forecast
    cover_days = settings.lead_time_days + settings.review_days
    w_cover = horizon_weights(cover_days)
    w_lead = horizon_weights(settings.lead_time_days)
    wide = fc.pivot(index="series_id", columns="horizon", values="forecast").sort_index(axis=1)
    f_cover = pd.Series(wide.to_numpy() @ w_cover, index=wide.index)
    f_lead = pd.Series(wide.to_numpy() @ w_lead, index=wide.index)
    sigma = demand.series.set_index("series_id")["sigma_weekly"]

    df = state.copy()
    df["forecast_cover_t"] = df["series_id"].map(f_cover).fillna(0)
    df["forecast_lead_t"] = df["series_id"].map(f_lead).fillna(0)
    df["sigma_weekly_t"] = df["series_id"].map(sigma).fillna(0)
    z = float(norm.ppf(np.clip(settings.service_level, 0.5, 0.999)))
    lt_weeks = cover_days / 7.0
    df["safety_stock_t"] = z * df["sigma_weekly_t"] * np.sqrt(lt_weeks)
    df["target_stock_t"] = df["forecast_cover_t"] + df["safety_stock_t"]
    df["net_position_t"] = df["current_stock_t"] + df["open_orders_t"]
    raw = (df["target_stock_t"] - df["net_position_t"]).clip(lower=0)
    lots = np.ceil(raw / settings.lot_size_t - 1e-9) * settings.lot_size_t
    lots = np.where((lots > 0) & (lots < settings.min_order_t), settings.min_order_t, lots)
    df["unconstrained_qty_t"] = lots

    # Credit headroom per dealer (shared across that dealer's products in proportion to need)
    rec = receivables.set_index("customer_id")
    headroom = (rec["credit_limit_inr"] - rec["outstanding_inr"]).clip(lower=0)
    df["credit_headroom_inr"] = df["dealer_id"].map(headroom).fillna(0)
    need_value = df["unconstrained_qty_t"] * net_price_per_t
    dealer_need = need_value.groupby(df["dealer_id"]).transform("sum")
    allowed = np.where(dealer_need > 0, df["credit_headroom_inr"] * need_value / dealer_need, 0)
    if settings.respect_credit_limit:
        capped = np.floor(allowed / net_price_per_t / settings.lot_size_t) * settings.lot_size_t
        df["recommended_qty_t"] = np.minimum(df["unconstrained_qty_t"], capped)
    else:
        df["recommended_qty_t"] = df["unconstrained_qty_t"]
    df["credit_constrained"] = df["recommended_qty_t"] < df["unconstrained_qty_t"]

    # Stock-out risk during lead time if nothing more is ordered
    lead_sigma = df["sigma_weekly_t"] * np.sqrt(settings.lead_time_days / 7.0)
    df["stockout_probability"] = np.where(
        lead_sigma > 0, 1 - norm.cdf((df["net_position_t"] - df["forecast_lead_t"]) / lead_sigma.replace(0, np.nan)),
        (df["net_position_t"] < df["forecast_lead_t"]).astype(float))
    df["stockout_probability"] = df["stockout_probability"].fillna(0).clip(0, 1)
    df["days_of_cover"] = np.where(df["forecast_cover_t"] > 0,
                                   df["net_position_t"] / (df["forecast_cover_t"] / cover_days), 99).clip(max=99)
    df["stockout_flag"] = np.where(df["stockout_probability"] >= 0.5, "LIKELY STOCKOUT",
                                   np.where(df["stockout_probability"] >= 0.2, "NEEDS REVIEW", "WITHIN LIMIT"))
    # Expected shortfall over the cover period without replenishment -> sales protected by acting
    shortfall = (df["forecast_cover_t"] - df["net_position_t"]).clip(lower=0)
    protected = np.minimum(shortfall, df["recommended_qty_t"])
    df["protected_volume_t"] = protected
    df["protected_revenue_inr"] = protected * net_price_per_t
    df["protected_contribution_inr"] = protected * contribution_per_t
    return df


def depot_replenishment_requirement(recs: pd.DataFrame, env) -> pd.DataFrame:
    """Aggregate recommended dealer replenishment to depot x product (feeds inventory planning)."""
    d = recs.merge(env.dealers[["dealer_id", "depot_id"]], on="dealer_id", how="left")
    d = d.merge(env.products[["product_id", "category"]], on="product_id", how="left")
    return d.groupby(["depot_id", "category"], as_index=False)["recommended_qty_t"].sum()
