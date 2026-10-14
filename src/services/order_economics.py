"""Integrated order economics: one definition of revenue, contribution and risk adjustment.

Definitions (all per order, INR):
    gross_revenue            = list price per t x quantity
    discount_amount          = discount per t x quantity
    net_revenue              = gross_revenue - discount_amount
    production_cost          = production / sourcing cost per t x quantity
    freight_cost             = (primary + secondary) freight per t x quantity
    handling_cost            = loading / re-handling cost per t x quantity
    gross_margin             = net_revenue - production_cost                 (before logistics)
    contribution_margin      = net_revenue - production_cost - freight_cost - handling_cost
    expected_credit_loss     = PD x LGD x exposure (exposure = net_revenue of the order)
    financing_cost           = net_revenue x annual rate x expected collection days / 365
    risk_adjusted_contribution = contribution_margin - expected_credit_loss - financing_cost

Each cost is counted once. The ECL covers loss of principal; the financing cost covers only the
time value of money for the expected collection period, so the two do not overlap. Taxes are
excluded (amounts are ex-GST).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANNUAL_FINANCING_RATE

COMPONENT_LABELS = {
    "gross_revenue": "Gross revenue (list price x qty)",
    "discount_amount": "Discounts and schemes",
    "net_revenue": "Net revenue",
    "production_cost": "Production / sourcing cost",
    "freight_cost": "Freight (primary + secondary)",
    "handling_cost": "Loading and handling",
    "contribution_margin": "Contribution margin",
    "expected_credit_loss": "Expected credit loss (PD x LGD x EAD)",
    "financing_cost": "Financing cost of credit period",
    "risk_adjusted_contribution": "Risk-adjusted contribution",
}


def compute_order_economics(df: pd.DataFrame, pd_col: str | None = None, lgd_col: str | None = None,
                            collection_days_col: str | None = None,
                            financing_rate: float = ANNUAL_FINANCING_RATE) -> pd.DataFrame:
    """Add economics columns to an order-level frame.

    Required columns: qty_t, list_price_per_t, discount_per_t, production_cost_per_t,
    freight_per_t, handling_per_t. Optional risk columns are applied when provided.
    """
    out = df.copy()
    q = out["qty_t"].astype(float)
    out["gross_revenue"] = q * out["list_price_per_t"]
    out["discount_amount"] = q * out["discount_per_t"]
    out["net_revenue"] = out["gross_revenue"] - out["discount_amount"]
    out["production_cost"] = q * out["production_cost_per_t"]
    out["freight_cost"] = q * out["freight_per_t"]
    out["handling_cost"] = q * out["handling_per_t"]
    out["gross_margin"] = out["net_revenue"] - out["production_cost"]
    out["contribution_margin"] = out["gross_margin"] - out["freight_cost"] - out["handling_cost"]
    out["contribution_margin_pct"] = np.where(out["net_revenue"] != 0,
                                              out["contribution_margin"] / out["net_revenue"], np.nan)
    if pd_col and lgd_col:
        pd_ = out[pd_col].astype(float).clip(0, 1)
        lgd = out[lgd_col].astype(float).clip(0, 1)
        out["expected_credit_loss"] = pd_ * lgd * out["net_revenue"].clip(lower=0)
    else:
        out["expected_credit_loss"] = 0.0
    if collection_days_col:
        days = out[collection_days_col].astype(float).clip(lower=0)
        out["financing_cost"] = out["net_revenue"].clip(lower=0) * financing_rate * days / 365.0
    else:
        out["financing_cost"] = 0.0
    out["risk_adjusted_contribution"] = out["contribution_margin"] - out["expected_credit_loss"] - out["financing_cost"]
    out["risk_adjusted_margin_pct"] = np.where(out["net_revenue"] != 0,
                                               out["risk_adjusted_contribution"] / out["net_revenue"], np.nan)
    return out


def summarise_economics(df: pd.DataFrame) -> dict[str, float]:
    """Totals and per-tonne figures for a frame produced by compute_order_economics."""
    qty = float(df["qty_t"].sum()) if len(df) else 0.0
    tot = {k: float(df[k].sum()) if len(df) else 0.0 for k in COMPONENT_LABELS if k in df.columns}
    tot["qty_t"] = qty
    tot["contribution_margin_pct"] = tot["contribution_margin"] / tot["net_revenue"] if tot.get("net_revenue") else 0.0
    tot["freight_per_t"] = tot["freight_cost"] / qty if qty else 0.0
    tot["contribution_per_t"] = tot["contribution_margin"] / qty if qty else 0.0
    tot["net_price_per_t"] = tot["net_revenue"] / qty if qty else 0.0
    return tot


def waterfall_frame(totals: dict[str, float]) -> pd.DataFrame:
    """Rows for a revenue-to-risk-adjusted-contribution waterfall chart."""
    steps = [("Gross revenue", totals.get("gross_revenue", 0.0), "absolute"),
             ("Discounts", -totals.get("discount_amount", 0.0), "relative"),
             ("Production cost", -totals.get("production_cost", 0.0), "relative"),
             ("Freight", -totals.get("freight_cost", 0.0), "relative"),
             ("Handling", -totals.get("handling_cost", 0.0), "relative"),
             ("Contribution", totals.get("contribution_margin", 0.0), "total")]
    if totals.get("expected_credit_loss", 0.0) or totals.get("financing_cost", 0.0):
        steps += [("Expected credit loss", -totals.get("expected_credit_loss", 0.0), "relative"),
                  ("Financing cost", -totals.get("financing_cost", 0.0), "relative"),
                  ("Risk-adjusted contribution", totals.get("risk_adjusted_contribution", 0.0), "total")]
    return pd.DataFrame(steps, columns=["step", "value", "measure"])
