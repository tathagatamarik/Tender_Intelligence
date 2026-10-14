"""Financial impact and ROI simulator.

Conventions (stated in the UI):
    Annual freight baseline      = monthly addressable tonnage x freight cost per t x 12
    Freight savings              = annual freight baseline x achievable saving %
    Inventory holding saving     = inventory value x reduction % x holding cost %
    Incremental contribution     = monthly volume x incremental volume % x contribution per t x 12
    Avoided credit losses        = expected credit loss x reduction %
    Gross annual benefit         = (sum of the four P&L items) x (1 - overlap adjustment)
    Annual recurring cost        = software / infrastructure + operating cost
    Annual net benefit           = gross annual benefit - annual recurring cost
    ROI (%)                      = annual net benefit / implementation cost x 100  (simple, year-one, undiscounted)
    Payback (months)             = implementation cost / (annual net benefit / 12), only when net benefit > 0
    Working-capital release      = one-off cash release (receivables + inventory) - NOT profit; its
                                   financing benefit is shown separately and is not added to P&L benefits.

All figures are illustrative estimates on simulated assumptions, not verified business outcomes.
"""
from __future__ import annotations

from src.data.schemas import ROIInputs

SCENARIO_MULTIPLIERS = {"Conservative": 0.5, "Base": 1.0, "Upside": 1.5}
# Improvement assumptions scaled per scenario, with their validation caps
SCALED_FIELDS = {"freight_saving_pct": 0.5, "inventory_reduction_pct": 0.8, "receivable_release_pct": 1.0,
                 "credit_loss_reduction_pct": 1.0, "incremental_volume_pct": 0.5}


def compute_roi(inp: ROIInputs) -> dict[str, float | None]:
    annual_freight_baseline = inp.monthly_volume_t * inp.addressable_share * inp.freight_cost_per_t * 12
    freight_savings = annual_freight_baseline * inp.freight_saving_pct
    inventory_saving = inp.inventory_value_inr * inp.inventory_reduction_pct * inp.holding_cost_pct
    incremental_contribution = inp.monthly_volume_t * inp.incremental_volume_pct * inp.contribution_per_t * 12
    avoided_credit_losses = inp.expected_credit_loss_inr * inp.credit_loss_reduction_pct
    gross_unadjusted = freight_savings + inventory_saving + incremental_contribution + avoided_credit_losses
    overlap = gross_unadjusted * inp.overlap_adjustment_pct
    gross = gross_unadjusted - overlap
    recurring = inp.annual_software_cost_inr + inp.annual_operating_cost_inr
    net = gross - recurring
    wc_release = inp.overdue_receivables_inr * inp.receivable_release_pct + inp.inventory_value_inr * inp.inventory_reduction_pct
    wc_financing = wc_release * inp.financing_rate
    roi = net / inp.implementation_cost_inr * 100
    payback = inp.implementation_cost_inr / (net / 12) if net > 0 else None
    return {
        "Annual freight baseline": annual_freight_baseline,
        "Freight savings": freight_savings,
        "Inventory holding-cost saving": inventory_saving,
        "Incremental contribution": incremental_contribution,
        "Avoided credit losses": avoided_credit_losses,
        "Gross benefit before overlap adjustment": gross_unadjusted,
        "Overlap adjustment": -overlap,
        "Gross annual benefit": gross,
        "Annual recurring cost": recurring,
        "Annual net benefit": net,
        "Implementation cost": inp.implementation_cost_inr,
        "ROI (%)": roi,
        "Payback (months)": payback,
        "Working-capital release (one-off, not profit)": wc_release,
        "Financing value of working-capital release (memo)": wc_financing,
    }


def scenario_table(base: ROIInputs) -> dict[str, dict]:
    """Conservative / base / upside results by scaling the improvement assumptions."""
    out = {}
    for name, m in SCENARIO_MULTIPLIERS.items():
        data = base.model_dump()
        for f, cap in SCALED_FIELDS.items():
            data[f] = min(data[f] * m, cap)
        out[name] = compute_roi(ROIInputs(**data))
    return out
