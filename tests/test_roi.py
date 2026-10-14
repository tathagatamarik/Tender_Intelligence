import pytest
from pydantic import ValidationError

from src.data.schemas import ROIInputs
from src.services.roi_calculator import compute_roi, scenario_table

BASE = dict(monthly_volume_t=100_000, freight_cost_per_t=1000, addressable_share=0.5, freight_saving_pct=0.05,
            inventory_value_inr=1e9, holding_cost_pct=0.14, inventory_reduction_pct=0.10, overdue_receivables_inr=5e8,
            receivable_release_pct=0.10, expected_credit_loss_inr=1e8, credit_loss_reduction_pct=0.20,
            incremental_volume_pct=0.01, contribution_per_t=900, overlap_adjustment_pct=0.10, financing_rate=0.10,
            implementation_cost_inr=1e8, annual_software_cost_inr=1e7, annual_operating_cost_inr=1e7)


def test_formulas():
    r = compute_roi(ROIInputs(**BASE))
    assert r["Annual freight baseline"] == pytest.approx(100_000 * 0.5 * 1000 * 12)
    assert r["Freight savings"] == pytest.approx(6e8 * 0.05)
    assert r["Inventory holding-cost saving"] == pytest.approx(1e9 * 0.10 * 0.14)
    assert r["Incremental contribution"] == pytest.approx(100_000 * 0.01 * 900 * 12)
    assert r["Avoided credit losses"] == pytest.approx(2e7)
    gross_un = 3e7 + 1.4e7 + 1.08e7 + 2e7
    assert r["Gross annual benefit"] == pytest.approx(gross_un * 0.9)
    assert r["Annual net benefit"] == pytest.approx(gross_un * 0.9 - 2e7)
    assert r["ROI (%)"] == pytest.approx(r["Annual net benefit"] / 1e8 * 100)
    assert r["Payback (months)"] == pytest.approx(1e8 / (r["Annual net benefit"] / 12))


def test_working_capital_not_added_to_benefits():
    a = compute_roi(ROIInputs(**BASE))
    b = compute_roi(ROIInputs(**{**BASE, "receivable_release_pct": 0.9}))
    assert a["Gross annual benefit"] == b["Gross annual benefit"]
    assert b["Working-capital release (one-off, not profit)"] > a["Working-capital release (one-off, not profit)"]


def test_no_payback_when_net_negative():
    r = compute_roi(ROIInputs(**{**BASE, "annual_operating_cost_inr": 1e10}))
    assert r["Annual net benefit"] < 0 and r["Payback (months)"] is None


def test_scenarios_are_ordered():
    s = scenario_table(ROIInputs(**BASE))
    assert s["Conservative"]["Annual net benefit"] < s["Base"]["Annual net benefit"] < s["Upside"]["Annual net benefit"]


def test_invalid_inputs_rejected():
    with pytest.raises(ValidationError):
        ROIInputs(**{**BASE, "implementation_cost_inr": 0})
    with pytest.raises(ValidationError):
        ROIInputs(**{**BASE, "freight_saving_pct": 2.0})
