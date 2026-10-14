import numpy as np
import pandas as pd
import pytest

from src.services.order_economics import compute_order_economics, summarise_economics, waterfall_frame


def _orders():
    return pd.DataFrame({"qty_t": [10.0, 20.0], "list_price_per_t": [7000.0, 6800.0], "discount_per_t": [200.0, 100.0],
                         "production_cost_per_t": [4300.0, 4200.0], "freight_per_t": [1100.0, 1500.0],
                         "handling_per_t": [60.0, 170.0], "pd": [0.1, 0.0], "lgd": [0.45, 0.45], "days": [30.0, 60.0]})


def test_components_reconcile():
    e = compute_order_economics(_orders(), pd_col="pd", lgd_col="lgd", collection_days_col="days", financing_rate=0.10)
    r = e.iloc[0]
    assert r["net_revenue"] == pytest.approx(10 * (7000 - 200))
    assert r["contribution_margin"] == pytest.approx(r["net_revenue"] - 10 * (4300 + 1100 + 60))
    assert r["expected_credit_loss"] == pytest.approx(0.1 * 0.45 * r["net_revenue"])
    assert r["financing_cost"] == pytest.approx(r["net_revenue"] * 0.10 * 30 / 365)
    assert r["risk_adjusted_contribution"] == pytest.approx(
        r["contribution_margin"] - r["expected_credit_loss"] - r["financing_cost"])
    # Gross revenue reconciles to all components without double counting
    rec = (e["discount_amount"] + e["production_cost"] + e["freight_cost"] + e["handling_cost"]
           + e["expected_credit_loss"] + e["financing_cost"] + e["risk_adjusted_contribution"])
    np.testing.assert_allclose(rec, e["gross_revenue"])


def test_no_risk_columns_means_zero_adjustment():
    e = compute_order_economics(_orders())
    assert (e["expected_credit_loss"] == 0).all() and (e["financing_cost"] == 0).all()
    np.testing.assert_allclose(e["risk_adjusted_contribution"], e["contribution_margin"])


def test_summary_and_waterfall():
    e = compute_order_economics(_orders(), pd_col="pd", lgd_col="lgd", collection_days_col="days")
    t = summarise_economics(e)
    assert t["qty_t"] == 30
    wf = waterfall_frame(t)
    rel = wf[wf["measure"] == "relative"]["value"].iloc[:4].sum()
    assert t["gross_revenue"] + rel == pytest.approx(t["contribution_margin"])


def test_historical_orders_reconcile(env):
    e = compute_order_economics(env.orders.head(500))
    np.testing.assert_allclose(e["gross_revenue"] - e["discount_amount"] - e["production_cost"] - e["freight_cost"]
                               - e["handling_cost"], e["contribution_margin"])
