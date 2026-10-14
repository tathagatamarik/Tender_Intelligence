"""End-to-end: every page renders with default synthetic data and no exceptions (Streamlit AppTest)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["1_Executive_Dashboard", "2_Dealer_Intelligence", "3_Freight_Optimisation", "4_Tender_Intelligence",
         "5_Production_Inventory", "6_Credit_Margin", "7_Product_Compliance", "8_Pricing_Intelligence",
         "9_Offtake_Early_Warning", "10_Sales_Knowledge_Assistant"]


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600)
    at.switch_page(f"pages/{page}.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert any("DEMO ENVIRONMENT" in m.value for m in at.markdown)
