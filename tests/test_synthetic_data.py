import pandas as pd

from src.config import DataSettings
from src.data.synthetic_data import generate_dealer_population, generate_environment
from src.data.validation import validate_environment


def test_reproducible_with_seed():
    s = DataSettings(seed=11, n_dealers=100, n_weeks=64, n_institutional=80)
    a, b = generate_environment(s), generate_environment(s)
    pd.testing.assert_frame_equal(a.orders, b.orders)
    pd.testing.assert_frame_equal(a.dealer_weekly, b.dealer_weekly)


def test_entity_ids_are_consistent(env):
    cust = set(env.customers["customer_id"])
    assert set(env.orders["customer_id"]) <= cust
    assert set(env.invoices["order_id"]) <= set(env.orders["order_id"])
    assert set(env.open_orders["customer_id"]) <= cust
    assert set(env.dealer_series["dealer_id"]) <= set(env.dealers["dealer_id"])
    assert set(env.orders["product_id"]) <= set(env.products["product_id"])
    src = set(env.plants["plant_id"]) | set(env.depots["depot_id"])
    assert set(env.orders["source_id"]) <= src
    assert set(env.trips["order_id"]) <= set(env.orders["order_id"])
    assert set(env.projects["customer_id"]) <= set(env.institutional_customers["customer_id"])


def test_value_ranges_and_scale(env):
    assert (env.orders["qty_t"] > 0).all()
    assert (env.dealer_weekly["qty_t"] >= 0).all()
    assert env.dealer_weekly["week_start"].nunique() == 72
    assert (env.invoices["due_date"] >= env.invoices["invoice_date"]).all()
    paid = env.invoices["paid_date"].dropna()
    assert (paid <= env.as_of).all()
    trade_share = env.orders.groupby("channel")["qty_t"].sum() / env.orders["qty_t"].sum()
    assert 0.5 < trade_share["Trade"] < 0.85


def test_relationships_are_not_independent(env):
    # Freight per tonne rises with distance (plant-direct orders: freight depends on secondary distance only)
    o = env.orders[env.orders["source_type"] == "Plant"]
    assert o["freight_per_t"].corr(o["secondary_km"]) > 0.99
    # Depot sourcing adds primary freight on top of secondary freight
    d = env.orders[env.orders["source_type"] == "Depot"]
    assert d["freight_per_t"].corr(d["primary_km"] * 1.15 + d["secondary_km"] * 4.1) > 0.95
    # Riskier customers pay later
    inv = env.invoices.dropna(subset=["paid_date"]).merge(env.customers[["customer_id", "latent_risk"]], on="customer_id")
    late = (inv["paid_date"] - inv["due_date"]).dt.days
    assert late.corr(inv["latent_risk"]) > 0.2


def test_validation_passes(env):
    report = validate_environment(env)
    assert (report["Status"] == "PASSED").all(), report


def test_population_scale():
    pop = generate_dealer_population()
    assert len(pop) == 49_300
    assert pop["dealer_id"].is_unique
