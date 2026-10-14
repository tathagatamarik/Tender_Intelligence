import numpy as np
import pandas as pd
import pytest

from src.config import VEHICLE_TYPES
from src.optimisation.freight_optimisation import FreightScenario, build_sources, optimise_dispatch
from src.optimisation.inventory_optimisation import PlanningScenario, build_planning_demand, run_planning
from src.optimisation.replenishment import ReplenishmentSettings, recommend_replenishment
from src.models.dealer_segmentation import receivables_position


@pytest.fixture(scope="module")
def plan(env, eta):
    return optimise_dispatch(env, eta.model, FreightScenario())


def test_freight_each_order_assigned_at_most_once(plan, env):
    assert plan.plan["open_order_id"].is_unique
    assert len(plan.plan) == len(env.open_orders)
    assert plan.solver_status.startswith(("Optimal", "Feasible"))


def test_freight_respects_deadlines_and_capacity(plan, env):
    p = plan.plan[plan.plan["opt_source_id"].notna()]
    assert (p["opt_delivery_hours"] <= p["deadline_hours"] + 1e-6).all()
    cap = p["opt_vehicle_type"].map(lambda v: VEHICLE_TYPES[v]["capacity_t"])
    assert (p["opt_trucks"] * cap >= p["qty_t"] - 1e-6).all()
    src, atp, veh = build_sources(env, FreightScenario())
    used = p.groupby(["opt_source_id", "product_id"])["qty_t"].sum()
    for (s, prod), q in used.items():
        assert q <= atp.get((s, prod), 0) + 1e-6
    load = p.groupby("opt_source_id")["qty_t"].sum()
    caps = src.set_index("source_id")["loading_capacity_t"]
    assert (load <= caps.reindex(load.index) + 1e-6).all()


def test_freight_not_worse_than_baseline_on_common_orders(plan):
    both = plan.plan[plan.plan["base_source_id"].notna() & plan.plan["opt_source_id"].notna()]
    # Baseline may violate deadlines; on orders where baseline was on time it is a feasible option, so cost cannot be lower
    feasible_base = both[both["base_delivery_hours"] <= both["deadline_hours"]]
    assert feasible_base["opt_cost_inr"].sum() <= feasible_base["base_cost_inr"].sum() * 1.02


def test_disabled_source_is_not_used(env, eta):
    res = optimise_dispatch(env, eta.model, FreightScenario(disabled_sources=("PL01", "DP08")))
    assert not res.plan["opt_source_id"].isin(["PL01", "DP08"]).any()


def test_higher_freight_rates_raise_cost(env, eta, plan):
    from src.services.freight_tariff import TariffParams
    hi = optimise_dispatch(env, eta.model, FreightScenario(tariff=TariffParams(rate_multiplier=1.4, diesel_price=120)))
    assert hi.optimised_kpis["Secondary freight cost (INR)"] > plan.optimised_kpis["Secondary freight cost (INR)"]


def test_replenishment_logic(env, demand):
    rs = ReplenishmentSettings(lot_size_t=5, min_order_t=5, respect_credit_limit=False)
    r = recommend_replenishment(env, demand, rs, receivables_position(env), 6800, 950)
    assert (r["recommended_qty_t"] >= 0).all()
    assert np.allclose(r["recommended_qty_t"] % 5, 0)
    pos = r[r["recommended_qty_t"] > 0]
    assert (pos["recommended_qty_t"] + pos["net_position_t"] >= pos["target_stock_t"] - 1e-6).all()
    zero = r[r["net_position_t"] >= r["target_stock_t"]]
    assert (zero["recommended_qty_t"] == 0).all()
    assert r["stockout_probability"].between(0, 1).all()
    capped = recommend_replenishment(env, demand, ReplenishmentSettings(), receivables_position(env), 6800, 950)
    assert capped["recommended_qty_t"].sum() <= r["recommended_qty_t"].sum()


def test_replenishment_responds_to_service_level(env, demand):
    rec = receivables_position(env)
    lo = recommend_replenishment(env, demand, ReplenishmentSettings(service_level=0.85, respect_credit_limit=False), rec, 6800, 950)
    hi = recommend_replenishment(env, demand, ReplenishmentSettings(service_level=0.99, respect_credit_limit=False), rec, 6800, 950)
    assert hi["safety_stock_t"].sum() > lo["safety_stock_t"].sum()


@pytest.fixture(scope="module")
def planning_demand(env, demand):
    return build_planning_demand(env, demand, None, None, None, PlanningScenario(horizon_weeks=6))


def test_planning_capacity_and_conservation(env, planning_demand):
    sc = PlanningScenario(horizon_weeks=6)
    res = run_planning(env, planning_demand, sc)
    prod = res.production.groupby(["plant_id", "week"])["regular_t"].sum()
    cap = env.plants.set_index("plant_id")["weekly_capacity_t"]
    for (p, w), q in prod.items():
        assert q <= cap[p] + 1e-4
    # Conservation: initial stock + production = shipments-consumed demand + closing stock (network level)
    inv = env.inventory
    init = inv["on_hand_t"].sum()
    produced = res.production[["regular_t", "overtime_t"]].fillna(0).sum().sum()
    served = res.optimised["Total demand (t)"] - res.optimised["Stockout quantity (t)"]
    closing_plant_depot = init + produced - served
    assert closing_plant_depot >= -1e-3
    assert 0 <= res.optimised["Service level (fill rate %)"] <= 100


def test_planning_optimum_not_worse_than_baseline(env, planning_demand):
    res = run_planning(env, planning_demand, PlanningScenario(horizon_weeks=6))
    assert res.optimised["Total operating cost (INR)"] <= res.baseline["Total operating cost (INR)"] + 1.0


def test_plant_shutdown_enforced(env, planning_demand):
    res = run_planning(env, planning_demand, PlanningScenario(horizon_weeks=6, shutdown_plant="PL04", shutdown_weeks=(1, 3)))
    p = res.production[(res.production["plant_id"] == "PL04") & (res.production["week"] <= 3)]
    assert p[["regular_t", "overtime_t"]].fillna(0).to_numpy().sum() < 1e-6
