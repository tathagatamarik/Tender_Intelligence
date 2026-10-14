"""Freight and dispatch optimisation (order-to-source-and-vehicle assignment MILP).

Decision: for each open order choose ONE (source, vehicle type, fleet mode) option or leave it
unserved for manual intervention.

Objective: minimise total fulfilment cost
    = sourcing cost (plant variable cost; depot stock valued at replacement landed cost)
    + handling + secondary freight (vehicle trips incl. empty return) + unserved-order penalties.

Constraints:
    * every order is assigned exactly once (or flagged unserved);
    * available-to-promise stock per source and product;
    * loading capacity per source over the planning window;
    * contracted fleet availability per source and vehicle type (with multi-trip cycles);
    * optional market-hire vehicles (premium cost, limited per source and type);
    * delivery deadline: an option is only feasible if loading + predicted transit time
      (from the ETA model) fits within the order's deadline;
    * product compatibility: a source can only ship products it holds.

Scope limitation: this is an assignment / allocation model, not a full vehicle-routing problem.
Multi-drop milk runs, driver shifts and dynamic re-dispatch are not modelled.
Solved with SciPy's HiGHS MILP solver; a transparent greedy heuristic is used as fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.config import (
    DEPOT_LOADING_HOURS,
    PLANT_LOADING_HOURS,
    VEHICLE_TYPES,
)
from src.data.geography import road_km
from src.data.synthetic_data import nearest_capable_plant
from src.models.eta_prediction import predict_transit_hours
from src.services.freight_tariff import TariffParams, empty_km, primary_freight_per_t, trip_cost

SCENARIOS = ["Normal dispatch", "Green Channel", "Stock-on-Wheels"]


@dataclass(frozen=True)
class FreightScenario:
    mode: str = "Normal dispatch"
    tariff: TariffParams = field(default_factory=TariffParams)
    vehicle_availability_pct: float = 1.0
    source_capacity_pct: float = 1.0
    disabled_sources: tuple[str, ...] = ()
    urgent_penalty_per_t: float = 4000.0
    standard_penalty_per_t: float = 2500.0
    flexible_penalty_per_t: float = 1500.0
    allow_market_hire: bool = True
    market_vehicles_per_source_type: int = 4
    plan_hours: float = 48.0
    sow_trucks: int = 8
    sow_holding_cost_per_truck_day: float = 2500.0
    green_channel_plant_loading_hours: float = 1.5
    green_channel_depot_loading_hours: float = 1.0
    green_channel_handling_saving_per_t: float = 20.0
    max_trucks_per_order: int = 8


@dataclass
class FreightPlanResult:
    plan: pd.DataFrame
    baseline_kpis: dict
    optimised_kpis: dict
    method: str
    solver_status: str
    objective_value: float
    n_candidates: int
    sources: pd.DataFrame
    scenario: FreightScenario
    fixed_scenario_cost: float = 0.0


# ---------------------------------------------------------------------------
# Sources and candidates
# ---------------------------------------------------------------------------
def build_sources(env, sc: FreightScenario) -> tuple[pd.DataFrame, dict, dict]:
    """Return sources frame, ATP dict {(source, product): t} and vehicle dict {(source, vtype): n}."""
    days = sc.plan_hours / 24.0
    gc = sc.mode == "Green Channel"
    plants, depots = env.plants, env.depots
    rows = []
    for _, p in plants.iterrows():
        rows.append({"source_id": p["plant_id"], "source_name": p["plant_name"], "source_type": "Plant",
                     "city": p["city"], "region": p["region"], "lat": p["lat"], "lon": p["lon"],
                     "hilly": p["terrain"] == "Hilly",
                     "loading_hours": sc.green_channel_plant_loading_hours if gc else PLANT_LOADING_HOURS,
                     "handling_per_t": sc.tariff.plant_handling_per_t - (sc.green_channel_handling_saving_per_t if gc else 0),
                     "loading_capacity_t": p["weekly_capacity_t"] / 7 * 1.3 * days * sc.source_capacity_pct})
    for _, d in depots.iterrows():
        rows.append({"source_id": d["depot_id"], "source_name": d["depot_name"], "source_type": "Depot",
                     "city": d["city"], "region": d["region"], "lat": d["lat"], "lon": d["lon"],
                     "hilly": d["terrain"] == "Hilly",
                     "loading_hours": sc.green_channel_depot_loading_hours if gc else DEPOT_LOADING_HOURS,
                     "handling_per_t": sc.tariff.plant_handling_per_t + sc.tariff.depot_handling_per_t,
                     "loading_capacity_t": d["storage_t"] / 3 * days * sc.source_capacity_pct})
    src = pd.DataFrame(rows)
    src = src[~src["source_id"].isin(sc.disabled_sources)].reset_index(drop=True)

    inv = env.inventory
    atp = {(r.location_id, r.product_id): float(r.available_t) for r in inv.itertuples() if r.available_t > 0}
    veh = env.vehicles[env.vehicles["status"] == "Available"].groupby(["home_source_id", "vehicle_type"]).size()
    vehicles = {k: int(np.floor(n * sc.vehicle_availability_pct)) for k, n in veh.items()}

    if sc.mode == "Stock-on-Wheels" and sc.sow_trucks > 0:
        oo = env.open_orders
        urgent = oo[oo["urgency"] == "Urgent"]
        dep = depots.copy()
        dep["urgent_t"] = [float(urgent["qty_t"][road_km(r.lat, r.lon, urgent["lat"], urgent["lon"]) < 250].sum())
                           for r in dep.itertuples()]
        top = dep.sort_values("urgent_t", ascending=False).head(4)
        top = top[top["urgent_t"] > 0]
        if len(top):
            alloc = np.maximum(1, np.round(sc.sow_trucks * top["urgent_t"] / top["urgent_t"].sum())).astype(int)
            for (_, d), n in zip(top.iterrows(), alloc):
                sid = f"SOW-{d['depot_id']}"
                src = pd.concat([src, pd.DataFrame([{
                    "source_id": sid, "source_name": f"Stock-on-Wheels near {d['city']}", "source_type": "Stock-on-Wheels",
                    "city": d["city"], "region": d["region"], "lat": d["lat"], "lon": d["lon"],
                    "hilly": d["terrain"] == "Hilly", "loading_hours": 0.5,
                    "handling_per_t": sc.tariff.plant_handling_per_t, "loading_capacity_t": n * 25.0}])],
                    ignore_index=True)
                for pid in ("P01", "P03"):
                    atp[(sid, pid)] = n * 25.0 / 2
                vehicles[(sid, "MAV-25")] = int(n)
    return src, atp, vehicles


def _cost_basis(env, src: pd.DataFrame, sc: FreightScenario) -> dict:
    """Per-tonne sourcing cost by (source, product). Depot / SoW stock valued at replacement landed cost."""
    prod = env.products.set_index("product_id")
    plants = env.plants.set_index("plant_id")
    out = {}
    for s in src.itertuples():
        for pid, p in prod.iterrows():
            if s.source_type == "Plant":
                if p["category"] in plants.loc[s.source_id, "categories"]:
                    out[(s.source_id, pid)] = p["base_cost"] + plants.loc[s.source_id, "cost_adj"]
            else:
                mp, km = nearest_capable_plant(s.lat, s.lon, p["category"], env.plants)
                out[(s.source_id, pid)] = (p["base_cost"] + plants.loc[mp, "cost_adj"]
                                           + float(primary_freight_per_t(km, sc.tariff)))
    return out


def build_candidates(env, eta_model, sc: FreightScenario, src: pd.DataFrame, atp: dict, vehicles: dict,
                     cost_basis: dict) -> pd.DataFrame:
    oo = env.open_orders.reset_index(drop=True)
    month = int(env.as_of.month)
    hilly_city = set(env.cities.loc[env.cities["terrain"] == "Hilly", "city"])
    rows = []
    for o in oo.itertuples():
        for s in src.itertuples():
            if atp.get((s.source_id, o.product_id), 0.0) <= 0 or (s.source_id, o.product_id) not in cost_basis:
                continue
            dist = float(road_km(s.lat, s.lon, o.lat, o.lon)) + float(o.last_mile_km)
            for vt, spec in VEHICLE_TYPES.items():
                trucks = int(np.ceil(o.qty_t / spec["capacity_t"] - 1e-9))
                if trucks > sc.max_trucks_per_order:
                    continue
                for market in (False, True):
                    if market and (not sc.allow_market_hire or s.source_type == "Stock-on-Wheels"):
                        continue
                    if not market and vehicles.get((s.source_id, vt), 0) <= 0:
                        continue
                    if s.source_type == "Stock-on-Wheels" and vt != "MAV-25":
                        continue
                    rows.append({"open_order_id": o.open_order_id, "order_idx": o.Index, "source_id": s.source_id,
                                 "source_type": s.source_type, "product_id": o.product_id, "qty_t": o.qty_t,
                                 "vehicle_type": vt, "market_hire": market, "trucks": trucks, "distance_km": dist,
                                 "hilly": bool(s.hilly or o.city in hilly_city),
                                 "loading_hours": s.loading_hours, "handling_per_t": s.handling_per_t,
                                 "deadline_hours": o.deadline_hours})
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand
    cand["transit_hours"] = predict_transit_hours(eta_model, cand["distance_km"].to_numpy(), cand["vehicle_type"].to_numpy(),
                                                  cand["hilly"].to_numpy(), (cand["source_type"] == "Plant").to_numpy(),
                                                  cand["qty_t"].to_numpy(), month)
    cand["delivery_hours"] = cand["loading_hours"] + cand["transit_hours"]
    cand = cand[cand["delivery_hours"] <= cand["deadline_hours"]].copy()
    cand["cycles"] = np.maximum(1, np.floor(sc.plan_hours / (cand["loading_hours"] + 2 * cand["transit_hours"]))).astype(int)
    cand.loc[cand["market_hire"] | (cand["source_type"] == "Stock-on-Wheels"), "cycles"] = 1
    freight = np.zeros(len(cand))
    for vt in VEHICLE_TYPES:
        for mk in (False, True):
            m = ((cand["vehicle_type"] == vt) & (cand["market_hire"] == mk)).to_numpy()
            if m.any():
                freight[m] = cand.loc[m, "trucks"].to_numpy() * trip_cost(cand.loc[m, "distance_km"].to_numpy(), vt,
                                                                          sc.tariff, market_hire=mk)
    cand["secondary_freight_inr"] = freight
    cand["sourcing_cost_per_t"] = [cost_basis[(s, p)] for s, p in zip(cand["source_id"], cand["product_id"])]
    cand["total_cost_inr"] = cand["qty_t"] * (cand["sourcing_cost_per_t"] + cand["handling_per_t"]) + cand["secondary_freight_inr"]
    return cand.reset_index(drop=True)


def _penalties(oo: pd.DataFrame, sc: FreightScenario) -> np.ndarray:
    urg = oo["urgency"].map({"Urgent": sc.urgent_penalty_per_t, "Standard": sc.standard_penalty_per_t,
                             "Flexible": sc.flexible_penalty_per_t}).fillna(sc.standard_penalty_per_t)
    # Lost revenue plus a service penalty: serving any feasible option is always preferred.
    return (oo["qty_t"] * (oo["list_price_per_t"] + urg)).to_numpy()


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def solve_assignment(cand: pd.DataFrame, oo: pd.DataFrame, src: pd.DataFrame, atp: dict, vehicles: dict,
                     sc: FreightScenario, time_limit: float = 30.0) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    n_c, n_o = len(cand), len(oo)
    n = n_c + n_o
    c = np.concatenate([cand["total_cost_inr"].to_numpy(), _penalties(oo, sc)])

    # Equality: each order assigned once (or unserved)
    rows = np.concatenate([cand["order_idx"].to_numpy(), np.arange(n_o)])
    cols = np.arange(n)
    A_eq = coo_matrix((np.ones(n), (rows, cols)), shape=(n_o, n)).tocsr()

    ub_rows, ub_cols, ub_vals, b = [], [], [], []
    r = 0
    qty = cand["qty_t"].to_numpy()
    for (sid, pid), g in cand.groupby(["source_id", "product_id"]).groups.items():
        idx = np.asarray(g)
        ub_rows += [r] * len(idx); ub_cols += list(idx); ub_vals += list(qty[idx]); b.append(atp.get((sid, pid), 0.0)); r += 1
    cap = src.set_index("source_id")["loading_capacity_t"]
    for sid, g in cand.groupby("source_id").groups.items():
        idx = np.asarray(g)
        ub_rows += [r] * len(idx); ub_cols += list(idx); ub_vals += list(qty[idx]); b.append(float(cap[sid])); r += 1
    occ = (cand["trucks"] / cand["cycles"]).to_numpy()
    for (sid, vt, mk), g in cand.groupby(["source_id", "vehicle_type", "market_hire"]).groups.items():
        idx = np.asarray(g)
        limit = sc.market_vehicles_per_source_type if mk else vehicles.get((sid, vt), 0)
        ub_rows += [r] * len(idx); ub_cols += list(idx); ub_vals += list(occ[idx]); b.append(float(limit)); r += 1
    A_ub = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(r, n)).tocsr()

    res = milp(c=c, constraints=[LinearConstraint(A_eq, 1, 1), LinearConstraint(A_ub, -np.inf, np.array(b))],
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-3, "disp": False})
    if res.x is not None and res.status in (0, 1):
        x = np.round(res.x).astype(int)
        status = "Optimal" if res.status == 0 else "Feasible (time limit reached)"
        return x[:n_c], x[n_c:], "Mixed-integer linear programme (SciPy HiGHS)", status, float(res.fun)
    xg, ug = greedy_assignment(cand, oo, src, atp, vehicles, sc)
    return xg, ug, "Greedy heuristic (MILP fallback)", f"MILP failed: {res.message}", float(
        (cand["total_cost_inr"].to_numpy() * xg).sum() + (_penalties(oo, sc) * ug).sum())


def greedy_assignment(cand, oo, src, atp, vehicles, sc) -> tuple[np.ndarray, np.ndarray]:
    """Transparent fallback: urgent orders first, each takes its cheapest option that still fits."""
    atp_left = dict(atp)
    cap_left = src.set_index("source_id")["loading_capacity_t"].to_dict()
    veh_left = {**vehicles}
    mkt_left: dict = {}
    x = np.zeros(len(cand), dtype=int)
    u = np.ones(len(oo), dtype=int)
    order_seq = oo.assign(rank=oo["urgency"].map({"Urgent": 0, "Standard": 1, "Flexible": 2})).sort_values("rank").index
    for oi in order_seq:
        opts = cand[cand["order_idx"] == oi].sort_values("total_cost_inr")
        for ci, o in opts.iterrows():
            occ = o["trucks"] / o["cycles"]
            key = (o["source_id"], o["vehicle_type"])
            fleet_ok = (mkt_left.get(key, sc.market_vehicles_per_source_type) >= occ) if o["market_hire"] else (veh_left.get(key, 0) >= occ)
            if atp_left.get((o["source_id"], o["product_id"]), 0) >= o["qty_t"] and cap_left[o["source_id"]] >= o["qty_t"] and fleet_ok:
                x[ci] = 1; u[oi] = 0
                atp_left[(o["source_id"], o["product_id"])] -= o["qty_t"]
                cap_left[o["source_id"]] -= o["qty_t"]
                if o["market_hire"]:
                    mkt_left[key] = mkt_left.get(key, sc.market_vehicles_per_source_type) - occ
                else:
                    veh_left[key] -= occ
                break
    return x, u


# ---------------------------------------------------------------------------
# Baseline (current practice)
# ---------------------------------------------------------------------------
def baseline_plan(env, eta_model, sc: FreightScenario, src: pd.DataFrame, atp: dict, vehicles: dict,
                  cost_basis: dict) -> pd.DataFrame:
    """Current practice: designated source first, else nearest stock point; standard 25 t trucks.

    Deadlines are not checked at planning time (on-time performance is measured afterwards).
    """
    oo = env.open_orders.reset_index(drop=True)
    srcx = src.set_index("source_id")
    atp_left = dict(atp)
    cap_left = srcx["loading_capacity_t"].to_dict()
    veh_left = dict(vehicles)
    hilly_city = set(env.cities.loc[env.cities["terrain"] == "Hilly", "city"])
    rows = []
    for o in oo.itertuples():
        d_all = road_km(srcx["lat"].to_numpy(), srcx["lon"].to_numpy(), o.lat, o.lon)
        order = list(srcx.index[np.argsort(d_all)])
        if o.designated_source_id in srcx.index:
            order.remove(o.designated_source_id)
            order.insert(0, o.designated_source_id)
        chosen = None
        for sid in order:
            if srcx.loc[sid, "source_type"] == "Stock-on-Wheels":
                continue
            if atp_left.get((sid, o.product_id), 0) >= o.qty_t and cap_left[sid] >= o.qty_t and (sid, o.product_id) in cost_basis:
                chosen = sid
                break
        if chosen is None:
            rows.append({"open_order_id": o.open_order_id, "base_source_id": None, "base_vehicle_type": None,
                         "base_trucks": 0, "base_market_hire": False, "base_distance_km": np.nan,
                         "base_delivery_hours": np.nan, "base_cost_inr": np.nan, "base_freight_inr": np.nan,
                         "base_sourcing_cost_per_t": np.nan, "base_handling_per_t": np.nan})
            continue
        atp_left[(chosen, o.product_id)] -= o.qty_t
        cap_left[chosen] -= o.qty_t
        s = srcx.loc[chosen]
        dist = float(road_km(s["lat"], s["lon"], o.lat, o.lon)) + float(o.last_mile_km)
        # Standard practice: 25 t truck from the contracted fleet; otherwise any contracted type
        # with enough vehicles; otherwise market-hire 25 t trucks.
        vt, market = "MAV-25", True
        for cand_vt in ("MAV-25", "TRL-35", "LCV-10"):
            n_req = int(np.ceil(o.qty_t / VEHICLE_TYPES[cand_vt]["capacity_t"] - 1e-9))
            if n_req <= sc.max_trucks_per_order and veh_left.get((chosen, cand_vt), 0) >= n_req:
                vt, market = cand_vt, False
                break
        trucks = int(np.ceil(o.qty_t / VEHICLE_TYPES[vt]["capacity_t"] - 1e-9))
        if not market:
            veh_left[(chosen, vt)] -= trucks
        transit = float(predict_transit_hours(eta_model, [dist], vt, bool(s["hilly"] or o.city in hilly_city),
                                              s["source_type"] == "Plant", o.qty_t, int(env.as_of.month))[0])
        freight = trucks * float(trip_cost(dist, vt, sc.tariff, market_hire=market))
        basis = cost_basis[(chosen, o.product_id)]
        rows.append({"open_order_id": o.open_order_id, "base_source_id": chosen, "base_vehicle_type": vt,
                     "base_trucks": trucks, "base_market_hire": market, "base_distance_km": dist,
                     "base_delivery_hours": s["loading_hours"] + transit,
                     "base_cost_inr": o.qty_t * (basis + s["handling_per_t"]) + freight,
                     "base_freight_inr": freight, "base_sourcing_cost_per_t": basis,
                     "base_handling_per_t": s["handling_per_t"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KPIs and orchestration
# ---------------------------------------------------------------------------
def plan_kpis(plan: pd.DataFrame, prefix: str, tariff: TariffParams, fixed_cost: float = 0.0) -> dict:
    served = plan[plan[f"{prefix}source_id"].notna()]
    q = served["qty_t"].sum()
    trucks = served[f"{prefix}trucks"].sum()
    cap = (served[f"{prefix}trucks"] * served[f"{prefix}vehicle_type"].map(lambda v: VEHICLE_TYPES[v]["capacity_t"])).sum()
    loaded_km = (served[f"{prefix}trucks"] * served[f"{prefix}distance_km"]).sum()
    freight = served[f"{prefix}freight_inr"].sum()
    primary = (served["qty_t"] * served[f"{prefix}primary_per_t"]).sum() if f"{prefix}primary_per_t" in served else 0.0
    on_time = (served[f"{prefix}delivery_hours"] <= served["deadline_hours"]).mean() if len(served) else np.nan
    return {
        "Orders served": int(len(served)),
        "Tonnes served": float(q),
        "Orders requiring manual intervention": int(len(plan) - len(served)),
        "Total fulfilment cost (INR)": float(served[f"{prefix}cost_inr"].sum() + fixed_cost),
        "Secondary freight cost (INR)": float(freight),
        "Freight cost per tonne (INR)": float(freight / q) if q else np.nan,
        "Logistics cost per tonne (INR)": float((freight + primary + (served["qty_t"] * served[f"{prefix}handling_per_t"]).sum() + fixed_cost) / q) if q else np.nan,
        "Cost per loaded km (INR)": float(freight / loaded_km) if loaded_km else np.nan,
        "Vehicle utilisation (%)": float(q / cap * 100) if cap else np.nan,
        "Vehicles dispatched": int(trucks),
        "On-time delivery (%)": float(on_time * 100) if on_time == on_time else np.nan,
        "Average delivery time (hours)": float(served[f"{prefix}delivery_hours"].mean()) if len(served) else np.nan,
        "Loaded km": float(loaded_km),
        "Empty return km (estimate)": float(empty_km(loaded_km, tariff)),
        "Market-hire vehicles": int(served.loc[served[f"{prefix}market_hire"].astype(bool), f"{prefix}trucks"].sum()),
    }


def optimise_dispatch(env, eta_model, sc: FreightScenario = FreightScenario()) -> FreightPlanResult:
    src, atp, vehicles = build_sources(env, sc)
    cost_basis = _cost_basis(env, src, sc)
    oo = env.open_orders.reset_index(drop=True)
    cand = build_candidates(env, eta_model, sc, src, atp, vehicles, cost_basis)
    base_sc_src, base_atp, base_veh = build_sources(env, FreightScenario(tariff=sc.tariff,
                                                                         vehicle_availability_pct=sc.vehicle_availability_pct,
                                                                         source_capacity_pct=sc.source_capacity_pct,
                                                                         disabled_sources=sc.disabled_sources,
                                                                         plan_hours=sc.plan_hours))
    base = baseline_plan(env, eta_model, sc, base_sc_src, base_atp, base_veh, _cost_basis(env, base_sc_src, sc))

    if cand.empty:
        x = np.zeros(0, dtype=int); u = np.ones(len(oo), dtype=int)
        method, status, obj = "No feasible options", "No candidates", float(_penalties(oo, sc).sum())
    else:
        x, u, method, status, obj = solve_assignment(cand, oo, src, atp, vehicles, sc)
    chosen = cand[x == 1] if len(cand) else cand
    opt = chosen.rename(columns={"source_id": "opt_source_id", "vehicle_type": "opt_vehicle_type", "trucks": "opt_trucks",
                                 "market_hire": "opt_market_hire", "distance_km": "opt_distance_km",
                                 "delivery_hours": "opt_delivery_hours", "total_cost_inr": "opt_cost_inr",
                                 "secondary_freight_inr": "opt_freight_inr", "sourcing_cost_per_t": "opt_sourcing_cost_per_t",
                                 "handling_per_t": "opt_handling_per_t", "transit_hours": "opt_transit_hours"})
    keep = ["open_order_id", "opt_source_id", "opt_vehicle_type", "opt_trucks", "opt_market_hire", "opt_distance_km",
            "opt_delivery_hours", "opt_cost_inr", "opt_freight_inr", "opt_sourcing_cost_per_t", "opt_handling_per_t"]
    plan = oo[["open_order_id", "customer_id", "customer_name", "channel", "customer_segment", "region", "city",
               "product_id", "category", "qty_t", "urgency", "deadline_hours", "list_price_per_t", "discount_per_t",
               "designated_source_id"]].merge(base, on="open_order_id", how="left")
    plan = plan.merge(opt[keep] if len(opt) else pd.DataFrame(columns=keep), on="open_order_id", how="left")
    plan["opt_market_hire"] = plan["opt_market_hire"].eq(True)
    plan["base_market_hire"] = plan["base_market_hire"].eq(True)
    plan["opt_trucks"] = plan["opt_trucks"].fillna(0).astype(int)

    # Primary freight embedded in depot sourcing cost (for logistics KPIs)
    prod_cost = env.products.set_index("product_id")["base_cost"]
    plant_adj = env.plants.set_index("plant_id")["cost_adj"]
    def primary(sid, pid, basis):
        if pd.isna(sid) or str(sid).startswith("PL"):
            return 0.0
        cat = env.products.set_index("product_id").loc[pid, "category"]
        s = src.set_index("source_id").loc[sid] if sid in set(src["source_id"]) else base_sc_src.set_index("source_id").loc[sid]
        mp, _ = nearest_capable_plant(s["lat"], s["lon"], cat, env.plants)
        return float(basis - prod_cost[pid] - plant_adj[mp])
    plan["base_primary_per_t"] = [primary(s, p, b) for s, p, b in zip(plan["base_source_id"], plan["product_id"], plan["base_sourcing_cost_per_t"])]
    plan["opt_primary_per_t"] = [primary(s, p, b) for s, p, b in zip(plan["opt_source_id"], plan["product_id"], plan["opt_sourcing_cost_per_t"])]

    plan["saving_inr"] = plan["base_cost_inr"] - plan["opt_cost_inr"]
    plan["base_on_time"] = np.where(plan["base_source_id"].notna(), plan["base_delivery_hours"] <= plan["deadline_hours"], np.nan)
    plan["opt_on_time"] = np.where(plan["opt_source_id"].notna(), plan["opt_delivery_hours"] <= plan["deadline_hours"], np.nan)
    plan["status"] = np.select(
        [plan["opt_source_id"].isna(), plan["opt_source_id"] != plan["base_source_id"],
         plan["opt_vehicle_type"] != plan["base_vehicle_type"]],
        ["ACTION REQUIRED - NO FEASIBLE OPTION", "RE-SOURCED", "VEHICLE CHANGED"], "UNCHANGED")
    plan["approval_status"] = "PENDING DISPATCH APPROVAL"

    fixed = 0.0
    if sc.mode == "Stock-on-Wheels":
        sow = src[src["source_type"] == "Stock-on-Wheels"]
        fixed = float(sum(vehicles.get((s, "MAV-25"), 0) for s in sow["source_id"])) * sc.sow_holding_cost_per_truck_day * sc.plan_hours / 24
    base_kpis = plan_kpis(plan, "base_", sc.tariff)
    opt_kpis = plan_kpis(plan, "opt_", sc.tariff, fixed_cost=fixed)
    return FreightPlanResult(plan=plan, baseline_kpis=base_kpis, optimised_kpis=opt_kpis, method=method,
                             solver_status=status, objective_value=obj, n_candidates=len(cand), sources=src,
                             scenario=sc, fixed_scenario_cost=fixed)


def savings_breakdown(plan: pd.DataFrame, by: str) -> pd.DataFrame:
    """Cost comparison for orders served in both plans, grouped by a dimension."""
    both = plan[plan["base_source_id"].notna() & plan["opt_source_id"].notna()].copy()
    if by == "route":
        both["route"] = both["opt_source_id"] + " to " + both["city"]
        key = "route"
    elif by == "source":
        key = "opt_source_id"
    else:
        key = by
    g = both.groupby(key).agg(orders=("open_order_id", "count"), tonnes=("qty_t", "sum"),
                              baseline_cost_inr=("base_cost_inr", "sum"), optimised_cost_inr=("opt_cost_inr", "sum"),
                              baseline_freight_inr=("base_freight_inr", "sum"), optimised_freight_inr=("opt_freight_inr", "sum"))
    g["saving_inr"] = g["baseline_cost_inr"] - g["optimised_cost_inr"]
    g["saving_pct"] = g["saving_inr"] / g["baseline_cost_inr"]
    return g.reset_index().sort_values("saving_inr", ascending=False)


def compare_scenarios(env, eta_model, base_sc: FreightScenario) -> pd.DataFrame:
    rows = []
    for mode in SCENARIOS:
        sc = FreightScenario(**{**base_sc.__dict__, "mode": mode})
        res = optimise_dispatch(env, eta_model, sc)
        k = res.optimised_kpis
        rows.append({"Scenario": mode, "Total fulfilment cost (INR)": k["Total fulfilment cost (INR)"],
                     "Freight cost per tonne (INR)": k["Freight cost per tonne (INR)"],
                     "On-time delivery (%)": k["On-time delivery (%)"],
                     "Average delivery time (hours)": k["Average delivery time (hours)"],
                     "Orders served": k["Orders served"],
                     "Orders requiring manual intervention": k["Orders requiring manual intervention"],
                     "Vehicle utilisation (%)": k["Vehicle utilisation (%)"],
                     "Scenario fixed cost (INR)": res.fixed_scenario_cost})
    return pd.DataFrame(rows)
