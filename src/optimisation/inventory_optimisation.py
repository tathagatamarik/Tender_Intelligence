"""Demand-driven multi-period production, allocation and inventory optimisation (LP).

Network: plants (production by product category) -> depots (demand nodes), with optional
depot-to-depot transfers between depots within a distance limit.

Decision variables per week t over the planning horizon:
    prod[p,k,t], overtime[p,k,t], ship[p,d,k,t], transfer[d,e,k,t], inv_plant[p,k,t],
    inv_depot[d,k,t], unmet[d,k,t], safety_shortfall[d,k,t]

Objective (minimise): production cost + overtime premium + primary freight and handling +
transfer cost + inventory holding cost + unmet-demand cost + safety-stock shortfall penalty.
Unmet demand is costed as replacement sourcing (production-cost proxy) plus a service penalty,
so baseline and optimised plans are compared on the same total relevant cost.

Constraints: inventory conservation at plants and depots, regular and overtime capacity,
plant shutdowns, storage limits, unmet demand <= demand.

Simplifying assumptions (stated in the UI): weekly buckets; shipments arrive within the same
week; unmet demand is lost (not back-ordered); linear costs; product categories aggregate grades.
Re-running weekly with updated forecasts gives a rolling-horizon plan; only week 1 is executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.config import HOLDING_COST_PCT_ANNUAL, PRODUCT_CATEGORIES
from src.data.geography import road_km
from src.services.freight_tariff import TariffParams, primary_freight_per_t, secondary_freight_per_t


@dataclass(frozen=True)
class PlanningScenario:
    horizon_weeks: int = 8
    demand_multiplier: float = 1.0
    region_surge: tuple[tuple[str, float], ...] = ()          # (region, multiplier)
    capacity_multiplier: float = 1.0
    shutdown_plant: str | None = None
    shutdown_weeks: tuple[int, int] = (1, 2)                    # inclusive, 1-based
    holding_cost_multiplier: float = 1.0
    overtime_share: float = 0.10
    overtime_premium_per_t: float = 600.0
    unmet_penalty_per_t: float = 3500.0
    safety_shortfall_penalty_per_t: float = 60.0
    transfer_max_km: float = 700.0
    include_tender_demand: bool = True
    apply_offtake_alerts: bool = True
    tariff: TariffParams = field(default_factory=TariffParams)


@dataclass
class PlanningResult:
    baseline: dict
    optimised: dict
    production: pd.DataFrame
    shipments: pd.DataFrame
    transfers: pd.DataFrame
    depot_status: pd.DataFrame
    weekly: pd.DataFrame
    demand: pd.DataFrame
    solver_status: str


# ---------------------------------------------------------------------------
# Demand assembly (cross-module integration)
# ---------------------------------------------------------------------------
def build_planning_demand(env, demand_result, replenishment: pd.DataFrame | None, offtake, tenders_scored,
                          sc: PlanningScenario) -> pd.DataFrame:
    """Depot x category x week demand from trade forecast, replenishment, projects and tenders."""
    H = sc.horizon_weeks
    cats = env.products.set_index("product_id")["category"]
    # 1. Trade: dealer forecasts by depot and category
    f = demand_result.forecast[demand_result.forecast["horizon"] <= H].merge(
        demand_result.series[["series_id", "depot_id", "product_id"]], on="series_id")
    f["category"] = f["product_id"].map(cats)
    trade = f.groupby(["depot_id", "category", "horizon"])["forecast"].sum().rename("trade_t")
    # 2. Week 1 trade = max(forecast, recommended dealer replenishment) - replenishment drives sell-in
    if replenishment is not None and len(replenishment):
        r = replenishment.merge(env.dealers[["dealer_id", "depot_id"]], on="dealer_id")
        r["category"] = r["product_id"].map(cats)
        rep = r.groupby(["depot_id", "category"])["recommended_qty_t"].sum()
    else:
        rep = pd.Series(dtype=float)
    # 3. Institutional projects: planned schedule, optionally adjusted by early-warning forecasts
    pm = env.project_monthly
    as_of_month = env.as_of.to_period("M").to_timestamp()
    fut = pm[pm["month"] >= as_of_month].merge(env.projects[["project_id", "lat", "lon", "product_category"]],
                                               on="project_id")
    if sc.apply_offtake_alerts and offtake is not None:
        fa = offtake.projects.set_index("project_id")["forecast_attainment"]
        fut["planned_t"] = fut["planned_t"] * fut["project_id"].map(fa).fillna(1.0)
    depots = env.depots
    def nearest_depot(lat, lon):
        return depots.iloc[int(np.argmin(road_km(lat, lon, depots["lat"].to_numpy(), depots["lon"].to_numpy())))]["depot_id"]
    proj_dep = {r.project_id: nearest_depot(r.lat, r.lon) for r in env.projects.itertuples()}
    fut["depot_id"] = fut["project_id"].map(proj_dep)
    weeks = pd.date_range(env.as_of, periods=H, freq="7D")
    inst_rows = []
    for h, wk in enumerate(weeks, start=1):
        m = wk.to_period("M").to_timestamp()
        g = fut[fut["month"] == m].groupby(["depot_id", "product_category"])["planned_t"].sum() / 4.345
        for (dep, cat), v in g.items():
            inst_rows.append((dep, cat, h, v))
    inst = pd.DataFrame(inst_rows, columns=["depot_id", "category", "horizon", "institutional_t"])
    inst = inst.groupby(["depot_id", "category", "horizon"])["institutional_t"].sum()
    # 4. Probability-weighted tender demand for tenders recommended to bid
    tend = pd.Series(dtype=float)
    if sc.include_tender_demand and tenders_scored is not None and len(tenders_scored):
        rows = []
        for t in tenders_scored[tenders_scored["recommendation"].str.startswith("BID")].itertuples():
            start = int(np.ceil(max(t.days_to_close, 0) / 7)) + 2
            weekly = t.p_win * t.quantity_t / (t.delivery_months * 4.345)
            city = env.cities[env.cities["city"] == t.project_location]
            if city.empty:
                continue
            dep = nearest_depot(city["lat"].iloc[0], city["lon"].iloc[0])
            for h in range(start, H + 1):
                rows.append((dep, t.product_category, h, weekly))
        if rows:
            tend = pd.DataFrame(rows, columns=["depot_id", "category", "horizon", "v"]).groupby(
                ["depot_id", "category", "horizon"])["v"].sum()

    idx = pd.MultiIndex.from_product([depots["depot_id"], PRODUCT_CATEGORIES, range(1, H + 1)],
                                     names=["depot_id", "category", "horizon"])
    d = pd.DataFrame(index=idx)
    d["trade_t"] = trade.reindex(idx).fillna(0.0)
    d["institutional_t"] = inst.reindex(idx).fillna(0.0)
    d["tender_t"] = tend.reindex(idx).fillna(0.0) if len(tend) else 0.0
    d = d.reset_index()
    if len(rep):
        w1 = d["horizon"] == 1
        rep_v = pd.Series([rep.get((a, b), 0.0) for a, b in zip(d.loc[w1, "depot_id"], d.loc[w1, "category"])],
                          index=d.index[w1])
        d.loc[w1, "replenishment_uplift_t"] = (rep_v - d.loc[w1, "trade_t"]).clip(lower=0)
    d["replenishment_uplift_t"] = d.get("replenishment_uplift_t", 0.0)
    d["replenishment_uplift_t"] = d["replenishment_uplift_t"].fillna(0.0)
    region = depots.set_index("depot_id")["region"]
    surge = dict(sc.region_surge)
    mult = sc.demand_multiplier * d["depot_id"].map(region).map(lambda r: surge.get(r, 1.0))
    d["demand_t"] = (d["trade_t"] + d["replenishment_uplift_t"] + d["institutional_t"] + d["tender_t"]) * mult
    return d


# ---------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------
def _network(env, sc: PlanningScenario):
    plants, depots = env.plants.reset_index(drop=True), env.depots.reset_index(drop=True)
    H = sc.horizon_weeks
    cost_by_cat = env.products.groupby("category")["base_cost"].mean()
    cap = np.tile((plants["weekly_capacity_t"].to_numpy() * sc.capacity_multiplier)[:, None], (1, H))
    if sc.shutdown_plant in set(plants["plant_id"]):
        i = int(plants.index[plants["plant_id"] == sc.shutdown_plant][0])
        a, b = sc.shutdown_weeks
        cap[i, max(a, 1) - 1:min(b, H)] = 0.0
    capable = {(i, k) for i, p in plants.iterrows() for k in p["categories"]}
    dist_pd = road_km(plants["lat"].to_numpy()[:, None], plants["lon"].to_numpy()[:, None],
                      depots["lat"].to_numpy()[None, :], depots["lon"].to_numpy()[None, :])
    dist_dd = road_km(depots["lat"].to_numpy()[:, None], depots["lon"].to_numpy()[:, None],
                      depots["lat"].to_numpy()[None, :], depots["lon"].to_numpy()[None, :])
    ship_cost = primary_freight_per_t(dist_pd, sc.tariff) + sc.tariff.depot_handling_per_t
    trans_cost = secondary_freight_per_t(dist_dd, sc.tariff) + sc.tariff.depot_handling_per_t
    inv = env.inventory
    init_d = inv[inv["location_type"] == "Depot"].groupby(["location_id", "category"])["on_hand_t"].sum()
    init_p = inv[inv["location_type"] == "Plant"].groupby(["location_id", "category"])["on_hand_t"].sum()
    ss = inv[inv["location_type"] == "Depot"].groupby(["location_id", "category"])["safety_stock_t"].sum()
    hold = cost_by_cat * HOLDING_COST_PCT_ANNUAL / 52 * sc.holding_cost_multiplier
    return plants, depots, cap, capable, dist_dd, ship_cost, trans_cost, init_d, init_p, ss, hold, cost_by_cat


def _cost_rows(plants, cost_by_cat):
    return {(i, k): cost_by_cat[k] + plants.loc[i, "cost_adj"] for i in plants.index for k in PRODUCT_CATEGORIES}


# ---------------------------------------------------------------------------
# Optimised plan (LP)
# ---------------------------------------------------------------------------
def optimise_plan(env, demand: pd.DataFrame, sc: PlanningScenario) -> tuple[dict, dict]:
    plants, depots, cap, capable, dist_dd, ship_cost, trans_cost, init_d, init_p, ss, hold, cost_by_cat = _network(env, sc)
    H, K = sc.horizon_weeks, PRODUCT_CATEGORIES
    P, D = len(plants), len(depots)
    pcost = _cost_rows(plants, cost_by_cat)
    dem = demand.set_index(["depot_id", "category", "horizon"])["demand_t"]
    pairs = [(a, b) for a in range(D) for b in range(D) if a != b and dist_dd[a, b] <= sc.transfer_max_km]

    var = {}
    costs = []
    ub = []

    def add(key, cost, upper=np.inf):
        var[key] = len(costs)
        costs.append(cost)
        ub.append(upper)

    for t in range(H):
        for (i, k) in sorted(capable):
            add(("prod", i, k, t), pcost[(i, k)])
            add(("ot", i, k, t), pcost[(i, k)] + sc.overtime_premium_per_t)
            add(("invp", i, k, t), hold[k])
            for j in range(D):
                add(("ship", i, j, k, t), ship_cost[i, j])
        for j in range(D):
            for k in K:
                add(("invd", j, k, t), hold[k])
                add(("unmet", j, k, t), sc.unmet_penalty_per_t + cost_by_cat[k],
                    float(dem.get((depots.loc[j, "depot_id"], k, t + 1), 0.0)))
                add(("ssshort", j, k, t), sc.safety_shortfall_penalty_per_t,
                    float(ss.get((depots.loc[j, "depot_id"], k), 0.0)))
                for (a, b) in pairs:
                    if a == j:
                        add(("tr", a, b, k, t), trans_cost[a, b])

    n = len(costs)
    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    ub_r, ub_c, ub_v, ub_b = [], [], [], []
    r_eq = r_ub = 0
    for t in range(H):
        # Plant inventory balance: invp[t] - invp[t-1] - prod - ot + sum ship = 0 (invp[-1] = initial)
        for (i, k) in sorted(capable):
            cols = [var[("invp", i, k, t)], var[("prod", i, k, t)], var[("ot", i, k, t)]]
            vals = [1.0, -1.0, -1.0]
            cols += [var[("ship", i, j, k, t)] for j in range(D)]
            vals += [1.0] * D
            rhs = float(init_p.get((plants.loc[i, "plant_id"], k), 0.0)) if t == 0 else 0.0
            if t > 0:
                cols.append(var[("invp", i, k, t - 1)]); vals.append(-1.0)
            eq_r += [r_eq] * len(cols); eq_c += cols; eq_v += vals; eq_b.append(rhs); r_eq += 1
        # Depot balance: invd[t] - invd[t-1] - inbound + outbound - unmet = -demand
        for j in range(D):
            dep_id = depots.loc[j, "depot_id"]
            for k in K:
                cols = [var[("invd", j, k, t)], var[("unmet", j, k, t)]]
                vals = [1.0, -1.0]
                for i in range(P):
                    if (i, k) in capable:
                        cols.append(var[("ship", i, j, k, t)]); vals.append(-1.0)
                for (a, b) in pairs:
                    if b == j:
                        cols.append(var[("tr", a, b, k, t)]); vals.append(-1.0)
                    if a == j:
                        cols.append(var[("tr", a, b, k, t)]); vals.append(1.0)
                rhs = -float(dem.get((dep_id, k, t + 1), 0.0))
                if t == 0:
                    rhs += float(init_d.get((dep_id, k), 0.0))
                else:
                    cols.append(var[("invd", j, k, t - 1)]); vals.append(-1.0)
                eq_r += [r_eq] * len(cols); eq_c += cols; eq_v += vals; eq_b.append(rhs); r_eq += 1
                # Safety stock (soft): -invd - ssshort <= -SS
                ub_r += [r_ub, r_ub]; ub_c += [var[("invd", j, k, t)], var[("ssshort", j, k, t)]]
                ub_v += [-1.0, -1.0]; ub_b.append(-float(ss.get((dep_id, k), 0.0))); r_ub += 1
            # Depot storage
            cols = [var[("invd", j, k, t)] for k in K]
            ub_r += [r_ub] * len(cols); ub_c += cols; ub_v += [1.0] * len(cols)
            ub_b.append(float(depots.loc[j, "storage_t"]) * 1.5); r_ub += 1
        for i in range(P):
            ks = [k for k in K if (i, k) in capable]
            ub_r += [r_ub] * len(ks); ub_c += [var[("prod", i, k, t)] for k in ks]; ub_v += [1.0] * len(ks)
            ub_b.append(float(cap[i, t])); r_ub += 1
            ub_r += [r_ub] * len(ks); ub_c += [var[("ot", i, k, t)] for k in ks]; ub_v += [1.0] * len(ks)
            ub_b.append(float(cap[i, t] * sc.overtime_share)); r_ub += 1
            ub_r += [r_ub] * len(ks); ub_c += [var[("invp", i, k, t)] for k in ks]; ub_v += [1.0] * len(ks)
            ub_b.append(float(plants.loc[i, "weekly_capacity_t"]) * 2.0); r_ub += 1

    A_eq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(r_eq, n)).tocsr()
    A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(r_ub, n)).tocsr()
    res = linprog(np.array(costs), A_ub=A_ub, b_ub=np.array(ub_b), A_eq=A_eq, b_eq=np.array(eq_b),
                  bounds=list(zip([0.0] * n, ub)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"Planning LP did not solve: {res.message}")
    sol = {k: res.x[v] for k, v in var.items() if res.x[v] > 1e-6}
    ctx = {"plants": plants, "depots": depots, "cap": cap, "capable": capable, "ship_cost": ship_cost,
           "trans_cost": trans_cost, "hold": hold, "pcost": pcost, "dem": dem, "ss": ss, "sc": sc,
           "init_d": init_d, "init_p": init_p, "status": "Optimal (HiGHS linear programme)", "objective": res.fun,
           "cost_by_cat": cost_by_cat}
    return sol, ctx


def _summarise(sol: dict, ctx: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plants, depots, sc = ctx["plants"], ctx["depots"], ctx["sc"]
    H = sc.horizon_weeks
    prod_rows, ship_rows, tr_rows, invd_rows, unmet_rows = [], [], [], [], []
    cost = {"Production cost": 0.0, "Overtime premium": 0.0, "Primary freight and handling": 0.0,
            "Inter-depot transfer cost": 0.0, "Inventory holding cost": 0.0,
            "Unmet-demand cost (replacement sourcing + service penalty)": 0.0}
    for key, v in sol.items():
        kind = key[0]
        if kind in ("prod", "ot"):
            _, i, k, t = key
            prod_rows.append({"plant_id": plants.loc[i, "plant_id"], "category": k, "week": t + 1,
                              "regular_t" if kind == "prod" else "overtime_t": v})
            cost["Production cost"] += v * ctx["pcost"][(i, k)]
            if kind == "ot":
                cost["Overtime premium"] += v * sc.overtime_premium_per_t
        elif kind == "ship":
            _, i, j, k, t = key
            ship_rows.append({"plant_id": plants.loc[i, "plant_id"], "depot_id": depots.loc[j, "depot_id"],
                              "category": k, "week": t + 1, "qty_t": v})
            cost["Primary freight and handling"] += v * ctx["ship_cost"][i, j]
        elif kind == "tr":
            _, a, b, k, t = key
            tr_rows.append({"from_depot": depots.loc[a, "depot_id"], "to_depot": depots.loc[b, "depot_id"],
                            "category": k, "week": t + 1, "qty_t": v, "cost_inr": v * ctx["trans_cost"][a, b]})
            cost["Inter-depot transfer cost"] += v * ctx["trans_cost"][a, b]
        elif kind in ("invd", "invp"):
            cost["Inventory holding cost"] += v * ctx["hold"][key[2]]
            if kind == "invd":
                _, j, k, t = key
                invd_rows.append({"depot_id": depots.loc[j, "depot_id"], "category": k, "week": t + 1, "inventory_t": v})
        elif kind == "unmet":
            _, j, k, t = key
            unmet_rows.append({"depot_id": depots.loc[j, "depot_id"], "category": k, "week": t + 1, "unmet_t": v})
            cost["Unmet-demand cost (replacement sourcing + service penalty)"] += v * (sc.unmet_penalty_per_t + ctx["cost_by_cat"][k])
    production = pd.DataFrame(prod_rows, columns=["plant_id", "category", "week", "regular_t", "overtime_t"]).groupby(
        ["plant_id", "category", "week"], as_index=False).sum()
    shipments = pd.DataFrame(ship_rows, columns=["plant_id", "depot_id", "category", "week", "qty_t"])
    transfers = pd.DataFrame(tr_rows, columns=["from_depot", "to_depot", "category", "week", "qty_t", "cost_inr"])
    invd = pd.DataFrame(invd_rows, columns=["depot_id", "category", "week", "inventory_t"])
    unmet = pd.DataFrame(unmet_rows, columns=["depot_id", "category", "week", "unmet_t"])
    total_dem = float(ctx["dem"].sum())
    total_unmet = float(unmet["unmet_t"].sum())
    total_prod = float(production[["regular_t", "overtime_t"]].sum().sum()) if len(production) else 0.0
    kpis = {
        "Total operating cost (INR)": float(sum(cost.values())),
        **{f"{k} (INR)": v for k, v in cost.items()},
        "Service level (fill rate %)": (1 - total_unmet / total_dem) * 100 if total_dem else 100.0,
        "Stockout quantity (t)": total_unmet,
        "Average depot inventory (t)": float(invd.groupby("week")["inventory_t"].sum().reindex(range(1, H + 1), fill_value=0).mean()),
        "Plant utilisation (%)": total_prod / float(ctx["cap"].sum()) * 100 if ctx["cap"].sum() else 0.0,
        "Inter-depot transfers (t)": float(transfers["qty_t"].sum()),
        "Overtime production (t)": float(production["overtime_t"].sum()) if len(production) else 0.0,
        "Total demand (t)": total_dem,
    }
    weekly = pd.DataFrame({"week": range(1, H + 1)})
    weekly["demand_t"] = [float(ctx["dem"].xs(w, level="horizon").sum()) for w in weekly["week"]]
    weekly["production_t"] = weekly["week"].map(production.groupby("week")[["regular_t", "overtime_t"]].sum().sum(axis=1)).fillna(0)
    weekly["depot_inventory_t"] = weekly["week"].map(invd.groupby("week")["inventory_t"].sum()).fillna(0)
    weekly["unmet_t"] = weekly["week"].map(unmet.groupby("week")["unmet_t"].sum()).fillna(0)
    return kpis, production, shipments, transfers, invd, unmet, weekly


# ---------------------------------------------------------------------------
# Baseline policy simulation
# ---------------------------------------------------------------------------
def baseline_plan(env, demand: pd.DataFrame, sc: PlanningScenario) -> tuple[dict, dict]:
    """Current practice: each depot is replenished from its nearest capable plant up to demand plus
    safety stock; plants ration capacity proportionally; no overtime and no inter-depot transfers."""
    plants, depots, cap, capable, dist_dd, ship_cost, trans_cost, init_d, init_p, ss, hold, cost_by_cat = _network(env, sc)
    H, K = sc.horizon_weeks, PRODUCT_CATEGORIES
    dem = demand.set_index(["depot_id", "category", "horizon"])["demand_t"]
    pcost = _cost_rows(plants, cost_by_cat)
    dist_pd = ship_cost  # cost proxy is monotone in distance
    home = {}
    for j in depots.index:
        for k in K:
            opts = [i for i in plants.index if (i, k) in capable]
            home[(j, k)] = min(opts, key=lambda i: dist_pd[i, j]) if opts else None
    inv_d = {(j, k): float(init_d.get((depots.loc[j, "depot_id"], k), 0.0)) for j in depots.index for k in K}
    inv_p = {(i, k): float(init_p.get((plants.loc[i, "plant_id"], k), 0.0)) for (i, k) in capable}
    sol = {}
    for t in range(H):
        req = {}
        for j in depots.index:
            for k in K:
                d = float(dem.get((depots.loc[j, "depot_id"], k, t + 1), 0.0))
                s = float(ss.get((depots.loc[j, "depot_id"], k), 0.0))
                req[(j, k)] = max(0.0, d + s - inv_d[(j, k)])
        for i in plants.index:
            ks = [k for k in K if (i, k) in capable]
            need = {k: sum(v for (j, kk), v in req.items() if kk == k and home[(j, kk)] == i) for k in ks}
            to_make = {k: max(0.0, need[k] - inv_p[(i, k)]) for k in ks}
            total = sum(to_make.values())
            scale = min(1.0, cap[i, t] / total) if total > 0 else 0.0
            for k in ks:
                made = to_make[k] * scale
                if made > 0:
                    sol[("prod", i, k, t)] = made
                avail = inv_p[(i, k)] + made
                deps = [j for j in depots.index if home[(j, k)] == i and req[(j, k)] > 0]
                tot_req = sum(req[(j, k)] for j in deps)
                ratio = min(1.0, avail / tot_req) if tot_req > 0 else 0.0
                for j in deps:
                    q = req[(j, k)] * ratio
                    if q > 0:
                        sol[("ship", i, j, k, t)] = q
                        inv_d[(j, k)] += q
                        avail -= q
                inv_p[(i, k)] = avail
                if avail > 0:
                    sol[("invp", i, k, t)] = avail
        for j in depots.index:
            for k in K:
                d = float(dem.get((depots.loc[j, "depot_id"], k, t + 1), 0.0))
                served = min(d, inv_d[(j, k)])
                if d - served > 0:
                    sol[("unmet", j, k, t)] = d - served
                inv_d[(j, k)] -= served
                if inv_d[(j, k)] > 0:
                    sol[("invd", j, k, t)] = inv_d[(j, k)]
    ctx = {"plants": plants, "depots": depots, "cap": cap, "capable": capable, "ship_cost": ship_cost,
           "trans_cost": trans_cost, "hold": hold, "pcost": pcost, "dem": dem, "ss": ss, "sc": sc,
           "cost_by_cat": cost_by_cat}
    return sol, ctx


def run_planning(env, demand: pd.DataFrame, sc: PlanningScenario) -> PlanningResult:
    b_sol, b_ctx = baseline_plan(env, demand, sc)
    b_kpis, *_ , b_weekly = _summarise(b_sol, b_ctx)
    o_sol, o_ctx = optimise_plan(env, demand, sc)
    o_kpis, production, shipments, transfers, invd, unmet, weekly = _summarise(o_sol, o_ctx)
    # Depot status at the end of week 1 vs safety stock and excess thresholds
    avg_dem = demand.groupby(["depot_id", "category"])["demand_t"].mean()
    w1 = invd[invd["week"] == 1].set_index(["depot_id", "category"])["inventory_t"]
    u_all = unmet.groupby(["depot_id", "category"])["unmet_t"].sum() if len(unmet) else pd.Series(dtype=float)
    idx = avg_dem.index
    st = pd.DataFrame(index=idx)
    st["avg_weekly_demand_t"] = avg_dem
    st["week1_closing_inventory_t"] = w1.reindex(idx).fillna(0.0)
    st["safety_stock_t"] = [float(o_ctx["ss"].get(i, 0.0)) for i in idx]
    st["horizon_unmet_t"] = u_all.reindex(idx).fillna(0.0)
    st["status"] = np.select(
        [st["horizon_unmet_t"] > 0.5, st["week1_closing_inventory_t"] < st["safety_stock_t"] * 0.8,
         st["week1_closing_inventory_t"] > 2 * st["avg_weekly_demand_t"] + st["safety_stock_t"]],
        ["STOCKOUT RISK", "BELOW SAFETY STOCK", "EXCESS INVENTORY"], "WITHIN LIMIT")
    st = st.reset_index()
    st = st[(st["avg_weekly_demand_t"] > 0.05) | (st["week1_closing_inventory_t"] > 0)]
    weekly = weekly.merge(b_weekly[["week", "unmet_t", "depot_inventory_t", "production_t"]].rename(
        columns={"unmet_t": "baseline_unmet_t", "depot_inventory_t": "baseline_depot_inventory_t",
                 "production_t": "baseline_production_t"}), on="week")
    return PlanningResult(baseline=b_kpis, optimised=o_kpis, production=production, shipments=shipments,
                          transfers=transfers, depot_status=st, weekly=weekly, demand=demand,
                          solver_status=o_ctx["status"])
