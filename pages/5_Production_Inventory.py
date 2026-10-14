"""Page 5 - Demand-Driven Production and Inventory Planning."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import REGIONS
from src.optimisation.inventory_optimisation import PlanningScenario
from src.ui import state
from src.ui.charts import SCENARIO_COLORS, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_pct, fmt_t

page_header("Demand-Driven Production and Inventory Planning",
            "Multi-period optimisation of production, plant-to-depot allocation, transfers and inventory against forecast demand.")
k = state.key()
env = state.env()
cur = state.planning_scenario()
plants = env.plants["plant_id"].tolist()

with st.form("plan_form"):
    section("Scenario simulator", "Demand, capacity, cost and disruption assumptions.")
    c1, c2, c3, c4 = st.columns(4)
    horizon = c1.slider("Planning horizon (weeks)", 4, 8, int(cur.horizon_weeks), key="p_pl_h")
    dmult = c2.slider("Network demand multiplier", 0.7, 1.4, float(cur.demand_multiplier), 0.05, key="p_pl_dm")
    surge_region = c3.selectbox("Regional demand surge", ["None"] + REGIONS,
                                index=(["None"] + REGIONS).index(cur.region_surge[0][0]) if cur.region_surge else 0, key="p_pl_sr")
    surge_pct = c4.slider("Surge size", 0.0, 0.6, float(cur.region_surge[0][1] - 1) if cur.region_surge else 0.25, 0.05, key="p_pl_sp")
    c1, c2, c3, c4 = st.columns(4)
    cap = c1.slider("Plant capacity multiplier", 0.5, 1.2, float(cur.capacity_multiplier), 0.05, key="p_pl_cap")
    shut = c2.selectbox("Plant shutdown / maintenance", ["None"] + plants,
                        index=(["None"] + plants).index(cur.shutdown_plant) if cur.shutdown_plant else 0, key="p_pl_shut")
    weeks = c3.slider("Shutdown weeks", 1, 8, tuple(cur.shutdown_weeks), key="p_pl_sw")
    hold = c4.slider("Holding cost multiplier", 0.5, 3.0, float(cur.holding_cost_multiplier), 0.1, key="p_pl_hold")
    c1, c2, c3, c4 = st.columns(4)
    ot = c1.slider("Overtime capacity (share of regular)", 0.0, 0.3, float(cur.overtime_share), 0.05, key="p_pl_ot")
    pen = c2.number_input("Unmet-demand service penalty (INR / t)", 0.0, 20000.0, float(cur.unmet_penalty_per_t), 250.0, key="p_pl_pen")
    trkm = c3.slider("Max inter-depot transfer distance (km)", 0, 1500, int(cur.transfer_max_km), 50, key="p_pl_tr")
    with c4:
        inc_t = st.toggle("Include probability-weighted tender demand", cur.include_tender_demand, key="p_pl_tend")
        off = st.toggle("Apply offtake early-warning adjustments", cur.apply_offtake_alerts, key="p_pl_off")
    submitted = st.form_submit_button("Run planning optimisation", type="primary")
if submitted:
    st.session_state["planning_scenario"] = PlanningScenario(
        horizon_weeks=int(horizon), demand_multiplier=float(dmult),
        region_surge=((surge_region, 1 + float(surge_pct)),) if surge_region != "None" else (),
        capacity_multiplier=float(cap), shutdown_plant=None if shut == "None" else shut, shutdown_weeks=tuple(int(w) for w in weeks),
        holding_cost_multiplier=float(hold), overtime_share=float(ot), unmet_penalty_per_t=float(pen),
        transfer_max_km=float(trkm), include_tender_demand=bool(inc_t), apply_offtake_alerts=bool(off))
try:
    res = state.planning()
except Exception as exc:
    st.error(f"Planning optimisation failed: {exc}")
    st.stop()
if submitted:
    st.success(f"Plan recomputed: {res.solver_status}.")

b, o = res.baseline, res.optimised
saving = b["Total operating cost (INR)"] - o["Total operating cost (INR)"]
kpi_cards([
    ("Total relevant cost saving", fmt_inr(saving), f"{fmt_pct(saving / b['Total operating cost (INR)'] if b['Total operating cost (INR)'] else 0)} vs current practice"),
    ("Service level (fill rate)", f"{o['Service level (fill rate %)']:.1f}%", f"Baseline {b['Service level (fill rate %)']:.1f}%"),
    ("Stock-out quantity", fmt_t(o["Stockout quantity (t)"]), f"Baseline {fmt_t(b['Stockout quantity (t)'])}"),
    ("Average depot inventory", fmt_t(o["Average depot inventory (t)"]), f"Baseline {fmt_t(b['Average depot inventory (t)'])}"),
    ("Plant utilisation", f"{o['Plant utilisation (%)']:.1f}%", f"Baseline {b['Plant utilisation (%)']:.1f}%"),
    ("Inter-depot transfers", fmt_t(o["Inter-depot transfers (t)"]), f"Overtime {fmt_t(o['Overtime production (t)'])}"),
])
note(f"SIMULATED RESULT. {res.solver_status}. Baseline = nearest-plant replenishment to demand plus safety stock, "
     "proportional capacity rationing, no overtime or transfers. Only week 1 would be executed; the plan is re-run weekly.")

t1, t2, t3, t4, t5 = st.tabs(["Baseline versus optimised", "Production plan", "Depot inventory and transfers",
                              "Replenishment schedule", "Demand inputs"])
with t1:
    comp = pd.DataFrame({"Baseline (current practice)": b, "Optimised": o})
    comp.index.name = "KPI"
    disp = comp.reset_index()
    for col in ["Baseline (current practice)", "Optimised"]:
        disp[col] = [fmt_inr(v) if "INR" in kpi else (f"{v:,.1f}%" if "%" in kpi else fmt_t(v)) for kpi, v in zip(disp["KPI"], disp[col])]
    a, c = st.columns([2, 3])
    with a:
        show_table(disp, height=520)
    with c:
        w = res.weekly
        fig = go.Figure()
        fig.add_trace(go.Bar(x=w["week"], y=w["baseline_unmet_t"], name="Baseline (current practice)", marker_color=SCENARIO_COLORS["Baseline (current practice)"]))
        fig.add_trace(go.Bar(x=w["week"], y=w["unmet_t"], name="Optimised", marker_color=SCENARIO_COLORS["Optimised"]))
        fig.update_layout(barmode="group")
        st.plotly_chart(style(fig, "Unmet demand by week (t)", "Week", "Tonnes", height=280), width="stretch", config=config())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["week"], y=w["baseline_depot_inventory_t"], name="Baseline (current practice)",
                                 line=dict(color=SCENARIO_COLORS["Baseline (current practice)"])))
        fig.add_trace(go.Scatter(x=w["week"], y=w["depot_inventory_t"], name="Optimised", line=dict(color=SCENARIO_COLORS["Optimised"])))
        st.plotly_chart(style(fig, "Closing depot inventory by week (t)", "Week", "Tonnes", height=280), width="stretch", config=config())
    cost_rows = [kk for kk in b if kk.endswith("(INR)") and kk != "Total operating cost (INR)"]
    cm = comp.loc[cost_rows].reset_index().melt(id_vars="KPI", var_name="Plan", value_name="INR")
    cm["INR crore"] = cm["INR"] / 1e7
    fig = px.bar(cm, x="INR crore", y="KPI", color="Plan", barmode="group", orientation="h", color_discrete_map=SCENARIO_COLORS,
                 labels={"KPI": ""})
    st.plotly_chart(style(fig, "Cost components (INR crore)", height=360), width="stretch", config=config())

with t2:
    prod = res.production.copy()
    prod["total_t"] = prod["regular_t"].fillna(0) + prod["overtime_t"].fillna(0)
    util = prod.groupby(["plant_id", "week"])["total_t"].sum().reset_index()
    capw = env.plants.set_index("plant_id")["weekly_capacity_t"]
    sc = state.planning_scenario()
    util["utilisation"] = util["total_t"] / (util["plant_id"].map(capw) * sc.capacity_multiplier)
    fig = px.density_heatmap(util, x="week", y="plant_id", z="utilisation", histfunc="avg", text_auto=".0%",
                             color_continuous_scale=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"],
                             labels={"week": "Week", "plant_id": "Plant", "utilisation": "Utilisation"})
    st.plotly_chart(style(fig, "Plant utilisation by week (regular + overtime vs regular capacity)", height=360, legend=False),
                    width="stretch", config=config())
    piv = prod.pivot_table(index=["plant_id", "category"], columns="week", values="total_t", aggfunc="sum").fillna(0).round(0)
    piv.columns = [f"Week {c}" for c in piv.columns]
    show_table(piv.reset_index())
    download_csv(prod, "production_plan.csv", "Download production plan (CSV)", key="dl_prod")

with t3:
    stt = res.depot_status.merge(env.depots[["depot_id", "city", "region"]], on="depot_id")
    show_table(display_frame(stt[["depot_id", "city", "region", "category", "avg_weekly_demand_t", "safety_stock_t",
                                  "week1_closing_inventory_t", "horizon_unmet_t", "status"]],
                             dec1=["avg_weekly_demand_t", "safety_stock_t", "week1_closing_inventory_t", "horizon_unmet_t"],
                             rename={"depot_id": "Depot", "city": "City", "region": "Region", "category": "Category",
                                     "avg_weekly_demand_t": "Avg weekly demand (t)", "safety_stock_t": "Safety stock (t)",
                                     "week1_closing_inventory_t": "Week-1 closing stock (t)", "horizon_unmet_t": "Unmet over horizon (t)",
                                     "status": "Status"}), status_cols=["Status"], height=380)
    section("Inter-depot transfers evaluated by the optimiser",
            "Transfers are used only when cheaper than fresh plant shipments or needed to avoid stock-outs.")
    if res.transfers.empty:
        note("No inter-depot transfers are cost-effective in this scenario. Try a plant shutdown or a regional surge.")
    else:
        show_table(display_frame(res.transfers, dec1=["qty_t"], inr=["cost_inr"]))
    download_csv(stt, "depot_inventory_status.csv", "Download depot status (CSV)", key="dl_dep")

with t4:
    sh = res.shipments[res.shipments["qty_t"] > 0.5].copy()
    wk = st.slider("Show weeks", 1, int(sh["week"].max()) if len(sh) else 1, (1, min(2, int(sh["week"].max()) if len(sh) else 1)),
                   key="p_pl_shw")
    sh = sh[(sh["week"] >= wk[0]) & (sh["week"] <= wk[1])].sort_values(["week", "plant_id", "qty_t"], ascending=[True, True, False])
    show_table(display_frame(sh, dec1=["qty_t"], rename={"plant_id": "From plant", "depot_id": "To depot", "category": "Category",
                                                          "week": "Week", "qty_t": "Quantity (t)"}), height=420)
    a, c = st.columns(2)
    with a:
        download_csv(sh, "replenishment_schedule.csv", "Download replenishment schedule (CSV)", key="dl_sched")
    with c:
        approval_button("Production Planning", "Week-1 plan", f"Optimised plan saving {fmt_inr(saving)} over horizon",
                        "Supply Chain Planning Head", key="ap_plan", label="Submit week-1 plan for approval")

with t5:
    d = res.demand
    comp_d = d.groupby("horizon")[["trade_t", "replenishment_uplift_t", "institutional_t", "tender_t"]].sum().reset_index()
    comp_d = comp_d.melt(id_vars="horizon", var_name="Source", value_name="Tonnes")
    comp_d["Source"] = comp_d["Source"].map({"trade_t": "Dealer forecast", "replenishment_uplift_t": "Replenishment uplift (week 1)",
                                             "institutional_t": "Project schedules (offtake-adjusted)",
                                             "tender_t": "Probability-weighted tenders"})
    fig = px.bar(comp_d, x="horizon", y="Tonnes", color="Source", color_discrete_sequence=SERIES, labels={"horizon": "Week"})
    st.plotly_chart(style(fig, "Planning demand by source (before scenario multipliers)", height=360), width="stretch", config=config())
    note("Cross-module integration: dealer forecasts (Page 2), recommended replenishment (Page 2), project offtake forecasts "
         "(Page 9) and tenders recommended to bid (Page 4) all feed this plan.")
    download_csv(d, "planning_demand.csv", "Download planning demand (CSV)", key="dl_pdem")

assumptions([
    "Linear programme solved with HiGHS (SciPy) over weekly buckets for the selected horizon.",
    "Constraints: inventory conservation at plants and depots, regular and overtime capacity, plant shutdowns, storage limits "
    "(depots at 1.5x nominal storage, plants at 2 weeks of capacity), unmet demand not exceeding demand.",
    "Costs: production (category cost + plant adjustment), overtime premium, primary freight and depot handling, inter-depot "
    "transfer freight, holding cost (14% per year of unit cost x multiplier), safety-stock shortfall penalty, and unmet demand "
    "costed as replacement sourcing plus a service penalty.",
    "Simplifications: shipments arrive in the same week; unmet demand is lost, not back-ordered; product grades are aggregated "
    "into categories; cement shelf life is respected implicitly by limiting storage to a few weeks of throughput.",
    "Plant capacities are allocated to the simulated sample and do not represent actual plant capacities.",
], "Optimisation model and simplifying assumptions")
