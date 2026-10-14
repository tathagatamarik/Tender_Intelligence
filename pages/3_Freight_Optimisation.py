"""Page 3 - AI Freight and Dispatch Optimisation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import DEFAULT_BACKHAUL_SHARE, DEFAULT_DIESEL_PRICE
from src.optimisation.freight_optimisation import SCENARIOS, FreightScenario, compare_scenarios, savings_breakdown
from src.services.freight_tariff import TariffParams
from src.ui import state
from src.ui.charts import SCENARIO_COLORS, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_num, fmt_pct

page_header("AI Freight and Dispatch Optimisation",
            "Allocate open orders to the lowest-cost feasible plant or depot and vehicle, within stock, fleet and deadline constraints.")
k = state.key()
env = state.env()
regions = state.global_regions()

cur = state.freight_scenario()
with st.form("freight_form"):
    section("Scenario assumptions", "Change assumptions and recompute; only the dispatch optimisation re-runs.")
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.selectbox("Dispatch mode", SCENARIOS, index=SCENARIOS.index(cur.mode), key="p_fr_mode")
    diesel = c2.number_input("Diesel price (INR / litre)", 60.0, 150.0, float(cur.tariff.diesel_price), 1.0, key="p_fr_diesel")
    rate = c3.slider("Freight rate multiplier", 0.7, 1.5, float(cur.tariff.rate_multiplier), 0.05, key="p_fr_rate")
    backhaul = c4.slider("Backhaul share of return km", 0.0, 0.8, float(cur.tariff.backhaul_share), 0.05, key="p_fr_backhaul")
    c1, c2, c3, c4 = st.columns(4)
    veh = c1.slider("Contracted fleet availability", 0.2, 1.0, float(cur.vehicle_availability_pct), 0.05, key="p_fr_veh")
    capp = c2.slider("Source loading capacity", 0.3, 1.5, float(cur.source_capacity_pct), 0.05, key="p_fr_cap")
    disabled = c3.multiselect("Plants or depots unavailable", env.plants["plant_id"].tolist() + env.depots["depot_id"].tolist(),
                              default=list(cur.disabled_sources), key="p_fr_disabled")
    market = c4.toggle("Allow market-hire vehicles", cur.allow_market_hire, key="p_fr_market")
    c1, c2, c3, c4 = st.columns(4)
    pu = c1.number_input("Urgent: service penalty (INR / t unserved)", 0.0, 20000.0, float(cur.urgent_penalty_per_t), 250.0, key="p_fr_pu")
    ps = c2.number_input("Standard: service penalty (INR / t)", 0.0, 20000.0, float(cur.standard_penalty_per_t), 250.0, key="p_fr_ps")
    pf = c3.number_input("Flexible: service penalty (INR / t)", 0.0, 20000.0, float(cur.flexible_penalty_per_t), 250.0, key="p_fr_pf")
    sow = c4.number_input("Stock-on-Wheels trucks", 0, 30, int(cur.sow_trucks), 1, key="p_fr_sow")
    submitted = st.form_submit_button("Run dispatch optimisation", type="primary")
sc = FreightScenario(mode=mode, tariff=TariffParams(diesel_price=float(diesel), rate_multiplier=float(rate),
                                                    backhaul_share=float(backhaul)),
                     vehicle_availability_pct=float(veh), source_capacity_pct=float(capp), disabled_sources=tuple(disabled),
                     urgent_penalty_per_t=float(pu), standard_penalty_per_t=float(ps), flexible_penalty_per_t=float(pf),
                     allow_market_hire=bool(market), sow_trucks=int(sow))
if submitted:
    st.session_state["freight_scenario"] = sc
sc = state.freight_scenario()
try:
    res = state.freight(sc)
except Exception as exc:
    st.error(f"Optimisation failed: {exc}")
    st.stop()
if submitted:
    st.success(f"Dispatch plan recomputed: {res.method}, status {res.solver_status}, {res.n_candidates:,} feasible options evaluated.")

plan = res.plan
plan_r = plan[plan["region"].isin(regions)]
both = plan_r[plan_r["base_source_id"].notna() & plan_r["opt_source_id"].notna()]
saving = (both["base_cost_inr"] - both["opt_cost_inr"]).sum()
fr_saving = both["base_freight_inr"].sum() - both["opt_freight_inr"].sum()
b, o = res.baseline_kpis, res.optimised_kpis
kpi_cards([
    ("Fulfilment cost saving", fmt_inr(saving), f"{fmt_pct(saving / both['base_cost_inr'].sum() if len(both) else 0)} on {len(both)} common orders"),
    ("Secondary freight saving", fmt_inr(fr_saving), f"{fmt_pct(fr_saving / both['base_freight_inr'].sum() if len(both) else 0)} of baseline freight"),
    ("Freight per tonne", f"{fmt_inr(o['Freight cost per tonne (INR)'], False)}", f"Baseline {fmt_inr(b['Freight cost per tonne (INR)'], False)}"),
    ("Vehicle utilisation", f"{o['Vehicle utilisation (%)']:.1f}%", f"Baseline {b['Vehicle utilisation (%)']:.1f}%"),
    ("On-time delivery (predicted)", f"{o['On-time delivery (%)']:.1f}%", f"Baseline {b['On-time delivery (%)']:.1f}%"),
    ("Manual intervention", fmt_num(o["Orders requiring manual intervention"]), "Orders with no feasible option"),
])
note(f"SIMULATED RESULT. Method: {res.method} ({res.solver_status}). Baseline = current practice (designated source, "
     "standard 25 t trucks, no deadline check). Savings depend on this order book and are not achieved results.")

t1, t2, t3, t4, t5, t6 = st.tabs(["Plan comparison", "Shipment table", "Savings breakdown", "Green Channel and Stock-on-Wheels",
                                  "ETA and route anomalies", "Model and constraints"])
with t1:
    comp = pd.DataFrame({"Baseline (current practice)": b, "Optimised": o})
    comp.index.name = "KPI"
    disp = comp.reset_index()
    for col in ["Baseline (current practice)", "Optimised"]:
        disp[col] = [fmt_inr(v) if "INR" in kpi and "per" not in kpi else
                     (fmt_num(v) if float(v).is_integer() else f"{v:,.1f}") for kpi, v in zip(disp["KPI"], disp[col])]
    a, c = st.columns([2, 3])
    with a:
        show_table(disp)
    with c:
        sel = ["Freight cost per tonne (INR)", "Logistics cost per tonne (INR)", "Cost per loaded km (INR)"]
        m = comp.loc[sel].reset_index().melt(id_vars="KPI", var_name="Plan", value_name="INR")
        fig = px.bar(m, x="KPI", y="INR", color="Plan", barmode="group", color_discrete_map=SCENARIO_COLORS, labels={"KPI": ""})
        st.plotly_chart(style(fig, "Unit cost comparison (INR)"), width="stretch", config=config())
        m = comp.loc[["Loaded km", "Empty return km (estimate)"]].reset_index().melt(id_vars="KPI", var_name="Plan", value_name="km")
        fig = px.bar(m, x="KPI", y="km", color="Plan", barmode="group", color_discrete_map=SCENARIO_COLORS, labels={"KPI": ""})
        st.plotly_chart(style(fig, "Loaded and empty kilometres", height=300), width="stretch", config=config())

with t2:
    c1, c2 = st.columns(2)
    stat = c1.multiselect("Plan status", sorted(plan_r["status"].unique()), key="p_fr_status", placeholder="All statuses")
    urg = c2.multiselect("Urgency", ["Urgent", "Standard", "Flexible"], key="p_fr_urg", placeholder="All urgencies")
    v = plan_r
    if stat:
        v = v[v["status"].isin(stat)]
    if urg:
        v = v[v["urgency"].isin(urg)]
    cols = ["open_order_id", "customer_name", "city", "region", "product_id", "qty_t", "urgency", "deadline_hours",
            "base_source_id", "base_vehicle_type", "base_trucks", "base_delivery_hours", "base_cost_inr",
            "opt_source_id", "opt_vehicle_type", "opt_trucks", "opt_delivery_hours", "opt_cost_inr", "saving_inr", "status",
            "approval_status"]
    show_table(display_frame(v[cols], dec1=["qty_t", "base_delivery_hours", "opt_delivery_hours"],
                             inr=["base_cost_inr", "opt_cost_inr", "saving_inr"],
                             rename={"open_order_id": "Order", "customer_name": "Customer", "city": "Destination",
                                     "region": "Region", "product_id": "Product", "qty_t": "Qty (t)", "urgency": "Urgency",
                                     "deadline_hours": "Deadline (h)", "base_source_id": "Baseline source",
                                     "base_vehicle_type": "Baseline vehicle", "base_trucks": "Baseline trucks",
                                     "base_delivery_hours": "Baseline ETA (h)", "base_cost_inr": "Baseline cost",
                                     "opt_source_id": "Recommended source", "opt_vehicle_type": "Recommended vehicle",
                                     "opt_trucks": "Trucks", "opt_delivery_hours": "Predicted ETA (h)",
                                     "opt_cost_inr": "Recommended cost", "saving_inr": "Saving", "status": "Plan status",
                                     "approval_status": "Approval"}),
               status_cols=["Plan status", "Approval"], height=460)
    a, c = st.columns(2)
    with a:
        download_csv(v[cols], "dispatch_plan.csv", "Download dispatch plan (CSV)", key="dl_disp")
    with c:
        approval_button("Freight Optimisation", f"{len(plan)} orders", f"Optimised dispatch plan saving {fmt_inr(saving)}",
                        "Logistics Planner", key="ap_disp", label="Submit plan for dispatch approval")

with t3:
    dim = st.radio("Savings by", ["region", "source", "route", "customer_name"], horizontal=True, key="p_fr_dim",
                   format_func=lambda x: {"region": "Region", "source": "Recommended plant / depot", "route": "Route",
                                          "customer_name": "Customer"}[x])
    sb = savings_breakdown(plan_r, dim)
    keycol = {"source": "opt_source_id"}.get(dim, dim)
    top = sb.head(15).sort_values("saving_inr")
    fig = px.bar(top, x="saving_inr", y=keycol, orientation="h", labels={"saving_inr": "Fulfilment cost saving (INR)", keycol: ""})
    fig.update_traces(marker_color=SERIES[0])
    st.plotly_chart(style(fig, "Largest savings (orders served in both plans)", height=420, legend=False), width="stretch", config=config())
    show_table(display_frame(sb, inr=["baseline_cost_inr", "optimised_cost_inr", "baseline_freight_inr", "optimised_freight_inr", "saving_inr"],
                             pct=["saving_pct"], dec1=["tonnes"]))
    download_csv(sb, f"savings_by_{dim}.csv", "Download savings breakdown (CSV)", key="dl_sav")

with t4:
    section("Normal dispatch versus Green Channel and Stock-on-Wheels (simulated)",
            "Each scenario re-runs the optimisation with the current assumptions.")
    if st.button("Run scenario comparison", key="fr_cmp"):
        with st.spinner("Optimising three dispatch scenarios..."):
            st.session_state["freight_cmp"] = compare_scenarios(env, state.eta_for(k).model, sc)
    cmp_df = st.session_state.get("freight_cmp")
    if cmp_df is not None:
        disp = cmp_df.copy()
        for ccol in ["Total fulfilment cost (INR)", "Scenario fixed cost (INR)"]:
            disp[ccol] = disp[ccol].map(fmt_inr)
        for ccol in ["Freight cost per tonne (INR)", "On-time delivery (%)", "Average delivery time (hours)", "Vehicle utilisation (%)"]:
            disp[ccol] = disp[ccol].map(lambda x: f"{x:,.1f}")
        show_table(disp)
        a, c = st.columns(2)
        fig = px.bar(cmp_df, x="Scenario", y="Average delivery time (hours)", text_auto=".1f")
        fig.update_traces(marker_color=SERIES[0])
        a.plotly_chart(style(fig, "Average delivery time (hours)", legend=False), width="stretch", config=config())
        fig = px.bar(cmp_df, x="Scenario", y="Freight cost per tonne (INR)", text_auto=",.0f")
        fig.update_traces(marker_color=SERIES[1])
        c.plotly_chart(style(fig, "Freight cost per tonne (INR)", legend=False), width="stretch", config=config())
        download_csv(cmp_df, "dispatch_scenarios.csv", "Download scenario comparison (CSV)", key="dl_scn")
    else:
        note("Select 'Run scenario comparison' to evaluate all three dispatch modes.")
    assumptions(["Green Channel (simulated): plant loading 1.5 h instead of 4 h, depot 1 h instead of 2 h, handling INR 20/t lower.",
                 "Stock-on-Wheels (simulated): pre-loaded 25 t trucks positioned at up to four depots with the most urgent demand "
                 "nearby, carrying OPC 53 and PPC, loading 0.5 h, holding cost INR 2,500 per truck-day.",
                 "Scenario costs include the SoW holding cost as a fixed cost."])

with t5:
    eta = state.eta_for(k)
    section("Delivery ETA model", f"Chronological holdout of {eta.test_size:,} trips (trained on {eta.train_size:,}).")
    mt = eta.metrics.reset_index().rename(columns={"index": "Method"})
    show_table(display_frame(mt, dec1=[c for c in mt.columns if c != "Method"]))
    a, c = st.columns(2)
    fig = px.histogram(eta.holdout, x="error_hours", nbins=60, labels={"error_hours": "Predicted - actual transit (hours)"})
    fig.update_traces(marker_color=SERIES[0])
    a.plotly_chart(style(fig, "ETA error distribution", y_title="Trips", legend=False), width="stretch", config=config())
    an = state.trip_anomalies_for(k)
    with c:
        section("Route anomaly detection", "Precision and recall against simulated anomaly labels.")
        show_table(display_frame(an.metrics.reset_index().rename(columns={"index": "Method"}), pct=["Precision", "Recall"],
                                 dec1=["F1"]))
    trips = an.trips[(an.trips["rule_flag"] | an.trips["iforest_flag"]) & an.trips["region"].isin(regions)]
    trips = trips.sort_values("iforest_score", ascending=False)
    cols = ["trip_id", "dispatch_date", "source_id", "destination_city", "vehicle_type", "planned_km", "gps_km", "detour_ratio",
            "eta_hours", "actual_transit_hours", "excess_stop_hours", "delay_hours", "empty_return_km", "anomaly_reason"]
    show_table(display_frame(trips[cols].head(300), dates=["dispatch_date"],
                             dec1=["planned_km", "gps_km", "eta_hours", "actual_transit_hours", "excess_stop_hours", "delay_hours",
                                   "empty_return_km"],
                             rename={"anomaly_reason": "Alert"}), height=360)
    download_csv(trips[cols], "route_anomalies.csv", "Download anomaly list (CSV)", key="dl_anom")
    emp = an.trips.groupby("source_id").agg(trips=("trip_id", "count"), empty_km=("empty_return_km", "sum"),
                                            loaded_km=("planned_km", "sum")).reset_index()
    emp["empty_share"] = emp["empty_km"] / (emp["empty_km"] + emp["loaded_km"])
    fig = px.bar(emp.sort_values("empty_km", ascending=False), x="source_id", y="empty_km", hover_data={"empty_share": ":.0%"},
                 labels={"source_id": "Source", "empty_km": "Empty return km (historical trips)"})
    fig.update_traces(marker_color=SERIES[1])
    st.plotly_chart(style(fig, "Empty kilometres by dispatch source (historical GPS trips)", legend=False), width="stretch", config=config())

with t6:
    assumptions([
        "Decision: each open order is assigned to exactly one (source, vehicle type, fleet mode) option or flagged for manual intervention.",
        "Objective: minimise sourcing cost + handling + secondary freight (incl. empty return) + service penalties for unserved orders.",
        "Constraints: available-to-promise stock per source and product; source loading capacity over the 48-hour window; contracted "
        "fleet availability per source and vehicle type (vehicles may complete several trips if cycle time allows); market-hire "
        "vehicles capped per source and type at an 18% premium; delivery deadline using ETA-model transit times; product compatibility.",
        "Depot stock is valued at replacement landed cost (mother-plant cost + primary freight) so plant-direct and depot options compare fairly.",
        "Scope limitation: assignment and allocation model, not a full vehicle-routing problem; multi-drop routes, driver hours and "
        "real-time re-dispatch are out of scope for the demo.",
        "Freight tariff is synthetic: (fixed rate per km x multiplier + litres per km x diesel price) per loaded km, plus 70% of that on "
        f"empty return km. Defaults: diesel INR {DEFAULT_DIESEL_PRICE}/litre, backhaul share {DEFAULT_BACKHAUL_SHARE:.0%}.",
        "All dispatch plans require planner approval; nothing is executed automatically.",
    ], "Optimisation model, constraints and scope")
    section("Dispatch sources in this scenario")
    show_table(display_frame(res.sources[["source_id", "source_name", "source_type", "region", "loading_hours", "handling_per_t",
                                          "loading_capacity_t"]], dec1=["loading_hours", "loading_capacity_t"], inr_full=["handling_per_t"]))
