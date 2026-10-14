"""Page 9 - Project Offtake Early-Warning System."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import state
from src.ui.charts import PRIMARY, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_date, fmt_inr, fmt_pct, fmt_t

page_header("Project Offtake Early-Warning System",
            "Detect slowing project offtake, estimate at-risk tonnage and working capital, and trigger follow-up and re-planning.")
k = state.key()
env = state.env()
regions = state.global_regions()
thr = st.slider("Attainment threshold for alerts (actual / planned, last 3 months)", 0.5, 0.95, 0.80, 0.05, key="p_ow_thr")
off = state.offtake_for(k, float(thr))
pj = off.projects[off.projects["region"].isin(regions)]
alerts = pj[pj["alert_level"] != "NORMAL"]
kpi_cards([
    ("Active projects monitored", str(len(pj)), f"{pj['committed_offtake_t'].sum():,.0f} t committed"),
    ("ACTION REQUIRED", str(int((pj["alert_level"] == "ACTION REQUIRED").sum())), f"WATCH: {int((pj['alert_level'] == 'WATCH').sum())}"),
    ("Estimated at-risk tonnage", fmt_t(alerts["at_risk_t"].sum()), "Next 3 months, estimate"),
    ("Inventory exposure", fmt_inr(alerts["inventory_exposure_inr"].sum()), "At production cost"),
    ("Working-capital implication", fmt_inr(alerts["working_capital_impact_inr"].sum()), "Inventory + delayed collections"),
])
note("Predicted shortfalls are ESTIMATES based on order patterns, not confirmed cancellations. Confirm with the customer before acting.")

t1, t2, t3, t4 = st.tabs(["Early-warning alerts", "Project drill-down", "Re-planning scenario", "Detection quality"])
with t1:
    lv = st.multiselect("Alert level", ["ACTION REQUIRED", "WATCH", "NORMAL"], default=["ACTION REQUIRED", "WATCH"], key="p_ow_lv")
    v = pj[pj["alert_level"].isin(lv)] if lv else pj
    cols = ["project_id", "project_name", "customer_segment", "region", "project_stage", "product_category", "recent_attainment",
            "attainment_slope", "change_point_month", "planned_next_t", "forecast_next_t", "at_risk_t", "inventory_exposure_inr",
            "alert_reasons", "recommended_action", "alert_level"]
    show_table(display_frame(v[cols], pct=["recent_attainment", "attainment_slope"], dates=["change_point_month"],
                             tonnes=["planned_next_t", "forecast_next_t", "at_risk_t"], inr=["inventory_exposure_inr"],
                             rename={"project_id": "Project", "project_name": "Name", "customer_segment": "Segment",
                                     "region": "Region", "project_stage": "Stage", "product_category": "Product",
                                     "recent_attainment": "3-month attainment", "attainment_slope": "Trend / month",
                                     "change_point_month": "Change point", "planned_next_t": "Planned next 3m",
                                     "forecast_next_t": "Forecast next 3m", "at_risk_t": "At risk (estimate)",
                                     "inventory_exposure_inr": "Inventory exposure", "alert_reasons": "Signals",
                                     "recommended_action": "Recommended follow-up", "alert_level": "Alert"}),
               status_cols=["Alert"], height=440)
    a, b = st.columns(2)
    with a:
        download_csv(v[cols], "offtake_alerts.csv", "Download alerts (CSV)", key="dl_ow")
    with b:
        approval_button("Offtake Early Warning", f"{len(alerts)} projects", "Sales follow-up and re-planning for flagged projects",
                        "Institutional Sales Head", key="ap_ow", label="Assign follow-up to institutional sales")
    agg = alerts.groupby("region")["at_risk_t"].sum().reset_index().sort_values("at_risk_t")
    if len(agg):
        fig = px.bar(agg, x="at_risk_t", y="region", orientation="h", labels={"at_risk_t": "Estimated at-risk tonnes", "region": ""})
        fig.update_traces(marker_color=SERIES[1])
        st.plotly_chart(style(fig, "Estimated at-risk tonnage by region (next 3 months)", height=280, legend=False),
                        width="stretch", config=config())

with t2:
    if pj.empty:
        note("No projects in the selected regions.")
    else:
        pid = st.selectbox("Project", pj["project_id"].tolist(), key="p_ow_proj",
                           format_func=lambda p: f"{p} - {pj.set_index('project_id').loc[p, 'project_name']} ({pj.set_index('project_id').loc[p, 'alert_level']})")
        r = pj.set_index("project_id").loc[pid]
        pm = env.project_monthly[env.project_monthly["project_id"] == pid].sort_values("month")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pm["month"], y=pm["planned_t"], name="Planned / committed", marker_color="#c3c2b7"))
        fig.add_trace(go.Scatter(x=pm["month"], y=pm["actual_t"], name="Actual orders", line=dict(color=PRIMARY), mode="lines+markers"))
        fut = pm[pm["actual_t"].isna()].head(3)
        fig.add_trace(go.Scatter(x=fut["month"], y=fut["planned_t"] * r["forecast_attainment"], name="Forecast (estimate)",
                                 line=dict(color=SERIES[1], dash="dash"), mode="lines+markers"))
        if pd.notna(r["change_point_month"]):
            fig.add_vline(x=r["change_point_month"], line_dash="dot", line_color="#d03b3b")
            fig.add_annotation(x=r["change_point_month"], y=pm["planned_t"].max(), text="CUSUM change point", showarrow=False,
                               yshift=10)
        st.plotly_chart(style(fig, f"{r['project_name']}: monthly offtake (t)", None, "Tonnes", height=360), width="stretch", config=config())
        kpi_cards([("Alert level", r["alert_level"], r["alert_reasons"][:70]),
                   ("3-month attainment", fmt_pct(r["recent_attainment"]), f"Trend {r['attainment_slope']:+.1%} per month"),
                   ("At-risk tonnage", fmt_t(r["at_risk_t"]), "Estimate, next 3 months"),
                   ("Avg payment delay", f"{r['avg_payment_delay_days']:.0f} days", "Last 3 months")])
        show_table(display_frame(pm, dates=["month"], dec1=["planned_t", "actual_t", "payment_delay_days"]), height=300)

with t3:
    section("Production and supply re-planning with and without offtake adjustments",
            "Re-runs the production and inventory optimisation (Page 5) with the same assumptions.")
    if st.button("Compare plans", key="ow_cmp"):
        sc = state.planning_scenario()
        with st.spinner("Running two planning scenarios..."):
            with_adj = state.planning(replace(sc, apply_offtake_alerts=True))
            without = state.planning(replace(sc, apply_offtake_alerts=False))
        st.session_state["ow_cmp_res"] = (with_adj, without)
    if "ow_cmp_res" in st.session_state:
        with_adj, without = st.session_state["ow_cmp_res"]
        rows = []
        for name, res in [("Plan to committed schedules", without), ("Plan with early-warning adjustments", with_adj)]:
            o = res.optimised
            rows.append({"Scenario": name, "Total demand planned (t)": o["Total demand (t)"],
                         "Average depot inventory (t)": o["Average depot inventory (t)"],
                         "Holding cost (INR)": o["Inventory holding cost (INR)"], "Total relevant cost (INR)": o["Total operating cost (INR)"],
                         "Plant utilisation (%)": o["Plant utilisation (%)"]})
        df = pd.DataFrame(rows)
        show_table(display_frame(df, tonnes=["Total demand planned (t)", "Average depot inventory (t)"],
                                 inr=["Holding cost (INR)", "Total relevant cost (INR)"], dec1=["Plant utilisation (%)"]))
        note("Planning to committed schedules builds production for offtake that the early-warning model expects not to materialise. "
             "The difference in planned tonnage approximates avoidable inventory build-up (SIMULATED RESULT).")
    else:
        note("Select 'Compare plans' to quantify the supply-planning effect of the alerts.")

with t4:
    section("Alert quality against simulated slowdown labels", "Possible only because the synthetic data records which projects slowed.")
    show_table(display_frame(off.metrics.reset_index().rename(columns={"index": "Alert set"}), pct=["Precision", "Recall"], dec1=["F1"]))
    assumptions(["Signals: 3-month attainment versus plan, linear trend of attainment over 6 months, one-sided CUSUM change point "
                 "on the attainment shortfall (k = 0.10, h = 0.45), cancellation and deferment z-scores versus all projects, and "
                 "payment delays.",
                 "ACTION REQUIRED when the combined score >= 0.9 or 3-month attainment < 60%; WATCH when score >= 0.45.",
                 "Forecast attainment uses the recent level (after a change point) or a smoothed level, adjusted by trend.",
                 "Inventory exposure = at-risk tonnage x average production cost; working capital adds delayed collection effects.",
                 "Thresholds are configurable demonstration choices, not validated policies."], "Detection method")
