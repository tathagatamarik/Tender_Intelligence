"""Page 1 - Executive Intelligence Dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import HOLDING_COST_PCT_ANNUAL, ANNUAL_FINANCING_RATE
from src.data.schemas import ROIInputs
from src.data.validation import validate_environment
from src.services.persistence import list_requests
from src.services.recommendation_engine import generate_recommendations
from src.services.roi_calculator import compute_roi, scenario_table
from src.ui import state
from src.ui.charts import CHANNEL_COLORS, PRIMARY, SEQUENTIAL, SERIES, config, style
from src.ui.components import (assumptions, download_csv, kpi_cards, note, page_header, recommendation_card,
                               section, show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_pct, fmt_t

page_header("Executive Intelligence Dashboard",
            "Consolidated view of sales, supply chain, profitability, working capital and AI-generated opportunities.")

k = state.key()
env = state.env()
orders = state.order_economics_for(k)
regions = state.global_regions()
d0, d1 = state.global_dates()

# ---------------------------------------------------------------- page filters
c1, c2, c3, c4 = st.columns(4)
plant_sel = c1.multiselect("Plant (production origin)", env.plants["plant_id"].tolist(), key="p_exec_plant",
                           placeholder="All plants")
cat_sel = c2.multiselect("Product category", sorted(env.products["category"].unique()), key="p_exec_cat",
                         placeholder="All categories")
ch_sel = c3.multiselect("Channel", ["Trade", "Non-Trade"], key="p_exec_channel", placeholder="Trade and non-trade")
seg_sel = c4.multiselect("Customer segment", sorted(orders["customer_segment"].unique()), key="p_exec_seg",
                         placeholder="All segments")

f = orders[(orders["order_date"] >= d0) & (orders["order_date"] <= d1) & orders["region"].isin(regions)]
if plant_sel:
    f = f[f["plant_id"].isin(plant_sel)]
if cat_sel:
    f = f[f["category"].isin(cat_sel)]
if ch_sel:
    f = f[f["channel"].isin(ch_sel)]
if seg_sel:
    f = f[f["customer_segment"].isin(seg_sel)]

dem = state.demand_for(k)
cust_region = env.customers.set_index("customer_id")["region"]
policy = state.credit_policy()
policy_f = policy[policy["customer_id"].map(cust_region).isin(regions)]
trips = env.trips[(env.trips["dispatch_date"] >= d0) & (env.trips["dispatch_date"] <= d1) & env.trips["region"].isin(regions)]
inv = env.inventory[env.inventory["region"].isin(regions)]

with st.spinner("Computing current-scenario results across all modules..."):
    results = state.all_results()
    recs = generate_recommendations(results)

tab_over, tab_ops, tab_recs, tab_roi, tab_gov = st.tabs(
    ["Overview", "Commercial and supply chain", "AI Recommended Actions", "Financial impact and ROI",
     "Data quality and approvals"])

qty = f["qty_t"].sum()
net_rev = f["net_revenue"].sum()
cm = f["contribution_margin"].sum()
high_risk = policy_f[policy_f["risk_band"].isin(["High", "Very High"])]
holding_value = inv["inventory_value_inr"].sum()
# Only cost savings and incremental contribution count as addressable value; exposures and revenue at risk do not.
addressable = float(recs.loc[recs["value_type"].isin(["Cost saving", "Contribution"]), "impact_inr"].sum()) if len(recs) else 0.0

with tab_over:
    if f.empty:
        note("No orders match the current filters. Widen the date range, region or filters.")
    kpi_cards([
        ("Sales volume", fmt_t(qty), f"{d0.strftime('%d-%b-%Y')} to {d1.strftime('%d-%b-%Y')}"),
        ("Net revenue", fmt_inr(net_rev), "After discounts, ex-GST"),
        ("Contribution margin", fmt_inr(cm), f"{fmt_pct(cm / net_rev if net_rev else 0)} of net revenue"),
        ("Average freight cost", f"{fmt_inr(f['freight_cost'].sum() / qty if qty else 0, compact=False)} / t", "Primary + secondary"),
        ("Demand forecast accuracy", fmt_pct(1 - dem.aggregated_wape("depot_id")),
         f"Depot-week, 1 - WAPE; dealer-week {fmt_pct(dem.forecast_accuracy)}"),
        ("On-time delivery", fmt_pct(trips["delivered_on_time"].mean() if len(trips) else 0), f"{len(trips):,} GPS-tracked trips"),
        ("Inventory holding value", fmt_inr(holding_value), f"Holding cost {fmt_inr(holding_value * HOLDING_COST_PCT_ANNUAL)} / year"),
        ("Overdue receivables", fmt_inr(policy_f["overdue_inr"].sum()), f"{int((policy_f['overdue_inr'] > 0).sum())} customers"),
        ("Credit-risk exposure", fmt_inr(high_risk["ead_inr"].sum()), f"High / very-high risk EAD, {len(high_risk)} customers"),
        ("Addressable AI value", fmt_inr(addressable), "Savings + contribution, annualised - SIMULATED"),
    ])
    note("All values are computed from the shared synthetic scenario. Savings are illustrative estimates, not achieved results. "
         "Monthly trend charts show complete calendar months only.")

    if not f.empty:
        # Trend charts show complete calendar months only, so partial months do not look like collapses.
        first_full = d0 if d0.day == 1 else (d0 + pd.offsets.MonthBegin(1))
        last_full_end = d1 if (d1 + pd.Timedelta(days=1)).day == 1 else (d1 - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1))
        ft = f[(f["order_date"] >= first_full) & (f["order_date"] <= last_full_end + pd.Timedelta(hours=23))]
        ft = ft if len(ft) else f
        monthly = ft.groupby([pd.Grouper(key="order_date", freq="MS"), "channel"]).agg(
            qty_t=("qty_t", "sum"), net_revenue=("net_revenue", "sum")).reset_index()
        a, b = st.columns(2)
        fig = px.bar(monthly, x="order_date", y="qty_t", color="channel", color_discrete_map=CHANNEL_COLORS,
                     labels={"order_date": "Month", "qty_t": "Tonnes", "channel": "Channel"})
        a.plotly_chart(style(fig, "Sales volume by month (t)", None, "Tonnes"), width="stretch", config=config())
        rev = monthly.groupby("order_date")["net_revenue"].sum().reset_index()
        fig = px.line(rev, x="order_date", y=rev["net_revenue"] / 1e7, markers=True,
                      labels={"order_date": "Month", "y": "INR crore"})
        fig.update_traces(line_color=PRIMARY, marker=dict(size=8))
        b.plotly_chart(style(fig, "Net revenue by month (INR crore)", None, "INR crore", legend=False),
                       width="stretch", config=config())

        a, b = st.columns([1, 2])
        mix = f.groupby("channel")["qty_t"].sum().reset_index()
        mix["share"] = mix["qty_t"] / mix["qty_t"].sum()
        fig = px.bar(mix, x="channel", y="share", color="channel", color_discrete_map=CHANNEL_COLORS,
                     text=mix["share"].map(fmt_pct), labels={"share": "Share of volume", "channel": ""})
        fig.update_yaxes(tickformat=".0%")
        a.plotly_chart(style(fig, "Trade versus non-trade mix", legend=False), width="stretch", config=config())
        heat = ft.assign(month=ft["order_date"].dt.to_period("M").dt.to_timestamp()).pivot_table(
            index="region", columns="month", values="qty_t", aggfunc="sum").fillna(0)
        fig = go.Figure(go.Heatmap(z=heat.to_numpy(), x=[m.strftime("%b-%y") for m in heat.columns], y=heat.index,
                                   colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                                   colorbar=dict(title="t"), xgap=2, ygap=2,
                                   hovertemplate="%{y} %{x}: %{z:,.0f} t<extra></extra>"))
        b.plotly_chart(style(fig, "Regional demand heatmap (tonnes per month)", legend=False), width="stretch",
                       config=config())

with tab_ops:
    if f.empty:
        note("No orders match the current filters.")
    else:
        a, b = st.columns(2)
        src = f.groupby("source_id").agg(qty_t=("qty_t", "sum"), cm=("contribution_margin", "sum"),
                                         freight=("freight_cost", "sum")).reset_index()
        src["cm_per_t"] = src["cm"] / src["qty_t"]
        src["type"] = src["source_id"].str[:2].map({"PL": "Plant", "DP": "Depot"})
        src = src.sort_values("qty_t", ascending=False)
        fig = px.bar(src, x="source_id", y="qty_t", color="type", color_discrete_map={"Plant": SERIES[0], "Depot": SERIES[2]},
                     hover_data={"cm_per_t": ":,.0f"}, labels={"source_id": "Dispatch source", "qty_t": "Tonnes", "type": ""})
        a.plotly_chart(style(fig, "Plant and depot dispatch volume"), width="stretch", config=config())
        fig = px.bar(src.sort_values("cm_per_t"), x="cm_per_t", y="source_id", orientation="h",
                     labels={"cm_per_t": "Contribution per tonne (INR)", "source_id": ""})
        fig.update_traces(marker_color=PRIMARY)
        b.plotly_chart(style(fig, "Contribution per tonne by dispatch source", height=420, legend=False),
                       width="stretch", config=config())

        a, b = st.columns(2)
        f2 = f.assign(route=f["source_id"] + " to " + f["city"])
        routes = f2.groupby("route").agg(qty_t=("qty_t", "sum"), freight=("freight_cost", "sum")).reset_index()
        routes = routes[routes["qty_t"] >= routes["qty_t"].quantile(0.5)]
        routes["freight_per_t"] = routes["freight"] / routes["qty_t"]
        top = routes.sort_values("freight_per_t", ascending=False).head(15).sort_values("freight_per_t")
        fig = px.bar(top, x="freight_per_t", y="route", orientation="h", hover_data={"qty_t": ":,.0f"},
                     labels={"freight_per_t": "Freight INR per tonne", "route": ""})
        fig.update_traces(marker_color=SERIES[1])
        a.plotly_chart(style(fig, "Highest freight cost routes (higher-volume routes)", height=440, legend=False),
                       width="stretch", config=config())
        band_order = ["Low", "Medium", "High", "Very High", "No recent history"]
        rb = policy_f.groupby("risk_band").agg(customers=("customer_id", "count"), ead=("ead_inr", "sum")).reindex(band_order).fillna(0).reset_index()
        fig = px.bar(rb, x="risk_band", y="customers", text="customers", hover_data={"ead": ":,.0f"},
                     labels={"risk_band": "Predicted late-payment risk band", "customers": "Customers"})
        fig.update_traces(marker_color=PRIMARY)
        b.plotly_chart(style(fig, "Customer credit-risk distribution", height=440, legend=False),
                       width="stretch", config=config())

    plan = results["planning"]
    dmd = plan.demand[plan.demand["horizon"] <= 4].merge(env.depots[["depot_id", "region"]], on="depot_id")
    dmd = dmd[dmd["region"].isin(regions)].groupby("region")["demand_t"].sum()
    stock = inv.groupby("region")["available_t"].sum()
    comp = pd.DataFrame({"Available stock (plants and depots)": stock, "Forecast demand, next 4 weeks": dmd}).fillna(0)
    comp.index.name = "region"
    comp = comp.reset_index().melt(id_vars="region", var_name="measure", value_name="tonnes")
    a, b = st.columns(2)
    fig = px.bar(comp, x="region", y="tonnes", color="measure", barmode="group",
                 color_discrete_sequence=[SERIES[0], SERIES[1]], labels={"region": "", "tonnes": "Tonnes"})
    a.plotly_chart(style(fig, "Inventory availability versus forecast demand"), width="stretch", config=config())
    if len(recs):
        port = recs.groupby(["source_module", "value_type"])["impact_inr"].sum().reset_index()
        port["INR crore"] = port["impact_inr"] / 1e7
        order = port.groupby("source_module")["impact_inr"].sum().sort_values().index.tolist()
        fig = px.bar(port, x="INR crore", y="source_module", color="value_type", orientation="h",
                     category_orders={"source_module": order[::-1],
                                      "value_type": ["Cost saving", "Contribution", "Exposure (not profit)", "Revenue at risk"]},
                     color_discrete_sequence=SERIES, labels={"INR crore": "Estimated value (INR crore)", "source_module": "",
                                                             "value_type": ""})
        b.plotly_chart(style(fig, "AI opportunity portfolio by use case (illustrative)", legend=False),
                       width="stretch", config=config())
    note("Forecast demand combines dealer forecasts, recommended replenishment, project schedules (adjusted for "
         "early-warning signals) and probability-weighted tender demand, from the production planning module.")

with tab_recs:
    section("AI Recommended Actions",
            "Generated from the current scenario outputs of each module, ranked by priority, estimated impact and confidence.")
    if recs.empty:
        note("No recommendations are generated under the current assumptions.")
    else:
        bu = st.multiselect("Business unit", sorted(recs["business_unit"].unique()), key="p_exec_bu", placeholder="All units")
        view = recs[recs["business_unit"].isin(bu)] if bu else recs
        for r in view.to_dict("records"):
            recommendation_card(r, fmt_inr)
        download_csv(recs, "ai_recommended_actions.csv", "Download recommendations (CSV)", key="dl_recs")
        assumptions(["Impacts are annualised where stated, using explicit factors shown with each item.",
                     "Confidence reflects model evaluation strength and assumption sensitivity (HIGH / MEDIUM / LOW).",
                     "Each item carries a value type: cost saving, contribution, exposure (not profit) or revenue at risk. Only cost savings and contribution are summed into addressable value.",
                     "Every recommendation requires review and approval by the accountable function."])

with tab_roi:
    section("Financial impact and ROI simulator", "Illustrative estimates on editable assumptions; not verified business outcomes.")
    base = state.commercial_baseline(k)
    fr = results["freight"]
    both = fr.plan[fr.plan["base_source_id"].notna() & fr.plan["opt_source_id"].notna()]
    sim_saving = 1 - both["opt_freight_inr"].sum() / both["base_freight_inr"].sum() if len(both) else 0.0
    sim_ecl = float(policy["ecl_inr"].sum())
    st.caption(f"Reference values from the simulation: freight saving on the current order book {fmt_pct(sim_saving)} "
               f"(against a simple rule-based baseline); portfolio expected credit loss {fmt_inr(sim_ecl)}; "
               f"sample contribution {fmt_inr(base['contribution_per_t'], compact=False)} per tonne.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Volume and freight**")
        vol = st.number_input("Monthly cement sales volume (lakh t, network)", 0.0, 1000.0, 25.0, 0.5, key="p_roi_vol") * 1e5
        fpt = st.number_input("Average freight cost per tonne (INR)", 0.0, 5000.0, float(round(base["freight_per_t"], -1)), 10.0, key="p_roi_fpt")
        share = st.slider("Addressable order volume (%)", 0, 100, 60, key="p_roi_share") / 100
        fsave = st.slider("Achievable freight saving (%)", 0.0, 15.0, 4.0, 0.5, key="p_roi_fsave") / 100
    with c2:
        st.markdown("**Inventory, receivables, credit**")
        invv = st.number_input("Inventory value (INR crore)", 0.0, 1e5, 900.0, 10.0, key="p_roi_inv") * 1e7
        hold = st.slider("Inventory holding cost (% per year)", 0.0, 30.0, HOLDING_COST_PCT_ANNUAL * 100, 0.5, key="p_roi_hold") / 100
        invred = st.slider("Inventory reduction (%)", 0.0, 30.0, 8.0, 0.5, key="p_roi_invred") / 100
        overdue = st.number_input("Overdue receivables (INR crore)", 0.0, 1e5, 600.0, 10.0, key="p_roi_overdue") * 1e7
        rel = st.slider("Receivables released (%)", 0.0, 50.0, 10.0, 1.0, key="p_roi_rel") / 100
        ecl = st.number_input("Annual expected credit loss (INR crore)", 0.0, 1e4, 90.0, 1.0, key="p_roi_ecl") * 1e7
        eclred = st.slider("Credit-loss reduction (%)", 0.0, 50.0, 15.0, 1.0, key="p_roi_eclred") / 100
    with c3:
        st.markdown("**Growth and costs**")
        incv = st.slider("Incremental sales volume (%)", 0.0, 5.0, 0.8, 0.1, key="p_roi_incv") / 100
        cpt = st.number_input("Contribution per incremental tonne (INR)", 0.0, 5000.0, float(round(base["contribution_per_t"], -1)), 10.0, key="p_roi_cpt")
        overlap = st.slider("Overlap adjustment between benefits (%)", 0.0, 50.0, 15.0, 1.0, key="p_roi_overlap") / 100
        impl = st.number_input("Implementation cost (INR crore)", 0.01, 1e4, 25.0, 1.0, key="p_roi_impl") * 1e7
        sw = st.number_input("Annual software and infrastructure cost (INR crore)", 0.0, 1e4, 6.0, 0.5, key="p_roi_sw") * 1e7
        opx = st.number_input("Annual operating cost (INR crore)", 0.0, 1e4, 4.0, 0.5, key="p_roi_opx") * 1e7
    try:
        inp = ROIInputs(monthly_volume_t=vol, freight_cost_per_t=fpt, addressable_share=share, freight_saving_pct=fsave,
                        inventory_value_inr=invv, holding_cost_pct=hold, inventory_reduction_pct=invred,
                        overdue_receivables_inr=overdue, receivable_release_pct=rel, expected_credit_loss_inr=ecl,
                        credit_loss_reduction_pct=eclred, incremental_volume_pct=incv, contribution_per_t=cpt,
                        overlap_adjustment_pct=overlap, financing_rate=ANNUAL_FINANCING_RATE,
                        implementation_cost_inr=impl, annual_software_cost_inr=sw, annual_operating_cost_inr=opx)
        res = compute_roi(inp)
        kpi_cards([
            ("Gross annual benefit", fmt_inr(res["Gross annual benefit"]), "After overlap adjustment"),
            ("Annual net benefit", fmt_inr(res["Annual net benefit"]), "Gross benefit - recurring cost"),
            ("ROI (year one)", f"{res['ROI (%)']:.0f}%", "Net benefit / implementation cost"),
            ("Payback period", f"{res['Payback (months)']:.1f} months" if res["Payback (months)"] else "Not reached",
             "Implementation cost / monthly net benefit"),
            ("Working-capital release", fmt_inr(res["Working-capital release (one-off, not profit)"]), "One-off cash, not profit"),
        ])
        scen = scenario_table(inp)
        tbl = pd.DataFrame(scen).reset_index().rename(columns={"index": "Line item"})
        disp = tbl.copy()
        for col in ["Conservative", "Base", "Upside"]:
            disp[col] = [f"{v:.0f}%" if li == "ROI (%)" else (f"{v:.1f}" if li == "Payback (months)" and v else
                         ("Not reached" if li == "Payback (months)" else fmt_inr(v)))
                         for li, v in zip(tbl["Line item"], tbl[col])]
        a, b = st.columns([3, 2])
        with a:
            show_table(disp)
            download_csv(tbl, "roi_scenarios.csv", "Download ROI scenarios (CSV)", key="dl_roi")
        with b:
            fig = px.bar(x=list(scen.keys()), y=[scen[s]["Annual net benefit"] / 1e7 for s in scen],
                         labels={"x": "Scenario", "y": "Annual net benefit (INR crore)"})
            fig.update_traces(marker_color=PRIMARY)
            st.plotly_chart(style(fig, "Annual net benefit by scenario", legend=False), width="stretch", config=config())
        assumptions(["Conservative and upside scenarios scale every improvement percentage by 0.5x and 1.5x.",
                     "ROI convention: simple year-one ROI = annual net benefit / implementation cost x 100 (undiscounted).",
                     "Payback = implementation cost / average monthly net benefit; shown only when net benefit is positive.",
                     "Working-capital release is one-off cash and is not added to annual benefits.",
                     "Overlap adjustment avoids double counting where freight, inventory and growth benefits interact.",
                     "Default network-scale inputs are editable demonstration assumptions, not company disclosures."],
                    "ROI formulas and conventions")
    except Exception as exc:
        st.error(f"Invalid ROI inputs: {exc}")

with tab_gov:
    section("Data validation", "Schema, type, missing-value and range checks on the synthetic tables.")
    show_table(validate_environment(env), status_cols=["Status"])
    section("Approval queue", "Recommendations submitted for human approval from any module (stored in local SQLite).")
    show_table(list_requests(), status_cols=["status"])
    summary = "\n".join([
        "DALMIA AI INTELLIGENCE HUB - EXECUTIVE SUMMARY (SIMULATED, SYNTHETIC DATA)",
        f"Period: {d0.strftime('%d-%b-%Y')} to {d1.strftime('%d-%b-%Y')} | Regions: {', '.join(regions)}",
        f"Sales volume: {fmt_t(qty)} | Net revenue: {fmt_inr(net_rev)} | Contribution: {fmt_inr(cm)} ({fmt_pct(cm / net_rev if net_rev else 0)})",
        f"Demand forecast accuracy (1 - WAPE): depot-week {fmt_pct(1 - dem.aggregated_wape('depot_id'))}, "
        f"dealer-week {fmt_pct(dem.forecast_accuracy)}",
        f"Overdue receivables: {fmt_inr(policy_f['overdue_inr'].sum())} | High-risk exposure: {fmt_inr(high_risk['ead_inr'].sum())}",
        f"Addressable AI value - savings and contribution (illustrative, annualised): {fmt_inr(addressable)}",
        "", "Top recommended actions:"] + [f"{r['rank']}. [{r['priority']}] {r['title']} - {fmt_inr(r['impact_inr'])}"
                                         for r in recs.head(8).to_dict("records")] +
        ["", "All figures are simulated and illustrative. Not connected to live company systems."])
    st.download_button("Download executive summary (TXT)", summary.encode("utf-8"), "executive_summary.txt",
                       "text/plain", key="dl_exec_summary")
