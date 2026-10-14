"""Page 2 - AI Dealer Demand and Replenishment."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.dealer_segmentation import SEGMENTS, decline_detection_quality, segment_population
from src.models.demand_forecasting import series_horizon_forecast
from src.optimisation.replenishment import ReplenishmentSettings
from src.ui import state
from src.ui.charts import PRIMARY, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_num, fmt_pct, fmt_t

page_header("AI Dealer Demand and Replenishment",
            "Forecast dealer demand, detect decline, recommend replenishment and prioritise sales visits.")
k = state.key()
env = state.env()
dem = state.demand_for(k)
regions = state.global_regions()

cur = state.replenishment_settings()
with st.expander("Forecast and replenishment settings", expanded=True):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    horizon = c1.radio("Forecast horizon", [7, 14, 30], index=[7, 14, 30].index(st.session_state.get("dl_horizon", 30)),
                       format_func=lambda d: f"{d} days", key="p_dl_h", horizontal=False)
    st.session_state["dl_horizon"] = horizon
    sl = c2.slider("Service level", 0.80, 0.99, float(cur.service_level), 0.01, key="p_dl_sl")
    lt = c3.number_input("Lead time (days)", 1, 21, int(cur.lead_time_days), key="p_dl_lt")
    rv = c4.number_input("Review period (days)", 1, 28, int(cur.review_days), key="p_dl_rv")
    lot = c5.number_input("Lot size (t)", 0.5, 25.0, float(cur.lot_size_t), 0.5, key="p_dl_lot")
    moq = c6.number_input("Minimum order (t)", 0.5, 25.0, float(cur.min_order_t), 0.5, key="p_dl_moq")
    credit = st.toggle("Cap recommendations by available credit headroom", bool(cur.respect_credit_limit), key="p_dl_credit")
rs = ReplenishmentSettings(service_level=float(sl), lead_time_days=int(lt), review_days=int(rv), lot_size_t=float(lot),
                           min_order_t=float(moq), respect_credit_limit=bool(credit))
st.session_state["replenishment_settings"] = rs  # feeds planning, dashboard and recommendations

rep = state.replenishment(rs)
seg = state.segments_for(k)
vp = state.visit_list(rs)
hf = series_horizon_forecast(dem, int(horizon))
base = state.commercial_baseline(k)
dealer_region = env.dealers.set_index("dealer_id")["region"]
rep_r = rep[rep["dealer_id"].map(dealer_region).isin(regions)]
hf_r = hf[hf["region"].isin(regions)]

kpi_cards([
    ("Dealers in interactive sample", fmt_num(env.dealers["dealer_id"].isin(dealer_region[dealer_region.isin(regions)].index).sum()), f"{len(hf_r):,} dealer-product series"),
    (f"Forecast demand, next {horizon} days", fmt_t(hf_r["forecast_t"].sum()), f"90% range {fmt_t(hf_r['lower_t'].sum())} to {fmt_t(hf_r['upper_t'].sum())}"),
    ("Forecast accuracy (1 - WAPE)", fmt_pct(dem.forecast_accuracy),
     f"Dealer-week; depot-week {fmt_pct(1 - dem.aggregated_wape('depot_id'))}"),
    ("Likely stock-outs", fmt_num((rep_r["stockout_flag"] == "LIKELY STOCKOUT").sum()), "dealer-product lines"),
    ("Recommended replenishment", fmt_t(rep_r["recommended_qty_t"].sum()), f"{int(rep_r['credit_constrained'].sum())} lines credit-capped"),
    ("Protected contribution", fmt_inr(rep_r["protected_contribution_inr"].sum()), f"Revenue {fmt_inr(rep_r['protected_revenue_inr'].sum())} - SIMULATED"),
])
note("Forecasts are estimates of dealer sell-in demand, not guaranteed sales outcomes.")

t1, t2, t3, t4, t5 = st.tabs(["Forecast accuracy", "Dealer drill-down", "Replenishment and stock-outs",
                              "Segmentation and visit plan", "Network simulation (49,300 partners)"])

with t1:
    section("Model versus baselines on the chronological holdout",
            f"Trained on weeks before the holdout ({dem.train_rows:,} training rows); one-week-ahead predictions on the last "
            f"{dem.holdout_weeks} weeks. Model: {dem.model_name}.")
    m = dem.metrics.reset_index().rename(columns={"index": "Method"})
    disp = m.copy()
    disp["MAE"] = disp["MAE"].map(lambda v: f"{v:.2f} t")
    disp["RMSE"] = disp["RMSE"].map(lambda v: f"{v:.2f} t")
    disp["WAPE"] = disp["WAPE"].map(fmt_pct)
    disp["Bias"] = disp["Bias"].map(lambda v: f"{v * 100:+.1f}%")
    show_table(disp)
    hold = dem.holdout.merge(dem.series[["series_id", "region"]], on="series_id")
    hold = hold[hold["region"].isin(regions)]
    agg = hold.groupby("week_start")[["actual", "model", "ma4", "snaive"]].sum().reset_index()
    hist = env.dealer_weekly.merge(dem.series[["series_id", "region"]], on="series_id")
    hist = hist[hist["region"].isin(regions)].groupby("week_start")["qty_t"].sum().reset_index().tail(52)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["qty_t"], name="Actual", line=dict(color="#52514e")))
    fig.add_trace(go.Scatter(x=agg["week_start"], y=agg["model"], name="Model (holdout)", line=dict(color=SERIES[0])))
    fig.add_trace(go.Scatter(x=agg["week_start"], y=agg["ma4"], name="4-week moving average", line=dict(color=SERIES[1], dash="dot")))
    fc = dem.forecast.merge(dem.series[["series_id", "region"]], on="series_id")
    fc = fc[fc["region"].isin(regions)].groupby("week_start")[["forecast", "lower", "upper"]].sum().reset_index()
    fig.add_trace(go.Scatter(x=fc["week_start"], y=fc["forecast"], name="Forward forecast", line=dict(color=SERIES[2])))
    st.plotly_chart(style(fig, "Weekly dealer sell-in: actual, holdout predictions and forward forecast (t)", None, "Tonnes",
                          height=380), width="stretch", config=config())
    a, b = st.columns(2)
    fi = dem.feature_importance.head(10).sort_values("importance")
    fig = px.bar(fi, x="importance", y="feature", orientation="h", labels={"importance": "Relative importance (split gain share)", "feature": ""})
    fig.update_traces(marker_color=PRIMARY)
    a.plotly_chart(style(fig, "Key forecast drivers", legend=False), width="stretch", config=config())
    err = hold.assign(error=hold["model"] - hold["actual"])
    fig = px.histogram(err, x="error", nbins=60, labels={"error": "Forecast error (t), predicted - actual"})
    fig.update_traces(marker_color=PRIMARY)
    b.plotly_chart(style(fig, "Holdout error distribution", y_title="Count", legend=False), width="stretch", config=config())
    assumptions([
        "Chronological split only; no random shuffling of time-series observations.",
        "WAPE = sum of absolute errors / sum of actual demand, robust to zero-demand weeks (no per-row division).",
        "Bias = (sum predicted - sum actual) / sum actual; positive means over-forecasting.",
        "Forward forecasts are recursive and come from a model refitted on the full history.",
        "Prediction ranges use holdout residual spread x sqrt(horizon); they are approximate, not guaranteed coverage.",
        "Day horizons are converted from weekly forecasts (30 days = 4 weeks + 2/7 of week 5).",
    ], "Methodology and limitations")

with t2:
    dealers = env.dealers[env.dealers["region"].isin(regions)].sort_values("dealer_name")
    if dealers.empty:
        note("No dealers in the selected regions.")
    else:
        q = st.text_input("Search dealer by name, ID or district", key="p_dl_search")
        if q:
            ql = q.lower()
            dealers = dealers[dealers["dealer_name"].str.lower().str.contains(ql, regex=False)
                              | dealers["dealer_id"].str.lower().str.contains(ql, regex=False)
                              | dealers["district"].str.lower().str.contains(ql, regex=False)]
        opts = dealers["dealer_id"].tolist()
        if not opts:
            note("No dealers match the search.")
        else:
            did = st.selectbox("Dealer", opts, key="p_dl_dealer",
                               format_func=lambda d: f"{d} - {env.dealers.set_index('dealer_id').loc[d, 'dealer_name']}")
            d = env.dealers.set_index("dealer_id").loc[did]
            s = seg.set_index("dealer_id").loc[did]
            kpi_cards([("Region / district", f"{d['region']}", d["district"]), ("Segment", s["segment"], s["segment_reason"][:60]),
                       ("Last 13 weeks", fmt_t(s["vol_last13_t"]), f"{s['growth_yoy']:+.0%} year-on-year"),
                       ("Credit status", s["credit_status"], f"Utilisation {fmt_pct(s['credit_utilisation'])}"),
                       ("Sales representative", d["sales_rep_id"], "Assigned rep")])
            ser = dem.series[dem.series["dealer_id"] == did]
            for _, sr in ser.iterrows():
                h = env.dealer_weekly[env.dealer_weekly["series_id"] == sr["series_id"]].tail(52)
                ho = dem.holdout[dem.holdout["series_id"] == sr["series_id"]]
                fo = dem.forecast[dem.forecast["series_id"] == sr["series_id"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(fo["week_start"]) + list(fo["week_start"][::-1]),
                                         y=list(fo["upper"]) + list(fo["lower"][::-1]), fill="toself",
                                         fillcolor="rgba(27,175,122,0.15)", line=dict(width=0), name="Approx. 90% range",
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=h["week_start"], y=h["qty_t"], name="Actual", line=dict(color="#52514e")))
                fig.add_trace(go.Scatter(x=ho["week_start"], y=ho["model"], name="Holdout prediction", line=dict(color=SERIES[0])))
                fig.add_trace(go.Scatter(x=fo["week_start"], y=fo["forecast"], name="Forecast", line=dict(color=SERIES[2])))
                pname = env.products.set_index("product_id").loc[sr["product_id"], "product_name"]
                st.plotly_chart(style(fig, f"{pname}: weekly demand (t)", None, "Tonnes", height=300), width="stretch",
                                config=config(), key=f"dl_chart_{sr['series_id']}")
            r = rep[rep["dealer_id"] == did].merge(env.products[["product_id", "product_name"]], on="product_id")
            r = r.merge(hf[["series_id", "forecast_t", "lower_t", "upper_t"]], on="series_id")
            show_table(display_frame(r[["product_name", "forecast_t", "lower_t", "upper_t", "current_stock_t", "open_orders_t",
                                        "safety_stock_t", "recommended_qty_t", "stockout_probability", "stockout_flag"]],
                                     dec1=["forecast_t", "lower_t", "upper_t", "current_stock_t", "open_orders_t", "safety_stock_t",
                                           "recommended_qty_t"], pct=["stockout_probability"],
                                     rename={"product_name": "Product", "forecast_t": f"Forecast {horizon}d (t)", "lower_t": "Low (t)",
                                             "upper_t": "High (t)", "current_stock_t": "Stock (t)", "open_orders_t": "Open orders (t)",
                                             "safety_stock_t": "Safety stock (t)", "recommended_qty_t": "Recommended order (t)",
                                             "stockout_probability": "Stock-out probability", "stockout_flag": "Status"}),
                       status_cols=["Status"])

with t3:
    c1, c2 = st.columns([1, 3])
    flag = c1.multiselect("Status", ["LIKELY STOCKOUT", "NEEDS REVIEW", "WITHIN LIMIT"], default=["LIKELY STOCKOUT", "NEEDS REVIEW"],
                          key="p_dl_flag")
    view = rep_r[rep_r["stockout_flag"].isin(flag)] if flag else rep_r
    view = view.merge(env.dealers[["dealer_id", "dealer_name", "region", "sales_rep_id"]], on="dealer_id").sort_values(
        ["stockout_probability", "protected_contribution_inr"], ascending=False)
    cols = ["dealer_id", "dealer_name", "region", "product_id", "current_stock_t", "open_orders_t", "forecast_cover_t",
            "safety_stock_t", "recommended_qty_t", "credit_constrained", "days_of_cover", "stockout_probability",
            "protected_contribution_inr", "stockout_flag"]
    show_table(display_frame(view[cols], dec1=["current_stock_t", "open_orders_t", "forecast_cover_t", "safety_stock_t",
                                                "recommended_qty_t", "days_of_cover"], pct=["stockout_probability"],
                             inr=["protected_contribution_inr"],
                             rename={"dealer_id": "Dealer", "dealer_name": "Name", "region": "Region", "product_id": "Product",
                                     "current_stock_t": "Stock (t)", "open_orders_t": "Open orders (t)",
                                     "forecast_cover_t": "Forecast over cover (t)", "safety_stock_t": "Safety stock (t)",
                                     "recommended_qty_t": "Recommended (t)", "credit_constrained": "Credit capped",
                                     "days_of_cover": "Days of cover", "stockout_probability": "Stock-out probability",
                                     "protected_contribution_inr": "Protected contribution", "stockout_flag": "Status"}),
               status_cols=["Status"], height=420)
    a, b = st.columns(2)
    with a:
        download_csv(view[cols], "replenishment_recommendations.csv", "Download recommendations (CSV)", key="dl_rep")
    with b:
        approval_button("Dealer Replenishment", f"{len(view)} lines", f"Recommended replenishment of {view['recommended_qty_t'].sum():,.0f} t",
                        "Regional Sales Manager", key="ap_rep", label="Submit recommended orders for sales approval")
    assumptions([f"Recommended order = forecast over lead time ({lt} d) + review period ({rv} d) + safety stock - stock - open orders, "
                 f"rounded up to {lot} t lots with a {moq} t minimum.",
                 f"Safety stock = z({sl:.0%}) x weekly forecast-error sigma x sqrt(cover weeks).",
                 "Credit cap allocates each dealer's available headroom across its products in proportion to need.",
                 "Stock-out probability assumes normally distributed lead-time demand.",
                 f"Protected contribution uses the sample average of {fmt_inr(base['contribution_per_t'], False)} per tonne.",
                 "Dealer stock and open orders are a simulated snapshot (as from a dealer ordering platform)."])

with t4:
    seg_r = vp[vp["region"].isin(regions)]
    counts = seg_r["segment"].value_counts().reindex(SEGMENTS).fillna(0).reset_index()
    counts.columns = ["segment", "dealers"]
    a, b = st.columns([2, 3])
    fig = px.bar(counts, x="segment", y="dealers", text="dealers", labels={"segment": "", "dealers": "Dealers"})
    fig.update_traces(marker_color=PRIMARY)
    a.plotly_chart(style(fig, "Dealers by segment (rules-based)", legend=False), width="stretch", config=config())
    with b:
        q = decline_detection_quality(env, seg)
        section("Decline detection quality", "Against the simulation's known declining-dealer patterns (possible only with synthetic data).")
        kpi_cards([("Precision", fmt_pct(q["Precision"]), f"{q['Flagged']} dealers flagged"),
                   ("Recall", fmt_pct(q["Recall"]), f"{q['Actual positives']} truly declining"),
                   ("F1", f"{q['F1']:.2f}", "Rules-based segmentation")])
    reps = sorted(seg_r["sales_rep_id"].unique())
    rep_sel = st.selectbox("Visit plan for sales representative", ["All representatives"] + reps, key="p_dl_rep")
    vv = seg_r if rep_sel == "All representatives" else seg_r[seg_r["sales_rep_id"] == rep_sel]
    seg_f = st.multiselect("Segments", SEGMENTS, key="p_dl_segf", placeholder="All segments")
    if seg_f:
        vv = vv[vv["segment"].isin(seg_f)]
    cols = ["visit_rank", "priority", "dealer_id", "dealer_name", "district", "sales_rep_id", "segment", "segment_reason",
            "vol_last52_t", "growth_yoy", "credit_status", "incremental_opportunity_inr"]
    show_table(display_frame(vv[cols].head(200), tonnes=["vol_last52_t"], pct=["growth_yoy"], inr=["incremental_opportunity_inr"],
                             rename={"visit_rank": "Rank", "priority": "Priority", "dealer_id": "Dealer", "dealer_name": "Name",
                                     "district": "District", "sales_rep_id": "Rep", "segment": "Segment",
                                     "segment_reason": "Reason", "vol_last52_t": "Volume 52w", "growth_yoy": "YoY",
                                     "credit_status": "Credit status", "incremental_opportunity_inr": "Estimated opportunity"}),
               status_cols=["Priority", "Credit status"], height=420)
    download_csv(vv[cols], "dealer_visit_plan.csv", "Download visit plan (CSV)", key="dl_visit")
    assumptions(["Segments use transparent thresholds: At-risk (overdue > 45 days, utilisation > 110%, or declining with no order "
                 "for 4+ weeks), Declining (volume down > 25% vs prior quarter and > 20% year-on-year, or falling order frequency), "
                 "Growing (> 15% year-on-year), High-value (top 15% by annual volume), otherwise Stable.",
                 "Estimated opportunity = 30% of the quarterly volume gap versus last year x contribution per tonne "
                 "(scenario assumption) + contribution protected by replenishment.",
                 "This is a rules-based method, not a trained ML model."], "Segmentation rules")

with t5:
    pop = segment_population(state.population())
    pop = pop[pop["region"].isin(regions)]
    section("Simulated network of 49,300 channel partners",
            "Vectorised simulation for network-level views; only aggregates are sent to the browser.")
    kpi_cards([("Channel partners (filtered)", fmt_num(len(pop)), "Synthetic population"),
               ("Annual volume", fmt_t(pop["annual_volume_t"].sum()), "Simulated"),
               ("Digital ordering adoption", fmt_pct(pop["uses_digital_ordering"].mean()), "Share of partners"),
               ("Partners at risk or declining", fmt_num(pop["segment"].isin(["At-risk", "Declining"]).sum()), "Rules-based")])
    agg = pop.groupby(["region", "segment"]).agg(partners=("dealer_id", "count"), volume_t=("annual_volume_t", "sum")).reset_index()
    fig = px.bar(agg, x="region", y="partners", color="segment", category_orders={"segment": SEGMENTS},
                 color_discrete_sequence=SERIES, labels={"region": "", "partners": "Channel partners"})
    st.plotly_chart(style(fig, "Channel partners by region and segment", height=380), width="stretch", config=config())
    tbl = agg.pivot(index="region", columns="segment", values="partners").fillna(0).astype(int).reset_index()
    show_table(tbl)
    download_csv(agg, "network_segment_summary.csv", "Download network summary (CSV)", key="dl_pop")
