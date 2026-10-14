"""Page 6 - Credit Risk and Profitable Order Intelligence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.credit_risk import explain_customer, payment_survival_by_band
from src.services.order_economics import compute_order_economics, summarise_economics, waterfall_frame
from src.ui import state
from src.ui.charts import PRIMARY, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_pct

page_header("Credit Risk and Profitable Order Intelligence",
            "Predict late payment, estimate expected credit loss, prioritise collections and test risk-adjusted order profitability.")
k = state.key()
env = state.env()
regions = state.global_regions()
cr = state.credit_for(k)
cp = state.credit_params()

with st.form("credit_form"):
    section("Credit scenario assumptions", "Changes are simulations only; actual credit limits change only with finance approval.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    pdc = c1.slider("PD conversion (late to default)", 0.05, 0.5, float(cp["pd_conversion"]), 0.01, key="p_cr_pdc")
    lt = c2.slider("LGD trade", 0.1, 0.9, float(cp["lgd_trade"]), 0.05, key="p_cr_lt")
    lc = c3.slider("LGD corporate", 0.1, 0.9, float(cp["lgd_corporate"]), 0.05, key="p_cr_lc")
    lg = c4.slider("LGD government", 0.0, 0.9, float(cp["lgd_government"]), 0.05, key="p_cr_lg")
    lm = c5.slider("Credit-limit scenario multiplier", 0.5, 2.0, float(cp["limit_multiplier"]), 0.05, key="p_cr_lm")
    extra = c6.slider("Payment-term change (days)", -15, 45, int(st.session_state.get("credit_extra_days", 0)), 5, key="p_cr_extra")
    if st.form_submit_button("Recalculate credit scenario", type="primary"):
        st.session_state["credit_params"] = {"pd_conversion": float(pdc), "lgd_trade": float(lt), "lgd_corporate": float(lc),
                                             "lgd_government": float(lg), "limit_multiplier": float(lm)}
        st.session_state["credit_extra_days"] = int(extra)
        st.success("Credit scenario recalculated.")
pol = state.credit_policy()
extra_days = int(st.session_state.get("credit_extra_days", 0))
pol_r = pol[pol["customer_id"].map(env.customers.set_index("customer_id")["region"]).isin(regions)]

kpi_cards([
    ("Outstanding receivables", fmt_inr(pol_r["outstanding_inr"].sum()), f"{int((pol_r['outstanding_inr'] > 0).sum())} customers"),
    ("Overdue receivables", fmt_inr(pol_r["overdue_inr"].sum()), f"{int((pol_r['overdue_inr'] > 0).sum())} customers overdue"),
    ("Expected credit loss", fmt_inr(pol_r["ecl_inr"].sum()), "Simplified PD x LGD x EAD"),
    ("High / very-high risk", str(int(pol_r["risk_band"].isin(["High", "Very High"]).sum())), "Predicted late payment > 35%"),
    ("Exposure above limit", str(int((pol_r["limit_headroom_inr"] < 0).sum())), "After pending orders, scenario limits"),
    ("Model ROC-AUC (holdout)", f"{cr.metrics.loc[cr.champion_name, 'ROC-AUC']:.2f}", cr.champion_name),
])
note("Expected credit loss here is a simplified demonstration formula (PD x LGD x EAD), not a regulatory accounting model. "
     "No protected personal attributes are used; features describe commercial payment behaviour only.")

t1, t2, t3, t4, t5 = st.tabs(["Risk overview", "Collections and credit review", "Customer drill-down",
                              "Order profitability", "Model performance"])
band_order = ["Low", "Medium", "High", "Very High", "No recent history"]
with t1:
    a, b = st.columns(2)
    rb = pol_r.groupby("risk_band").agg(customers=("customer_id", "count"), overdue=("overdue_inr", "sum"),
                                        ecl=("ecl_inr", "sum")).reindex(band_order).fillna(0).reset_index()
    fig = px.bar(rb, x="risk_band", y="customers", text="customers", labels={"risk_band": "Risk band", "customers": "Customers"})
    fig.update_traces(marker_color=PRIMARY)
    a.plotly_chart(style(fig, "Customers by predicted late-payment risk", legend=False), width="stretch", config=config())
    fig = px.histogram(pol_r.dropna(subset=["p_late_90d"]), x="p_late_90d", nbins=30,
                       labels={"p_late_90d": "Probability of 30+ day late payment in next 90 days"})
    fig.update_traces(marker_color=PRIMARY)
    b.plotly_chart(style(fig, "Distribution of late-payment probability", y_title="Customers", legend=False), width="stretch", config=config())
    inv = env.invoices[(env.invoices["status"] != "Paid") & env.invoices["customer_id"].isin(pol_r["customer_id"])].copy()
    inv["bucket"] = pd.cut(inv["days_past_due"], [-1, 0, 30, 60, 90, 10_000],
                           labels=["Not yet due", "1-30 days", "31-60 days", "61-90 days", "Over 90 days"])
    ag = inv.groupby(["bucket", "channel"], observed=False)["amount_inr"].sum().reset_index()
    ag["INR crore"] = ag["amount_inr"] / 1e7
    a, b = st.columns(2)
    fig = px.bar(ag, x="bucket", y="INR crore", color="channel", color_discrete_sequence=[SERIES[0], SERIES[1]],
                 labels={"bucket": "Invoice ageing"})
    a.plotly_chart(style(fig, "Invoice ageing (open receivables, INR crore)"), width="stretch", config=config())
    seg = pol_r.groupby("customer_segment").agg(overdue=("overdue_inr", "sum"), ecl=("ecl_inr", "sum")).reset_index().sort_values("overdue")
    fig = px.bar(seg, x=seg["overdue"] / 1e7, y="customer_segment", orientation="h",
                 labels={"x": "Overdue receivables (INR crore)", "customer_segment": ""})
    fig.update_traces(marker_color=SERIES[1])
    b.plotly_chart(style(fig, "Overdue exposure by customer segment", legend=False), width="stretch", config=config())

with t2:
    act = st.multiselect("Recommended action", sorted(pol_r["recommended_action"].unique()),
                         default=[a_ for a_ in pol_r["recommended_action"].unique() if a_ != "WITHIN LIMIT"], key="p_cr_act")
    v = pol_r[pol_r["recommended_action"].isin(act)] if act else pol_r
    v = v.merge(env.customers[["customer_id", "customer_name", "region"]], on="customer_id").sort_values("collection_priority_score", ascending=False)
    cols = ["customer_id", "customer_name", "customer_segment", "region", "risk_band", "p_late_90d_filled", "outstanding_inr",
            "overdue_inr", "max_dpd", "pending_orders_inr", "exposure_after_orders_inr", "scenario_credit_limit_inr",
            "limit_headroom_inr", "ecl_inr", "key_risk_drivers", "recommended_action"]
    show_table(display_frame(v[cols], pct=["p_late_90d_filled"], inr=["outstanding_inr", "overdue_inr", "pending_orders_inr",
                                                                      "exposure_after_orders_inr", "scenario_credit_limit_inr",
                                                                      "limit_headroom_inr", "ecl_inr"], num=["max_dpd"],
                             rename={"customer_id": "Customer", "customer_name": "Name", "customer_segment": "Segment",
                                     "region": "Region", "risk_band": "Risk band", "p_late_90d_filled": "P(late 30+ d)",
                                     "outstanding_inr": "Outstanding", "overdue_inr": "Overdue", "max_dpd": "Max DPD",
                                     "pending_orders_inr": "Pending orders", "exposure_after_orders_inr": "Exposure after orders",
                                     "scenario_credit_limit_inr": "Credit limit (scenario)", "limit_headroom_inr": "Headroom",
                                     "ecl_inr": "ECL", "key_risk_drivers": "Key risk drivers", "recommended_action": "Action"}),
               status_cols=["Action", "Risk band"], height=460)
    a, b = st.columns(2)
    with a:
        download_csv(v[cols], "collection_priority.csv", "Download collection priority list (CSV)", key="dl_coll")
    with b:
        approval_button("Credit Risk", f"{len(v)} customers", "Credit review / order escalation list", "Finance Manager",
                        key="ap_credit", label="Send list to finance for approval")
    note("Order release and credit-limit changes require finance approval. The platform does not change customer terms.")

with t3:
    cand = pol_r.merge(env.customers[["customer_id", "customer_name"]], on="customer_id").sort_values("ecl_inr", ascending=False)
    if cand.empty:
        note("No customers in the selected regions.")
    else:
        cid = st.selectbox("Customer", cand["customer_id"].tolist(), key="p_cr_cust",
                           format_func=lambda c: f"{c} - {cand.set_index('customer_id').loc[c, 'customer_name']}")
        r = pol.set_index("customer_id").loc[cid]
        kpi_cards([("Risk band", r["risk_band"], f"P(late) {fmt_pct(r['p_late_90d_filled'])}"),
                   ("Exposure after orders", fmt_inr(r["exposure_after_orders_inr"]), f"Limit {fmt_inr(r['scenario_credit_limit_inr'])}"),
                   ("Overdue", fmt_inr(r["overdue_inr"]), f"Max {int(r['max_dpd'])} days past due"),
                   ("Expected credit loss", fmt_inr(r["ecl_inr"]), f"PD {fmt_pct(r['pd_12m'])} x LGD {fmt_pct(r['lgd'])}"),
                   ("Action", r["recommended_action"], "Requires finance approval")])
        contrib = explain_customer(cr.logistic, pd.concat([r, pd.Series({"customer_id": cid})]), top_n=8)
        ex = pd.DataFrame(contrib, columns=["factor", "contribution"]).sort_values("contribution")
        fig = px.bar(ex, x="contribution", y="factor", orientation="h",
                     labels={"contribution": "Contribution to log-odds of late payment (+ raises risk)", "factor": ""})
        fig.update_traces(marker_color=[SERIES[7] if v_ > 0 else SERIES[0] for v_ in ex["contribution"]])
        st.plotly_chart(style(fig, "Explainable risk factors (transparent logistic model)", height=340, legend=False),
                        width="stretch", config=config())
        ci = env.invoices[env.invoices["customer_id"] == cid].sort_values("invoice_date", ascending=False).head(40)
        show_table(display_frame(ci[["invoice_id", "invoice_date", "due_date", "paid_date", "amount_inr", "status", "days_past_due"]],
                                 dates=["invoice_date", "due_date", "paid_date"], inr_full=["amount_inr"]),
                   status_cols=["status"], height=300)

with t4:
    fr = state.freight()
    p = fr.plan.copy()
    served_opt = p["opt_source_id"].notna()
    p["production_cost_per_t"] = np.where(served_opt, p["opt_sourcing_cost_per_t"] - p["opt_primary_per_t"],
                                          p["base_sourcing_cost_per_t"] - p["base_primary_per_t"])
    p["freight_per_t"] = np.where(served_opt, p["opt_freight_inr"] / p["qty_t"] + p["opt_primary_per_t"],
                                  p["base_freight_inr"] / p["qty_t"] + p["base_primary_per_t"])
    p["handling_per_t"] = np.where(served_opt, p["opt_handling_per_t"], p["base_handling_per_t"])
    p["freight_basis"] = np.where(served_opt, "Optimised dispatch plan", "Baseline (no feasible optimised option)")
    p = p[p["production_cost_per_t"].notna()]
    pr = pol.set_index("customer_id")
    p["pd_12m"] = p["customer_id"].map(pr["pd_12m"]).fillna(pr["pd_12m"].median())
    p["lgd"] = p["customer_id"].map(pr["lgd"]).fillna(0.45)
    p["collection_days"] = p["customer_id"].map(pr["payment_terms_days"] + pr["avg_days_late_12m"].clip(lower=0)).fillna(30) + extra_days
    econ = compute_order_economics(p, pd_col="pd_12m", lgd_col="lgd", collection_days_col="collection_days")
    thr = st.slider("Minimum risk-adjusted margin before escalation", 0.0, 0.15, 0.05, 0.01, key="p_cr_thr")
    econ["profitability_flag"] = np.select(
        [econ["contribution_margin"] < 0, (econ["gross_margin"] > 0) & (econ["risk_adjusted_margin_pct"] < thr)],
        ["NEGATIVE CONTRIBUTION - ESCALATE", "WEAK RISK-ADJUSTED CONTRIBUTION - REVIEW"], "WITHIN LIMIT")
    econ_r = econ[econ["region"].isin(regions)]
    tot = summarise_economics(econ_r)
    kpi_cards([("Pending order net revenue", fmt_inr(tot["net_revenue"]), f"{len(econ_r)} open orders"),
               ("Contribution margin", fmt_inr(tot["contribution_margin"]), fmt_pct(tot["contribution_margin_pct"])),
               ("Risk-adjusted contribution", fmt_inr(tot["risk_adjusted_contribution"]), "After ECL and financing"),
               ("Orders flagged", str(int((econ_r["profitability_flag"] != "WITHIN LIMIT").sum())),
                "Positive gross margin, weak after costs and risk")])
    wf = waterfall_frame(tot)
    fig = go.Figure(go.Waterfall(x=wf["step"], y=wf["value"] / 1e5, measure=wf["measure"],
                                 increasing=dict(marker_color=SERIES[0]), decreasing=dict(marker_color=SERIES[1]),
                                 totals=dict(marker_color="#52514e"), connector=dict(line=dict(color="#c3c2b7"))))
    st.plotly_chart(style(fig, "From gross revenue to risk-adjusted contribution (pending orders, INR lakh)", y_title="INR lakh",
                          height=360, legend=False), width="stretch", config=config())
    cols = ["open_order_id", "customer_id", "customer_name", "channel", "product_id", "qty_t", "net_revenue", "production_cost",
            "freight_cost", "handling_cost", "gross_margin", "contribution_margin", "expected_credit_loss", "financing_cost",
            "risk_adjusted_contribution", "risk_adjusted_margin_pct", "freight_basis", "profitability_flag"]
    show_table(display_frame(econ_r.sort_values("risk_adjusted_margin_pct")[cols], dec1=["qty_t"], pct=["risk_adjusted_margin_pct"],
                             inr=["net_revenue", "production_cost", "freight_cost", "handling_cost", "gross_margin",
                                  "contribution_margin", "expected_credit_loss", "financing_cost", "risk_adjusted_contribution"],
                             rename={"open_order_id": "Order", "customer_id": "Customer", "customer_name": "Name",
                                     "channel": "Channel", "product_id": "Product", "qty_t": "Qty (t)",
                                     "net_revenue": "Net revenue", "production_cost": "Production cost",
                                     "freight_cost": "Freight", "handling_cost": "Handling", "gross_margin": "Gross margin",
                                     "contribution_margin": "Contribution", "expected_credit_loss": "ECL",
                                     "financing_cost": "Financing", "risk_adjusted_contribution": "Risk-adj. contribution",
                                     "risk_adjusted_margin_pct": "Risk-adj. margin", "freight_basis": "Freight basis",
                                     "profitability_flag": "Flag"}),
               status_cols=["Flag"], height=420)
    download_csv(econ_r[cols], "order_profitability.csv", "Download order profitability (CSV)", key="dl_op")
    assumptions(["Freight per order comes from the current dispatch optimisation scenario (Page 3), so freight changes flow into margins.",
                 "Expected credit loss per order = PD (12 months) x LGD x order net revenue (incremental exposure).",
                 f"Financing cost = net revenue x 10% per year x expected collection days / 365 (terms + historical lateness"
                 f"{' + scenario change of ' + str(extra_days) + ' days' if extra_days else ''}).",
                 "ECL and financing do not overlap: ECL is loss of principal; financing is the time value of the collection period.",
                 "Production cost excludes the primary freight that is already included in freight for depot-sourced orders."],
                "Order economics definitions")

with t5:
    section("Late-payment classifier", f"Features observed at snapshot {cr.snapshot_date.strftime('%d-%b-%Y')}; label = any invoice "
            f"due in the next 90 days paid 30+ days late. Train {cr.train_size} / holdout {cr.test_size} customers.")
    m = cr.metrics.reset_index().rename(columns={"index": "Model"})
    show_table(display_frame(m[["Model", "ROC-AUC", "Precision", "Recall", "F1", "Brier score", "Positive rate (actual)", "Threshold"]],
                             pct=["Precision", "Recall", "Positive rate (actual)"]).round(3))
    st.caption(f"Champion: {cr.champion_name} (the transparent model is preferred unless the challenger improves ROC-AUC by more than 0.02).")
    a, b = st.columns(2)
    fig = px.line(cr.calibration, x="mean_predicted", y="observed_rate", markers=True,
                  labels={"mean_predicted": "Mean predicted probability", "observed_rate": "Observed late-payment rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#898781", dash="dot"))
    fig.update_traces(line_color=PRIMARY, marker=dict(size=9))
    a.plotly_chart(style(fig, "Calibration on holdout (dotted = perfect)", legend=False), width="stretch", config=config())
    co = cr.coefficients.sort_values("coefficient")
    fig = px.bar(co, x="coefficient", y="label", orientation="h", labels={"coefficient": "Standardised coefficient", "label": ""})
    fig.update_traces(marker_color=PRIMARY)
    b.plotly_chart(style(fig, "Risk drivers (logistic regression)", height=420, legend=False), width="stretch", config=config())
    surv = payment_survival_by_band(env, cr.scored)
    bands = [bb for bb in band_order if bb in set(surv["risk_band"])]
    fig = px.line(surv, x="days_after_due", y="prob_unpaid", color="risk_band", category_orders={"risk_band": bands},
                  color_discrete_sequence=SERIES, labels={"days_after_due": "Days after due date",
                                                          "prob_unpaid": "Probability invoice still unpaid", "risk_band": "Risk band"})
    st.plotly_chart(style(fig, "Payment timing: Kaplan-Meier survival of unpaid invoices by risk band", height=360),
                    width="stretch", config=config())
