"""Page 8 - Pricing, Discount and Sales Intelligence."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MAX_DISCOUNT_PER_T_POLICY, MIN_MARGIN_PCT_POLICY, REGIONS
from src.data.synthetic_data import TRUE_ELASTICITY
from src.models.pricing_analysis import (baseline_position, discount_frontier, scheme_effectiveness, simulate_scenario,
                                         suggest_discount_range)
from src.ui import state
from src.ui.charts import PRIMARY, SERIES, config, style
from src.ui.components import (approval_button, assumptions, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_inr, fmt_pct, fmt_t

page_header("Pricing, Discount and Sales Intelligence",
            "Price and margin trends, estimated price sensitivity, discount scenarios within policy, and contribution opportunities.")
k = state.key()
env = state.env()
panel = env.pricing_panel
regions = state.global_regions()

c1, c2 = st.columns(2)
min_m = c1.slider("Policy: minimum contribution margin", 0.0, 0.3, float(st.session_state.get("policy_min_margin", MIN_MARGIN_PCT_POLICY)),
                  0.01, key="p_pr_minm")
max_d = c2.slider("Policy: maximum discount (INR / t)", 0, 800, int(st.session_state.get("policy_max_discount", MAX_DISCOUNT_PER_T_POLICY)),
                  10, key="p_pr_maxd")
st.session_state["policy_min_margin"] = float(min_m)
st.session_state["policy_max_discount"] = float(max_d)
note("Competitor prices are a simulated market indicator only. The platform never recommends coordinated or collusive pricing. "
     "Elasticities are correlational estimates from simulated data, used as explicit scenario assumptions.")

pr = panel[panel["region"].isin(regions)]
el = state.elasticities_for(k)
t1, t2, t3, t4, t5, t6 = st.tabs(["Regional price and margin", "Price elasticity", "Discount scenario simulator",
                                  "Discount effectiveness", "Customer and product margins", "Sales opportunities"])
with t1:
    m = pr.groupby(["month", "region"]).apply(lambda g: pd.Series({
        "net_price": (g["net_price_per_t"] * g["volume_t"]).sum() / g["volume_t"].sum(),
        "margin": g["contribution_inr"].sum() / (g["net_price_per_t"] * g["volume_t"]).sum()}), include_groups=False).reset_index()
    a, b = st.columns(2)
    fig = px.line(m, x="month", y="net_price", color="region", color_discrete_sequence=SERIES,
                  category_orders={"region": REGIONS}, labels={"month": "", "net_price": "INR per tonne"})
    a.plotly_chart(style(fig, "Volume-weighted net price by region"), width="stretch", config=config())
    fig = px.line(m, x="month", y="margin", color="region", color_discrete_sequence=SERIES,
                  category_orders={"region": REGIONS}, labels={"month": "", "margin": "Contribution margin"})
    fig.update_yaxes(tickformat=".0%")
    b.plotly_chart(style(fig, "Contribution margin by region"), width="stretch", config=config())
    last = pr[pr["month"] == pr["month"].max()].groupby("region").agg(
        net_price=("net_price_per_t", "mean"), competitor=("competitor_price_per_t", "mean"), discount=("discount_per_t", "mean"),
        freight=("freight_per_t", "mean"), margin=("contribution_margin_pct", "mean")).reset_index()
    show_table(display_frame(last, inr_full=["net_price", "competitor", "discount", "freight"], pct=["margin"],
                             rename={"region": "Region", "net_price": "Net price / t", "competitor": "Competitor indicator / t",
                                     "discount": "Discount / t", "freight": "Freight / t", "margin": "Margin"}))

with t2:
    e = el.copy()
    e["simulation_assumption"] = e["region"].map(TRUE_ELASTICITY)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e["elasticity"], y=e["region"], mode="markers", name="Estimate (95% CI)",
                             marker=dict(color=PRIMARY, size=10),
                             error_x=dict(type="data", symmetric=False, array=e["ci_high"] - e["elasticity"],
                                          arrayminus=e["elasticity"] - e["ci_low"], color=PRIMARY)))
    fig.add_trace(go.Scatter(x=e["simulation_assumption"], y=e["region"], mode="markers", name="Simulation ground truth",
                             marker=dict(color=SERIES[1], size=10, symbol="diamond")))
    st.plotly_chart(style(fig, "Estimated price elasticity of volume by region", "Elasticity (% volume change per 1% relative price change)",
                          height=340), width="stretch", config=config())
    show_table(display_frame(e, dec1=[], rename={"region": "Region", "elasticity": "Estimate", "ci_low": "CI low", "ci_high": "CI high",
                                                 "p_value": "p-value", "r_squared": "R-squared", "n_obs": "Observations",
                                                 "simulation_assumption": "Simulated true value"}).round(3))
    assumptions(["Log-log OLS per region: log(volume) on log(net price / competitor price indicator), log(market demand index), "
                 "micro-market and product fixed effects (statsmodels).",
                 "Estimates are correlational. Real data would need controls for promotions, supply constraints and price endogeneity.",
                 "The simulated true elasticity is shown only because the data is synthetic."], "Estimation method")

with t3:
    c1, c2, c3 = st.columns(3)
    reg = c1.selectbox("Region", [r for r in REGIONS if r in regions] or REGIONS, key="p_pr_reg")
    prods = panel[["product_id", "product_name"]].drop_duplicates()
    pid = c2.selectbox("Product", prods["product_id"].tolist(), key="p_pr_prod",
                       format_func=lambda p: prods.set_index("product_id").loc[p, "product_name"])
    est = float(el.set_index("region").loc[reg, "elasticity"])
    eps = c3.number_input("Elasticity assumption", -4.0, -0.1, round(est, 2), 0.05, key=f"p_pr_eps_{reg}",
                          help="Defaults to the regional estimate; edit to test sensitivity.")
    c1, c2 = st.columns(2)
    dp = c1.slider("List price change", -0.08, 0.08, 0.0, 0.005, key="p_pr_dp", format="%.3f")
    dd = c2.slider("Discount change (INR / t)", -300, 300, 0, 10, key="p_pr_dd")
    base = baseline_position(panel, reg, pid)
    new = simulate_scenario(base, eps, dp, dd)
    cur = simulate_scenario(base, eps)
    viol = []
    if new["margin_pct"] < min_m:
        viol.append(f"margin {fmt_pct(new['margin_pct'])} below policy minimum {fmt_pct(min_m)}")
    if new["discount"] > max_d:
        viol.append(f"discount INR {new['discount']:,.0f} above policy maximum INR {max_d:,.0f}")
    kpi_cards([("Net price / t", fmt_inr(new["net_price"], False), f"Current {fmt_inr(cur['net_price'], False)}"),
               ("Monthly volume (scenario)", fmt_t(new["monthly_volume_t"]), f"{new['volume_change_pct']:+.1%} vs current"),
               ("Monthly revenue", fmt_inr(new["revenue"]), f"Current {fmt_inr(cur['revenue'])}"),
               ("Monthly contribution", fmt_inr(new["contribution"]), f"{fmt_inr(new['contribution_change'])} change"),
               ("Contribution margin", fmt_pct(new["margin_pct"]), "POLICY BREACH" if viol else "WITHIN LIMIT")])
    if viol:
        st.warning("Outside configured policy: " + "; ".join(viol) + ". This scenario cannot be recommended.")
    fr = discount_frontier(base, eps, max_d, min_m)
    sug = suggest_discount_range(fr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr["discount_per_t"], y=fr["contribution"] / 1e5, name="Monthly contribution (INR lakh)",
                             line=dict(color=SERIES[0])))
    if sug:
        fig.add_vrect(x0=sug["low"], x1=sug["high"], fillcolor="rgba(27,175,122,0.15)", line_width=0,
                      annotation_text="Suggested range", annotation_position="top left")
    st.plotly_chart(style(fig, "Contribution versus discount (within policy maximum)", "Discount (INR / t)", "INR lakh per month",
                          height=320), width="stretch", config=config())
    fig = px.scatter(fr, x="revenue", y="contribution", color="within_policy", hover_data={"discount_per_t": True},
                     color_discrete_map={True: SERIES[0], False: SERIES[7]},
                     labels={"revenue": "Monthly revenue (INR)", "contribution": "Monthly contribution (INR)",
                             "within_policy": "Within policy"})
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
    st.plotly_chart(style(fig, "Revenue versus margin trade-off across discount levels", height=320), width="stretch", config=config())
    if sug:
        st.markdown(f"**Suggested discount range within policy:** INR {sug['low']:,.0f} to {sug['high']:,.0f} per tonne "
                    f"(best INR {sug['best_discount']:,.0f}, margin {fmt_pct(sug['margin_pct'])}). SIMULATED RESULT - requires "
                    "pricing committee and management approval; contractual commitments override this suggestion.")
        approval_button("Pricing", f"{reg}/{pid}", f"Discount range INR {sug['low']:,.0f}-{sug['high']:,.0f} per t",
                        "Pricing Committee", key="ap_price", label="Submit discount proposal for management approval")
    else:
        note("No discount level satisfies the configured policy for this region and product.")

with t4:
    se = scheme_effectiveness(pr)
    show_table(display_frame(se, inr_full=["avg_discount_per_t", "avg_contribution_per_t"], pct=["avg_conversion", "avg_margin_pct",
                                                                                                 "volume_uplift_vs_no_scheme"],
                             rename={"scheme": "Scheme", "observations": "Observations", "avg_discount_per_t": "Avg discount / t",
                                     "avg_volume_index": "Volume index", "avg_conversion": "Quote conversion",
                                     "avg_contribution_per_t": "Contribution / t", "avg_margin_pct": "Margin",
                                     "volume_uplift_vs_no_scheme": "Volume vs no scheme"}).round(3))
    fig = px.scatter(se, x="volume_uplift_vs_no_scheme", y="avg_contribution_per_t", text="scheme",
                     labels={"volume_uplift_vs_no_scheme": "Volume index vs no scheme", "avg_contribution_per_t": "Contribution per tonne (INR)"})
    fig.update_traces(marker=dict(size=12, color=PRIMARY), textposition="top center")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "Volume growth versus margin erosion by scheme", height=360, legend=False), width="stretch", config=config())
    note("Scheme comparisons are descriptive (correlational) and not evidence that a scheme caused the volume difference.")

with t5:
    o = state.order_economics_for(k)
    o = o[(o["order_date"] >= env.as_of - pd.Timedelta(days=180)) & o["region"].isin(regions)]
    byp = o.groupby("product_id").agg(qty_t=("qty_t", "sum"), net_revenue=("net_revenue", "sum"),
                                      contribution=("contribution_margin", "sum")).reset_index()
    byp["margin"] = byp["contribution"] / byp["net_revenue"]
    byp = byp.merge(env.products[["product_id", "product_name"]], on="product_id")
    fig = px.bar(byp.sort_values("margin"), x="margin", y="product_name", orientation="h",
                 labels={"margin": "Contribution margin (last 6 months)", "product_name": ""})
    fig.update_traces(marker_color=PRIMARY)
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, "Contribution margin by product", legend=False), width="stretch", config=config())
    byc = o.groupby(["customer_id", "channel", "customer_segment"]).agg(
        qty_t=("qty_t", "sum"), net_revenue=("net_revenue", "sum"), contribution=("contribution_margin", "sum"),
        avg_discount=("discount_per_t", "mean"), avg_freight=("freight_per_t", "mean")).reset_index()
    byc["margin"] = byc["contribution"] / byc["net_revenue"]
    byc["flag"] = byc["margin"].apply(lambda v: "BELOW POLICY MARGIN" if v < min_m else "WITHIN LIMIT")
    low = byc.sort_values("margin").head(100)
    section("Lowest-margin customers (last 6 months)", f"{int((byc['flag'] != 'WITHIN LIMIT').sum())} customers below the "
            f"{fmt_pct(min_m)} policy margin, mostly driven by freight distance and discount levels.")
    show_table(display_frame(low, tonnes=["qty_t"], inr=["net_revenue", "contribution"], inr_full=["avg_discount", "avg_freight"],
                             pct=["margin"]), status_cols=["flag"], height=380)
    download_csv(byc, "customer_margins.csv", "Download customer margins (CSV)", key="dl_cm")

with t6:
    opp = state.pricing_opportunities()
    opp = opp[opp["region"].isin(regions)]
    section("Micro-market opportunities ranked by expected contribution gain within policy",
            "Gain = contribution at the policy-best discount minus current contribution, using the regional elasticity estimate.")
    show_table(display_frame(opp, inr_full=["current_discount_per_t", "suggested_discount_per_t"], pct=["current_margin_pct"],
                             inr=["monthly_contribution_now_inr", "monthly_contribution_gain_inr"]).round(2), height=420)
    download_csv(opp, "pricing_opportunities.csv", "Download opportunities (CSV)", key="dl_po")
