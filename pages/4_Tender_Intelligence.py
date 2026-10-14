"""Page 4 - Institutional Tender and Sales Intelligence."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.models.credit_risk import LGD_BY_CLASS
from src.models.tender_scoring import (EXCLUDED_POST_OUTCOME, _design, bid_recommendation, eligibility_checklist,
                                       tender_briefing, tender_economics)
from src.services.document_parser import DocumentError, extract_tender_fields, read_document, text_to_pdf_bytes
from src.ui import state
from src.ui.charts import PRIMARY, SERIES, config, style
from src.ui.components import (approval_button, assumptions, badge, download_csv, kpi_cards, note, page_header,
                               section, show_table)
from src.ui.formatting import display_frame, fmt_date, fmt_inr, fmt_pct, fmt_t

page_header("Institutional Tender and Sales Intelligence",
            "Extract tender requirements, check eligibility, estimate win probability and risk-adjusted value, and brief the bid committee.")
k = state.key()
env = state.env()
regions = state.global_regions()

c1, c2 = st.columns([1, 3])
mm = c1.slider("Minimum risk-adjusted margin for BID", 0.0, 0.25, float(st.session_state.get("tender_min_margin", 0.08)), 0.01,
               key="p_td_margin")
st.session_state["tender_min_margin"] = mm
tm = state.tender_model_for(k)
scored = state.tenders_scored()
sv = scored[scored["region"].isin(regions)]
kpi_cards([
    ("Open tenders", str(len(sv)), f"Closing within {sv['days_to_close'].max() if len(sv) else 0} days"),
    ("Pipeline quantity", fmt_t(sv["quantity_t"].sum()), "Total requirement"),
    ("Expected revenue", fmt_inr(sv["expected_revenue_inr"].sum()), "Win probability x contract value"),
    ("Risk-adjusted expected value", fmt_inr(sv["risk_adjusted_ev_inr"].sum()), "After credit, financing, guarantee, penalty"),
    ("Recommended to bid", str(int(sv["recommendation"].str.startswith("BID").sum())), "Subject to bid-committee approval"),
    ("Holdout ROC-AUC", f"{tm.metrics.loc['Holdout (later tenders)', 'ROC-AUC']:.2f}", "Win-probability model"),
])
note("Recommendations are advisory. The platform never submits a tender or makes a legally binding commitment.")

t1, t2, t3, t4 = st.tabs(["Pipeline and closing calendar", "Tender evaluation and briefing", "Upload a tender document",
                          "Win-probability model"])

with t1:
    rec_f = st.multiselect("Recommendation", sorted(sv["recommendation"].unique()), key="p_td_rec", placeholder="All")
    v = sv[sv["recommendation"].isin(rec_f)] if rec_f else sv
    cal = v.sort_values("closing_date").copy()
    fig = px.scatter(cal, x="closing_date", y="institution_type", size="quantity_t", color="recommendation",
                     hover_data={"tender_id": True, "quantity_t": ":,.0f", "p_win": ":.0%"},
                     color_discrete_sequence=SERIES, labels={"closing_date": "Closing date", "institution_type": ""})
    fig.update_traces(marker=dict(line=dict(width=2, color="white")))
    st.plotly_chart(style(fig, "Tender closing calendar (bubble size = quantity)", height=360), width="stretch", config=config())
    cols = ["tender_id", "institution", "institution_type", "portal", "project_location", "product_category", "quantity_t",
            "closing_date", "days_to_close", "p_win", "contract_value_inr", "contribution_margin_pct", "risk_adjusted_margin_pct",
            "risk_adjusted_ev_inr", "opportunity_rank_score", "recommendation"]
    show_table(display_frame(v[cols], tonnes=["quantity_t"], dates=["closing_date"], pct=["p_win", "contribution_margin_pct",
                                                                                           "risk_adjusted_margin_pct"],
                             inr=["contract_value_inr", "risk_adjusted_ev_inr"],
                             rename={"tender_id": "Tender", "institution": "Institution", "institution_type": "Type",
                                     "portal": "Portal", "project_location": "Location", "product_category": "Product",
                                     "quantity_t": "Quantity", "closing_date": "Closes", "days_to_close": "Days left",
                                     "p_win": "Win probability", "contract_value_inr": "Contract value",
                                     "contribution_margin_pct": "Margin", "risk_adjusted_margin_pct": "Risk-adj. margin",
                                     "risk_adjusted_ev_inr": "Risk-adj. EV", "opportunity_rank_score": "Rank score",
                                     "recommendation": "Recommendation"}),
               status_cols=["Recommendation"], height=420)
    download_csv(v[cols], "tender_pipeline.csv", "Download tender pipeline (CSV)", key="dl_tp")
    assumptions(["Rank score = 0.5 x normalised risk-adjusted EV + 0.2 x win probability + 0.15 x strategic flag + 0.15 x (1 - eligibility risk).",
                 "Risk-adjusted value deducts expected credit loss (PD x LGD x average exposure), financing of the payment period, "
                 "performance-guarantee bank charges, expected liquidated damages and EMD carrying cost.",
                 "Quotes, market prices and freight are simulated assumptions."])

with t2:
    if sv.empty:
        note("No open tenders in the selected regions.")
    else:
        tid = st.selectbox("Select tender", sv["tender_id"].tolist(), key="p_td_sel",
                           format_func=lambda t: f"{t} - {sv.set_index('tender_id').loc[t, 'institution']}")
        t = sv.set_index("tender_id").loc[tid].to_dict()
        t["tender_id"] = tid
        doc = env.tender_documents[tid]
        ext = extract_tender_fields(doc)
        a, b = st.columns(2)
        with a:
            section("Source document (synthetic)", "Retrieved evidence: the text the extraction ran on.")
            st.code(doc, language=None)
            st.download_button("Download as PDF (for upload testing)", text_to_pdf_bytes(doc, tid), f"{tid}.pdf",
                               "application/pdf", key="dl_tpdf")
        with b:
            section("Extracted fields", "Deterministic regular-expression extraction with evidence snippets.")
            rows = [(f, str(getattr(ext, f)), ext.evidence.get(f, "")) for f in
                    ["tender_id", "issuing_authority", "quantity_t", "cement_requirement", "product_category", "delivery_months",
                     "closing_date", "emd_inr", "performance_guarantee_pct", "payment_terms_days", "penalty_pct_per_week",
                     "penalty_cap_pct", "min_turnover_cr", "min_experience_years"]]
            show_table(pd.DataFrame(rows, columns=["Field", "Extracted value", "Evidence"]), height=380)
            st.markdown("**Mandatory documents:** " + ", ".join(ext.required_documents))
            if ext.missing_fields:
                st.markdown(badge("NEEDS REVIEW") + " Fields not found: " + ", ".join(ext.missing_fields), unsafe_allow_html=True)
        checklist = eligibility_checklist({**t, "penalty_pct_per_week": ext.penalty_pct_per_week,
                                           "penalty_cap_pct": ext.penalty_cap_pct}, env.company_documents)
        section("Eligibility checklist and missing-document alerts")
        show_table(checklist, status_cols=["Status"])
        kpi_cards([("Win probability", fmt_pct(t["p_win"]), "Model estimate"),
                   ("Contract value", fmt_inr(t["contract_value_inr"]), fmt_t(t["quantity_t"])),
                   ("Contribution if won", fmt_inr(t["contribution_if_won_inr"]), fmt_pct(t["contribution_margin_pct"])),
                   ("Risk-adjusted if won", fmt_inr(t["risk_adjusted_if_won_inr"]), fmt_pct(t["risk_adjusted_margin_pct"])),
                   ("Risk-adjusted EV", fmt_inr(t["risk_adjusted_ev_inr"]), "Probability-weighted")])
        st.markdown(f"**Recommendation:** {badge(t['recommendation'])} {t['recommendation_reason']}", unsafe_allow_html=True)
        risk = pd.DataFrame({"Component": ["Expected credit loss", "Financing of payment period", "Guarantee charges",
                                           "Expected penalties", "EMD carrying cost"],
                             "INR": [t["ecl_inr"], t["financing_cost_inr"], t["guarantee_cost_inr"], t["expected_penalty_inr"],
                                     t["emd_carrying_cost_inr"]]})
        fig = px.bar(risk, x="INR", y="Component", orientation="h", labels={"Component": ""})
        fig.update_traces(marker_color=SERIES[1])
        st.plotly_chart(style(fig, "Risk and cost deductions (INR)", height=260, legend=False), width="stretch", config=config())
        brief = tender_briefing(t, checklist)
        section("Tender briefing (generated from extracted fields and simulated data)")
        st.code(brief, language=None)
        a, b = st.columns(2)
        with a:
            st.download_button("Download briefing (TXT)", brief.encode("utf-8"), f"{tid}_briefing.txt", "text/plain", key="dl_brief")
        with b:
            approval_button("Tender Intelligence", tid, f"{t['recommendation']}: {t['recommendation_reason']}",
                            "Bid Committee", key="ap_tender", label="Submit bid / no-bid to bid committee")

with t3:
    section("Upload a synthetic tender (TXT or PDF, max 2 MB)",
            "Content is parsed as data only; it is never executed or treated as instructions.")
    up = st.file_uploader("Tender document", type=["txt", "pdf"], key="p_td_upload")
    if up is not None:
        try:
            text = read_document(up.name, up.getvalue())
            ext = extract_tender_fields(text)
            st.success(f"Extracted {len(ext.model_dump(exclude_none=True)) - 3} fields from {up.name}.")
            fields = ext.model_dump()
            ev = fields.pop("evidence")
            missing = fields.pop("missing_fields")
            show_table(pd.DataFrame([(k2, str(v2), ev.get(k2, "")) for k2, v2 in fields.items()],
                                    columns=["Field", "Extracted value", "Evidence"]))
            if missing:
                st.markdown(badge("NEEDS REVIEW") + " Missing or unreadable: " + ", ".join(missing), unsafe_allow_html=True)
            with st.expander("Review and correct extracted values before scoring", expanded=bool(missing)):
                c1, c2, c3, c4 = st.columns(4)
                qty = c1.number_input("Quantity (t)", 0.0, 1e7, float(ext.quantity_t or 1000.0), key="up_qty")
                cat = c2.selectbox("Product category", ["OPC", "PPC", "PSC", "Composite", "Specialty"],
                                   index=["OPC", "PPC", "PSC", "Composite", "Specialty"].index(ext.product_category or "PPC"), key="up_cat")
                pay = c3.number_input("Payment terms (days)", 0, 365, int(ext.payment_terms_days or 60), key="up_pay")
                months = c4.number_input("Delivery months", 1, 60, int(ext.delivery_months or 6), key="up_m")
                c1, c2, c3, c4 = st.columns(4)
                quote = c1.number_input("Our quote (INR / t)", 3000.0, 15000.0, 6900.0, 10.0, key="up_quote")
                market = c2.number_input("Estimated market price (INR / t)", 3000.0, 15000.0, 6900.0, 10.0, key="up_mkt")
                dist = c3.number_input("Distance from nearest capable plant (km)", 0.0, 3000.0, 300.0, key="up_dist")
                bidders = c4.number_input("Expected bidders", 1, 30, 5, key="up_bid")
            from src.services.freight_tariff import secondary_freight_per_t
            tdict = {"tender_id": ext.tender_id or up.name, "institution": ext.issuing_authority or "Unknown",
                     "institution_type": "State PWD / Municipal", "portal": ext.portal or "n/a",
                     "project_location": ext.project_location or "n/a", "quantity_t": qty, "product_category": cat,
                     "product_requirement": ext.cement_requirement or cat, "delivery_months": months,
                     "closing_date": pd.Timestamp(ext.closing_date) if ext.closing_date else None,
                     "min_turnover_cr": ext.min_turnover_cr, "min_experience_years": ext.min_experience_years,
                     "required_documents": ext.required_documents, "emd_inr": ext.emd_inr or 0.0,
                     "performance_guarantee_pct": ext.performance_guarantee_pct or 5.0,
                     "penalty_pct_per_week": ext.penalty_pct_per_week, "penalty_cap_pct": ext.penalty_cap_pct,
                     "payment_terms_days": pay, "customer_payment_score": 0.7, "estimated_market_price_per_t": market,
                     "our_quote_per_t": quote, "price_index_vs_market": quote / market, "distance_km": dist,
                     "estimated_freight_per_t": float(secondary_freight_per_t(dist + 15)), "delivery_capability_ratio": 1.0,
                     "eligibility_score": 0.9, "documents_complete": True, "expected_bidders": bidders, "strategic_account": False}
            chk = eligibility_checklist(tdict, env.company_documents)
            tdict["eligibility_score"] = float((chk["Status"] == "PASS").mean())
            tdict["documents_complete"] = not chk[chk["Criterion"].str.startswith("Document")]["Status"].ne("PASS").any()
            p = float(tm.model.predict_proba(_design(pd.DataFrame([tdict]), tm.columns))[:, 1][0])
            cost = float(env.products.groupby("category")["base_cost"].mean()[cat])
            econ = tender_economics(tdict, p, cost, 0.3 * 0.18, LGD_BY_CLASS["Government"], mm)
            rec, why = bid_recommendation(chk, econ, p, False)
            tdict.update(econ, p_win=p, recommendation=rec, recommendation_reason=why)
            show_table(chk, status_cols=["Status"])
            kpi_cards([("Win probability", fmt_pct(p), "Model estimate; unknown fields use stated defaults"),
                       ("Risk-adjusted EV", fmt_inr(econ["risk_adjusted_ev_inr"]), "Illustrative"),
                       ("Risk-adjusted margin", fmt_pct(econ["risk_adjusted_margin_pct"]), f"Threshold {fmt_pct(mm)}")])
            st.markdown(f"**Recommendation:** {badge(rec)} {why}", unsafe_allow_html=True)
            st.code(tender_briefing(tdict, chk), language=None)
        except DocumentError as exc:
            st.error(f"Document could not be processed: {exc}")
    else:
        note("Tip: download a sample PDF from the 'Tender evaluation and briefing' tab and upload it here.")

with t4:
    section("Tender win-probability model", f"{tm.approach}. Chronological split: trained on {tm.train_size} tenders closing before "
            f"{fmt_date(tm.split_date)}, evaluated on the {tm.test_size} later tenders.")
    met = tm.metrics.reset_index().rename(columns={"index": "Dataset"})
    show_table(display_frame(met[["Dataset", "ROC-AUC", "Precision", "Recall", "F1", "Brier score", "Positive rate (actual)"]],
                             pct=["Precision", "Recall", "Positive rate (actual)"], dec1=[]).round(3))
    a, b = st.columns(2)
    cal = tm.calibration
    fig = px.line(cal, x="mean_predicted", y="observed_rate", markers=True,
                  labels={"mean_predicted": "Mean predicted win probability", "observed_rate": "Observed win rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#898781", dash="dot"))
    fig.update_traces(line_color=PRIMARY, marker=dict(size=9))
    a.plotly_chart(style(fig, "Calibration on holdout (dotted = perfect)", legend=False), width="stretch", config=config())
    co = tm.coefficients.head(12).sort_values("coefficient")
    fig = px.bar(co, x="coefficient", y="feature", orientation="h", labels={"coefficient": "Standardised coefficient", "feature": ""})
    fig.update_traces(marker_color=PRIMARY)
    b.plotly_chart(style(fig, "Drivers of win probability", legend=False), width="stretch", config=config())
    assumptions([f"Excluded post-outcome fields (target-leakage control): {', '.join(EXCLUDED_POST_OUTCOME)}.",
                 "Features are attributes known at bid time: price position, eligibility, documents, delivery capability, distance, "
                 "competition, strategic flag, payment terms, quantity, customer payment score, guarantee and penalty terms.",
                 "The model is refitted on all history before scoring open tenders.",
                 "Probabilities come from logistic regression on simulated outcomes; with real data they must be re-validated."],
                "Model design")
