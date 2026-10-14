"""Tender win-probability model, eligibility checks, economics and bid / no-bid logic.

Leakage control: only attributes known at bid-submission time are used. Post-outcome fields
(L1 price, our rank, award date, outcome) are explicitly excluded. Evaluation uses a
chronological split by closing date.

All recommendations are advisory. The platform never submits a bid or makes a commitment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ANNUAL_FINANCING_RATE, PLANT_HANDLING_PER_T
from src.models.metrics import calibration_table, classification_metrics

NUMERIC_FEATURES = ["price_index_vs_market", "eligibility_score", "documents_complete", "delivery_capability_ratio",
                    "distance_km", "expected_bidders", "strategic_account", "payment_terms_days", "log_quantity",
                    "customer_payment_score", "performance_guarantee_pct", "penalty_pct_per_week"]
CATEGORICAL_FEATURES = ["institution_type", "product_category"]
EXCLUDED_POST_OUTCOME = ["outcome", "won", "l1_price_per_t", "our_rank", "award_date"]
COMPANY_TURNOVER_CR = 950.0
COMPANY_EXPERIENCE_YEARS = 15
BANK_GUARANTEE_FEE_PA = 0.015


def _design(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = df.copy()
    X["log_quantity"] = np.log1p(X["quantity_t"].astype(float))
    X["documents_complete"] = X["documents_complete"].astype(float)
    X["strategic_account"] = X["strategic_account"].astype(float)
    X["delivery_capability_ratio"] = X["delivery_capability_ratio"].clip(upper=3)
    D = pd.get_dummies(X[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float)
    if columns is not None:
        D = D.reindex(columns=columns, fill_value=0.0)
    return D.astype(float)


@dataclass
class TenderModelResult:
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    coefficients: pd.DataFrame
    columns: list[str]
    model: object
    train_size: int
    test_size: int
    split_date: pd.Timestamp
    approach: str


def train_tender_model(env, test_share: float = 0.25) -> TenderModelResult:
    hist = env.tenders_hist.sort_values("closing_date")
    n_test = max(int(len(hist) * test_share), 1)
    train, test = hist.iloc[:-n_test], hist.iloc[-n_test:]
    if len(train) < 80 or train["won"].nunique() < 2:
        raise ValueError("Insufficient tender history; use rules-based opportunity score instead.")
    X_tr = _design(train)
    cols = list(X_tr.columns)
    X_te = _design(test, cols)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    model.fit(X_tr, train["won"])
    p = model.predict_proba(X_te)[:, 1]
    p_tr = model.predict_proba(X_tr)[:, 1]
    metrics = pd.DataFrame({"Training (in-sample)": classification_metrics(train["won"], p_tr),
                            "Holdout (later tenders)": classification_metrics(test["won"], p)}).T
    calib = calibration_table(test["won"], p, n_bins=5)
    lr = model.named_steps["logisticregression"]
    coefs = pd.DataFrame({"feature": cols, "coefficient": lr.coef_[0]}).sort_values(
        "coefficient", key=np.abs, ascending=False)
    # Refit on all history for scoring open tenders
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    final.fit(_design(hist, cols), hist["won"])
    return TenderModelResult(metrics=metrics, calibration=calib, coefficients=coefs, columns=cols, model=final,
                             train_size=len(train), test_size=len(test), split_date=test["closing_date"].min(),
                             approach="Logistic regression trained on simulated historical tender outcomes")


def eligibility_checklist(tender: dict, company_docs: pd.DataFrame) -> pd.DataFrame:
    """Criterion-level eligibility check with explicit PASS / FAIL / NEEDS REVIEW outcomes."""
    rows = []
    turnover = tender.get("min_turnover_cr")
    if turnover is None:
        rows.append(("Minimum turnover", "Not found in document", "NEEDS REVIEW", "Extraction"))
    else:
        ok = COMPANY_TURNOVER_CR >= turnover
        rows.append(("Minimum turnover", f"Required Rs. {turnover:,.0f} Cr; company (synthetic) Rs. {COMPANY_TURNOVER_CR:,.0f} Cr",
                     "PASS" if ok else "FAIL", "Company profile (synthetic)"))
    exp = tender.get("min_experience_years")
    if exp is None:
        rows.append(("Supply experience", "Not found in document", "NEEDS REVIEW", "Extraction"))
    else:
        ok = COMPANY_EXPERIENCE_YEARS >= exp
        rows.append(("Supply experience", f"Required {exp} years; company {COMPANY_EXPERIENCE_YEARS} years",
                     "PASS" if ok else "FAIL", "Company profile (synthetic)"))
    reg = company_docs.set_index("document")
    for doc in tender.get("required_documents", []) or []:
        match = next((d for d in reg.index if d.lower() in doc.lower() or doc.lower() in d.lower()), None)
        if match is None:
            rows.append((f"Document: {doc}", "Not held in document repository", "FAIL", "Document repository"))
            continue
        status = reg.loc[match, "status"]
        res = {"Valid": "PASS", "Valid (no expiry)": "PASS", "Expiring within 30 days": "NEEDS REVIEW",
               "Expired": "FAIL"}.get(status, "NEEDS REVIEW")
        exp_date = reg.loc[match, "expiry_date"]
        detail = status if pd.isna(exp_date) else f"{status} (expiry {pd.Timestamp(exp_date).strftime('%d-%b-%Y')})"
        rows.append((f"Document: {doc}", detail, res, reg.loc[match, "repository_ref"]))
    cap = tender.get("delivery_capability_ratio")
    if cap is not None:
        # Availability is a delivery risk (priced through expected penalties), not an eligibility criterion.
        res = "PASS" if cap >= 1.0 else "NEEDS REVIEW"
        detail = f"Coverage ratio {cap:.2f}" + ("" if cap >= 1.0 else " - production / allocation plan change required")
        rows.append(("Product availability vs monthly requirement", detail, res, "Inventory snapshot (synthetic)"))
    if tender.get("penalty_pct_per_week") is None:
        rows.append(("Penalty clause", "Liquidated damages clause not found", "NEEDS REVIEW", "Extraction"))
    else:
        rows.append(("Penalty clause", f"{tender['penalty_pct_per_week']}% per week, cap {tender.get('penalty_cap_pct', 'n/a')}%",
                     "PASS" if tender["penalty_pct_per_week"] <= 1.5 else "NEEDS REVIEW", "Extraction"))
    return pd.DataFrame(rows, columns=["Criterion", "Detail", "Status", "Evidence source"])


def tender_economics(tender: dict, p_win: float, production_cost_per_t: float, pd_customer: float,
                     lgd: float, min_margin_pct: float = 0.08) -> dict:
    """Expected and risk-adjusted value of bidding (all figures illustrative)."""
    qty = float(tender["quantity_t"])
    price = float(tender["our_quote_per_t"])
    freight = float(tender["estimated_freight_per_t"])
    months = max(int(tender.get("delivery_months") or 6), 1)
    revenue = qty * price
    contribution = qty * (price - production_cost_per_t - freight - PLANT_HANDLING_PER_T)
    pay_days = float(tender.get("payment_terms_days") or 60)
    avg_exposure = revenue * min(pay_days / (months * 30.0), 1.0)
    ecl = pd_customer * lgd * avg_exposure
    financing = revenue * ANNUAL_FINANCING_RATE * pay_days / 365.0
    pg_cost = revenue * float(tender.get("performance_guarantee_pct") or 0) / 100 * BANK_GUARANTEE_FEE_PA * months / 12
    cap_ratio = float(tender.get("delivery_capability_ratio") or 1.0)
    expected_delay_weeks = max(0.0, (1.0 - cap_ratio)) * 4
    penalty_rate = float(tender.get("penalty_pct_per_week") or 1.0) / 100
    penalty_cap = float(tender.get("penalty_cap_pct") or 10) / 100
    expected_penalty = revenue * min(penalty_rate * expected_delay_weeks, penalty_cap)
    emd_cost = float(tender.get("emd_inr") or 0) * ANNUAL_FINANCING_RATE * 2 / 12  # EMD blocked ~2 months
    risk_costs = ecl + financing + pg_cost + expected_penalty
    risk_adj_if_won = contribution - risk_costs
    return {
        "expected_revenue_inr": p_win * revenue,
        "contract_value_inr": revenue,
        "contribution_if_won_inr": contribution,
        "contribution_margin_pct": contribution / revenue if revenue else 0.0,
        "expected_contribution_inr": p_win * contribution,
        "ecl_inr": ecl, "financing_cost_inr": financing, "guarantee_cost_inr": pg_cost,
        "expected_penalty_inr": expected_penalty, "emd_carrying_cost_inr": emd_cost,
        "risk_adjusted_if_won_inr": risk_adj_if_won,
        "risk_adjusted_margin_pct": risk_adj_if_won / revenue if revenue else 0.0,
        "risk_adjusted_ev_inr": p_win * risk_adj_if_won - emd_cost,
        "min_margin_pct": min_margin_pct,
    }


def bid_recommendation(checklist: pd.DataFrame, econ: dict, p_win: float, strategic: bool) -> tuple[str, str]:
    fails = checklist[checklist["Status"] == "FAIL"]["Criterion"].tolist()
    reviews = checklist[checklist["Status"] == "NEEDS REVIEW"]["Criterion"].tolist()
    if fails:
        return ("NO BID - ELIGIBILITY GAP",
                f"Disqualification risk on: {', '.join(fails)}. Resolve before considering a bid.")
    if econ["risk_adjusted_margin_pct"] < econ["min_margin_pct"]:
        return ("NO BID - INSUFFICIENT RISK-ADJUSTED MARGIN",
                f"Risk-adjusted margin {econ['risk_adjusted_margin_pct']:.1%} is below the "
                f"{econ['min_margin_pct']:.0%} threshold after credit, financing, guarantee and penalty costs.")
    if p_win < 0.25 and not strategic:
        return ("NO BID - LOW WIN PROBABILITY",
                f"Estimated win probability {p_win:.0%} gives low expected value relative to bid effort.")
    if p_win < 0.25 and strategic:
        return ("BID - STRATEGIC, REVIEW PRICING",
                f"Strategic account but estimated win probability is only {p_win:.0%}; review price position.")
    if reviews:
        return ("BID WITH CONDITIONS", f"Commercially attractive; confirm before submission: {', '.join(reviews)}.")
    return ("BID", f"Eligible with risk-adjusted margin {econ['risk_adjusted_margin_pct']:.1%} and "
                   f"win probability {p_win:.0%}.")


def score_open_tenders(env, model_res: TenderModelResult, credit_scored: pd.DataFrame,
                       min_margin_pct: float = 0.08, pd_conversion: float = 0.18) -> pd.DataFrame:
    """Score, evaluate and rank all open tenders."""
    from src.models.credit_risk import LGD_BY_CLASS

    open_ = env.tenders_open.copy()
    open_["p_win"] = model_res.model.predict_proba(_design(open_, model_res.columns))[:, 1]
    cost = env.products.groupby("category")["base_cost"].mean()
    cs = credit_scored.set_index("customer_id")
    rows = []
    for t in open_.to_dict("records"):
        checklist = eligibility_checklist(t, env.company_documents)
        p_late = cs["p_late_90d"].get(t["customer_id"], np.nan)
        p_late = 0.3 if pd.isna(p_late) else float(p_late)
        lgd = LGD_BY_CLASS.get(cs["lgd_class"].get(t["customer_id"], "Corporate"), 0.35)
        econ = tender_economics(t, t["p_win"], float(cost.get(t["product_category"], 4300)),
                                p_late * pd_conversion, lgd, min_margin_pct)
        rec, why = bid_recommendation(checklist, econ, t["p_win"], bool(t["strategic_account"]))
        rows.append({**t, **econ, "recommendation": rec, "recommendation_reason": why,
                     "checks_failed": int((checklist["Status"] == "FAIL").sum()),
                     "checks_review": int((checklist["Status"] == "NEEDS REVIEW").sum()),
                     "days_to_close": (pd.Timestamp(t["closing_date"]) - env.as_of).days})
    df = pd.DataFrame(rows)
    ev = df["risk_adjusted_ev_inr"]
    ev_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)
    risk = (df["checks_failed"] * 0.5 + df["checks_review"] * 0.15).clip(0, 1)
    df["opportunity_rank_score"] = (0.5 * ev_norm + 0.2 * df["p_win"] + 0.15 * df["strategic_account"].astype(float)
                                    + 0.15 * (1 - risk)).round(3)
    return df.sort_values("opportunity_rank_score", ascending=False).reset_index(drop=True)


def tender_briefing(t: dict, checklist: pd.DataFrame) -> str:
    """Deterministic, template-based tender briefing (generated text, grounded in extracted fields)."""
    closing = pd.Timestamp(t["closing_date"]).strftime("%d-%b-%Y") if t.get("closing_date") is not None else "not found"
    fails = checklist[checklist["Status"] == "FAIL"]["Criterion"].tolist()
    reviews = checklist[checklist["Status"] == "NEEDS REVIEW"]["Criterion"].tolist()
    lines = [
        f"TENDER BRIEFING - {t.get('tender_id', 'Unknown')} (SIMULATED RESULT, FOR HUMAN REVIEW)",
        f"Institution: {t.get('institution', 'n/a')} | Portal: {t.get('portal', 'n/a')} | Location: {t.get('project_location', 'n/a')}",
        f"Requirement: {t.get('quantity_t', 0):,.0f} t of {t.get('product_requirement', 'n/a')} over {t.get('delivery_months', 'n/a')} months.",
        f"Closing date: {closing}. Payment terms: {t.get('payment_terms_days', 'n/a')} days.",
        f"Win probability (model estimate): {t.get('p_win', 0):.0%}. Contract value: Rs. {t.get('contract_value_inr', 0):,.0f}.",
        f"Contribution if won: Rs. {t.get('contribution_if_won_inr', 0):,.0f} ({t.get('contribution_margin_pct', 0):.1%}); "
        f"risk-adjusted: Rs. {t.get('risk_adjusted_if_won_inr', 0):,.0f} ({t.get('risk_adjusted_margin_pct', 0):.1%}).",
        f"Risk-adjusted expected value of bidding: Rs. {t.get('risk_adjusted_ev_inr', 0):,.0f}.",
        f"Eligibility failures: {', '.join(fails) if fails else 'none identified'}.",
        f"Items needing review: {', '.join(reviews) if reviews else 'none'}.",
        f"Recommendation: {t.get('recommendation', 'n/a')} - {t.get('recommendation_reason', '')}",
        "This briefing is generated from extracted fields and simulated data. Bid decisions, pricing and "
        "submission remain with authorised personnel.",
    ]
    return "\n".join(lines)
