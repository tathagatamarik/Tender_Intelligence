"""AI Recommended Actions: generated from module outputs, never from a hard-coded list.

Each generator inspects one analytical result and emits zero or more recommendations with a
business unit, reason, estimated financial impact, confidence, required action and priority.
Annualisation factors and confidence levels are explicit assumptions shown in the UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.config import WORKING_DAYS_PER_YEAR


@dataclass
class Recommendation:
    title: str
    business_unit: str
    reason: str
    impact_inr: float
    impact_basis: str
    confidence: str          # HIGH / MEDIUM / LOW
    confidence_score: float  # 0-1, used for ranking
    action: str
    priority: str = ""
    source_module: str = ""
    value_type: str = "Contribution"  # Cost saving | Contribution | Exposure (not profit) | Revenue at risk


def _priority(impact: float, conf: float, urgent: bool = False) -> str:
    score = impact * conf
    if urgent or score >= 2_500_000:
        return "HIGH PRIORITY"
    if score >= 500_000:
        return "MEDIUM PRIORITY"
    return "LOW PRIORITY"


def from_replenishment(rep: pd.DataFrame | None, seg: pd.DataFrame | None) -> list[Recommendation]:
    if rep is None or rep.empty:
        return []
    out = []
    risky = rep[rep["stockout_flag"] == "LIKELY STOCKOUT"].merge(seg[["dealer_id", "region"]], on="dealer_id", how="left")
    for region, g in risky.groupby("region"):
        val = float(g["protected_contribution_inr"].sum())
        if val <= 0:
            continue
        out.append(Recommendation(
            title=f"Prioritise replenishment for {g['dealer_id'].nunique()} dealers at stock-out risk in {region}",
            business_unit="Trade Sales", reason=f"Projected stock below lead-time demand for {len(g)} dealer-product lines; "
                                                 f"recommended {g['recommended_qty_t'].sum():,.0f} t of orders.",
            impact_inr=val, impact_basis="Contribution protected over the replenishment cover period",
            confidence="MEDIUM", confidence_score=0.6, action="Sales reps to confirm recommended orders with dealers",
            source_module="Dealer Demand and Replenishment"))
    return out


def from_segments(vp: pd.DataFrame | None) -> list[Recommendation]:
    if vp is None or vp.empty:
        return []
    dec = vp[vp["segment"].isin(["Declining", "At-risk"]) & (vp["recovery_opportunity_inr"] > 0)]
    if dec.empty:
        return []
    top = dec.head(15)
    return [Recommendation(
        title=f"Schedule visits to the {len(top)} highest-value declining or at-risk dealers",
        business_unit="Trade Sales", reason=f"{len(dec)} dealers show year-on-year volume decline or payment stress; "
                                            f"top names include {', '.join(top['dealer_name'].head(3))}.",
        impact_inr=float(top["recovery_opportunity_inr"].sum()), impact_basis="Assumes 30% of the quarterly volume gap is recoverable",
        confidence="LOW", confidence_score=0.4, action="Regional managers to assign visits this week",
        source_module="Dealer Demand and Replenishment")]


def from_freight(plan_result) -> list[Recommendation]:
    if plan_result is None:
        return []
    plan = plan_result.plan
    both = plan[plan["base_source_id"].notna() & plan["opt_source_id"].notna()]
    saving = float((both["base_cost_inr"] - both["opt_cost_inr"]).sum())
    out = []
    if saving > 0:
        annual = saving / (plan_result.scenario.plan_hours / 24) * WORKING_DAYS_PER_YEAR
        out.append(Recommendation(
            title=f"Adopt optimised dispatch plan ({(both['status'] != 'UNCHANGED').sum()} orders re-sourced or re-vehicled)",
            business_unit="Logistics", reason=f"Plan saves Rs. {saving:,.0f} on this order book versus current practice while meeting deadlines.",
            impact_inr=annual, impact_basis="Plan saving annualised over 300 working days (illustrative; order books vary)",
            confidence="MEDIUM", confidence_score=0.55, action="Dispatch planner to review and approve plan",
            source_module="Freight and Dispatch Optimisation", value_type="Cost saving"))
    manual = plan[plan["opt_source_id"].isna()]
    if len(manual):
        out.append(Recommendation(
            title=f"Resolve {len(manual)} orders with no feasible dispatch option",
            business_unit="Logistics", reason="No source can meet the deadline with available stock and fleet.",
            impact_inr=float((manual["qty_t"] * (manual["list_price_per_t"] - manual["discount_per_t"])).sum()),
            impact_basis="Net revenue of affected orders at risk", confidence="HIGH", confidence_score=0.8,
            action="Negotiate revised delivery date or authorise exceptional dispatch", priority="HIGH PRIORITY",
            source_module="Freight and Dispatch Optimisation", value_type="Revenue at risk"))
    return out


def from_credit(policy: pd.DataFrame | None) -> list[Recommendation]:
    if policy is None or policy.empty:
        return []
    out = []
    coll = policy[policy["recommended_action"] == "COLLECTION INTERVENTION"]
    if len(coll):
        out.append(Recommendation(
            title=f"Collection intervention for {len(coll)} high-risk accounts",
            business_unit="Finance", reason=f"Overdue Rs. {coll['overdue_inr'].sum():,.0f}; high predicted late-payment risk.",
            impact_inr=float(coll["ecl_inr"].sum()), impact_basis="Expected credit loss on these accounts (simplified PD x LGD x EAD)",
            confidence="MEDIUM", confidence_score=0.65, action="Finance to initiate collection calls and review terms",
            source_module="Credit Risk and Profitable Orders", value_type="Exposure (not profit)"))
    esc = policy[policy["recommended_action"].isin(["ESCALATE ORDER RELEASE TO FINANCE", "CREDIT REVIEW - EXPOSURE ABOVE LIMIT"])]
    if len(esc):
        out.append(Recommendation(
            title=f"Finance approval required before releasing orders for {len(esc)} customers above credit limit",
            business_unit="Finance", reason="Exposure after pending orders exceeds the approved credit limit.",
            impact_inr=float(esc["ecl_inr"].sum()), impact_basis="Expected credit loss at stake",
            confidence="HIGH", confidence_score=0.8, action="Hold order release pending finance review",
            priority="HIGH PRIORITY", source_module="Credit Risk and Profitable Orders", value_type="Exposure (not profit)"))
    return out


def from_tenders(scored: pd.DataFrame | None) -> list[Recommendation]:
    if scored is None or scored.empty:
        return []
    out = []
    bids = scored[scored["recommendation"].str.startswith("BID")]
    for t in bids.head(3).itertuples():
        out.append(Recommendation(
            title=f"Prepare bid for {t.tender_id} ({t.quantity_t:,.0f} t, closes in {t.days_to_close} days)",
            business_unit="Institutional Sales", reason=t.recommendation_reason,
            impact_inr=float(max(t.risk_adjusted_ev_inr, 0)), impact_basis="Risk-adjusted expected value of bidding",
            confidence="MEDIUM" if t.p_win >= 0.4 else "LOW", confidence_score=0.5 if t.p_win >= 0.4 else 0.35,
            action="Bid committee to review pricing and documents", priority="HIGH PRIORITY" if t.days_to_close <= 7 else "",
            source_module="Tender Intelligence"))
    gaps = scored[scored["recommendation"] == "NO BID - ELIGIBILITY GAP"]
    if len(gaps):
        out.append(Recommendation(
            title=f"Close document gaps blocking {len(gaps)} tenders",
            business_unit="Institutional Sales", reason="Expired or missing mandatory documents create disqualification risk.",
            impact_inr=float(gaps["expected_contribution_inr"].clip(lower=0).sum()),
            impact_basis="Expected contribution of blocked tenders if gaps were resolved", confidence="LOW",
            confidence_score=0.3, action="Renew expired certificates via compliance team", source_module="Tender Intelligence"))
    return out


def from_planning(planning) -> list[Recommendation]:
    if planning is None:
        return []
    b, o = planning.baseline, planning.optimised
    saving = b["Total operating cost (INR)"] - o["Total operating cost (INR)"]
    if saving <= 0:
        return []
    weeks = planning.demand["horizon"].max()
    return [Recommendation(
        title="Adopt optimised production and depot allocation plan",
        business_unit="Supply Chain", reason=f"Fill rate {o['Service level (fill rate %)']:.1f}% vs {b['Service level (fill rate %)']:.1f}% "
                                             f"under current practice; stock-outs reduced by {b['Stockout quantity (t)'] - o['Stockout quantity (t)']:,.0f} t.",
        impact_inr=float(saving * 52 / weeks), impact_basis=f"{weeks}-week plan saving annualised (illustrative)",
        confidence="MEDIUM", confidence_score=0.5, action="Supply planning to review week-1 production and shipments",
        source_module="Production and Inventory Planning", value_type="Cost saving")]


def from_offtake(off) -> list[Recommendation]:
    if off is None:
        return []
    act = off.projects[off.projects["alert_level"] == "ACTION REQUIRED"]
    if act.empty:
        return []
    return [Recommendation(
        title=f"Follow up {len(act)} projects with offtake shortfall warnings",
        business_unit="Institutional Sales", reason=f"Estimated {act['at_risk_t'].sum():,.0f} t at risk over the next 3 months (estimate, not confirmed cancellations).",
        impact_inr=float(act["working_capital_impact_inr"].sum()), impact_basis="Working capital tied in at-risk inventory (not profit)",
        confidence="MEDIUM", confidence_score=0.6, action="Sales to confirm schedules; planning to hold dedicated production",
        source_module="Project Offtake Early Warning", value_type="Exposure (not profit)")]


def from_pricing(opps: pd.DataFrame | None) -> list[Recommendation]:
    if opps is None or opps.empty:
        return []
    top = opps[opps["monthly_contribution_gain_inr"] > 0].head(5)
    if top.empty:
        return []
    return [Recommendation(
        title=f"Review discount levels in {len(top)} micro-markets within policy",
        business_unit="Pricing", reason="Scenario model indicates higher contribution at alternative discount levels within policy limits "
                                         "(correlational elasticity estimate; not causal).",
        impact_inr=float(top["monthly_contribution_gain_inr"].sum() * 12), impact_basis="Monthly scenario gain x 12 (assumption)",
        confidence="LOW", confidence_score=0.3, action="Pricing committee review; requires management approval",
        source_module="Pricing and Discount Intelligence")]


def generate_recommendations(results: dict) -> pd.DataFrame:
    recs: list[Recommendation] = []
    recs += from_replenishment(results.get("replenishment"), results.get("segments"))
    recs += from_segments(results.get("visit_priority"))
    recs += from_freight(results.get("freight"))
    recs += from_credit(results.get("credit_policy"))
    recs += from_tenders(results.get("tenders"))
    recs += from_planning(results.get("planning"))
    recs += from_offtake(results.get("offtake"))
    recs += from_pricing(results.get("pricing_opps"))
    for r in recs:
        if not r.priority:
            r.priority = _priority(r.impact_inr, r.confidence_score)
    df = pd.DataFrame([asdict(r) for r in recs])
    if df.empty:
        return df
    df["rank_score"] = df["impact_inr"] * df["confidence_score"] + df["priority"].map(
        {"HIGH PRIORITY": 1e12, "MEDIUM PRIORITY": 1e6, "LOW PRIORITY": 0})
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.drop(columns=["rank_score"])
