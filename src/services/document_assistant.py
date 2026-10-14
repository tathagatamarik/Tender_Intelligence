"""Offline sales knowledge assistant: synthetic knowledge base, TF-IDF retrieval, grounded answers.

* Documents are generated from the shared synthetic environment and analytics results.
* Documents are chunked and tagged with metadata (category, sensitivity, source).
* Retrieval uses TF-IDF cosine similarity; results are filtered by simulated role permissions.
* Answers are composed by deterministic templates ONLY from retrieved excerpts, and every
  answer lists the exact chunks it used. If nothing relevant is retrieved, the assistant says so.
* ``AnswerGenerator`` is the adapter interface for a future approved enterprise / local LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import MAX_DISCOUNT_PER_T_POLICY, MIN_MARGIN_PCT_POLICY

PUBLIC, INTERNAL, RESTRICTED = "Public", "Internal", "Restricted - Finance"
ROLE_PERMISSIONS = {
    "Sales Representative": {PUBLIC, INTERNAL},
    "Logistics Planner": {PUBLIC, INTERNAL},
    "Regional Sales Manager": {PUBLIC, INTERNAL},
    "Finance Manager": {PUBLIC, INTERNAL, RESTRICTED},
}
MIN_RELEVANCE = 0.08


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    sensitivity: str
    text: str


@dataclass
class Answer:
    question: str
    intent: str
    text: str
    sources: list[dict] = field(default_factory=list)
    grounded: bool = False
    withheld_for_permissions: int = 0


class AnswerGenerator(Protocol):
    """Adapter interface: produce an answer from a question and retrieved chunks only."""

    def generate(self, question: str, intent: str, chunks: list[tuple[Chunk, float]]) -> str: ...


# ---------------------------------------------------------------------------
# Knowledge base construction
# ---------------------------------------------------------------------------
def _policy_documents(env) -> list[tuple[str, str, str, str, str]]:
    prods = env.products
    docs = [
        ("POL-DISC", "Trade Discount and Scheme Policy (Simulated)", "Policy", INTERNAL,
         f"Approved discount policy in this simulation: the maximum trade discount is Rs. {MAX_DISCOUNT_PER_T_POLICY:,.0f} "
         f"per tonne including all schemes. Any price or discount proposal must keep contribution margin at or above "
         f"{MIN_MARGIN_PCT_POLICY:.0%} of net revenue.\n\n"
         "Discount tiers by dealer segment (simulated): Key Account up to Rs. 330 per tonne, Large up to Rs. 270, "
         "Medium up to Rs. 210, Small up to Rs. 150, before seasonal schemes.\n\n"
         "Seasonal schemes such as the Monsoon Volume Scheme may add up to Rs. 80 per tonne during July to September, "
         "subject to the overall cap. All pricing and discount changes require regional sales manager and finance "
         "approval. Competitor price information is a simulated market indicator only and must never be used to "
         "coordinate prices."),
        ("POL-CREDIT", "Customer Credit Policy (Simulated)", "Policy", INTERNAL,
         "Orders that take a customer's exposure (outstanding receivables plus pending orders) above the approved "
         "credit limit require finance approval before release.\n\n"
         "Accounts overdue by more than 45 days are placed on credit hold. Accounts with a predicted late-payment "
         "probability above 35 percent are flagged for credit review.\n\n"
         "Expected credit loss in this demo is a simplified PD x LGD x EAD estimate and is not a regulatory "
         "accounting measure. Credit limit changes are never applied automatically."),
        ("SOP-DISP", "Dispatch Planning SOP incl. Green Channel and Stock-on-Wheels (Simulated)", "Logistics", PUBLIC,
         "Dispatch recommendations choose the lowest total fulfilment cost option that meets the delivery deadline, "
         "considering source stock, loading capacity, fleet availability, vehicle size and predicted transit time.\n\n"
         "Green Channel (simulated) pre-clears documentation and gate formalities so plant loading time falls from "
         "about 4 hours to 1.5 hours, improving deadline feasibility for urgent orders.\n\n"
         "Stock-on-Wheels (simulated) pre-positions loaded 25 tonne trucks near demand clusters with high urgent "
         "demand. It shortens delivery time at a holding cost per truck-day. All dispatch plans require planner "
         "approval before execution."),
        ("CHK-TENDER", "Tender Documentation Checklist (Simulated)", "Tender", PUBLIC,
         "Typical mandatory documents for institutional cement tenders in this simulation: BIS Licence, Mill Test "
         "Certificate, Third-Party Test Report, ISO 9001 Certificate, GST Registration, PAN Card, Audited Financial "
         "Statements, Past Supply Experience Certificate, Undertaking on Blacklisting and Power of Attorney.\n\n"
         "Expired documents create a disqualification risk and must be renewed before bid submission. The platform "
         "never submits tenders; bid decisions remain with authorised personnel."),
        ("GUIDE-REPL", "Dealer Replenishment Guideline (Simulated)", "Sales", PUBLIC,
         "Recommended dealer orders cover forecast demand over lead time plus review period, plus safety stock, minus "
         "current stock and open orders, rounded to 5 tonne lots and limited by available credit headroom. Forecasts "
         "are estimates, not guaranteed sales."),
    ]
    spec = env.product_specs.set_index("product_id")
    for _, p in prods.iterrows():
        s = spec.loc[p["product_id"]]
        docs.append((f"PRD-{p['product_id']}", f"Product Note: {p['product_name']} (Synthetic)", "Product", PUBLIC,
                     f"{p['product_name']} ({p['product_id']}), category {p['category']}. Reference standard: "
                     f"{s['standard_reference']}. Declared synthetic values: 28-day compressive strength "
                     f"{s['compressive_strength_28d_mpa']} MPa, total alkali {s['total_alkali_pct']} percent, chloride "
                     f"{s['chloride_pct']} percent. Values are illustrative and do not represent any actual product."))
    return docs


def _data_documents(env, context: dict) -> list[tuple[str, str, str, str, str]]:
    docs = []
    inv = env.inventory.merge(env.products[["product_id", "product_name"]], on="product_id")
    for region, g in inv.groupby("region"):
        lines = [f"Stock availability report for {region} region as of {env.as_of.strftime('%d-%b-%Y')} (simulated)."]
        for (pid, pname), gg in g.groupby(["product_id", "product_name"]):
            locs = ", ".join(f"{r.location_id} {r.available_t:,.0f} t" for r in gg.itertuples())
            lines.append(f"{pname} ({pid}) in {region}: {gg['available_t'].sum():,.0f} t available ({locs}).")
        docs.append((f"STK-{region}", f"Stock Availability - {region}", "Inventory", INTERNAL, "\n".join(lines)))

    seg = context.get("segments")
    if seg is not None and len(seg):
        dec = seg[seg["segment"].isin(["Declining", "At-risk"])].sort_values("growth_yoy").head(25)
        for region, g in dec.groupby("region"):
            lines = [f"Dealers with declining orders this month in {region} region (simulated analysis):"]
            for r in g.itertuples():
                lines.append(f"{r.dealer_name} ({r.dealer_id}), {r.district}: volume {r.growth_yoy:+.0%} year-on-year; "
                             f"segment {r.segment}; reason: {r.segment_reason}.")
            docs.append((f"DEC-{region}", f"Declining Dealer Report - {region}", "Sales", INTERNAL, "\n".join(lines)))

    credit = context.get("credit")
    if credit is not None and len(credit):
        rev = credit[credit["recommended_action"] != "WITHIN LIMIT"].sort_values("ecl_inr", ascending=False).head(30)
        lines = ["Customers requiring credit review or collection action (simulated, restricted to finance):"]
        for r in rev.itertuples():
            lines.append(f"{r.customer_id} ({r.customer_segment}): {r.recommended_action}; late-payment probability "
                         f"{r.p_late_90d_filled:.0%}; overdue Rs. {r.overdue_inr:,.0f}; expected credit loss Rs. {r.ecl_inr:,.0f}.")
        for i in range(0, len(lines), 10):
            docs.append((f"CRV-{i // 10 + 1}", "Credit Review List: customers requiring credit review or collection action",
                         "Finance", RESTRICTED, "\n".join(lines[i:i + 10])))

    plan = context.get("freight_plan")
    if plan is not None and len(plan):
        chg = plan[plan["status"].isin(["RE-SOURCED", "VEHICLE CHANGED"])].sort_values("saving_inr", ascending=False).head(20)
        for r in chg.itertuples():
            docs.append((f"DSP-{r.open_order_id}", f"Dispatch Recommendation {r.open_order_id}", "Logistics", INTERNAL,
                         f"Dispatch recommendation for order {r.open_order_id} ({r.qty_t:.0f} t {r.product_id} to {r.city}, "
                         f"{r.urgency}, deadline {r.deadline_hours} h): ship from {r.opt_source_id} using "
                         f"{r.opt_trucks} x {r.opt_vehicle_type} instead of {r.base_source_id} with {r.base_trucks} x "
                         f"{r.base_vehicle_type}. It is preferred because total fulfilment cost falls by Rs. {r.saving_inr:,.0f} "
                         f"and predicted delivery time is {r.opt_delivery_hours:.1f} h versus {r.base_delivery_hours:.1f} h, "
                         f"within the deadline. Requires planner approval."))

    comp = context.get("compliance")
    if comp is not None and len(comp):
        for (proj, prod), g in comp.groupby(["compliance_project_id", "product_id"]):
            lines = [f"Compliance evidence for product {prod} against project {proj} (simulated, pending authorised approval):"]
            for r in g.itertuples():
                lines.append(f"{r.criterion}: {r.status} - {r.detail} (evidence {r.evidence}).")
            docs.append((f"CMP-{proj}-{prod}", f"Compliance Record {proj} / {prod}", "Technical", INTERNAL, "\n".join(lines)))

    tenders = context.get("tenders")
    if tenders is not None and len(tenders):
        for r in tenders.head(15).itertuples():
            docs.append((f"TND-{r.tender_id}", f"Open Tender {r.tender_id}", "Tender", INTERNAL,
                         f"Sample tender {r.tender_id} from {r.institution} closes {pd.Timestamp(r.closing_date).strftime('%d-%b-%Y')}. "
                         f"Required documents for this tender: {', '.join(r.required_documents)}. Recommendation: {r.recommendation}."))
    return docs


def _chunk(doc_id: str, title: str, category: str, sensitivity: str, text: str, max_words: int = 90) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks, buf = [], []
    for p in paras:
        if buf and sum(len(x.split()) for x in buf) + len(p.split()) > max_words:
            chunks.append(" ".join(buf)); buf = []
        buf.append(p)
    if buf:
        chunks.append(" ".join(buf))
    return [Chunk(f"{doc_id}#{i + 1}", doc_id, title, category, sensitivity, c) for i, c in enumerate(chunks)]


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform([f"{c.title}. {c.text}" for c in chunks])

    def search(self, query: str, role: str, k: int = 5, boost_category: str | None = None,
               boost: float = 0.10) -> tuple[list[tuple[Chunk, float]], int]:
        if not query.strip():
            return [], 0
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        if boost_category:
            # Intent-aware re-ranking: favour the document category the question is about (only if relevant at all)
            cat = np.array([c.category == boost_category for c in self.chunks])
            scores = np.where(cat & (scores >= MIN_RELEVANCE), scores + boost, scores)
        order = np.argsort(-scores)
        allowed = ROLE_PERMISSIONS.get(role, {PUBLIC})
        hits, withheld = [], 0
        for i in order[: k * 4]:
            if scores[i] < MIN_RELEVANCE:
                break
            c = self.chunks[i]
            if c.sensitivity not in allowed:
                withheld += 1
                continue
            hits.append((c, float(scores[i])))
            if len(hits) >= k:
                break
        return hits, withheld

    def documents_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"chunk_id": c.chunk_id, "document": c.title, "category": c.category,
                              "sensitivity": c.sensitivity, "words": len(c.text.split())} for c in self.chunks])


def build_knowledge_base(env, context: dict | None = None) -> KnowledgeBase:
    context = context or {}
    chunks: list[Chunk] = []
    for doc in _policy_documents(env) + _data_documents(env, context):
        chunks.extend(_chunk(*doc))
    return KnowledgeBase(chunks)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
INTENTS = [
    ("Stock availability", r"\b(stock|availab|inventory)\b"),
    ("Tender documents", r"\b(tender|bid)\b.*\b(document|certificate|required|need)|\b(document|certificate)s?\b.*\btender"),
    ("Discount policy", r"\b(discount|scheme|pricing polic|price polic)"),
    ("Declining dealers", r"\b(declin|dropping|falling|lost)\b.*\bdealer|\bdealer.*\b(declin|dropping|falling)"),
    ("Dispatch rationale", r"\b(dispatch|shipment|freight|route)\b"),
    ("Credit review", r"\b(credit|overdue|collection|receivable)"),
    ("Compliance evidence", r"\b(complian|evidence|specification|certificate|alkali)"),
]


INTENT_CATEGORY = {"Stock availability": "Inventory", "Tender documents": "Tender", "Discount policy": "Policy",
                   "Declining dealers": "Sales", "Dispatch rationale": "Logistics", "Credit review": "Finance",
                   "Compliance evidence": "Technical"}
SENTENCE_SPLIT = re.compile(r"(?<=[a-z0-9\)%])\.\s+(?=[A-Z])")


def detect_intent(question: str) -> str:
    q = question.lower()
    for name, pat in INTENTS:
        if re.search(pat, q):
            return name
    return "General"


class TemplateAnswerGenerator:
    """Deterministic, extractive answer composer (default; no language model)."""

    def generate(self, question: str, intent: str, chunks: list[tuple[Chunk, float]]) -> str:
        keywords = {w for w in re.findall(r"[a-z0-9\-]{3,}", question.lower())} - {
            "what", "which", "the", "for", "this", "that", "are", "is", "current", "selected", "does", "why",
            "with", "have", "and", "how", "required", "month", "simulation", "particular", "region", "product"}
        scored = []
        for ci, (c, _) in enumerate(chunks):
            for si, s in enumerate(SENTENCE_SPLIT.split(c.text)):
                hits = sum(k in s.lower() for k in keywords)
                if hits:
                    scored.append((-hits, ci, si, s.strip(), c.chunk_id))
        # Most specific sentences first (most query terms matched), then retrieval order
        sentences = [(s, cid) for _, _, _, s, cid in sorted(scored)]
        if not sentences:
            sentences = [(SENTENCE_SPLIT.split(c.text)[0], c.chunk_id) for c, _ in chunks[:2]]
        seen, lines = set(), []
        for s, cid in sentences:
            if s not in seen:
                seen.add(s)
                lines.append(f"- {s} [{cid}]")
            if len(lines) >= 8:
                break
        header = {"Stock availability": "Stock position from the retrieved stock reports:",
                  "Tender documents": "Documents identified in the retrieved tender sources:",
                  "Discount policy": "Approved discount policy (simulation) from retrieved policy text:",
                  "Declining dealers": "Dealers with declining orders from retrieved sales analysis:",
                  "Dispatch rationale": "Dispatch rationale from retrieved recommendation records:",
                  "Credit review": "Credit information from retrieved sources:",
                  "Compliance evidence": "Compliance evidence from retrieved records:"}.get(intent, "Relevant information found:")
        return header + "\n" + "\n".join(lines)


class EnterpriseLLMAdapter:
    """Placeholder for a future approved enterprise or local LLM. Not used in the demo.

    An implementation must (1) receive only the retrieved chunks, (2) be instructed to answer
    solely from them, (3) return chunk citations, and (4) run inside approved infrastructure.
    """

    def generate(self, question: str, intent: str, chunks: list[tuple[Chunk, float]]) -> str:  # pragma: no cover
        raise NotImplementedError("No LLM is configured. The demo uses TemplateAnswerGenerator.")


def answer_question(kb: KnowledgeBase, question: str, role: str,
                    generator: AnswerGenerator | None = None, k: int = 5) -> Answer:
    generator = generator or TemplateAnswerGenerator()
    question = question.strip()[:500]
    intent = detect_intent(question)
    hits, withheld = kb.search(question, role, k, boost_category=INTENT_CATEGORY.get(intent))
    if not hits:
        msg = ("The requested information is not available in the knowledge base for your role. "
               if withheld else "The requested information is not available in the knowledge base. ")
        if withheld:
            msg += f"{withheld} relevant item(s) are restricted to finance roles."
        else:
            msg += "Try rephrasing, or consult the relevant system owner."
        return Answer(question=question, intent=intent, text=msg, sources=[], grounded=False,
                      withheld_for_permissions=withheld)
    text = generator.generate(question, intent, hits)
    if withheld:
        text += f"\n\nNote: {withheld} additional relevant item(s) were withheld because they are restricted to finance roles."
    sources = [{"chunk_id": c.chunk_id, "document": c.title, "category": c.category, "sensitivity": c.sensitivity,
                "relevance": round(s, 3), "excerpt": c.text[:400]} for c, s in hits]
    return Answer(question=question, intent=intent, text=text, sources=sources, grounded=True,
                  withheld_for_permissions=withheld)
