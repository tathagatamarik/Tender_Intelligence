"""Page 10 - AI Sales Knowledge Assistant (offline, retrieval-grounded)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.compliance import evaluate_project
from src.services.document_assistant import ROLE_PERMISSIONS, answer_question, build_knowledge_base
from src.ui import state
from src.ui.components import assumptions, badge, kpi_cards, note, page_header, section, show_table

page_header("AI Sales Knowledge Assistant",
            "Offline question answering over a synthetic knowledge base, grounded in retrieved excerpts with source references.")


@st.cache_resource(show_spinner="Building knowledge base from the current scenario...")
def _kb(k, freight_repr: str, credit_repr: str, tender_repr: str):
    env = state.env_for(k)
    comp = pd.concat([evaluate_project(env, p)[1] for p in env.compliance_projects["compliance_project_id"]], ignore_index=True)
    ctx = {"segments": state.segments_for(k), "credit": state.credit_policy(), "freight_plan": state.freight().plan,
           "compliance": comp, "tenders": state.tenders_scored()}
    return build_knowledge_base(env, ctx)


k = state.key()
kb = _kb(k, repr(state.freight_scenario()), repr(sorted(state.credit_params().items())),
         repr(st.session_state.get("tender_min_margin", 0.08)))
docs = kb.documents_frame()
kpi_cards([("Documents", str(docs["document"].nunique()), "Synthetic policies, reports and records"),
           ("Chunks indexed", str(len(docs)), "TF-IDF, unigrams and bigrams"),
           ("Restricted chunks", str(int((docs["sensitivity"] == "Restricted - Finance").sum())), "Finance roles only"),
           ("Language model", "None (offline)", "Deterministic templates")])

role = st.selectbox("Your role (simulated permissions)", list(ROLE_PERMISSIONS), key="p_ka_role")
examples = [
    "What is the current stock availability of PPC in the East region?",
    "Which documents are required for a sample tender?",
    "What is the approved discount policy in this simulation?",
    "Which dealers have declining orders this month?",
    "Why is a particular dispatch recommendation preferred?",
    "Which customers require a credit review?",
    "What evidence supports a technical compliance decision for P04 on CP01?",
]
section("Example questions")
cols = st.columns(4)
for i, q in enumerate(examples):
    if cols[i % 4].button(q, key=f"ka_ex_{i}", width="stretch"):
        st.session_state["ka_question"] = q
with st.form("ka_form"):
    question = st.text_input("Ask a question", value=st.session_state.get("ka_question", ""), max_chars=500, key="p_ka_q")
    asked = st.form_submit_button("Ask", type="primary")
if asked:
    st.session_state["ka_question"] = question

q = st.session_state.get("ka_question", "")
if q:
    ans = answer_question(kb, q, role)
    section("Answer", f"Intent: {ans.intent} | Role: {role}")
    st.markdown(badge("GROUNDED IN RETRIEVED SOURCES", "good") if ans.grounded else badge("NOT AVAILABLE", "warning"),
                unsafe_allow_html=True)
    st.markdown(ans.text)
    st.caption("Generated explanation: composed by a deterministic template from the retrieved excerpts below. "
               "Citations in square brackets refer to chunk IDs.")
    if ans.sources:
        section("Retrieved evidence", "Exact excerpts used, with relevance scores.")
        for s in ans.sources:
            with st.expander(f"[{s['chunk_id']}] {s['document']} | {s['category']} | {s['sensitivity']} | relevance {s['relevance']:.2f}"):
                st.write(s["excerpt"])

with st.expander("Browse knowledge base index"):
    show_table(docs, height=360)
assumptions(["Knowledge base documents are generated from the shared synthetic scenario (inventory, dealer analysis, credit, dispatch, "
             "compliance and tenders) plus simulated policy documents.",
             "Retrieval: TF-IDF cosine similarity with a minimum relevance threshold; results are filtered by role before use.",
             "Answers only quote retrieved chunks and always list them; if nothing relevant is retrieved the assistant says the "
             "information is unavailable.",
             "An AnswerGenerator adapter interface allows a future approved enterprise or local LLM; the demo uses no LLM and makes "
             "no network calls.",
             "Restricted finance documents are withheld from non-finance roles, and the answer states that items were withheld."],
            "How the assistant works")
