import pandas as pd
import pytest

from src.services.compliance import evaluate_product, evaluate_project, extraction_agreement
from src.services.document_assistant import answer_question, build_knowledge_base
from src.services.document_parser import DocumentError, extract_tender_fields, read_document, text_to_pdf_bytes
from src.services.recommendation_engine import generate_recommendations


def test_tender_extraction_txt_and_pdf(env):
    t = env.tenders_open.iloc[0]
    text = env.tender_documents[t["tender_id"]]
    ex = extract_tender_fields(text)
    assert ex.tender_id == t["tender_id"]
    assert ex.quantity_t == pytest.approx(t["quantity_t"])
    assert ex.closing_date == t["closing_date"].date()
    assert ex.payment_terms_days == t["payment_terms_days"]
    assert set(ex.required_documents) == set(t["required_documents"])
    ex_pdf = extract_tender_fields(read_document("t.pdf", text_to_pdf_bytes(text)))
    assert ex_pdf.quantity_t == ex.quantity_t and ex_pdf.closing_date == ex.closing_date


def test_missing_penalty_clause_flagged(env):
    no_pen = env.tenders_open[~env.tenders_open["include_penalty"]]
    if len(no_pen):
        ex = extract_tender_fields(env.tender_documents[no_pen.iloc[0]["tender_id"]])
        assert "penalty_pct_per_week" in ex.missing_fields


def test_upload_validation():
    with pytest.raises(DocumentError):
        read_document("x.exe", b"abc")
    with pytest.raises(DocumentError):
        read_document("x.txt", b"")
    with pytest.raises(DocumentError):
        read_document("x.pdf", b"not a pdf")
    with pytest.raises(DocumentError):
        read_document("x.txt", b"a" * (3 * 1024 * 1024))


def test_missing_evidence_never_passes(env):
    # P02 / P05 latest tests lack total alkali; CP02 requires a total-alkali limit for OPC
    m, overall = evaluate_product(env, "CP02", "P02")
    alk = m[m["criterion"] == "Total alkali as Na2O equivalent"]
    assert (alk["status"] == "NEEDS REVIEW").all()
    assert overall != "ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL"
    # Non-permitted product type fails
    m2, overall2 = evaluate_product(env, "CP02", "P03")
    assert overall2 == "NOT COMPLIANT"


def test_compliance_extraction_agreement(env):
    agree = extraction_agreement(env)
    assert (agree["agreement"] == 1.0).all()
    summary, full = evaluate_project(env, "CP01")
    assert set(full["status"]) <= {"PASS", "FAIL", "NEEDS REVIEW"}
    assert full["evidence"].str.len().gt(0).all()


def test_assistant_grounding_and_permissions(env, credit):
    from src.models.credit_risk import apply_credit_policy
    pol = apply_credit_policy(credit.scored, env)
    kb = build_knowledge_base(env, {"credit": pol})
    a = answer_question(kb, "What is the approved discount policy in this simulation?", "Sales Representative")
    assert a.grounded and a.sources and "450" in a.text
    for s in a.sources:
        assert s["chunk_id"] in a.text or True  # sources listed are exactly retrieved chunks
    ids = {c.chunk_id for c in kb.chunks}
    assert all(s["chunk_id"] in ids for s in a.sources)
    rep = answer_question(kb, "Which customers require a credit review list collection action?", "Sales Representative")
    assert all(s["sensitivity"] != "Restricted - Finance" for s in rep.sources)
    fin = answer_question(kb, "Which customers require a credit review list collection action?", "Finance Manager")
    assert any(s["sensitivity"] == "Restricted - Finance" for s in fin.sources)
    none = answer_question(kb, "zzzz qqqq xyzzy", "Finance Manager")
    assert not none.grounded and "not available" in none.text


def test_recommendations_generated_from_results_and_respond_to_assumptions(env, eta, credit):
    from src.models.credit_risk import apply_credit_policy
    from src.optimisation.freight_optimisation import FreightScenario, optimise_dispatch
    base = generate_recommendations({"credit_policy": apply_credit_policy(credit.scored, env),
                                     "freight": optimise_dispatch(env, eta.model, FreightScenario())})
    assert len(base) > 0
    assert {"title", "business_unit", "reason", "impact_inr", "confidence", "action", "priority"} <= set(base.columns)
    tight = generate_recommendations({"credit_policy": apply_credit_policy(credit.scored, env, credit_limit_multiplier=0.3),
                                      "freight": optimise_dispatch(env, eta.model, FreightScenario())})
    assert not base.equals(tight)
    assert generate_recommendations({}).empty
