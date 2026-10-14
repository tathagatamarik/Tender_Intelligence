"""Technical product compliance engine (deterministic rules over synthetic evidence).

Status rules:
    PASS          - evidence present, in date, and within the extracted limit
    FAIL          - evidence present and in date but outside the limit, or product type not permitted,
                    or consultant approval rejected
    NEEDS REVIEW  - evidence missing, expired, older than the permitted test age, certificate absent
                    or expired, or consultant approval pending / not submitted

A product is never labelled compliant because some fields match. The best possible overall
outcome is "ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL"; final technical approval always
remains with authorised personnel. All limits are illustrative and synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.services.document_parser import extract_spec_requirements

TYPE_TO_PRODUCTS = {"PSC": ["P04"], "PPC": ["P03"], "OPC": ["P01", "P02", "P07"], "Composite": ["P05"],
                    "Sulphate Resisting": ["P06"], "Low-Alkali OPC": ["P07"]}


def permitted_products(types: list[str]) -> list[str]:
    out: list[str] = []
    for t in types:
        out += TYPE_TO_PRODUCTS.get(t, [])
    return sorted(set(out))


def evaluate_product(env, project_id: str, product_id: str, spec_text: str | None = None) -> tuple[pd.DataFrame, str]:
    """Compliance matrix for one product against one project specification."""
    spec_text = spec_text or env.compliance_documents[project_id]
    spec = extract_spec_requirements(spec_text, f"SPEC-{project_id}")
    as_of = env.as_of
    max_age = spec["max_test_age_days"]
    rows = []

    allowed = permitted_products(spec["permitted_types"])
    rows.append({"criterion": "Permitted cement type", "requirement": " or ".join(spec["permitted_types"]) or "Not found",
                 "observed": product_id,
                 "status": "NEEDS REVIEW" if not spec["permitted_types"] else ("PASS" if product_id in allowed else "FAIL"),
                 "detail": "type requirement not found in specification" if not spec["permitted_types"] else
                 ("product type permitted" if product_id in allowed else "product type not permitted by specification"),
                 "evidence": f"SPEC-{project_id}; product catalogue"})

    tests = env.lab_tests[env.lab_tests["product_id"] == product_id].sort_values("test_date", ascending=False)
    latest = tests.iloc[0] if len(tests) else None
    for req in spec["requirements"]:
        label = f"{req.parameter_label} ({req.operator} {req.limit} {req.unit})"
        if latest is None:
            rows.append({"criterion": req.parameter_label, "requirement": label, "observed": "No test record",
                         "status": "NEEDS REVIEW", "detail": "no laboratory evidence available", "evidence": "None"})
            continue
        val = latest[req.parameter]
        age = (as_of - latest["test_date"]).days
        ev = f"{latest['certificate_id']} ({latest['laboratory']}, {latest['test_date'].strftime('%d-%b-%Y')})"
        if pd.isna(val):
            status, detail, observed = "NEEDS REVIEW", "value missing from latest test certificate", "Missing"
        else:
            within = val >= req.limit if req.operator == ">=" else val <= req.limit
            observed = f"{val:g}"
            if max_age is not None and age > max_age:
                status, detail = "NEEDS REVIEW", f"test is {age} days old; specification allows {max_age} days"
            else:
                status = "PASS" if within else "FAIL"
                detail = "within limit" if within else f"outside limit by {abs(val - req.limit):.3g} {req.unit}"
        rows.append({"criterion": req.parameter_label, "requirement": label, "observed": observed, "status": status,
                     "detail": detail, "evidence": ev})

    certs = env.product_certificates[env.product_certificates["product_id"] == product_id].set_index("certificate")
    for doc in spec["required_documents"]:
        if doc not in certs.index or not bool(certs.loc[doc, "available"]):
            rows.append({"criterion": f"Document: {doc}", "requirement": "Valid document on file", "observed": "Not on file",
                         "status": "NEEDS REVIEW", "detail": "required document missing", "evidence": "Document repository"})
            continue
        exp = certs.loc[doc, "expiry_date"]
        ok = pd.notna(exp) and exp >= as_of
        rows.append({"criterion": f"Document: {doc}", "requirement": "Valid document on file",
                     "observed": f"Expires {exp.strftime('%d-%b-%Y')}" if pd.notna(exp) else "No expiry recorded",
                     "status": "PASS" if ok else "NEEDS REVIEW",
                     "detail": "valid" if ok else "document expired - renewal required",
                     "evidence": certs.loc[doc, "document_ref"]})

    if spec["consultant"]:
        ap = env.consultant_approvals
        m = ap[(ap["compliance_project_id"] == project_id) & (ap["product_id"] == product_id)]
        status_txt = m["approval_status"].iloc[0] if len(m) else "Not submitted"
        res = {"Approved": "PASS", "Rejected": "FAIL"}.get(status_txt, "NEEDS REVIEW")
        rows.append({"criterion": "Consultant approval", "requirement": f"Approval from {spec['consultant']}",
                     "observed": status_txt, "status": res,
                     "detail": "approval on record" if res == "PASS" else f"approval {status_txt.lower()}",
                     "evidence": m["approval_ref"].iloc[0] if len(m) and m["approval_ref"].iloc[0] else "Approval register"})
    else:
        rows.append({"criterion": "Consultant approval", "requirement": "Not required by specification",
                     "observed": "Not required", "status": "PASS", "detail": "not required", "evidence": f"SPEC-{project_id}"})

    matrix = pd.DataFrame(rows)
    matrix.insert(0, "product_id", product_id)
    matrix.insert(0, "compliance_project_id", project_id)
    if (matrix["status"] == "FAIL").any():
        overall = "NOT COMPLIANT"
    elif (matrix["status"] == "NEEDS REVIEW").any():
        overall = "NEEDS REVIEW"
    else:
        overall = "ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL"
    return matrix, overall


def evaluate_project(env, project_id: str, spec_text: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every catalogue product against a project; returns (summary, full matrix)."""
    mats, summ = [], []
    for pid in env.products["product_id"]:
        m, overall = evaluate_product(env, project_id, pid, spec_text)
        mats.append(m)
        summ.append({"product_id": pid, "overall_status": overall, "pass": int((m["status"] == "PASS").sum()),
                     "fail": int((m["status"] == "FAIL").sum()), "needs_review": int((m["status"] == "NEEDS REVIEW").sum())})
    summary = pd.DataFrame(summ).merge(env.products[["product_id", "product_name"]], on="product_id")
    order = {"ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL": 0, "NEEDS REVIEW": 1, "NOT COMPLIANT": 2}
    summary = summary.sort_values(["overall_status", "fail", "needs_review"], key=lambda s: s.map(order) if s.name == "overall_status" else s)
    return summary, pd.concat(mats, ignore_index=True)


def extraction_agreement(env) -> pd.DataFrame:
    """Criterion-level agreement between extracted requirements and the synthetic ground truth."""
    rows = []
    for pid, g in env.compliance_requirements.groupby("compliance_project_id"):
        ext = extract_spec_requirements(env.compliance_documents[pid], f"SPEC-{pid}")["requirements"]
        got = {(r.parameter, r.operator, round(r.limit, 6)) for r in ext}
        truth = {(r.parameter, r.operator, round(float(r.limit), 6)) for r in g.itertuples()}
        rows.append({"compliance_project_id": pid, "ground_truth_criteria": len(truth), "extracted": len(got),
                     "matched": len(got & truth), "agreement": len(got & truth) / len(truth) if truth else np.nan})
    return pd.DataFrame(rows)


def compliance_report_text(env, project_id: str, product_id: str, matrix: pd.DataFrame, overall: str) -> str:
    proj = env.compliance_projects.set_index("compliance_project_id").loc[project_id]
    lines = ["SYNTHETIC TECHNICAL COMPLIANCE REPORT - DEMONSTRATION ONLY",
             f"Generated: {env.as_of.strftime('%d-%b-%Y')} (simulation date)",
             f"Project: {proj['project_name']} ({project_id})",
             f"Product: {product_id}",
             f"Overall status: {overall}", "",
             "Criterion | Requirement | Observed | Status | Detail | Evidence"]
    for r in matrix.itertuples():
        lines.append(f"{r.criterion} | {r.requirement} | {r.observed} | {r.status} | {r.detail} | {r.evidence}")
    lines += ["", "Missing or unresolved items:"]
    unresolved = matrix[matrix["status"] != "PASS"]
    lines += [f"- {r.criterion}: {r.detail}" for r in unresolved.itertuples()] or ["- None"]
    lines += ["", "All limits and standard references are illustrative and synthetic. This report does not certify "
                  "regulatory or technical compliance. Final technical approval rests with authorised personnel."]
    return "\n".join(lines)
