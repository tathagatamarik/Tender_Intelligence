"""Page 7 - Technical Product Compliance Copilot."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.compliance import (compliance_report_text, evaluate_product, evaluate_project, extraction_agreement,
                                     permitted_products)
from src.services.document_parser import DocumentError, extract_spec_requirements, read_document
from src.ui import state
from src.ui.components import (approval_button, assumptions, badge, download_csv, kpi_cards, note, page_header, section,
                               show_table)
from src.ui.formatting import display_frame, fmt_pct

page_header("Technical Product Compliance Copilot",
            "Extract project requirements, compare candidate products against synthetic evidence and build a traceable compliance matrix.")
env = state.env()
note("All standard references and limits are ILLUSTRATIVE and synthetic; they are not taken from any Indian Standard and do not "
     "describe actual products. Final technical approval remains with authorised personnel.")

projects = env.compliance_projects
c1, c2 = st.columns([2, 1])
pid = c1.selectbox("Project specification", projects["compliance_project_id"].tolist(), key="p_cp_proj",
                   format_func=lambda p: f"{p} - {projects.set_index('compliance_project_id').loc[p, 'project_name']}")
up = c2.file_uploader("Or upload a specification (TXT / PDF)", type=["txt", "pdf"], key="p_cp_up")
spec_text = env.compliance_documents[pid]
source = f"SPEC-{pid}"
if up is not None:
    try:
        spec_text = read_document(up.name, up.getvalue())
        source = up.name
        st.success(f"Using uploaded specification {up.name}; evidence is evaluated against project {pid}'s approval register.")
    except DocumentError as exc:
        st.error(f"Specification could not be read: {exc}")

spec = extract_spec_requirements(spec_text, source)
summary, full = evaluate_project(env, pid, spec_text)
agree = extraction_agreement(env)
kpi_cards([
    ("Numeric requirements extracted", str(len(spec["requirements"])), f"Source {source}"),
    ("Permitted products", ", ".join(permitted_products(spec["permitted_types"])) or "None found", " or ".join(spec["permitted_types"])),
    ("Max test-certificate age", f"{spec['max_test_age_days']} days" if spec["max_test_age_days"] else "Not specified", "From specification"),
    ("Consultant approval", spec["consultant"] or "Not required", "Approval register"),
    ("Extraction agreement", fmt_pct(agree["matched"].sum() / agree["ground_truth_criteria"].sum()),
     "Criterion-level vs synthetic ground truth"),
])

t1, t2, t3, t4 = st.tabs(["Requirements and candidates", "Compliance matrix", "Evidence and documents", "Extraction quality"])
with t1:
    a, b = st.columns([2, 3])
    with a:
        section("Project specification (retrieved source text)")
        st.code(spec_text, language=None)
    with b:
        section("Extracted requirement summary", "Deterministic extraction; each row cites its source.")
        reqs = pd.DataFrame([r.model_dump() for r in spec["requirements"]])
        show_table(reqs.rename(columns={"parameter_label": "Requirement", "operator": "Operator", "limit": "Limit", "unit": "Unit",
                                        "source": "Source"}).drop(columns=["parameter"]) if len(reqs) else reqs)
        st.markdown("**Required documents:** " + (", ".join(spec["required_documents"]) or "None found"))
        section("Candidate product comparison", "Overall status per product; a product is never compliant on partial matches.")
        show_table(summary[["product_id", "product_name", "overall_status", "pass", "fail", "needs_review"]].rename(
            columns={"product_id": "Product", "product_name": "Name", "overall_status": "Overall status", "pass": "Pass",
                     "fail": "Fail", "needs_review": "Needs review"}), status_cols=["Overall status"])

with t2:
    prod = st.selectbox("Product", summary["product_id"].tolist(), key="p_cp_prod",
                        format_func=lambda p: f"{p} - {env.products.set_index('product_id').loc[p, 'product_name']}")
    matrix, overall = evaluate_product(env, pid, prod, spec_text)
    st.markdown(f"**Overall status:** {badge(overall)}", unsafe_allow_html=True)
    show_table(matrix[["criterion", "requirement", "observed", "status", "detail", "evidence"]].rename(
        columns={"criterion": "Criterion", "requirement": "Requirement", "observed": "Observed", "status": "Status",
                 "detail": "Explanation", "evidence": "Evidence source"}), status_cols=["Status"])
    unresolved = matrix[matrix["status"] != "PASS"]
    if len(unresolved):
        section("Failed checks and unresolved requirements")
        for r in unresolved.itertuples():
            st.markdown(f"{badge(r.status)} **{r.criterion}** - {r.detail}. Evidence: {r.evidence}.", unsafe_allow_html=True)
    report = compliance_report_text(env, pid, prod, matrix, overall)
    a, b, c = st.columns(3)
    with a:
        st.download_button("Download compliance report (TXT)", report.encode("utf-8"), f"compliance_{pid}_{prod}.txt",
                           "text/plain", key="dl_cr_txt")
    with b:
        download_csv(matrix, f"compliance_matrix_{pid}_{prod}.csv", "Download matrix (CSV)", key="dl_cr_csv")
    with c:
        approval_button("Technical Compliance", f"{pid}/{prod}", f"Compliance status {overall}", "Head of Technical Services",
                        key="ap_comp", label="Submit for technical approval")

with t3:
    section("Synthetic laboratory test results (latest batches)")
    tests = env.lab_tests[env.lab_tests["product_id"] == prod].sort_values("test_date", ascending=False)
    show_table(display_frame(tests.drop(columns=["product_id"]), dates=["test_date"]), height=260)
    section("Missing-document checklist")
    certs = env.product_certificates[env.product_certificates["product_id"] == prod].copy()
    certs["status"] = [("NEEDS REVIEW - MISSING" if not a_ else ("PASS - VALID" if pd.notna(e) and e >= env.as_of else "NEEDS REVIEW - EXPIRED"))
                       for a_, e in zip(certs["available"], certs["expiry_date"])]
    certs["required_by_spec"] = certs["certificate"].isin(spec["required_documents"])
    show_table(display_frame(certs[["certificate", "required_by_spec", "issue_date", "expiry_date", "document_ref", "status"]],
                             dates=["issue_date", "expiry_date"]), status_cols=["status"])
    section("Consultant approval status")
    ap = env.consultant_approvals[env.consultant_approvals["compliance_project_id"] == pid]
    show_table(ap.rename(columns={"approval_status": "status"}), status_cols=["status"])

with t4:
    section("Requirement extraction versus synthetic ground truth",
            "Measures whether the deterministic parser recovered every criterion used to generate each specification.")
    show_table(display_frame(agree, pct=["agreement"]))
    section("All products across this project")
    download_csv(full, f"compliance_all_products_{pid}.csv", "Download full matrix for all products (CSV)", key="dl_cr_all")
    assumptions(["PASS requires evidence present, in date and within the limit.",
                 "Missing values, expired or over-age certificates, missing documents and pending consultant approvals produce NEEDS REVIEW.",
                 "Out-of-limit values, non-permitted product types and rejected approvals produce FAIL.",
                 "The best overall outcome is 'ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL'; no automatic certification occurs.",
                 "Uploaded specifications are parsed as text only and never executed."], "Compliance rules")
