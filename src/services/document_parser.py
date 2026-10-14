"""Deterministic document reading and field extraction (no external LLM).

Uploaded files are treated strictly as data: text is extracted and matched with regular
expressions. Content is never executed or interpreted as instructions.
"""
from __future__ import annotations

import io
import re
from datetime import datetime

from src.config import MAX_UPLOAD_BYTES
from src.data.schemas import ComplianceRequirement, TenderExtraction

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


class DocumentError(ValueError):
    """Raised for unreadable, oversized or unsupported documents."""


def read_document(name: str, data: bytes) -> str:
    """Extract plain text from an uploaded TXT or PDF file with size and type validation."""
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentError(f"Unsupported file type '{ext or 'none'}'. Upload a .txt or .pdf file.")
    if len(data) == 0:
        raise DocumentError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB demo limit.")
    if ext == ".txt":
        for enc in ("utf-8", "latin-1"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        raise DocumentError("Could not decode the text file.")
    if not data.startswith(b"%PDF"):
        raise DocumentError("The file does not look like a valid PDF.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > 50:
            raise DocumentError("PDF has more than 50 pages; please upload a shorter tender extract.")
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except DocumentError:
        raise
    except Exception as exc:  # malformed PDFs
        raise DocumentError(f"Could not read the PDF: {exc.__class__.__name__}.") from exc
    if not text.strip():
        raise DocumentError("No extractable text found in the PDF (scanned images are not supported in the demo).")
    return text


def text_to_pdf_bytes(text: str, title: str = "Synthetic document") -> bytes:
    """Render text into a simple PDF (used to create sample tender PDFs for upload testing)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_title(title)
    pdf.set_font("Helvetica", size=10)
    safe = text.replace("—", "-").replace("–", "-").replace("₹", "Rs.")
    safe = safe.encode("latin-1", "replace").decode("latin-1")
    for line in safe.split("\n"):
        pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _num(s: str) -> float:
    return float(s.replace(",", "").strip())


def _parse_date(s: str):
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


_TENDER_PATTERNS = {
    "tender_id": r"Tender\s+(?:Reference\s+)?(?:No|ID|Number)\.?\s*[:\-]\s*([A-Z0-9\-/]+)",
    "issuing_authority": r"Issuing\s+Authority\s*[:\-]\s*(.+)",
    "portal": r"Procurement\s+Portal\s*[:\-]\s*(.+)",
    "project_location": r"Project\s+Location\s*[:\-]\s*(.+)",
    "quantity_t": r"(?:Estimated\s+)?Quantity\s*[:\-]\s*([\d,\.]+)\s*(?:MT|tonnes|tons|T)\b",
    "cement_requirement": r"Cement\s+Type(?:\s*/\s*Grade)?\s*[:\-]\s*(.+)",
    "delivery_months": r"Delivery\s+Period\s*[:\-]\s*(\d+)\s*months",
    "closing_date": r"Closing\s+Date\s*[:\-]\s*([0-9]{1,2}[\-/ ][A-Za-z0-9]{2,9}[\-/ ][0-9]{4})",
    "emd_inr": r"(?:EMD|Earnest\s+Money\s+Deposit)[^:]*[:\-]\s*Rs\.?\s*([\d,\.]+)",
    "performance_guarantee_pct": r"Performance\s+(?:Security|Guarantee)\s*[:\-]\s*([\d\.]+)\s*%",
    "payment_terms_days": r"Payment\s+Terms\s*[:\-].*?(\d+)\s*days",
    "penalty_pct_per_week": r"Liquidated\s+Damages\s*[:\-]\s*([\d\.]+)\s*%\s*per\s*week",
    "penalty_cap_pct": r"maximum\s+of\s+([\d\.]+)\s*%",
    "min_turnover_cr": r"turnover\s+of\s+Rs\.?\s*([\d,\.]+)\s*Crore",
    "min_experience_years": r"Minimum\s+(\d+)\s+years\s+of\s+experience",
}
_CATEGORY_KEYWORDS = [("Sulphate Resisting", "Specialty"), ("Low-Alkali", "Specialty"), ("Slag", "PSC"),
                      ("PSC", "PSC"), ("Pozzolana", "PPC"), ("PPC", "PPC"), ("Composite", "Composite"),
                      ("OPC", "OPC"), ("Ordinary Portland", "OPC")]


def extract_tender_fields(text: str) -> TenderExtraction:
    """Extract structured tender fields using deterministic regular expressions."""
    found: dict = {}
    evidence: dict[str, str] = {}
    for field, pat in _TENDER_PATTERNS.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        raw = m.group(1).strip()
        evidence[field] = m.group(0).strip()[:160]
        try:
            if field in ("quantity_t", "emd_inr", "performance_guarantee_pct", "penalty_pct_per_week",
                         "penalty_cap_pct", "min_turnover_cr"):
                found[field] = _num(raw)
            elif field in ("delivery_months", "payment_terms_days", "min_experience_years"):
                found[field] = int(_num(raw))
            elif field == "closing_date":
                d = _parse_date(raw)
                if d:
                    found[field] = d
            else:
                found[field] = raw
        except ValueError:
            continue
    req = found.get("cement_requirement", "")
    for kw, cat in _CATEGORY_KEYWORDS:
        if kw.lower() in req.lower():
            found["product_category"] = cat
            break
    docs: list[str] = []
    block = re.search(r"MANDATORY\s+DOCUMENTS\s*:?(.*?)(?:\n\s*\n|Bids\s+not|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s*(?:\(?[a-z0-9]{1,2}[\)\.]|-|\*)\s*(.+)", line.strip(), flags=re.IGNORECASE)
            if m and len(m.group(1)) > 2:
                docs.append(m.group(1).strip().rstrip("."))
    found["required_documents"] = docs
    critical = ["tender_id", "quantity_t", "cement_requirement", "closing_date", "payment_terms_days",
                "penalty_pct_per_week", "emd_inr", "performance_guarantee_pct", "delivery_months"]
    found["missing_fields"] = [f for f in critical if f not in found]
    if not docs:
        found["missing_fields"].append("required_documents")
    found["evidence"] = evidence
    return TenderExtraction(**found)


_PARAM_PATTERNS = {
    "compressive_strength_28d_mpa": r"28-day\s+compressive\s+strength",
    "fineness_m2_per_kg": r"Fineness",
    "initial_setting_time_min": r"Initial\s+setting\s+time",
    "magnesia_pct": r"Magnesia",
    "sulphuric_anhydride_pct": r"Sulphuric\s+anhydride",
    "total_alkali_pct": r"Total\s+alkali",
    "chloride_pct": r"Chloride",
}


def extract_spec_requirements(text: str, source: str) -> dict:
    """Extract numeric limits, test-age rule, documents and consultant requirement from a spec."""
    from src.data.synthetic_data import COMPLIANCE_PARAMETERS

    reqs: list[ComplianceRequirement] = []
    for line in text.splitlines():
        for param, pat in _PARAM_PATTERNS.items():
            if re.search(pat, line, flags=re.IGNORECASE):
                m = re.search(r"(minimum|maximum|min\.?|max\.?|not less than|not more than)\s*([\d\.]+)",
                              line, flags=re.IGNORECASE)
                if m:
                    word = m.group(1).lower()
                    op = ">=" if word.startswith("min") or "less" in word else "<="
                    label, unit = COMPLIANCE_PARAMETERS[param]
                    reqs.append(ComplianceRequirement(parameter=param, parameter_label=label, operator=op,
                                                      limit=float(m.group(2)), unit=unit, source=source))
                break
    age = re.search(r"not\s+be\s+older\s+than\s+(\d+)\s+days", text, flags=re.IGNORECASE)
    docs = re.search(r"Required\s+documents\s*:\s*(.+)", text, flags=re.IGNORECASE)
    types = re.search(r"Permitted\s+cement\s+type\s*:\s*(.+)", text, flags=re.IGNORECASE)
    consultant = re.search(r"Consultant\s+approval\s+required\s+from\s+(.+?)\s+before", text, flags=re.IGNORECASE)
    return {
        "requirements": reqs,
        "max_test_age_days": int(age.group(1)) if age else None,
        "required_documents": [d.strip().rstrip(".") for d in docs.group(1).split(";")] if docs else [],
        "permitted_types": [t.strip() for t in types.group(1).split(" or ")] if types else [],
        "consultant": consultant.group(1).strip() if consultant else None,
    }
