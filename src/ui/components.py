"""Reusable Streamlit UI components (text-based status labels; no emojis or icons)."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.config import DEMO_BANNER
from src.services.persistence import submit_for_approval

_CRITICAL = ("HIGH PRIORITY", "FAIL", "NO BID", "ACTION REQUIRED", "LIKELY STOCKOUT", "ESCALATE", "NOT COMPLIANT",
             "STOCKOUT RISK", "ON HOLD", "OVER LIMIT", "VERY HIGH", "NO FEASIBLE")
_WARNING = ("NEEDS REVIEW", "MEDIUM", "WATCH", "CREDIT REVIEW", "BELOW SAFETY", "BID WITH CONDITIONS", "COLLECTION",
            "PENDING", "HIGH", "EXCESS", "OVERDUE", "STRATEGIC", "RE-SOURCED", "VEHICLE CHANGED")
_GOOD = ("PASS", "WITHIN LIMIT", "BID", "ALL CHECKS PASSED", "NORMAL", "UNCHANGED", "LOW", "ROUTINE", "APPROVED",
         "NOT REQUIRED")

_TINTS = {"critical": ("#fbe4e4", "#8a1f1f"), "warning": ("#fdf1d6", "#6b4a00"), "good": ("#e3f4e3", "#0b5a0b"),
          "info": ("#e8f0fb", "#184f95"), "neutral": ("#f0efec", "#52514e")}


def status_kind(text: str) -> str:
    t = str(text).upper()
    for kind, keys in (("critical", _CRITICAL), ("warning", _WARNING), ("good", _GOOD)):
        if any(t.startswith(k) or f" {k}" in f" {t}" for k in keys):
            return kind
    return "neutral"


CSS = """
<style>
:root { --ink:#1f2328; --ink2:#52514e; --muted:#6b6a65; --line:#e4e3dd; --navy:#1f3a5f; --panel:#f7f7f5; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }
.dh-banner { background:#fff8e6; border:1px solid #f0d58a; color:#5c4300; padding:6px 12px; border-radius:6px;
             font-size:12px; font-weight:600; letter-spacing:0.04em; margin-bottom:10px; text-align:center; }
.dh-header { border-bottom:1px solid var(--line); padding-bottom:8px; margin-bottom:14px; }
.dh-header .eyebrow { font-size:11px; font-weight:600; color:var(--muted); letter-spacing:0.08em; text-transform:uppercase; }
.dh-header h1 { font-size:26px; margin:2px 0 2px 0; padding:0; }
.dh-header p { color:var(--ink2); margin:0; font-size:14px; }
.dh-kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin:6px 0 14px 0; }
.dh-kpi { border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fff; }
.dh-kpi .lbl { font-size:11.5px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.04em; }
.dh-kpi .val { font-size:22px; color:var(--ink); font-weight:650; margin-top:2px; line-height:1.2; }
.dh-kpi .sub { font-size:12px; color:var(--ink2); margin-top:2px; }
.dh-badge { display:inline-block; padding:1px 8px; border-radius:4px; font-size:11px; font-weight:700;
            letter-spacing:0.03em; border:1px solid rgba(0,0,0,0.08); }
.dh-note { background:var(--panel); border-left:3px solid var(--navy); padding:8px 12px; border-radius:4px;
           font-size:13px; color:var(--ink2); margin:6px 0 12px 0; }
.dh-section { font-size:16px; font-weight:650; color:var(--ink); margin:14px 0 2px 0; }
.dh-caption { font-size:12.5px; color:var(--muted); margin-bottom:6px; }
.dh-rec { border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:8px; background:#fff; }
.dh-rec .t { font-weight:650; color:var(--ink); font-size:14px; }
.dh-rec .m { font-size:12.5px; color:var(--ink2); margin-top:3px; }
.dh-side-title { font-size:17px; font-weight:700; color:var(--navy); margin-bottom:0; }
.dh-side-sub { font-size:11.5px; color:var(--muted); margin-bottom:6px; }
[data-testid="stSidebar"] { background:#fafaf8; border-right:1px solid var(--line); }
div[data-testid="stTabs"] button p { font-size:14px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def banner() -> None:
    st.markdown(f'<div class="dh-banner">{html.escape(DEMO_BANNER)}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "SIMULATED RESULT") -> None:
    banner()
    st.markdown(f'<div class="dh-header"><div class="eyebrow">{html.escape(eyebrow)}</div>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def badge(text: str, kind: str | None = None) -> str:
    bg, fg = _TINTS[kind or status_kind(text)]
    return f'<span class="dh-badge" style="background:{bg};color:{fg}">{html.escape(str(text))}</span>'


def kpi_cards(items: list[tuple[str, str, str]]) -> None:
    """Render KPI cards: list of (label, value, sub-text)."""
    cells = "".join(f'<div class="dh-kpi"><div class="lbl">{html.escape(l)}</div><div class="val">{html.escape(v)}</div>'
                    f'<div class="sub">{html.escape(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="dh-kpis">{cells}</div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="dh-section">{html.escape(title)}</div>'
                + (f'<div class="dh-caption">{html.escape(caption)}</div>' if caption else ""), unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="dh-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def assumptions(items: list[str], title: str = "Assumptions and limitations") -> None:
    with st.expander(title, expanded=False):
        st.markdown("\n".join(f"- {i}" for i in items))


def style_status(df: pd.DataFrame, columns: list[str]):
    """Pandas Styler that tints status cells; the text label always carries the meaning."""
    def _fmt(v):
        bg, fg = _TINTS[status_kind(v)]
        return f"background-color:{bg};color:{fg};font-weight:600"
    cols = [c for c in columns if c in df.columns]
    return df.style.map(_fmt, subset=cols) if cols else df.style


def show_table(df: pd.DataFrame, status_cols: list[str] | None = None, height: int | None = None, key: str | None = None) -> None:
    if df is None or df.empty:
        st.markdown('<div class="dh-note">No records match the current filters.</div>', unsafe_allow_html=True)
        return
    data = style_status(df, status_cols) if status_cols else df
    kwargs = {"width": "stretch", "hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(data, key=key, **kwargs)


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    data = df.to_csv(index=False).encode("utf-8") if df is not None else b""
    st.download_button(label, data=data, file_name=filename, mime="text/csv", key=key or f"dl_{filename}",
                       disabled=df is None or df.empty)


def approval_button(module: str, item_ref: str, summary: str, approver_role: str, key: str,
                    label: str = "Submit for approval", payload: dict | None = None) -> None:
    """Human-in-the-loop: records an approval request; never executes the action."""
    if st.button(label, key=key):
        try:
            rid = submit_for_approval(module, item_ref, summary, approver_role, payload)
            st.success(f"Approval request {rid} recorded for {approver_role}. No action has been executed.")
        except Exception as exc:  # pragma: no cover
            st.error(f"Could not record the approval request: {exc}")


def recommendation_card(r: dict, fmt_inr) -> None:
    st.markdown(
        f'<div class="dh-rec"><div>{badge(r["priority"])} {badge("CONFIDENCE " + r["confidence"], "info")} '
        f'<span class="dh-caption">#{r["rank"]} | {html.escape(r["business_unit"])} | {html.escape(r["source_module"])}</span></div>'
        f'<div class="t" style="margin-top:4px">{html.escape(r["title"])}</div>'
        f'<div class="m"><b>Reason:</b> {html.escape(r["reason"])}</div>'
        f'<div class="m"><b>Estimated impact:</b> {html.escape(fmt_inr(r["impact_inr"]))} - {html.escape(r.get("value_type", ""))} ({html.escape(r["impact_basis"])})</div>'
        f'<div class="m"><b>Required action:</b> {html.escape(r["action"])}</div></div>', unsafe_allow_html=True)
