"""Consistent Indian-convention formatting for currency, tonnes, percentages and dates."""
from __future__ import annotations

import math

import pandas as pd

from src.config import DATE_FMT


def indian_group(n: float, decimals: int = 0) -> str:
    """Format a number with Indian digit grouping (12,34,56,789)."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "-"
    neg = n < 0
    n = abs(n)
    whole = int(round(n)) if decimals == 0 else int(n)
    frac = "" if decimals == 0 else f"{n - int(n):.{decimals}f}"[1:]
    s = str(whole)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if neg else "") + s + frac


def fmt_inr(x: float, compact: bool = True) -> str:
    """Rupee amounts: compact uses crore (Cr) and lakh (L)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if compact and a >= 1e7:
        return f"{sign}₹{a / 1e7:,.2f} Cr"
    if compact and a >= 1e5:
        return f"{sign}₹{a / 1e5:,.2f} L"
    return f"{sign}₹{indian_group(a)}"


def fmt_t(x: float, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{indian_group(x, decimals)} t"


def fmt_pct(x: float, decimals: int = 1) -> str:
    """Format a fraction (0.123) as a percentage (12.3%)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 0) -> str:
    return indian_group(x, decimals)


def fmt_date(d) -> str:
    if d is None or (isinstance(d, float) and math.isnan(d)) or pd.isna(d):
        return "-"
    return pd.Timestamp(d).strftime(DATE_FMT)


def display_frame(df: pd.DataFrame, inr: list[str] = (), tonnes: list[str] = (), pct: list[str] = (),
                  dates: list[str] = (), num: list[str] = (), dec1: list[str] = (), rename: dict | None = None,
                  inr_full: list[str] = ()) -> pd.DataFrame:
    """Return a display copy with formatted text columns (raw data stays available for CSV export)."""
    out = df.copy()
    for c in inr:
        if c in out:
            out[c] = out[c].map(fmt_inr)
    for c in inr_full:
        if c in out:
            out[c] = out[c].map(lambda v: fmt_inr(v, compact=False))
    for c in tonnes:
        if c in out:
            out[c] = out[c].map(fmt_t)
    for c in pct:
        if c in out:
            out[c] = out[c].map(fmt_pct)
    for c in dates:
        if c in out:
            out[c] = out[c].map(fmt_date)
    for c in num:
        if c in out:
            out[c] = out[c].map(fmt_num)
    for c in dec1:
        if c in out:
            out[c] = out[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.1f}")
    if rename:
        out = out.rename(columns=rename)
    return out
