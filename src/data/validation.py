"""Input schema and data-quality validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schemas import TABLE_SCHEMAS


@dataclass
class ValidationReport:
    table: str
    rows: int
    missing_columns: list[str] = field(default_factory=list)
    type_issues: list[str] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)
    negative_quantity_rows: int = 0

    @property
    def passed(self) -> bool:
        return not self.missing_columns and not self.type_issues and self.negative_quantity_rows == 0

    def as_row(self) -> dict:
        return {
            "Table": self.table,
            "Rows": self.rows,
            "Missing required columns": ", ".join(self.missing_columns) or "None",
            "Type issues": ", ".join(self.type_issues) or "None",
            "Nulls in required columns": sum(self.null_counts.values()),
            "Negative quantities": self.negative_quantity_rows,
            "Status": "PASSED" if self.passed else "NEEDS REVIEW",
        }


def _type_ok(series: pd.Series, expected: str) -> bool:
    if expected == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    if expected == "float":
        return pd.api.types.is_numeric_dtype(series)
    if expected == "int":
        return pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series)
    return True


def validate_table(name: str, df: pd.DataFrame) -> ValidationReport:
    """Validate a table against its declared required columns and types."""
    schema = TABLE_SCHEMAS.get(name, {})
    report = ValidationReport(table=name, rows=len(df))
    for col, typ in schema.items():
        if col not in df.columns:
            report.missing_columns.append(col)
            continue
        if not _type_ok(df[col], typ):
            report.type_issues.append(f"{col} (expected {typ})")
        n_null = int(df[col].isna().sum())
        if n_null:
            report.null_counts[col] = n_null
    for qcol in ("qty_t", "quantity_t", "on_hand_t", "volume_t"):
        if qcol in df.columns and pd.api.types.is_numeric_dtype(df[qcol]):
            report.negative_quantity_rows += int((df[qcol] < 0).sum())
    return report


def validate_environment(env) -> pd.DataFrame:
    """Validate every declared table in the environment and return a summary frame."""
    rows = [validate_table(name, getattr(env, name)).as_row() for name in TABLE_SCHEMAS if hasattr(env, name)]
    return pd.DataFrame(rows)


def clamp(value: float, lo: float, hi: float) -> float:
    """Bound a numeric user input to a safe range."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if v != v:  # NaN
        return lo
    return max(lo, min(hi, v))
