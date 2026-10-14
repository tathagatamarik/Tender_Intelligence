"""Pydantic schemas for validated inputs and structured extraction outputs."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Required columns per table, used by validation.validate_table
TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "dealers": {"dealer_id": "object", "region": "object", "dealer_segment": "object", "depot_id": "object",
                "credit_limit_inr": "float"},
    "dealer_weekly": {"series_id": "object", "week_start": "datetime", "qty_t": "float"},
    "orders": {"order_id": "object", "customer_id": "object", "product_id": "object", "qty_t": "float",
               "order_date": "datetime", "list_price_per_t": "float", "discount_per_t": "float",
               "production_cost_per_t": "float", "freight_per_t": "float", "handling_per_t": "float"},
    "invoices": {"invoice_id": "object", "customer_id": "object", "invoice_date": "datetime",
                 "due_date": "datetime", "amount_inr": "float"},
    "trips": {"trip_id": "object", "planned_km": "float", "actual_transit_hours": "float",
              "vehicle_type": "object"},
    "tenders_hist": {"tender_id": "object", "quantity_t": "float", "eligibility_score": "float", "won": "int"},
    "inventory": {"location_id": "object", "product_id": "object", "on_hand_t": "float", "available_t": "float"},
    "open_orders": {"open_order_id": "object", "customer_id": "object", "qty_t": "float", "deadline_hours": "int"},
    "pricing_panel": {"region": "object", "net_price_per_t": "float", "volume_t": "float",
                      "competitor_price_per_t": "float"},
    "project_monthly": {"project_id": "object", "month": "datetime", "planned_t": "float"},
}


class TenderExtraction(BaseModel):
    """Fields extracted deterministically from a tender document."""

    tender_id: Optional[str] = None
    issuing_authority: Optional[str] = None
    portal: Optional[str] = None
    project_location: Optional[str] = None
    quantity_t: Optional[float] = Field(default=None, ge=0)
    cement_requirement: Optional[str] = None
    product_category: Optional[str] = None
    delivery_months: Optional[int] = Field(default=None, ge=0, le=120)
    closing_date: Optional[date] = None
    emd_inr: Optional[float] = Field(default=None, ge=0)
    performance_guarantee_pct: Optional[float] = Field(default=None, ge=0, le=100)
    payment_terms_days: Optional[int] = Field(default=None, ge=0, le=720)
    penalty_pct_per_week: Optional[float] = Field(default=None, ge=0, le=100)
    penalty_cap_pct: Optional[float] = Field(default=None, ge=0, le=100)
    min_turnover_cr: Optional[float] = Field(default=None, ge=0)
    min_experience_years: Optional[int] = Field(default=None, ge=0, le=100)
    required_documents: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)


class ComplianceRequirement(BaseModel):
    """One numeric requirement extracted from a project specification."""

    parameter: str
    parameter_label: str
    operator: Literal[">=", "<="]
    limit: float
    unit: str
    source: str


class ROIInputs(BaseModel):
    """Validated ROI simulator inputs. All values are editable demonstration assumptions."""

    monthly_volume_t: float = Field(ge=0)
    freight_cost_per_t: float = Field(ge=0)
    addressable_share: float = Field(ge=0, le=1)
    freight_saving_pct: float = Field(ge=0, le=0.5)
    inventory_value_inr: float = Field(ge=0)
    holding_cost_pct: float = Field(ge=0, le=1)
    inventory_reduction_pct: float = Field(ge=0, le=0.8)
    overdue_receivables_inr: float = Field(ge=0)
    receivable_release_pct: float = Field(ge=0, le=1)
    expected_credit_loss_inr: float = Field(ge=0)
    credit_loss_reduction_pct: float = Field(ge=0, le=1)
    incremental_volume_pct: float = Field(ge=0, le=0.5)
    contribution_per_t: float = Field(ge=0)
    overlap_adjustment_pct: float = Field(ge=0, le=0.9)
    financing_rate: float = Field(ge=0, le=0.5)
    implementation_cost_inr: float = Field(gt=0)
    annual_software_cost_inr: float = Field(ge=0)
    annual_operating_cost_inr: float = Field(ge=0)

    @field_validator("implementation_cost_inr")
    @classmethod
    def _positive_impl(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Implementation cost must be positive for ROI and payback calculations.")
        return v
