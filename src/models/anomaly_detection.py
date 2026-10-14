"""Anomaly and early-warning detection.

1. Shipment anomalies: transparent GPS rules plus an Isolation Forest over trip behaviour,
   evaluated against simulated anomaly labels.
2. Project offtake early warning: forecast-versus-actual attainment, CUSUM change-point
   detection, order-velocity slope and cancellation / deferment z-scores. Predicted shortfalls
   are ESTIMATES, not confirmed cancellations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.eta_prediction import eta_features
from src.models.metrics import detection_metrics


# ---------------------------------------------------------------------------
# Shipment anomalies
# ---------------------------------------------------------------------------
@dataclass
class TripAnomalyResult:
    trips: pd.DataFrame
    metrics: pd.DataFrame


def detect_trip_anomalies(env, eta_model, contamination: float = 0.10, detour_threshold: float = 1.20,
                          halt_threshold_hours: float = 6.0, random_state: int = 0) -> TripAnomalyResult:
    t = env.trips.copy()
    t["eta_hours"] = eta_model.predict(eta_features(t))
    t["detour_ratio"] = t["gps_km"] / t["planned_km"].clip(lower=1)
    t["delay_ratio"] = t["actual_transit_hours"] / t["eta_hours"].clip(lower=0.5)
    expected_rest = np.floor(t["planned_transit_hours"] / 8.0)
    t["excess_stop_hours"] = (t["stop_hours"] - expected_rest).clip(lower=0)
    t["avg_speed_kmph"] = t["gps_km"] / t["actual_transit_hours"].clip(lower=0.5)

    t["rule_flag"] = (t["detour_ratio"] > detour_threshold) | (t["excess_stop_hours"] > halt_threshold_hours)
    feats = t[["detour_ratio", "delay_ratio", "excess_stop_hours", "avg_speed_kmph"]].to_numpy()
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso.fit(feats)
    t["iforest_score"] = -iso.score_samples(feats)
    t["iforest_flag"] = iso.predict(feats) == -1
    t["anomaly_reason"] = np.select(
        [t["detour_ratio"] > detour_threshold, t["excess_stop_hours"] > halt_threshold_hours, t["iforest_flag"]],
        ["ROUTE DEVIATION: GPS distance " + (t["detour_ratio"] * 100 - 100).round(0).astype(int).astype(str) + "% above plan",
         "UNSCHEDULED HALT: " + t["excess_stop_hours"].round(1).astype(str) + " h beyond expected rest",
         "STATISTICAL OUTLIER (Isolation Forest)"], "")
    t["delay_hours"] = (t["loading_hours"] + t["actual_transit_hours"] - t["promised_hours"]).clip(lower=0)
    metrics = pd.DataFrame({
        "GPS rules (transparent)": detection_metrics(t["true_anomaly"], t["rule_flag"]),
        "Isolation Forest (unsupervised)": detection_metrics(t["true_anomaly"], t["iforest_flag"]),
        "Rules OR Isolation Forest": detection_metrics(t["true_anomaly"], t["rule_flag"] | t["iforest_flag"]),
    }).T
    return TripAnomalyResult(trips=t, metrics=metrics)


# ---------------------------------------------------------------------------
# Project offtake early warning
# ---------------------------------------------------------------------------
def cusum_change_point(shortfall: np.ndarray, k: float = 0.10, h: float = 0.45) -> int | None:
    """One-sided CUSUM on attainment shortfall. Returns index where the alarm triggers, else None."""
    s = 0.0
    for i, x in enumerate(shortfall):
        s = max(0.0, s + x - k)
        if s > h:
            return i
    return None


@dataclass
class OfftakeResult:
    projects: pd.DataFrame
    metrics: pd.DataFrame


def offtake_early_warning(env, lookback_months: int = 6, forward_months: int = 3, attain_threshold: float = 0.80,
                          production_cost_per_t: float = 4300.0) -> OfftakeResult:
    pm = env.project_monthly
    as_of_month = env.as_of.to_period("M").to_timestamp()
    hist = pm[pm["actual_t"].notna()].copy()
    hist["attainment"] = hist["actual_t"] / hist["planned_t"].clip(lower=1)
    fleet_cancel = hist.groupby("project_id")["cancellations"].apply(lambda s: s.tail(3).mean())
    c_mu, c_sd = fleet_cancel.mean(), fleet_cancel.std() or 1.0
    defer = hist.groupby("project_id")["deferments"].apply(lambda s: s.tail(3).mean())
    d_mu, d_sd = defer.mean(), defer.std() or 1.0

    rows = []
    for pid, g in hist.groupby("project_id"):
        g = g.sort_values("month").tail(lookback_months)
        att = g["attainment"].to_numpy()
        recent3 = float(att[-3:].mean()) if len(att) else 1.0
        slope = float(np.polyfit(np.arange(len(att)), att, 1)[0]) if len(att) >= 3 else 0.0
        cp = cusum_change_point(1.0 - att)
        cancel_z = (fleet_cancel.get(pid, 0) - c_mu) / c_sd
        defer_z = (defer.get(pid, 0) - d_mu) / d_sd
        pay_delay = float(g["payment_delay_days"].tail(3).mean())
        # Forecast attainment: recent level when a change point fired, otherwise smoothed level
        f_att = recent3 if cp is not None else float(0.6 * recent3 + 0.4 * att.mean())
        f_att = float(np.clip(f_att + slope * 1.5, 0.0, 1.3))
        future = pm[(pm["project_id"] == pid) & (pm["month"] >= as_of_month)].sort_values("month").head(forward_months)
        planned_fwd = float(future["planned_t"].sum())
        forecast_fwd = planned_fwd * f_att
        at_risk = max(0.0, planned_fwd - forecast_fwd)
        score = (2.0 * max(0.0, attain_threshold - recent3) + 3.0 * max(0.0, -slope)
                 + 0.25 * max(0.0, cancel_z) + 0.2 * max(0.0, defer_z) + (0.5 if cp is not None else 0.0)
                 + 0.01 * max(0.0, pay_delay - 30))
        if score >= 0.9 or (recent3 < 0.6):
            level = "ACTION REQUIRED"
        elif score >= 0.45:
            level = "WATCH"
        else:
            level = "NORMAL"
        reasons = []
        if recent3 < attain_threshold:
            reasons.append(f"3-month offtake at {recent3:.0%} of plan")
        if slope < -0.03:
            reasons.append(f"order velocity falling {abs(slope):.0%} of plan per month")
        if cp is not None:
            reasons.append(f"CUSUM change point in {g['month'].iloc[cp].strftime('%b-%Y')}")
        if cancel_z > 1.5:
            reasons.append(f"cancellations unusually high (z = {cancel_z:.1f})")
        if defer_z > 1.5:
            reasons.append(f"deferments unusually high (z = {defer_z:.1f})")
        if pay_delay > 45:
            reasons.append(f"payments averaging {pay_delay:.0f} days late")
        rows.append({"project_id": pid, "recent_attainment": recent3, "attainment_slope": slope,
                     "change_point_month": g["month"].iloc[cp] if cp is not None else pd.NaT,
                     "cancellation_z": cancel_z, "deferment_z": defer_z, "avg_payment_delay_days": pay_delay,
                     "forecast_attainment": f_att, "planned_next_t": planned_fwd, "forecast_next_t": forecast_fwd,
                     "at_risk_t": at_risk, "alert_score": score, "alert_level": level,
                     "alert_reasons": "; ".join(reasons) if reasons else "Offtake tracking plan"})
    res = pd.DataFrame(rows).merge(env.projects, on="project_id", how="left")
    res["inventory_exposure_inr"] = res["at_risk_t"] * production_cost_per_t
    res["working_capital_impact_inr"] = res["inventory_exposure_inr"] * (1 + res["avg_payment_delay_days"].clip(0, 120) / 365)
    res["recommended_action"] = np.select(
        [res["alert_level"] == "ACTION REQUIRED", res["alert_level"] == "WATCH"],
        ["Sales follow-up within 7 days; hold project-dedicated production; re-plan dispatch schedule",
         "Confirm next month's schedule with customer; review allocated stock"], "No action")
    flagged = res["alert_level"].isin(["ACTION REQUIRED", "WATCH"])
    metrics = pd.DataFrame({
        "ACTION REQUIRED only": detection_metrics(res["true_slowdown"], res["alert_level"] == "ACTION REQUIRED"),
        "ACTION REQUIRED or WATCH": detection_metrics(res["true_slowdown"], flagged),
    }).T
    return OfftakeResult(projects=res.sort_values("alert_score", ascending=False).reset_index(drop=True),
                         metrics=metrics)
