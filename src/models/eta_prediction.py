"""Delivery ETA (transit time) prediction from historical GPS-tracked trips.

Model: gradient boosting regression with absolute-error loss (robust to GPS anomalies),
evaluated on a chronological holdout. Baseline: planned transit time = distance / nominal speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import VEHICLE_TYPES

FEATURES = ["planned_km", "vehicle_code", "route_code", "monsoon", "night_departure", "from_plant", "qty_t"]
ROUTE_CODES = {"Local / District": 0, "State Highway": 1, "National Highway": 2, "Hilly": 3}
VEHICLE_CODES = {v: i for i, v in enumerate(VEHICLE_TYPES)}


def route_type(distance_km: float, hilly: bool) -> str:
    if hilly:
        return "Hilly"
    if distance_km > 250:
        return "National Highway"
    if distance_km > 80:
        return "State Highway"
    return "Local / District"


def eta_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "planned_km": df["planned_km"].astype(float),
        "vehicle_code": df["vehicle_type"].map(VEHICLE_CODES).astype(float),
        "route_code": df["route_type"].map(ROUTE_CODES).astype(float),
        "monsoon": df["month"].isin([7, 8, 9]).astype(float),
        "night_departure": ((df["departure_hour"] >= 20) | (df["departure_hour"] <= 4)).astype(float),
        "from_plant": (df["source_type"] == "Plant").astype(float),
        "qty_t": df["qty_t"].astype(float),
    })


@dataclass
class ETAModelResult:
    model: HistGradientBoostingRegressor
    metrics: pd.DataFrame
    holdout: pd.DataFrame
    train_size: int
    test_size: int


def train_eta_model(env, test_share: float = 0.2, random_state: int = 0) -> ETAModelResult:
    trips = env.trips.sort_values("dispatch_date")
    n_test = int(len(trips) * test_share)
    train, test = trips.iloc[:-n_test], trips.iloc[-n_test:]
    model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, random_state=random_state)
    model.fit(eta_features(train), train["actual_transit_hours"])
    pred = model.predict(eta_features(test))
    base = test["planned_transit_hours"].to_numpy()
    actual = test["actual_transit_hours"].to_numpy()
    err_m, err_b = pred - actual, base - actual

    def summary(err):
        return {"MAE (hours)": float(np.mean(np.abs(err))), "Median abs error (hours)": float(np.median(np.abs(err))),
                "P90 abs error (hours)": float(np.percentile(np.abs(err), 90)), "Mean error (hours)": float(np.mean(err))}

    metrics = pd.DataFrame({"Gradient boosting ETA model": summary(err_m),
                            "Baseline: distance / nominal speed": summary(err_b)}).T
    holdout = test[["trip_id", "planned_km", "route_type", "vehicle_type", "actual_transit_hours",
                    "true_anomaly"]].assign(predicted_hours=pred, baseline_hours=base, error_hours=err_m)
    return ETAModelResult(model=model, metrics=metrics, holdout=holdout, train_size=len(train), test_size=len(test))


def predict_transit_hours(model, distance_km, vehicle_type, hilly, from_plant, qty_t, month: int,
                          departure_hour: int = 8) -> np.ndarray:
    """Vectorised ETA predictions for candidate dispatch options."""
    distance_km = np.asarray(distance_km, dtype=float)
    n = len(distance_km)
    df = pd.DataFrame({
        "planned_km": distance_km,
        "vehicle_type": np.broadcast_to(np.asarray(vehicle_type), n),
        "route_type": [route_type(d, h) for d, h in zip(distance_km, np.broadcast_to(np.asarray(hilly), n))],
        "month": month, "departure_hour": departure_hour,
        "source_type": np.where(np.broadcast_to(np.asarray(from_plant), n), "Plant", "Depot"),
        "qty_t": np.broadcast_to(np.asarray(qty_t, dtype=float), n),
    })
    return np.clip(model.predict(eta_features(df)), 0.3, None)
