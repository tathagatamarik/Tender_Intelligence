"""Geographic helpers: road-distance approximation between demonstration cities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CITIES, ROAD_DISTANCE_FACTOR

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres. Accepts scalars or NumPy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def road_km(lat1, lon1, lat2, lon2, min_km: float = 0.0):
    """Approximate road distance = great-circle distance x circuity factor."""
    return np.maximum(haversine_km(lat1, lon1, lat2, lon2) * ROAD_DISTANCE_FACTOR, min_km)


def cities_frame() -> pd.DataFrame:
    """Master table of demonstration cities."""
    df = pd.DataFrame(CITIES, columns=["city", "state", "region", "lat", "lon", "terrain"])
    df.insert(0, "city_id", [f"C{i + 1:03d}" for i in range(len(df))])
    return df


def distance_matrix(origins: pd.DataFrame, destinations: pd.DataFrame) -> np.ndarray:
    """Road-distance matrix (origins x destinations) using lat/lon columns."""
    return road_km(
        origins["lat"].to_numpy()[:, None],
        origins["lon"].to_numpy()[:, None],
        destinations["lat"].to_numpy()[None, :],
        destinations["lon"].to_numpy()[None, :],
    )
