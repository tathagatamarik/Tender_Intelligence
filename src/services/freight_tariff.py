"""Synthetic freight tariff model shared by data generation, optimisation and order economics.

Trip cost = loaded km x (fixed rate per km x rate multiplier + litres per km x diesel price)
          + empty return km x the same per-km cost x empty-return cost ratio.
Empty return km = distance x (1 - backhaul share).
All rates are illustrative demonstration assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    DEFAULT_BACKHAUL_SHARE,
    DEFAULT_DIESEL_PRICE,
    DEPOT_HANDLING_PER_T,
    EMPTY_RETURN_COST_RATIO,
    MARKET_HIRE_PREMIUM,
    PLANT_HANDLING_PER_T,
    PRIMARY_FREIGHT_PER_TKM,
    STANDARD_UTILISATION,
    VEHICLE_TYPES,
)


@dataclass(frozen=True)
class TariffParams:
    """Adjustable freight assumptions (hashable so it can key caches)."""

    diesel_price: float = DEFAULT_DIESEL_PRICE
    rate_multiplier: float = 1.0
    backhaul_share: float = DEFAULT_BACKHAUL_SHARE
    plant_handling_per_t: float = PLANT_HANDLING_PER_T
    depot_handling_per_t: float = DEPOT_HANDLING_PER_T
    primary_freight_per_tkm: float = PRIMARY_FREIGHT_PER_TKM


def loaded_cost_per_km(vehicle_type: str, params: TariffParams = TariffParams()) -> float:
    """Cost of one loaded kilometre for a vehicle type."""
    v = VEHICLE_TYPES[vehicle_type]
    return v["fixed_rate_per_km"] * params.rate_multiplier + v["litres_per_km"] * params.diesel_price


def trip_cost(distance_km, vehicle_type: str, params: TariffParams = TariffParams(),
              market_hire: bool = False):
    """Total round-trip cost for one vehicle (scalar or array distances)."""
    per_km = loaded_cost_per_km(vehicle_type, params)
    distance_km = np.asarray(distance_km, dtype=float)
    empty_km = distance_km * (1.0 - params.backhaul_share)
    cost = distance_km * per_km + empty_km * per_km * EMPTY_RETURN_COST_RATIO
    if market_hire:
        cost = cost * (1.0 + MARKET_HIRE_PREMIUM)
    return cost


def empty_km(distance_km, params: TariffParams = TariffParams()):
    """Expected empty return kilometres for a trip."""
    return np.asarray(distance_km, dtype=float) * (1.0 - params.backhaul_share)


def secondary_freight_per_t(distance_km, params: TariffParams = TariffParams(),
                            vehicle_type: str = "MAV-25"):
    """Per-tonne secondary freight tariff at the standard load factor."""
    cap = VEHICLE_TYPES[vehicle_type]["capacity_t"]
    return trip_cost(distance_km, vehicle_type, params) / (cap * STANDARD_UTILISATION)


def primary_freight_per_t(distance_km, params: TariffParams = TariffParams()):
    """Per-tonne plant-to-depot (primary) freight."""
    return np.asarray(distance_km, dtype=float) * params.primary_freight_per_tkm


def trucks_needed(quantity_t, vehicle_type: str):
    """Number of vehicles of a type required to move a quantity."""
    cap = VEHICLE_TYPES[vehicle_type]["capacity_t"]
    return np.ceil(np.asarray(quantity_t, dtype=float) / cap - 1e-9).astype(int)
