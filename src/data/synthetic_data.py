"""Coherent synthetic business environment for the Dalmia AI Intelligence Hub demo.

One call to :func:`generate_environment` produces every master and transactional table used
by the application, from a single random seed, so that entity IDs and business definitions
are consistent across pages. Relationships between variables are modelled explicitly
(seasonality, dealer size, price, distance, payment behaviour, eligibility and so on) rather
than drawn independently.

ALL DATA IS SYNTHETIC. No real customer, dealer, plant or company performance data is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    AS_OF_DATE,
    DEALER_POPULATION,
    DEPOTS,
    NON_TRADE_PAYMENT_TERMS,
    PLANTS,
    PRODUCTS,
    REGION_DEMAND_WEIGHT,
    REGIONS,
    TRADE_PAYMENT_TERMS,
    VEHICLE_TYPES,
    DataSettings,
)
from src.data.geography import cities_frame, distance_matrix, road_km
from src.services.freight_tariff import (
    TariffParams,
    primary_freight_per_t,
    secondary_freight_per_t,
    trip_cost,
)

SYNTHETIC_LABEL = "SIMULATED"

DEALER_TIERS = ["Key Account", "Large", "Medium", "Small"]
INSTITUTIONAL_SEGMENTS = [
    "Government Department", "Infrastructure EPC", "RMC Producer", "Real Estate Developer", "Industrial",
]
PROJECT_STAGES = ["Mobilisation", "Foundation", "Structure", "Finishing"]

_NAME_PREFIX = ["Sri", "Shree", "New", "Maa", "Jai", "Om", "Royal", "Sai", "Bharat", "Annapurna",
                "Laxmi", "Ganesh", "Balaji", "Durga", "Kaveri", "Ganga", "Vinayak", "Mahalaxmi"]
_NAME_CORE = ["Cement", "Building", "Hardware", "Construction", "Infra", "Steel and Cement", "Traders",
              "Agencies", "Build Mart", "Materials"]
_NAME_SUFFIX = ["Traders", "Enterprises", "Agency", "Stores", "Distributors", "and Sons", "Depot", "Corporation"]
_INST_CORE = ["Adarsh", "Bharat", "Pragati", "Unnati", "Sagar", "Shikhar", "Vikas", "Navyug", "Samrat",
              "Kalpataru", "Meghdoot", "Sahyadri", "Kaveri", "Brahmaputra", "Mahanadi", "Deccan"]
_INST_SUFFIX = {
    "Infrastructure EPC": ["Infracon Pvt Ltd", "Infra Projects Ltd", "Constructions Ltd", "Engineering Ltd"],
    "RMC Producer": ["Ready Mix Pvt Ltd", "RMC Solutions", "Concrete Works"],
    "Real Estate Developer": ["Developers Pvt Ltd", "Realty Ltd", "Housing Pvt Ltd", "Estates"],
    "Industrial": ["Industries Ltd", "Precast Products", "Pipes and Poles Ltd"],
}
_GOV_BODIES = ["Public Works Department", "Rural Development Department", "Water Resources Department",
               "Municipal Corporation", "Housing Board", "Irrigation Department", "Road Development Authority",
               "Urban Development Authority"]
_SCHEMES = ["Monsoon Volume Scheme", "Festive Loyalty Scheme", "Quarter-End Target Scheme",
            "New Dealer Activation Scheme", "No Scheme"]


@dataclass
class Environment:
    """Container for all synthetic tables (treat as read-only once generated)."""

    settings: DataSettings
    as_of: pd.Timestamp
    cities: pd.DataFrame
    plants: pd.DataFrame
    depots: pd.DataFrame
    products: pd.DataFrame
    sales_reps: pd.DataFrame
    dealers: pd.DataFrame
    dealer_series: pd.DataFrame
    dealer_weekly: pd.DataFrame
    institutional_customers: pd.DataFrame
    customers: pd.DataFrame
    projects: pd.DataFrame
    project_monthly: pd.DataFrame
    orders: pd.DataFrame
    invoices: pd.DataFrame
    vehicles: pd.DataFrame
    trips: pd.DataFrame
    inventory: pd.DataFrame
    open_orders: pd.DataFrame
    company_documents: pd.DataFrame
    tenders_hist: pd.DataFrame
    tenders_open: pd.DataFrame
    tender_documents: dict[str, str]
    product_specs: pd.DataFrame
    lab_tests: pd.DataFrame
    product_certificates: pd.DataFrame
    compliance_projects: pd.DataFrame
    compliance_requirements: pd.DataFrame
    compliance_documents: dict[str, str]
    consultant_approvals: pd.DataFrame
    pricing_panel: pd.DataFrame
    region_price_index: pd.DataFrame
    location_distance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def sources(self) -> pd.DataFrame:
        """Union of plants and depots as dispatch sources."""
        p = self.plants.rename(columns={"plant_id": "source_id", "plant_name": "source_name"})
        p["source_type"] = "Plant"
        d = self.depots.rename(columns={"depot_id": "source_id", "depot_name": "source_name"})
        d["source_type"] = "Depot"
        cols = ["source_id", "source_name", "source_type", "city", "region", "lat", "lon", "terrain"]
        return pd.concat([p[cols], d[cols]], ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seasonality(week_of_year: np.ndarray) -> np.ndarray:
    """Indian cement seasonality: strong Jan-May, monsoon dip Jul-Sep, festive recovery."""
    w = np.asarray(week_of_year, dtype=float)
    base = 1.0 + 0.16 * np.cos(2 * np.pi * (w - 12) / 52.0)
    monsoon = np.where((w >= 27) & (w <= 38), -0.20, 0.0)
    festive = np.where((w >= 41) & (w <= 46), 0.07, 0.0)
    return base + monsoon + festive


def _month_seasonality(month: np.ndarray) -> np.ndarray:
    return _seasonality((np.asarray(month) - 0.5) * 52.0 / 12.0)


def _dealer_name(rng: np.random.Generator) -> str:
    return f"{rng.choice(_NAME_PREFIX)} {rng.choice(_NAME_CORE)} {rng.choice(_NAME_SUFFIX)}"


def _nearest(lat, lon, targets: pd.DataFrame, mask: np.ndarray | None = None) -> tuple[int, float]:
    d = road_km(lat, lon, targets["lat"].to_numpy(), targets["lon"].to_numpy())
    if mask is not None:
        d = np.where(mask, d, np.inf)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def _attach_city(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    return df.merge(cities[["city", "state", "region", "lat", "lon", "terrain"]], on="city", how="left")


# ---------------------------------------------------------------------------
# Master data
# ---------------------------------------------------------------------------
def _build_master(rng: np.random.Generator):
    cities = cities_frame()
    plants = _attach_city(pd.DataFrame(PLANTS), cities)
    depots = pd.DataFrame(DEPOTS)
    depots["depot_name"] = depots["city"] + " Depot (Synthetic)"
    depots = _attach_city(depots, cities)
    # Mother plant per depot and category: nearest plant producing that category.
    products = pd.DataFrame(PRODUCTS)
    mother = []
    for _, dep in depots.iterrows():
        idx, dist = _nearest(dep["lat"], dep["lon"], plants)
        mother.append((plants.iloc[idx]["plant_id"], dist))
    depots["mother_plant_id"] = [m[0] for m in mother]
    depots["mother_plant_km"] = [round(m[1], 1) for m in mother]
    return cities, plants, depots, products


def nearest_capable_plant(lat: float, lon: float, category: str, plants: pd.DataFrame) -> tuple[str, float]:
    """Nearest plant that manufactures a product category."""
    mask = plants["categories"].apply(lambda c: category in c).to_numpy()
    idx, dist = _nearest(lat, lon, plants, mask)
    return plants.iloc[idx]["plant_id"], dist


def _build_sales_reps(rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        for k in range(int(rng.integers(3, 6))):
            rows.append({"rep_id": f"SR-{region[:2].upper()}{k + 1:02d}", "rep_name": fake.name(),
                         "region": region})
    return pd.DataFrame(rows)


def _build_dealers(rng, fake, n_dealers, cities, depots, plants, reps, as_of) -> pd.DataFrame:
    regions = np.array(REGIONS)
    w = np.array([REGION_DEMAND_WEIGHT[r] for r in REGIONS])
    dealer_regions = rng.choice(regions, size=n_dealers, p=w / w.sum())
    tiers = rng.choice(DEALER_TIERS, size=n_dealers, p=[0.07, 0.20, 0.38, 0.35])
    tier_size = {"Key Account": 3.0, "Large": 1.8, "Medium": 1.0, "Small": 0.55}
    rows = []
    for i in range(n_dealers):
        region = dealer_regions[i]
        cand = cities[cities["region"] == region]
        city = cand.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        d_idx, d_km = _nearest(city["lat"], city["lon"], depots)
        rep_pool = reps[reps["region"] == region]["rep_id"].to_numpy()
        size = tier_size[tiers[i]] * float(rng.lognormal(0, 0.35))
        risk = float(np.clip(rng.beta(2.0, 6.0) + (0.08 if tiers[i] == "Small" else 0.0), 0.01, 0.95))
        rows.append({
            "dealer_id": f"D{i + 1:05d}",
            "dealer_name": _dealer_name(rng),
            "region": region,
            "state": city["state"],
            "city": city["city"],
            "district": f"{city['city']} District",
            "micro_market": f"{city['city']} {rng.choice(['North', 'South', 'Central'])}",
            "lat": city["lat"], "lon": city["lon"], "terrain": city["terrain"],
            "dealer_segment": tiers[i],
            "size_factor": round(size, 3),
            "depot_id": depots.iloc[d_idx]["depot_id"],
            "depot_km": round(d_km, 1),
            "last_mile_km": round(float(rng.uniform(6, 45)), 1),
            "sales_rep_id": rng.choice(rep_pool),
            "onboarded_on": as_of - pd.Timedelta(days=int(rng.integers(400, 4000))),
            "payment_terms_days": int(rng.choice(TRADE_PAYMENT_TERMS, p=[0.15, 0.45, 0.25, 0.15])),
            "credit_limit_inr": float(round(size * rng.uniform(9, 16) * 1e5, -4)),
            "base_discount_per_t": float(round(
                {"Key Account": 330, "Large": 270, "Medium": 210, "Small": 150}[tiers[i]]
                + rng.normal(0, 30), 0)),
            "latent_risk": risk,
            "uses_digital_ordering": bool(rng.random() < 0.78),
            "data_label": SYNTHETIC_LABEL,
        })
    return pd.DataFrame(rows)


def _region_price_index(rng, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly regional price index (1.0 = base list price). Shared by orders and pricing."""
    rows = []
    region_level = {"East": 0.98, "North-East": 1.07, "South": 1.02, "West": 1.00, "Central": 0.97, "North": 1.01}
    for region in REGIONS:
        level = 1.0
        for m in months:
            shock = rng.normal(0, 0.012)
            seasonal = 0.02 if m.month in (1, 2, 3, 4) else (-0.025 if m.month in (7, 8, 9) else 0.0)
            level = 0.85 * level + 0.15 * 1.0 + shock
            rows.append({"region": region, "month": m,
                         "price_index": round(region_level[region] * (level + seasonal), 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dealer demand history
# ---------------------------------------------------------------------------
def _build_dealer_series(rng, dealers, products) -> pd.DataFrame:
    prod_ids = products["product_id"].to_numpy()
    base_share = products["trade_share"].to_numpy()
    rows = []
    for _, d in dealers.iterrows():
        share = base_share.copy()
        if d["region"] == "East":
            share[3] *= 1.8  # PSC more common near slag sources (simulation assumption)
        if d["region"] == "South":
            share[2] *= 1.3
        share = share / share.sum()
        n_prod = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
        chosen = rng.choice(prod_ids, size=n_prod, replace=False, p=share)
        weights = rng.dirichlet(np.ones(n_prod) * 2.0)
        for pid, wgt in zip(chosen, weights):
            rows.append({"dealer_id": d["dealer_id"], "product_id": pid, "mix_weight": float(wgt)})
    s = pd.DataFrame(rows)
    s.insert(0, "series_id", [f"S{i + 1:05d}" for i in range(len(s))])
    return s


def _build_dealer_weekly(rng, dealers, series, weeks: pd.DatetimeIndex):
    """Weekly dealer-product sell-in with seasonality, trend, regime changes and intermittency."""
    d_idx = dealers.set_index("dealer_id")
    n_s, n_w = len(series), len(weeks)
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    season = _seasonality(woy)[None, :]
    size = d_idx.loc[series["dealer_id"], "size_factor"].to_numpy()
    region_w = np.array([REGION_DEMAND_WEIGHT[r] for r in d_idx.loc[series["dealer_id"], "region"]])
    level = 4.4 * size * region_w * (0.4 + series["mix_weight"].to_numpy())  # tonnes per week
    t = np.arange(n_w)[None, :] / 52.0

    pattern = rng.choice(["stable", "growing", "declining", "volatile"], size=n_s, p=[0.62, 0.13, 0.15, 0.10])
    slope = rng.normal(0.0, 0.05, size=n_s)
    slope[pattern == "growing"] = rng.uniform(0.15, 0.35, size=(pattern == "growing").sum())
    trend = 1.0 + slope[:, None] * t

    decline = np.ones((n_s, n_w))
    change_week = np.full(n_s, -1)
    for i in np.where(pattern == "declining")[0]:
        cw = int(rng.integers(n_w - 26, n_w - 6))
        change_week[i] = cw
        depth = rng.uniform(0.35, 0.65)
        ramp = np.clip((np.arange(n_w) - cw) / 8.0, 0, 1)
        decline[i] = 1.0 - depth * ramp

    vol = np.where(pattern == "volatile", 0.55, 0.28)[:, None]
    mean = np.clip(level[:, None] * season * trend * decline, 0.05, None)
    p_order = np.clip(0.45 + 0.18 * size, 0.35, 0.97)[:, None]
    ordered = rng.random((n_s, n_w)) < p_order
    shape = 1.0 / vol ** 2
    noise = rng.gamma(shape, 1.0 / shape)
    qty = np.where(ordered, mean / p_order * noise, 0.0)
    qty = np.round(qty * 2) / 2  # multiples of 0.5 t (10 bags of 50 kg)

    long = pd.DataFrame({
        "series_id": np.repeat(series["series_id"].to_numpy(), n_w),
        "week_start": np.tile(weeks.to_numpy(), n_s),
        "qty_t": qty.ravel(),
    })
    long = long.merge(series[["series_id", "dealer_id", "product_id"]], on="series_id", how="left")
    meta = pd.DataFrame({
        "series_id": series["series_id"],
        "true_pattern": pattern,
        "decline_start_week": change_week,
        "expected_weekly_t": mean[:, -8:].mean(axis=1).round(2),
    })
    return long, meta


# ---------------------------------------------------------------------------
# Institutional customers and projects
# ---------------------------------------------------------------------------
def _build_institutional(rng, n_inst, cities) -> pd.DataFrame:
    seg = rng.choice(INSTITUTIONAL_SEGMENTS, size=n_inst, p=[0.25, 0.25, 0.20, 0.20, 0.10])
    rows = []
    for i in range(n_inst):
        city = cities.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        if seg[i] == "Government Department":
            name = f"{rng.choice(_GOV_BODIES)} - {city['city']} Division {int(rng.integers(1, 9))} (Simulated)"
            channel = rng.choice(["GeM", "GePNIC", "Direct"], p=[0.45, 0.45, 0.10])
            risk = float(np.clip(rng.beta(3.0, 5.0), 0.05, 0.9))  # slow payers, low default
            lgd_class = "Government"
        else:
            name = f"{rng.choice(_INST_CORE)} {rng.choice(_INST_SUFFIX[seg[i]])} (Synthetic)"
            channel = rng.choice(["Direct", "GeM"], p=[0.85, 0.15])
            risk = float(np.clip(rng.beta(2.0, 5.5) + (0.07 if seg[i] == "Real Estate Developer" else 0.0),
                                 0.02, 0.95))
            lgd_class = "Corporate"
        size = float(rng.lognormal(0.0, 0.5))
        rows.append({
            "customer_id": f"INS{i + 1:04d}",
            "customer_name": name,
            "customer_segment": seg[i],
            "institution_type": "Government" if seg[i] == "Government Department" else "Private",
            "procurement_channel": channel,
            "region": city["region"], "state": city["state"], "city": city["city"],
            "lat": city["lat"], "lon": city["lon"], "terrain": city["terrain"],
            "size_factor": round(size, 3),
            "payment_terms_days": int(rng.choice(NON_TRADE_PAYMENT_TERMS, p=[0.25, 0.3, 0.3, 0.15])),
            "credit_limit_inr": float(round(size * rng.uniform(25, 60) * 1e5, -4)),
            "latent_risk": risk,
            "lgd_class": lgd_class,
            "data_label": SYNTHETIC_LABEL,
        })
    return pd.DataFrame(rows)


def _build_projects(rng, inst, as_of: pd.Timestamp, n_projects: int):
    months_back = 18
    month_start = (as_of - pd.DateOffset(months=months_back)).to_period("M").to_timestamp()
    as_of_month = as_of.to_period("M").to_timestamp()
    cust = inst.sample(n_projects, replace=True, random_state=int(rng.integers(1e9)),
                       weights=inst["size_factor"]).reset_index(drop=True)
    category_choice = ["OPC", "PPC", "PSC", "Composite", "Specialty"]
    proj_rows, month_rows = [], []
    for i in range(n_projects):
        c = cust.iloc[i]
        start = month_start + pd.DateOffset(months=int(rng.integers(0, months_back - 2)))
        duration = int(rng.integers(12, 31))
        peak = float(rng.uniform(70, 200) * c["size_factor"] ** 0.5)
        category = rng.choice(category_choice, p=[0.30, 0.30, 0.20, 0.12, 0.08])
        slowdown = bool(rng.random() < 0.22)
        months = pd.date_range(start, periods=duration, freq="MS")
        x = np.arange(duration) / max(duration - 1, 1)
        curve = np.clip(np.minimum(x / 0.25, 1.0) * np.minimum((1 - x) / 0.2 + 0.25, 1.0), 0.15, 1.0)
        planned = np.round(peak * curve, 1)
        hist_mask = months < as_of_month
        n_hist = int(hist_mask.sum())
        slow_start = None
        attain = rng.normal(1.0, 0.16, size=duration).clip(0.45, 1.40)
        if slowdown and n_hist >= 4:
            slow_start = int(rng.integers(max(1, n_hist - 6), n_hist - 1))
            depth = rng.uniform(0.25, 0.70)
            ramp = np.clip((np.arange(duration) - slow_start + 1) / 3.0, 0, 1)
            attain = attain * (1 - depth * ramp)
        else:
            slowdown = False
        elapsed = n_hist / duration
        stage = PROJECT_STAGES[min(int(elapsed / 0.25), 3)]
        proj_rows.append({
            "project_id": f"PRJ{i + 1:04d}",
            "project_name": f"{c['city']} {rng.choice(['Bridge', 'Housing', 'Highway', 'Metro', 'Canal', 'Industrial Park', 'Hospital', 'Commercial Complex'])} Project {i + 1} (Simulated)",
            "customer_id": c["customer_id"], "customer_segment": c["customer_segment"],
            "region": c["region"], "city": c["city"], "lat": c["lat"], "lon": c["lon"],
            "product_category": category,
            "start_month": start, "duration_months": duration,
            "committed_offtake_t": float(planned.sum()),
            "project_stage": stage,
            "true_slowdown": slowdown,
            "slowdown_start_month": months[slow_start] if slow_start is not None else pd.NaT,
            "data_label": SYNTHETIC_LABEL,
        })
        for k, m in enumerate(months):
            is_hist = m < as_of_month
            actual = float(np.round(planned[k] * attain[k], 1)) if is_hist else np.nan
            in_slow = slow_start is not None and k >= slow_start
            cancels = int(rng.poisson(1.6 if in_slow else 0.25)) if is_hist else 0
            defers = int(rng.poisson(1.8 if in_slow else 0.35)) if is_hist else 0
            pay_delay = float(max(0.0, rng.normal(8 + 40 * c["latent_risk"] + (18 if in_slow else 0), 6))) if is_hist else np.nan
            month_rows.append({
                "project_id": f"PRJ{i + 1:04d}", "month": m,
                "planned_t": float(planned[k]), "actual_t": actual,
                "scheduled_deliveries": int(max(1, round(planned[k] / 25))),
                "cancellations": cancels, "deferments": defers,
                "payment_delay_days": pay_delay,
            })
    return pd.DataFrame(proj_rows), pd.DataFrame(month_rows)


# ---------------------------------------------------------------------------
# Orders and invoices
# ---------------------------------------------------------------------------
def _build_orders(rng, dealers, dealer_weekly, projects, project_monthly, products, plants, depots,
                  price_idx: pd.DataFrame) -> pd.DataFrame:
    prod = products.set_index("product_id")
    plants_idx = plants.set_index("plant_id")
    depots_idx = depots.set_index("depot_id")
    pidx = price_idx.set_index(["region", "month"])["price_index"]
    tariff = TariffParams()

    # --- Dealer (trade) orders from weekly sell-in
    dw = dealer_weekly[dealer_weekly["qty_t"] > 0].copy()
    dw = dw.merge(dealers[["dealer_id", "region", "city", "lat", "lon", "dealer_segment", "depot_id",
                           "depot_km", "last_mile_km", "base_discount_per_t"]], on="dealer_id", how="left")
    dw["order_date"] = dw["week_start"] + pd.to_timedelta(rng.integers(0, 6, size=len(dw)), unit="D")
    dw["category"] = prod.loc[dw["product_id"], "category"].to_numpy()

    # Direct plant dispatch for larger orders; otherwise via the mapped depot.
    direct = (dw["qty_t"] >= 20) & (rng.random(len(dw)) < 0.6)
    # Pre-compute nearest capable plant per (dealer, category)
    key = dw[["dealer_id", "category"]].drop_duplicates()
    near = {}
    for _, r in key.merge(dealers[["dealer_id", "lat", "lon"]], on="dealer_id").iterrows():
        near[(r["dealer_id"], r["category"])] = nearest_capable_plant(r["lat"], r["lon"], r["category"], plants)
    depot_plant = {}
    for dep_id, dep in depots_idx.iterrows():
        for cat in products["category"].unique():
            depot_plant[(dep_id, cat)] = nearest_capable_plant(dep["lat"], dep["lon"], cat, plants)

    plant_ids, src_ids, src_type, primary_km, secondary_km = [], [], [], [], []
    for (did, cat, dep, dep_km, lm, is_direct) in zip(dw["dealer_id"], dw["category"], dw["depot_id"],
                                                        dw["depot_km"], dw["last_mile_km"], direct):
        if is_direct:
            pid, km = near[(did, cat)]
            plant_ids.append(pid); src_ids.append(pid); src_type.append("Plant")
            primary_km.append(0.0); secondary_km.append(km + lm)
        else:
            pid, pkm = depot_plant[(dep, cat)]
            plant_ids.append(pid); src_ids.append(dep); src_type.append("Depot")
            primary_km.append(pkm); secondary_km.append(dep_km + lm)
    dw["plant_id"] = plant_ids
    dw["source_id"] = src_ids
    dw["source_type"] = src_type
    dw["primary_km"] = np.round(primary_km, 1)
    dw["secondary_km"] = np.round(secondary_km, 1)
    dw["month"] = dw["order_date"].dt.to_period("M").dt.to_timestamp()
    dw["list_price_per_t"] = np.round(
        prod.loc[dw["product_id"], "base_price"].to_numpy()
        * pidx.reindex(list(zip(dw["region"], dw["month"]))).fillna(1.0).to_numpy(), 0)
    scheme_bonus = np.where(dw["month"].dt.month.isin([7, 8, 9]), 60.0, 0.0)
    dw["discount_per_t"] = np.round(np.clip(dw["base_discount_per_t"] + scheme_bonus
                                            + rng.normal(0, 25, len(dw)), 0, None), 0)
    dw["customer_id"] = dw["dealer_id"]
    dw["channel"] = "Trade"
    dw["customer_segment"] = dw["dealer_segment"]
    dw["project_id"] = None

    # --- Institutional (non-trade) orders from project monthly actuals
    pm = project_monthly[project_monthly["actual_t"] > 0].merge(
        projects[["project_id", "customer_id", "customer_segment", "region", "city", "lat", "lon",
                  "product_category"]], on="project_id", how="left")
    inst_rows = []
    cat_to_product = {"OPC": ["P01", "P02"], "PPC": ["P03"], "PSC": ["P04"], "Composite": ["P05"],
                      "Specialty": ["P06", "P07"]}
    plant_cache = {}
    for _, r in pm.iterrows():
        k = int(rng.integers(2, 6))
        parts = rng.dirichlet(np.ones(k) * 3) * r["actual_t"]
        key2 = (r["project_id"], r["product_category"])
        if key2 not in plant_cache:
            plant_cache[key2] = nearest_capable_plant(r["lat"], r["lon"], r["product_category"], plants)
        pid, km = plant_cache[key2]
        product_id = cat_to_product[r["product_category"]][int(r["project_id"][-1]) % len(cat_to_product[r["product_category"]])]
        for q in parts:
            inst_rows.append({
                "customer_id": r["customer_id"], "project_id": r["project_id"], "product_id": product_id,
                "category": r["product_category"], "qty_t": float(np.round(max(q, 1.0), 1)),
                "order_date": r["month"] + pd.Timedelta(days=int(rng.integers(0, 27))),
                "region": r["region"], "city": r["city"], "customer_segment": r["customer_segment"],
                "plant_id": pid, "source_id": pid, "source_type": "Plant", "primary_km": 0.0,
                "secondary_km": round(km + float(rng.uniform(5, 30)), 1),
            })
    inst = pd.DataFrame(inst_rows)
    if len(inst):
        inst["month"] = inst["order_date"].dt.to_period("M").dt.to_timestamp()
        inst["list_price_per_t"] = np.round(
            prod.loc[inst["product_id"], "base_price"].to_numpy() * 0.955
            * pidx.reindex(list(zip(inst["region"], inst["month"]))).fillna(1.0).to_numpy(), 0)
        inst["discount_per_t"] = np.round(np.clip(rng.normal(110, 40, len(inst)), 0, None), 0)
        inst["channel"] = "Non-Trade"

    cols = ["customer_id", "project_id", "product_id", "category", "qty_t", "order_date", "month", "region",
            "city", "customer_segment", "channel", "plant_id", "source_id", "source_type", "primary_km",
            "secondary_km", "list_price_per_t", "discount_per_t"]
    orders = pd.concat([dw[cols], inst[cols]], ignore_index=True) if len(inst) else dw[cols].copy()
    orders = orders.sort_values("order_date").reset_index(drop=True)
    orders.insert(0, "order_id", [f"ORD{i + 1:06d}" for i in range(len(orders))])

    # Cost components
    orders["production_cost_per_t"] = np.round(
        prod.loc[orders["product_id"], "base_cost"].to_numpy()
        + plants_idx.loc[orders["plant_id"], "cost_adj"].to_numpy()
        + rng.normal(0, 40, len(orders)), 0)
    sec_freight = secondary_freight_per_t(orders["secondary_km"].to_numpy(), tariff)
    pri_freight = primary_freight_per_t(orders["primary_km"].to_numpy(), tariff)
    orders["freight_per_t"] = np.round(sec_freight + pri_freight, 0)
    orders["handling_per_t"] = np.where(orders["source_type"] == "Depot",
                                        tariff.plant_handling_per_t + tariff.depot_handling_per_t,
                                        tariff.plant_handling_per_t)
    orders["dispatch_region"] = orders["region"]
    orders["data_label"] = SYNTHETIC_LABEL
    return orders


def _build_invoices(rng, orders, customers, as_of: pd.Timestamp) -> pd.DataFrame:
    """One invoice per order; payment timing driven by latent risk with time drift."""
    c = customers.set_index("customer_id")
    inv = orders[["order_id", "customer_id", "order_date", "qty_t", "list_price_per_t",
                  "discount_per_t", "channel"]].copy()
    inv["invoice_id"] = "INV" + inv["order_id"].str[3:]
    inv["invoice_date"] = inv["order_date"] + pd.to_timedelta(rng.integers(0, 2, len(inv)), unit="D")
    inv["amount_inr"] = np.round(inv["qty_t"] * (inv["list_price_per_t"] - inv["discount_per_t"]), 0)
    terms = c.loc[inv["customer_id"], "payment_terms_days"].to_numpy()
    inv["due_date"] = inv["invoice_date"] + pd.to_timedelta(terms, unit="D")

    risk0 = c.loc[inv["customer_id"], "latent_risk"].to_numpy()
    drift = c.loc[inv["customer_id"], "risk_drift"].to_numpy()
    start = inv["invoice_date"].min()
    progress = ((inv["invoice_date"] - start).dt.days / max((as_of - start).days, 1)).to_numpy()
    risk = np.clip(risk0 + drift * progress, 0.01, 0.98)
    monsoon = inv["invoice_date"].dt.month.isin([7, 8, 9]).to_numpy()
    is_gov = (c.loc[inv["customer_id"], "customer_segment"].to_numpy() == "Government Department")
    p_late = np.clip(0.03 + 0.85 * risk ** 1.3 + 0.05 * monsoon + 0.10 * is_gov, 0, 0.98)
    late = rng.random(len(inv)) < p_late
    late_days = rng.exponential(3 + 70 * risk ** 2.2 + 12 * is_gov)
    early_days = -rng.uniform(0, terms * 0.4)
    delay = np.where(late, np.ceil(late_days) + 1, np.floor(early_days))
    # A small share of very high-risk invoices remain unpaid (potential default)
    never = (risk > 0.6) & (rng.random(len(inv)) < 0.25 * risk)
    paid = inv["due_date"] + pd.to_timedelta(delay, unit="D")
    paid = paid.where(~never, pd.NaT)
    paid = paid.where(paid <= as_of, pd.NaT)
    inv["paid_date"] = paid
    inv["status"] = np.where(inv["paid_date"].notna(), "Paid",
                             np.where(inv["due_date"] < as_of, "Overdue", "Open - Not Due"))
    inv["days_past_due"] = np.where(inv["status"] == "Overdue", (as_of - inv["due_date"]).dt.days, 0)
    inv["data_label"] = SYNTHETIC_LABEL
    return inv[["invoice_id", "order_id", "customer_id", "invoice_date", "due_date", "paid_date",
                "amount_inr", "status", "days_past_due", "channel", "data_label"]]


# ---------------------------------------------------------------------------
# Logistics: vehicles, trips, inventory, open orders
# ---------------------------------------------------------------------------
def _build_vehicles(rng, plants, depots) -> pd.DataFrame:
    rows = []
    vid = 1
    for loc_id, loc_type, city in list(zip(plants["plant_id"], ["Plant"] * len(plants), plants["city"])) + \
            list(zip(depots["depot_id"], ["Depot"] * len(depots), depots["city"])):
        counts = ({"LCV-10": rng.integers(2, 5), "MAV-25": rng.integers(4, 9), "TRL-35": rng.integers(2, 6)}
                  if loc_type == "Plant" else
                  {"LCV-10": rng.integers(3, 7), "MAV-25": rng.integers(2, 5), "TRL-35": rng.integers(0, 2)})
        for vt, n in counts.items():
            for _ in range(int(n)):
                status = rng.choice(["Available", "Available", "Available", "Available", "In Transit",
                                     "Under Maintenance"])
                rows.append({"vehicle_id": f"VH{vid:04d}", "vehicle_type": vt,
                             "capacity_t": VEHICLE_TYPES[vt]["capacity_t"], "home_source_id": loc_id,
                             "home_city": city, "status": status,
                             "contract_type": "Contracted fleet",
                             "gps_enabled": bool(rng.random() < 0.92),
                             "data_label": SYNTHETIC_LABEL})
                vid += 1
    return pd.DataFrame(rows)


def _build_trips(rng, orders, sources, cities, as_of, n_trips: int) -> pd.DataFrame:
    recent = orders[orders["order_date"] >= as_of - pd.Timedelta(days=540)]
    sample = recent.sample(min(n_trips, len(recent)), random_state=int(rng.integers(1e9))).copy()
    src = sources.set_index("source_id")
    city_terrain = cities.set_index("city")["terrain"]
    sample["origin_terrain"] = src.loc[sample["source_id"], "terrain"].to_numpy()
    sample["dest_terrain"] = city_terrain.loc[sample["city"]].to_numpy()
    sample["route_type"] = np.where((sample["origin_terrain"] == "Hilly") | (sample["dest_terrain"] == "Hilly"),
                                    "Hilly", np.where(sample["secondary_km"] > 250, "National Highway",
                                                      np.where(sample["secondary_km"] > 80, "State Highway",
                                                               "Local / District")))
    sample["vehicle_type"] = np.where(sample["qty_t"] <= 10, "LCV-10",
                                      np.where(sample["qty_t"] <= 25, "MAV-25", "TRL-35"))
    speed = sample["vehicle_type"].map(lambda v: VEHICLE_TYPES[v]["avg_speed_kmph"]).to_numpy(dtype=float)
    route_factor = sample["route_type"].map({"Hilly": 0.68, "National Highway": 1.08, "State Highway": 0.95,
                                             "Local / District": 0.80}).to_numpy()
    month = sample["order_date"].dt.month.to_numpy()
    monsoon = np.isin(month, [7, 8, 9])
    dep_hour = rng.integers(0, 24, len(sample))
    night = (dep_hour >= 20) | (dep_hour <= 4)
    dist = sample["secondary_km"].to_numpy()
    planned_hours = dist / speed
    base_hours = dist / (speed * route_factor) * np.where(monsoon, 1.17, 1.0) * np.where(night, 0.93, 1.0)
    rest = np.floor(base_hours / 8.0) * 1.0
    actual = base_hours + rest + rng.lognormal(-0.3, 0.6, len(sample))

    anomaly_type = rng.choice(["None", "Route deviation", "Unscheduled halt"], len(sample), p=[0.90, 0.05, 0.05])
    gps_km = dist * rng.uniform(1.0, 1.10, len(sample))
    # Legitimate operational variation that is NOT an anomaly (diversions, no-entry waits)
    benign_divert = (anomaly_type == "None") & (rng.random(len(sample)) < 0.04)
    gps_km[benign_divert] = dist[benign_divert] * rng.uniform(1.12, 1.25, benign_divert.sum())
    dev = anomaly_type == "Route deviation"
    gps_km[dev] = dist[dev] * rng.uniform(1.15, 1.70, dev.sum())
    actual[dev] = actual[dev] * gps_km[dev] / dist[dev].clip(1)
    halt = anomaly_type == "Unscheduled halt"
    stop_hours = rest + rng.uniform(0, 1.5, len(sample))
    stop_hours[halt] += rng.uniform(4, 28, halt.sum())
    benign_wait = (anomaly_type == "None") & (rng.random(len(sample)) < 0.04)
    stop_hours[benign_wait] += rng.uniform(4, 9, benign_wait.sum())
    actual[benign_wait] += stop_hours[benign_wait] - rest[benign_wait]
    actual[halt] += stop_hours[halt] - rest[halt]

    loading = np.where(sample["source_type"] == "Plant", rng.normal(4.2, 1.2, len(sample)),
                       rng.normal(2.1, 0.6, len(sample))).clip(0.8)
    promised = np.ceil((loading + planned_hours * 1.25 + 2) / 6) * 6
    backhaul = rng.random(len(sample)) < 0.25
    trips = pd.DataFrame({
        "trip_id": [f"TRP{i + 1:05d}" for i in range(len(sample))],
        "order_id": sample["order_id"].to_numpy(),
        "customer_id": sample["customer_id"].to_numpy(),
        "source_id": sample["source_id"].to_numpy(),
        "source_type": sample["source_type"].to_numpy(),
        "destination_city": sample["city"].to_numpy(),
        "region": sample["region"].to_numpy(),
        "dispatch_date": sample["order_date"].to_numpy(),
        "month": month,
        "departure_hour": dep_hour,
        "vehicle_type": sample["vehicle_type"].to_numpy(),
        "qty_t": sample["qty_t"].to_numpy(),
        "route_type": sample["route_type"].to_numpy(),
        "planned_km": np.round(dist, 1),
        "gps_km": np.round(gps_km, 1),
        "loading_hours": np.round(loading, 2),
        "planned_transit_hours": np.round(planned_hours, 2),
        "actual_transit_hours": np.round(actual, 2),
        "stop_hours": np.round(stop_hours, 2),
        "promised_hours": promised,
        "delivered_on_time": (loading + actual) <= promised,
        "backhaul_found": backhaul,
        "empty_return_km": np.round(np.where(backhaul, 0.0, dist), 1),
        "true_anomaly": anomaly_type != "None",
        "true_anomaly_type": anomaly_type,
        "data_label": SYNTHETIC_LABEL,
    })
    trips["trip_freight_inr"] = np.round(
        [float(trip_cost(d, v)) for d, v in zip(trips["planned_km"], trips["vehicle_type"])], 0)
    return trips.sort_values("dispatch_date").reset_index(drop=True)


def _build_inventory(rng, plants, depots, products, dealer_weekly, dealers, projects) -> pd.DataFrame:
    """Inventory snapshot by location and product with safety-stock targets."""
    recent = dealer_weekly[dealer_weekly["week_start"] >= dealer_weekly["week_start"].max() - pd.Timedelta(weeks=7)]
    by_depot = (recent.merge(dealers[["dealer_id", "depot_id"]], on="dealer_id")
                .groupby(["depot_id", "product_id"])["qty_t"].sum() / 8.0)
    prod = products.set_index("product_id")
    rows = []
    for _, dep in depots.iterrows():
        for pid in products["product_id"]:
            wk = float(by_depot.get((dep["depot_id"], pid), 0.0))
            if wk <= 0 and prod.loc[pid, "category"] != "Specialty":
                wk = float(rng.uniform(1, 6))
            on_hand = float(np.round(wk * rng.uniform(0.3, 1.8), 1))
            rows.append({"location_id": dep["depot_id"], "location_type": "Depot", "location_name": dep["depot_name"],
                         "region": dep["region"], "product_id": pid, "category": prod.loc[pid, "category"],
                         "on_hand_t": on_hand, "reserved_t": float(np.round(on_hand * rng.uniform(0, 0.25), 1)),
                         "safety_stock_t": float(np.round(wk * 0.6, 1)), "avg_weekly_demand_t": round(wk, 1),
                         "storage_capacity_t": float(dep["storage_t"])})
    for _, pl in plants.iterrows():
        cats = pl["categories"]
        prod_ids = [p for p in products["product_id"] if prod.loc[p, "category"] in cats]
        for pid in prod_ids:
            share = prod.loc[pid, "trade_share"] + 0.05
            on_hand = float(np.round(pl["weekly_capacity_t"] * share * rng.uniform(0.6, 1.4), 1))
            rows.append({"location_id": pl["plant_id"], "location_type": "Plant", "location_name": pl["plant_name"],
                         "region": pl["region"], "product_id": pid, "category": prod.loc[pid, "category"],
                         "on_hand_t": on_hand, "reserved_t": float(np.round(on_hand * rng.uniform(0.05, 0.3), 1)),
                         "safety_stock_t": float(np.round(pl["weekly_capacity_t"] * share * 0.3, 1)),
                         "avg_weekly_demand_t": float(np.round(pl["weekly_capacity_t"] * share * 0.8, 1)),
                         "storage_capacity_t": float(pl["weekly_capacity_t"] * 2.0)})
    inv = pd.DataFrame(rows)
    inv["available_t"] = (inv["on_hand_t"] - inv["reserved_t"]).clip(lower=0).round(1)
    inv["unit_cost_inr"] = prod.loc[inv["product_id"], "base_cost"].to_numpy()
    inv["inventory_value_inr"] = (inv["on_hand_t"] * inv["unit_cost_inr"]).round(0)
    inv["data_label"] = SYNTHETIC_LABEL
    return inv


def _build_open_orders(rng, dealers, series, institutional, projects, products, plants, depots, price_idx,
                       as_of, n_orders: int = 120) -> pd.DataFrame:
    prod = products.set_index("product_id")
    last_month = price_idx["month"].max()
    pidx = price_idx[price_idx["month"] == last_month].set_index("region")["price_index"]
    active_projects = projects[projects["project_stage"] != "Finishing"]
    cat_to_product = {"OPC": "P01", "PPC": "P03", "PSC": "P04", "Composite": "P05", "Specialty": "P06"}
    rows = []
    n_trade = int(n_orders * 0.72)
    dealer_pick = dealers.sample(n_trade, replace=True, weights=dealers["size_factor"],
                                 random_state=int(rng.integers(1e9)))
    for _, d in dealer_pick.iterrows():
        prods = series[series["dealer_id"] == d["dealer_id"]]["product_id"].to_numpy()
        pid = rng.choice(prods)
        qty = float(rng.choice([8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50],
                               p=[0.08, 0.14, 0.1, 0.14, 0.08, 0.14, 0.12, 0.08, 0.05, 0.05, 0.02]))
        rows.append({"customer_id": d["dealer_id"], "customer_name": d["dealer_name"], "channel": "Trade",
                     "customer_segment": d["dealer_segment"], "region": d["region"], "city": d["city"],
                     "lat": d["lat"], "lon": d["lon"], "last_mile_km": d["last_mile_km"],
                     "product_id": pid, "qty_t": qty, "designated_source_id": d["depot_id"],
                     "project_id": None, "discount_per_t": d["base_discount_per_t"]})
    proj_pick = active_projects.sample(n_orders - n_trade, replace=True, random_state=int(rng.integers(1e9)))
    inst_idx = institutional.set_index("customer_id")
    for _, p in proj_pick.iterrows():
        cust = inst_idx.loc[p["customer_id"]]
        cap_plant, _ = nearest_capable_plant(p["lat"], p["lon"], p["product_category"], plants)
        region_plants = plants[(plants["region"] == p["region"])
                               & plants["categories"].apply(lambda c, cat=p["product_category"]: cat in c)]
        designated = region_plants.iloc[0]["plant_id"] if len(region_plants) else cap_plant
        rows.append({"customer_id": p["customer_id"], "customer_name": cust["customer_name"],
                     "channel": "Non-Trade", "customer_segment": cust["customer_segment"],
                     "region": p["region"], "city": p["city"], "lat": p["lat"], "lon": p["lon"],
                     "last_mile_km": float(rng.uniform(5, 30)),
                     "product_id": cat_to_product[p["product_category"]],
                     "qty_t": float(rng.choice([30, 40, 50, 60, 70, 80, 100])),
                     "designated_source_id": designated, "project_id": p["project_id"],
                     "discount_per_t": float(round(rng.normal(110, 30), 0))})
    oo = pd.DataFrame(rows)
    oo.insert(0, "open_order_id", [f"OO{i + 1:04d}" for i in range(len(oo))])
    oo["category"] = prod.loc[oo["product_id"], "category"].to_numpy()
    urgency = rng.choice(["Urgent", "Standard", "Flexible"], len(oo), p=[0.22, 0.53, 0.25])
    oo["urgency"] = urgency
    oo["deadline_hours"] = np.select([urgency == "Urgent", urgency == "Standard"], [24, 48], 72)
    oo["list_price_per_t"] = np.round(prod.loc[oo["product_id"], "base_price"].to_numpy()
                                      * pidx.reindex(oo["region"]).fillna(1.0).to_numpy()
                                      * np.where(oo["channel"] == "Non-Trade", 0.955, 1.0), 0)
    oo["order_date"] = as_of
    oo["data_label"] = SYNTHETIC_LABEL
    return oo


# ---------------------------------------------------------------------------
# Tenders
# ---------------------------------------------------------------------------
REQUIRED_CERT_POOL = ["BIS Licence", "ISO 9001 Certificate", "ISO 14001 Certificate", "Mill Test Certificate",
                      "Third-Party Test Report", "GST Registration", "PAN Card", "Audited Financial Statements",
                      "Past Supply Experience Certificate", "Undertaking on Blacklisting", "Power of Attorney"]


def _build_company_documents(rng, as_of) -> pd.DataFrame:
    rows = []
    for doc in REQUIRED_CERT_POOL:
        if doc == "ISO 14001 Certificate":
            expiry = as_of - pd.Timedelta(days=21)   # expired: demonstrates missing-document alerts
        elif doc == "Power of Attorney":
            expiry = as_of + pd.Timedelta(days=12)   # expiring soon
        elif doc in ("GST Registration", "PAN Card"):
            expiry = pd.NaT
        else:
            expiry = as_of + pd.Timedelta(days=int(rng.integers(60, 700)))
        status = ("Valid (no expiry)" if pd.isna(expiry) else "Expired" if expiry < as_of else
                  "Expiring within 30 days" if expiry < as_of + pd.Timedelta(days=30) else "Valid")
        rows.append({"document": doc, "expiry_date": expiry, "status": status, "repository_ref": f"DOCREG-{len(rows) + 1:03d}",
                     "data_label": SYNTHETIC_LABEL})
    return pd.DataFrame(rows)


def _tender_document_text(t: dict) -> str:
    """Render a synthetic tender notice. Some fields are deliberately omitted to exercise extraction."""
    lines = [
        "SYNTHETIC TENDER NOTICE - FOR DEMONSTRATION ONLY",
        f"Tender Reference No: {t['tender_id']}",
        f"Issuing Authority: {t['institution']}",
        f"Procurement Portal: {t['portal']}",
        f"Name of Work: Supply of cement for {t['project_location']} works",
        f"Project Location: {t['project_location']}, {t['state']}",
        f"Estimated Quantity: {t['quantity_t']:,.0f} MT",
        f"Cement Type / Grade: {t['product_requirement']}",
        f"Delivery Period: {t['delivery_months']} months from date of supply order",
        f"Bid Submission Closing Date: {t['closing_date'].strftime('%d-%b-%Y')} 15:00 hrs",
        f"Earnest Money Deposit (EMD): Rs. {t['emd_inr']:,.0f}",
        f"Performance Security: {t['performance_guarantee_pct']:.0f}% of contract value",
        f"Payment Terms: Within {t['payment_terms_days']} days of receipt of material and invoice",
    ]
    if t.get("include_penalty", True):
        lines.append(f"Liquidated Damages: {t['penalty_pct_per_week']:.1f}% per week of delay, subject to a "
                     f"maximum of {t['penalty_cap_pct']:.0f}% of contract value")
    lines += [
        "ELIGIBILITY CRITERIA:",
        f"1. Minimum average annual turnover of Rs. {t['min_turnover_cr']:.0f} Crore in the last three financial years.",
        f"2. Minimum {t['min_experience_years']} years of experience in supply of cement to government or "
        "infrastructure projects.",
        "3. Bidder must hold a valid BIS licence for the offered product (illustrative requirement).",
        "MANDATORY DOCUMENTS:",
    ]
    for i, doc in enumerate(t["required_documents"], start=1):
        lines.append(f"({chr(96 + i)}) {doc}")
    lines.append("Bids not accompanied by mandatory documents are liable to be rejected.")
    lines.append("This document is synthetic and does not represent any real procurement.")
    return "\n".join(lines)


def _build_tenders(rng, institutional, plants, inventory, company_docs, as_of, n_hist=420, n_open=30):
    inst = institutional
    grade_map = {"OPC": "OPC 53 Grade conforming to IS 269 (illustrative reference)",
                 "PPC": "PPC (fly ash based) conforming to IS 1489 Part 1 (illustrative reference)",
                 "PSC": "Portland Slag Cement conforming to IS 455 (illustrative reference)",
                 "Composite": "Composite Cement conforming to IS 16415 (illustrative reference)",
                 "Specialty": "Sulphate Resisting Cement conforming to IS 12330 (illustrative reference)"}
    company_turnover_cr = 950.0   # synthetic sample-business turnover used for eligibility checks
    company_experience = 15
    valid_docs = set(company_docs[company_docs["status"].isin(["Valid", "Valid (no expiry)",
                                                                "Expiring within 30 days"])]["document"])
    atp_by_cat = inventory.groupby("category")["available_t"].sum()

    def make(i: int, closed: bool) -> dict:
        c = inst.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        inst_type = ("Central Govt Department" if c["institution_type"] == "Government" and rng.random() < 0.4 else
                     "State PWD / Municipal" if c["institution_type"] == "Government" else
                     rng.choice(["PSU", "Private EPC", "Private Developer"], p=[0.3, 0.45, 0.25]))
        cat = rng.choice(["OPC", "PPC", "PSC", "Composite", "Specialty"], p=[0.32, 0.30, 0.20, 0.10, 0.08])
        qty = float(np.round(rng.lognormal(7.2, 0.6), -1))
        _, dist = nearest_capable_plant(c["lat"], c["lon"], cat, plants)
        market_price = float(np.round(rng.normal(6900, 220), 0))
        price_index = float(np.clip(rng.normal(1.0, 0.035), 0.9, 1.12))
        our_price = float(np.round(market_price * price_index, 0))
        n_docs = int(rng.integers(4, 8))
        doc_w = np.array([0.4 if d in ("ISO 14001 Certificate", "Power of Attorney") else 1.0 for d in REQUIRED_CERT_POOL])
        req_docs = list(rng.choice(REQUIRED_CERT_POOL, size=n_docs, replace=False, p=doc_w / doc_w.sum()))
        if "BIS Licence" not in req_docs:
            req_docs[0] = "BIS Licence"
        docs_ok = sum(d in valid_docs for d in req_docs) / len(req_docs)
        min_turnover = float(rng.choice([50, 100, 200, 400, 800, 1200], p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05]))
        min_exp = int(rng.choice([3, 5, 7, 10, 20], p=[0.25, 0.3, 0.2, 0.17, 0.08]))
        turnover_ok = company_turnover_cr >= min_turnover
        exp_ok = company_experience >= min_exp
        elig = (0.4 * turnover_ok + 0.2 * exp_ok + 0.4 * docs_ok)
        delivery_months = int(rng.integers(3, 13))
        monthly_need = qty / delivery_months
        # Share of category stock that can be allocated to one contract (simulation assumption: 8-15%)
        delivery_cap = float(np.clip(atp_by_cat.get(cat, 0) * rng.uniform(0.08, 0.15) / max(monthly_need, 1), 0.2, 3.0))
        payment_days = int(rng.choice([30, 45, 60, 90, 120], p=[0.2, 0.3, 0.25, 0.17, 0.08]))
        bidders = int(rng.integers(2, 11))
        strategic = bool(rng.random() < 0.25)
        pay_score = float(np.round(1 - c["latent_risk"], 3))
        closing = (as_of - pd.Timedelta(days=int(rng.integers(20, 1050))) if closed
                   else as_of + pd.Timedelta(days=int(rng.integers(4, 46))))
        t = {
            "tender_id": f"TND-{closing.year}-{i:04d}", "institution": c["customer_name"],
            "customer_id": c["customer_id"], "institution_type": inst_type, "portal": c["procurement_channel"]
            if c["procurement_channel"] != "Direct" else "Direct invitation",
            "project_location": c["city"], "state": c["state"], "region": c["region"],
            "quantity_t": qty, "product_category": cat, "product_requirement": grade_map[cat],
            "delivery_months": delivery_months, "closing_date": closing,
            "min_turnover_cr": min_turnover, "min_experience_years": min_exp,
            "required_documents": req_docs,
            "emd_inr": float(np.round(qty * market_price * rng.uniform(0.01, 0.02), -3)),
            "performance_guarantee_pct": float(rng.choice([3, 5, 10], p=[0.3, 0.5, 0.2])),
            "penalty_pct_per_week": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "penalty_cap_pct": float(rng.choice([5, 10])),
            "payment_terms_days": payment_days,
            "customer_payment_score": pay_score,
            "estimated_market_price_per_t": market_price,
            "our_quote_per_t": our_price,
            "price_index_vs_market": round(price_index, 4),
            "distance_km": round(dist, 1),
            "estimated_freight_per_t": float(np.round(secondary_freight_per_t(dist + 15), 0)),
            "delivery_capability_ratio": round(delivery_cap, 3),
            "eligibility_score": round(elig, 3),
            "documents_complete": docs_ok == 1.0,
            "expected_bidders": bidders,
            "strategic_account": strategic,
            "include_penalty": bool(rng.random() > 0.12),
        }
        if closed:
            z = (-0.35 - 10.0 * (price_index - 1.0) + 2.2 * (elig - 0.85) + 0.7 * t["documents_complete"]
                 + 0.35 * min(delivery_cap, 1.5) - 0.0018 * (dist - 300) - 0.17 * (bidders - 5)
                 + 0.45 * strategic - 0.004 * (payment_days - 60))
            p = 1 / (1 + np.exp(-z))
            disqualified = elig < 0.7
            won = bool((rng.random() < p) and not disqualified)
            t.update({
                "outcome": "Won" if won else ("Disqualified" if disqualified else "Lost"),
                "won": int(won),
                # Post-outcome fields - known only after the award; never used as model features.
                "l1_price_per_t": float(np.round(our_price * (1.0 if won else rng.uniform(0.94, 0.995)), 0)),
                "our_rank": 1 if won else int(rng.integers(2, max(3, bidders + 1))),
                "award_date": closing + pd.Timedelta(days=int(rng.integers(10, 60))),
            })
        return t

    hist = pd.DataFrame([make(i + 1, True) for i in range(n_hist)])
    open_ = pd.DataFrame([make(n_hist + i + 1, False) for i in range(n_open)])
    docs = {r["tender_id"]: _tender_document_text(r) for r in open_.to_dict("records")}
    hist["data_label"] = SYNTHETIC_LABEL
    open_["data_label"] = SYNTHETIC_LABEL
    return hist.sort_values("closing_date").reset_index(drop=True), \
        open_.sort_values("closing_date").reset_index(drop=True), docs


# ---------------------------------------------------------------------------
# Technical compliance
# ---------------------------------------------------------------------------
# Illustrative parameter definitions. Limits are synthetic and NOT taken from any standard.
COMPLIANCE_PARAMETERS = {
    "compressive_strength_28d_mpa": ("28-day compressive strength", "MPa"),
    "fineness_m2_per_kg": ("Fineness (specific surface)", "m2/kg"),
    "initial_setting_time_min": ("Initial setting time", "minutes"),
    "magnesia_pct": ("Magnesia (MgO)", "% by mass"),
    "sulphuric_anhydride_pct": ("Sulphuric anhydride (SO3)", "% by mass"),
    "total_alkali_pct": ("Total alkali as Na2O equivalent", "% by mass"),
    "chloride_pct": ("Chloride content", "% by mass"),
}

_PRODUCT_PROFILES = {
    "P01": {"standard": "IS 269 (illustrative reference)", "compressive_strength_28d_mpa": 58, "fineness_m2_per_kg": 330,
            "initial_setting_time_min": 140, "magnesia_pct": 2.6, "sulphuric_anhydride_pct": 2.5,
            "total_alkali_pct": 0.72, "chloride_pct": 0.025},
    "P02": {"standard": "IS 269 (illustrative reference)", "compressive_strength_28d_mpa": 50, "fineness_m2_per_kg": 315,
            "initial_setting_time_min": 155, "magnesia_pct": 2.4, "sulphuric_anhydride_pct": 2.4,
            "total_alkali_pct": 0.68, "chloride_pct": 0.022},
    "P03": {"standard": "IS 1489 Part 1 (illustrative reference)", "compressive_strength_28d_mpa": 45, "fineness_m2_per_kg": 365,
            "initial_setting_time_min": 175, "magnesia_pct": 2.1, "sulphuric_anhydride_pct": 2.3,
            "total_alkali_pct": 0.62, "chloride_pct": 0.020},
    "P04": {"standard": "IS 455 (illustrative reference)", "compressive_strength_28d_mpa": 46, "fineness_m2_per_kg": 360,
            "initial_setting_time_min": 185, "magnesia_pct": 4.6, "sulphuric_anhydride_pct": 2.2,
            "total_alkali_pct": 0.48, "chloride_pct": 0.018},
    "P05": {"standard": "IS 16415 (illustrative reference)", "compressive_strength_28d_mpa": 44, "fineness_m2_per_kg": 370,
            "initial_setting_time_min": 180, "magnesia_pct": 3.9, "sulphuric_anhydride_pct": 2.3,
            "total_alkali_pct": 0.55, "chloride_pct": 0.020},
    "P06": {"standard": "IS 12330 (illustrative reference)", "compressive_strength_28d_mpa": 48, "fineness_m2_per_kg": 320,
            "initial_setting_time_min": 150, "magnesia_pct": 2.2, "sulphuric_anhydride_pct": 2.1,
            "total_alkali_pct": 0.58, "chloride_pct": 0.015},
    "P07": {"standard": "IS 269 (illustrative reference)", "compressive_strength_28d_mpa": 54, "fineness_m2_per_kg": 325,
            "initial_setting_time_min": 150, "magnesia_pct": 2.3, "sulphuric_anhydride_pct": 2.4,
            "total_alkali_pct": 0.45, "chloride_pct": 0.020},
}


def _build_compliance(rng, products, as_of):
    spec_rows, test_rows, cert_rows = [], [], []
    for pid, prof in _PRODUCT_PROFILES.items():
        spec_rows.append({"product_id": pid, "product_name": products.set_index("product_id").loc[pid, "product_name"],
                          "standard_reference": prof["standard"], "standard_version": "Illustrative edition (synthetic)",
                          **{k: prof[k] for k in COMPLIANCE_PARAMETERS}, "data_label": SYNTHETIC_LABEL})
        for b in range(6):
            test_date = as_of - pd.Timedelta(days=int(30 * b + rng.integers(0, 20)))
            row = {"test_id": f"LT-{pid}-{b + 1:02d}", "product_id": pid, "batch_id": f"B{pid[1:]}{as_of.year % 100}{b + 1:03d}",
                   "test_date": test_date, "laboratory": rng.choice(["Plant QC Lab (Synthetic)", "Independent Lab A (Synthetic)"]),
                   "certificate_id": f"TC-{pid}-{b + 1:02d}"}
            for k in COMPLIANCE_PARAMETERS:
                base = prof[k]
                sd = abs(base) * (0.035 if k != "initial_setting_time_min" else 0.07)
                row[k] = float(np.round(rng.normal(base, sd), 3))
            # Deliberate evidence gaps to exercise NEEDS REVIEW logic
            if pid in ("P02", "P05") and b == 0:
                row["total_alkali_pct"] = np.nan
            if pid == "P03" and b == 0:
                row["chloride_pct"] = np.nan
            test_rows.append(row)
        for cert, validity in [("BIS Licence", 900), ("Mill Test Certificate", 90), ("Third-Party Test Report", 180)]:
            issued = as_of - pd.Timedelta(days=int(rng.integers(10, 200)))
            expiry = issued + pd.Timedelta(days=validity)
            present = not (pid == "P05" and cert == "Third-Party Test Report")
            if pid == "P04" and cert == "Third-Party Test Report":
                issued = as_of - pd.Timedelta(days=220)
                expiry = issued + pd.Timedelta(days=validity)
            cert_rows.append({"product_id": pid, "certificate": cert, "available": present,
                              "issue_date": issued if present else pd.NaT, "expiry_date": expiry if present else pd.NaT,
                              "document_ref": f"CERT-{pid}-{cert[:3].upper()}" if present else None,
                              "data_label": SYNTHETIC_LABEL})

    project_defs = [
        ("CP01", "Coastal Bridge Package (Simulated)", "PSC or PPC", {"compressive_strength_28d_mpa": (">=", 43), "total_alkali_pct": ("<=", 0.60), "chloride_pct": ("<=", 0.03), "initial_setting_time_min": (">=", 60)}, "Consultant A (Synthetic)"),
        ("CP02", "Metro Viaduct Segment (Simulated)", "OPC", {"compressive_strength_28d_mpa": (">=", 53), "fineness_m2_per_kg": (">=", 300), "magnesia_pct": ("<=", 6.0), "total_alkali_pct": ("<=", 0.60)}, "Consultant B (Synthetic)"),
        ("CP03", "Irrigation Canal Lining (Simulated)", "PPC or PSC", {"compressive_strength_28d_mpa": (">=", 33), "sulphuric_anhydride_pct": ("<=", 3.0), "initial_setting_time_min": (">=", 30)}, "Consultant A (Synthetic)"),
        ("CP04", "Sewage Treatment Plant (Simulated)", "Sulphate Resisting or PSC", {"compressive_strength_28d_mpa": (">=", 43), "chloride_pct": ("<=", 0.05), "total_alkali_pct": ("<=", 0.60), "sulphuric_anhydride_pct": ("<=", 2.5)}, "Consultant C (Synthetic)"),
        ("CP05", "Affordable Housing Cluster (Simulated)", "PPC or Composite", {"compressive_strength_28d_mpa": (">=", 33), "fineness_m2_per_kg": (">=", 300), "total_alkali_pct": ("<=", 0.80)}, None),
        ("CP06", "Dam Spillway Rehabilitation (Simulated)", "Low-Alkali OPC", {"compressive_strength_28d_mpa": (">=", 43), "total_alkali_pct": ("<=", 0.50), "magnesia_pct": ("<=", 5.0), "chloride_pct": ("<=", 0.05)}, "Consultant B (Synthetic)"),
    ]
    category_text_to_products = {
        "PSC": ["P04"], "PPC": ["P03"], "OPC": ["P01", "P02", "P07"], "Composite": ["P05"],
        "Sulphate Resisting": ["P06"], "Low-Alkali OPC": ["P07"],
    }
    proj_rows, req_rows, docs, approvals = [], [], {}, []
    for pid_, name, types, reqs, consultant in project_defs:
        type_list = [x.strip() for x in types.split(" or ")]
        eligible = sorted({p for t_ in type_list for p in category_text_to_products.get(t_, [])})
        max_test_age = int(rng.choice([90, 120, 180]))
        required_certs = ["BIS Licence", "Mill Test Certificate"] + (["Third-Party Test Report"] if rng.random() < 0.8 else [])
        proj_rows.append({"compliance_project_id": pid_, "project_name": name, "permitted_cement_types": types,
                          "eligible_product_ids": eligible, "consultant": consultant or "Not specified",
                          "consultant_approval_required": consultant is not None,
                          "max_test_age_days": max_test_age, "required_certificates": required_certs,
                          "data_label": SYNTHETIC_LABEL})
        lines = [f"SYNTHETIC PROJECT TECHNICAL SPECIFICATION - {name}", f"Specification Ref: SPEC-{pid_}",
                 f"Permitted cement type: {types}", "Requirements (illustrative limits, not taken from any standard):"]
        for k, (op, lim) in reqs.items():
            label, unit = COMPLIANCE_PARAMETERS[k]
            word = "minimum" if op == ">=" else "maximum"
            lines.append(f"- {label}: {word} {lim} {unit}")
            req_rows.append({"compliance_project_id": pid_, "parameter": k, "parameter_label": label,
                             "operator": op, "limit": lim, "unit": unit, "source": f"SPEC-{pid_}"})
        lines.append(f"Test certificates shall not be older than {max_test_age} days at the time of supply.")
        lines.append("Required documents: " + "; ".join(required_certs) + ".")
        if consultant:
            lines.append(f"Consultant approval required from {consultant} before first supply.")
        else:
            lines.append("Consultant approval: not required for this package.")
        docs[pid_] = "\n".join(lines)
        for p in eligible:
            status = rng.choice(["Approved", "Approved", "Pending", "Not submitted"]) if consultant else "Not required"
            approvals.append({"compliance_project_id": pid_, "product_id": p, "consultant": consultant or "Not specified",
                              "approval_status": status,
                              "approval_ref": f"APR-{pid_}-{p}" if status == "Approved" else None})
    return (pd.DataFrame(spec_rows), pd.DataFrame(test_rows), pd.DataFrame(cert_rows), pd.DataFrame(proj_rows),
            pd.DataFrame(req_rows), docs, pd.DataFrame(approvals))


# ---------------------------------------------------------------------------
# Pricing panel
# ---------------------------------------------------------------------------
TRUE_ELASTICITY = {"East": -1.6, "North-East": -1.1, "South": -1.9, "West": -1.4, "Central": -1.5, "North": -1.3}


def _build_pricing_panel(rng, products, plants, cities, price_idx: pd.DataFrame) -> pd.DataFrame:
    prod = products[products["category"] != "Specialty"]
    months = sorted(price_idx["month"].unique())[-24:]
    pidx = price_idx.set_index(["region", "month"])["price_index"]
    rows = []
    for region in REGIONS:
        region_cities = cities[cities["region"] == region]["city"].tolist()
        micro = region_cities[:3] if len(region_cities) >= 3 else region_cities
        comp = 1.0
        demand_ind = 1.0
        for m in months:
            comp = 0.7 * comp + 0.3 * 1.0 + rng.normal(0, 0.015)
            demand_ind = 0.8 * demand_ind + 0.2 * 1.0 + rng.normal(0, 0.03)
            month_season = float(_month_seasonality(np.array([pd.Timestamp(m).month]))[0])
            for city in micro:
                c = cities[cities["city"] == city].iloc[0]
                for _, p in prod.iterrows():
                    _, dist = nearest_capable_plant(c["lat"], c["lon"], p["category"], plants)
                    freight = float(secondary_freight_per_t(dist + 20))
                    scheme = rng.choice(_SCHEMES, p=[0.15, 0.15, 0.15, 0.05, 0.50])
                    discount = float(np.clip(rng.normal(230, 70) + (80 if scheme != "No Scheme" else 0), 0, 600))
                    list_price = p["base_price"] * pidx[(region, m)] * (1 + rng.normal(0, 0.01))
                    net_price = list_price - discount
                    comp_price = p["base_price"] * pidx[(region, m)] * comp
                    eps = TRUE_ELASTICITY[region]
                    base_vol = 260 * REGION_DEMAND_WEIGHT[region] * p["trade_share"] * 5
                    volume = (base_vol * (net_price / comp_price) ** eps * demand_ind * month_season
                              * rng.lognormal(0, 0.08))
                    cost = p["base_cost"]
                    conv = float(np.clip(1 / (1 + np.exp(-(0.4 + 0.004 * (discount - 230) - 6 * (net_price / comp_price - 0.97)))), 0.05, 0.95))
                    rows.append({"region": region, "micro_market": f"{city} Market", "month": m,
                                 "product_id": p["product_id"], "product_name": p["product_name"],
                                 "category": p["category"], "list_price_per_t": round(list_price, 0),
                                 "discount_per_t": round(discount, 0), "net_price_per_t": round(net_price, 0),
                                 "scheme": scheme, "competitor_price_index": round(comp, 4),
                                 "competitor_price_per_t": round(comp_price, 0),
                                 "market_demand_index": round(demand_ind * month_season, 4),
                                 "volume_t": round(volume, 1), "freight_per_t": round(freight, 0),
                                 "production_cost_per_t": cost, "quote_conversion_rate": round(conv, 3)})
    df = pd.DataFrame(rows)
    df["contribution_per_t"] = df["net_price_per_t"] - df["production_cost_per_t"] - df["freight_per_t"]
    df["contribution_inr"] = (df["contribution_per_t"] * df["volume_t"]).round(0)
    df["contribution_margin_pct"] = (df["contribution_per_t"] / df["net_price_per_t"]).round(4)
    df["data_label"] = SYNTHETIC_LABEL
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_environment(settings: DataSettings = DataSettings()) -> Environment:
    """Generate the complete, internally consistent synthetic business environment."""
    rng = np.random.default_rng(settings.seed)
    fake = Faker("en_IN")
    Faker.seed(settings.seed)
    as_of = pd.Timestamp(AS_OF_DATE)

    cities, plants, depots, products = _build_master(rng)
    reps = _build_sales_reps(rng, fake)
    dealers = _build_dealers(rng, fake, settings.n_dealers, cities, depots, plants, reps, as_of)
    weeks = pd.date_range(end=as_of - pd.Timedelta(days=7), periods=settings.n_weeks, freq="W-MON")
    months = pd.date_range(weeks[0].to_period("M").to_timestamp(), as_of, freq="MS")
    price_idx = _region_price_index(rng, months)

    series = _build_dealer_series(rng, dealers, products)
    dealer_weekly, series_meta = _build_dealer_weekly(rng, dealers, series, weeks)
    series = series.merge(series_meta, on="series_id")

    inst = _build_institutional(rng, settings.n_institutional, cities)
    n_projects = max(20, int(settings.n_institutional * 0.48))
    projects, project_monthly = _build_projects(rng, inst, as_of, n_projects)

    # Unified customer master for credit (no personal or protected attributes).
    dealer_cust = dealers.rename(columns={"dealer_id": "customer_id", "dealer_name": "customer_name",
                                          "dealer_segment": "customer_segment"})
    dealer_cust["channel"] = "Trade"
    dealer_cust["lgd_class"] = "Trade"
    inst_cust = inst.copy()
    inst_cust["channel"] = "Non-Trade"
    cust_cols = ["customer_id", "customer_name", "channel", "customer_segment", "region", "city",
                 "payment_terms_days", "credit_limit_inr", "latent_risk", "lgd_class", "size_factor"]
    customers = pd.concat([dealer_cust[cust_cols], inst_cust[cust_cols]], ignore_index=True)
    drift = rng.normal(0.0, 0.05, len(customers))
    deteriorating = rng.random(len(customers)) < 0.15
    drift[deteriorating] = rng.uniform(0.15, 0.40, deteriorating.sum())
    customers["risk_drift"] = drift
    customers["data_label"] = SYNTHETIC_LABEL

    orders = _build_orders(rng, dealers, dealer_weekly, projects, project_monthly, products, plants, depots, price_idx)
    invoices = _build_invoices(rng, orders, customers, as_of)
    vehicles = _build_vehicles(rng, plants, depots)

    env_sources = pd.concat([
        plants.rename(columns={"plant_id": "source_id"})[["source_id", "city", "lat", "lon", "terrain"]],
        depots.rename(columns={"depot_id": "source_id"})[["source_id", "city", "lat", "lon", "terrain"]],
    ], ignore_index=True)
    trips = _build_trips(rng, orders, env_sources, cities, as_of, n_trips=4000)
    inventory = _build_inventory(rng, plants, depots, products, dealer_weekly, dealers, projects)
    open_orders = _build_open_orders(rng, dealers, series, inst, projects, products, plants, depots, price_idx, as_of)
    company_docs = _build_company_documents(rng, as_of)
    tenders_hist, tenders_open, tender_docs = _build_tenders(rng, inst, plants, inventory, company_docs, as_of)
    specs, tests, certs, cprojects, creqs, cdocs, approvals = _build_compliance(rng, products, as_of)
    pricing = _build_pricing_panel(rng, products, plants, cities, price_idx)

    loc = pd.concat([env_sources, cities[["city", "lat", "lon"]].assign(source_id=lambda d: "CITY:" + d["city"])],
                    ignore_index=True)
    loc_dist = pd.DataFrame(distance_matrix(loc, loc), index=loc["source_id"], columns=loc["source_id"])

    return Environment(
        settings=settings, as_of=as_of, cities=cities, plants=plants, depots=depots, products=products,
        sales_reps=reps, dealers=dealers, dealer_series=series, dealer_weekly=dealer_weekly,
        institutional_customers=inst, customers=customers, projects=projects, project_monthly=project_monthly,
        orders=orders, invoices=invoices, vehicles=vehicles, trips=trips, inventory=inventory,
        open_orders=open_orders, company_documents=company_docs, tenders_hist=tenders_hist,
        tenders_open=tenders_open, tender_documents=tender_docs, product_specs=specs, lab_tests=tests,
        product_certificates=certs, compliance_projects=cprojects, compliance_requirements=creqs,
        compliance_documents=cdocs, consultant_approvals=approvals, pricing_panel=pricing,
        region_price_index=price_idx, location_distance=loc_dist,
    )


def generate_dealer_population(n: int = DEALER_POPULATION, seed: int = 7) -> pd.DataFrame:
    """Lightweight vectorised simulation of the full channel-partner network (~49,300 dealers).

    Used only for network-level aggregates; never rendered row-by-row in the browser.
    """
    rng = np.random.default_rng(seed)
    w = np.array([REGION_DEMAND_WEIGHT[r] for r in REGIONS])
    region = rng.choice(REGIONS, size=n, p=w / w.sum())
    tier = rng.choice(DEALER_TIERS, size=n, p=[0.03, 0.12, 0.35, 0.50])
    tier_size = pd.Series({"Key Account": 3.0, "Large": 1.8, "Medium": 1.0, "Small": 0.45})
    size = tier_size.loc[tier].to_numpy() * rng.lognormal(0, 0.4, n)
    annual_t = np.round(size * 7.5 * 52 * rng.uniform(0.6, 1.2, n), 0)
    growth = rng.normal(0.02, 0.15, n)
    risk = np.clip(rng.beta(2, 6, n), 0, 1)
    active_digital = rng.random(n) < 0.78
    days_since_order = np.round(rng.exponential(9 / np.sqrt(size)), 0)
    return pd.DataFrame({
        "dealer_id": [f"N{i + 1:06d}" for i in range(n)], "region": region, "dealer_segment": tier,
        "annual_volume_t": annual_t, "yoy_growth": growth.round(3), "payment_risk_index": risk.round(3),
        "uses_digital_ordering": active_digital, "days_since_last_order": days_since_order,
    })
