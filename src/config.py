"""Central configuration, master reference data and explicit demo assumptions.

Every number in this module is a synthetic demonstration assumption. None of the
values represent actual Dalmia Bharat plants, capacities, prices or costs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

APP_NAME = "Dalmia AI Intelligence Hub"
APP_SUBTITLE = (
    "AI-powered commercial, supply-chain and profitability intelligence for the cement industry."
)
DEMO_BANNER = "DEMO ENVIRONMENT — SYNTHETIC DATA — NOT CONNECTED TO LIVE COMPANY SYSTEMS"
POSITIONING_NOTE = (
    "Management decision-support proof of concept. Not an official company product and not a "
    "replacement for ERP, sales, transport, finance or technical approval systems. All results "
    "are simulated and illustrative."
)

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_DEALER_SUBSET = 500
DEFAULT_INSTITUTIONAL_CUSTOMERS = 250
DEFAULT_HISTORY_WEEKS = 104
DEALER_POPULATION = 49_300
AS_OF_DATE = date(2026, 10, 12)  # Monday; the "today" of the simulation

MIN_DEALERS, MAX_DEALERS = 100, 1500
MIN_WEEKS, MAX_WEEKS = 60, 156

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB limit for uploaded documents
DATE_FMT = "%d-%b-%Y"

# ---------------------------------------------------------------------------
# Geography (Indian cities used for demonstration only)
# city, state, region, latitude, longitude, terrain
# ---------------------------------------------------------------------------
CITIES: list[tuple[str, str, str, float, float, str]] = [
    ("Kolkata", "West Bengal", "East", 22.57, 88.36, "Plain"),
    ("Durgapur", "West Bengal", "East", 23.52, 87.31, "Plain"),
    ("Siliguri", "West Bengal", "East", 26.73, 88.40, "Plain"),
    ("Bhubaneswar", "Odisha", "East", 20.30, 85.82, "Plain"),
    ("Cuttack", "Odisha", "East", 20.46, 85.88, "Plain"),
    ("Rourkela", "Odisha", "East", 22.26, 84.85, "Plain"),
    ("Sambalpur", "Odisha", "East", 21.47, 83.97, "Plain"),
    ("Ranchi", "Jharkhand", "East", 23.34, 85.31, "Plateau"),
    ("Jamshedpur", "Jharkhand", "East", 22.80, 86.20, "Plateau"),
    ("Bokaro", "Jharkhand", "East", 23.67, 86.15, "Plateau"),
    ("Patna", "Bihar", "East", 25.59, 85.14, "Plain"),
    ("Gaya", "Bihar", "East", 24.79, 85.00, "Plain"),
    ("Muzaffarpur", "Bihar", "East", 26.12, 85.39, "Plain"),
    ("Guwahati", "Assam", "North-East", 26.14, 91.74, "Hilly"),
    ("Silchar", "Assam", "North-East", 24.83, 92.78, "Hilly"),
    ("Jorhat", "Assam", "North-East", 26.75, 94.20, "Plain"),
    ("Dibrugarh", "Assam", "North-East", 27.47, 94.91, "Plain"),
    ("Tezpur", "Assam", "North-East", 26.63, 92.80, "Plain"),
    ("Shillong", "Meghalaya", "North-East", 25.58, 91.89, "Hilly"),
    ("Agartala", "Tripura", "North-East", 23.83, 91.29, "Hilly"),
    ("Chennai", "Tamil Nadu", "South", 13.08, 80.27, "Plain"),
    ("Tiruchirappalli", "Tamil Nadu", "South", 10.79, 78.70, "Plain"),
    ("Ariyalur", "Tamil Nadu", "South", 11.14, 79.08, "Plain"),
    ("Coimbatore", "Tamil Nadu", "South", 11.02, 76.96, "Plain"),
    ("Madurai", "Tamil Nadu", "South", 9.93, 78.12, "Plain"),
    ("Salem", "Tamil Nadu", "South", 11.66, 78.15, "Plain"),
    ("Bengaluru", "Karnataka", "South", 12.97, 77.59, "Plateau"),
    ("Mysuru", "Karnataka", "South", 12.30, 76.64, "Plateau"),
    ("Belagavi", "Karnataka", "South", 15.85, 74.50, "Plateau"),
    ("Hyderabad", "Telangana", "South", 17.39, 78.49, "Plateau"),
    ("Vijayawada", "Andhra Pradesh", "South", 16.51, 80.65, "Plain"),
    ("Visakhapatnam", "Andhra Pradesh", "South", 17.69, 83.22, "Plain"),
    ("Kadapa", "Andhra Pradesh", "South", 14.47, 78.82, "Plain"),
    ("Kochi", "Kerala", "South", 9.93, 76.27, "Plain"),
    ("Thiruvananthapuram", "Kerala", "South", 8.52, 76.94, "Hilly"),
    ("Pune", "Maharashtra", "West", 18.52, 73.86, "Plateau"),
    ("Mumbai", "Maharashtra", "West", 19.08, 72.88, "Plain"),
    ("Kolhapur", "Maharashtra", "West", 16.70, 74.24, "Plateau"),
    ("Solapur", "Maharashtra", "West", 17.66, 75.91, "Plateau"),
    ("Nagpur", "Maharashtra", "Central", 21.15, 79.09, "Plain"),
    ("Raipur", "Chhattisgarh", "Central", 21.25, 81.63, "Plain"),
    ("Bilaspur", "Chhattisgarh", "Central", 22.08, 82.15, "Plain"),
    ("Jabalpur", "Madhya Pradesh", "Central", 23.18, 79.99, "Plain"),
    ("Bhopal", "Madhya Pradesh", "Central", 23.26, 77.41, "Plain"),
    ("Lucknow", "Uttar Pradesh", "North", 26.85, 80.95, "Plain"),
    ("Varanasi", "Uttar Pradesh", "North", 25.32, 82.97, "Plain"),
    ("Prayagraj", "Uttar Pradesh", "North", 25.44, 81.85, "Plain"),
]
REGIONS = ["East", "North-East", "South", "West", "Central", "North"]

# Relative trade-demand weight by region (simulation assumption).
REGION_DEMAND_WEIGHT = {
    "East": 1.30, "North-East": 0.80, "South": 1.35, "West": 0.70, "Central": 0.75, "North": 0.70,
}

# ---------------------------------------------------------------------------
# Products (synthetic catalogue; grades are generic industry categories)
# ---------------------------------------------------------------------------
PRODUCTS: list[dict] = [
    {"product_id": "P01", "product_name": "OPC 53 Grade", "category": "OPC",
     "base_price": 7350, "base_cost": 4450, "trade_share": 0.18},
    {"product_id": "P02", "product_name": "OPC 43 Grade", "category": "OPC",
     "base_price": 7150, "base_cost": 4320, "trade_share": 0.12},
    {"product_id": "P03", "product_name": "Portland Pozzolana Cement (PPC)", "category": "PPC",
     "base_price": 7000, "base_cost": 4050, "trade_share": 0.38},
    {"product_id": "P04", "product_name": "Portland Slag Cement (PSC)", "category": "PSC",
     "base_price": 6900, "base_cost": 3960, "trade_share": 0.17},
    {"product_id": "P05", "product_name": "Composite Cement", "category": "Composite",
     "base_price": 6950, "base_cost": 4000, "trade_share": 0.12},
    {"product_id": "P06", "product_name": "Sulphate Resisting Cement", "category": "Specialty",
     "base_price": 7900, "base_cost": 4900, "trade_share": 0.02},
    {"product_id": "P07", "product_name": "Low-Alkali OPC", "category": "Specialty",
     "base_price": 7800, "base_cost": 4850, "trade_share": 0.01},
]
PRODUCT_CATEGORIES = ["OPC", "PPC", "PSC", "Composite", "Specialty"]

# ---------------------------------------------------------------------------
# Plants and depots (synthetic; capacities are allocated to the simulated sample)
# ---------------------------------------------------------------------------
PLANTS: list[dict] = [
    {"plant_id": "PL01", "plant_name": "Ariyalur Plant (Synthetic)", "city": "Ariyalur",
     "weekly_capacity_t": 1500, "categories": ["OPC", "PPC", "Composite", "Specialty"], "cost_adj": -60},
    {"plant_id": "PL02", "plant_name": "Kadapa Plant (Synthetic)", "city": "Kadapa",
     "weekly_capacity_t": 1250, "categories": ["OPC", "PPC", "Composite"], "cost_adj": -20},
    {"plant_id": "PL03", "plant_name": "Belagavi Plant (Synthetic)", "city": "Belagavi",
     "weekly_capacity_t": 850, "categories": ["OPC", "PPC", "Composite"], "cost_adj": 40},
    {"plant_id": "PL04", "plant_name": "Rourkela Plant (Synthetic)", "city": "Rourkela",
     "weekly_capacity_t": 1350, "categories": ["PSC", "PPC", "Composite"], "cost_adj": -80},
    {"plant_id": "PL05", "plant_name": "Bokaro Plant (Synthetic)", "city": "Bokaro",
     "weekly_capacity_t": 1200, "categories": ["PSC", "OPC", "PPC"], "cost_adj": -40},
    {"plant_id": "PL06", "plant_name": "Cuttack Plant (Synthetic)", "city": "Cuttack",
     "weekly_capacity_t": 900, "categories": ["PPC", "PSC", "Composite", "Specialty"], "cost_adj": 10},
    {"plant_id": "PL07", "plant_name": "Guwahati Plant (Synthetic)", "city": "Guwahati",
     "weekly_capacity_t": 950, "categories": ["OPC", "PPC"], "cost_adj": 120},
    {"plant_id": "PL08", "plant_name": "Nagpur Grinding Unit (Synthetic)", "city": "Nagpur",
     "weekly_capacity_t": 700, "categories": ["OPC", "PPC"], "cost_adj": 70},
]

DEPOTS: list[dict] = [
    {"depot_id": "DP01", "city": "Kolkata", "storage_t": 900},
    {"depot_id": "DP02", "city": "Patna", "storage_t": 700},
    {"depot_id": "DP03", "city": "Ranchi", "storage_t": 500},
    {"depot_id": "DP04", "city": "Bhubaneswar", "storage_t": 650},
    {"depot_id": "DP05", "city": "Siliguri", "storage_t": 450},
    {"depot_id": "DP06", "city": "Silchar", "storage_t": 400},
    {"depot_id": "DP07", "city": "Dibrugarh", "storage_t": 400},
    {"depot_id": "DP08", "city": "Chennai", "storage_t": 900},
    {"depot_id": "DP09", "city": "Bengaluru", "storage_t": 850},
    {"depot_id": "DP10", "city": "Hyderabad", "storage_t": 700},
    {"depot_id": "DP11", "city": "Kochi", "storage_t": 500},
    {"depot_id": "DP12", "city": "Pune", "storage_t": 550},
    {"depot_id": "DP13", "city": "Raipur", "storage_t": 450},
    {"depot_id": "DP14", "city": "Lucknow", "storage_t": 500},
]

# ---------------------------------------------------------------------------
# Freight assumptions (synthetic tariff model)
# ---------------------------------------------------------------------------
VEHICLE_TYPES: dict[str, dict] = {
    "LCV-10": {"capacity_t": 10, "fixed_rate_per_km": 20.0, "litres_per_km": 0.20, "avg_speed_kmph": 42},
    "MAV-25": {"capacity_t": 25, "fixed_rate_per_km": 30.0, "litres_per_km": 0.33, "avg_speed_kmph": 38},
    "TRL-35": {"capacity_t": 35, "fixed_rate_per_km": 38.0, "litres_per_km": 0.42, "avg_speed_kmph": 34},
}
DEFAULT_DIESEL_PRICE = 92.0          # INR per litre
ROAD_DISTANCE_FACTOR = 1.25          # road km / great-circle km
EMPTY_RETURN_COST_RATIO = 0.70       # cost of an empty return km relative to a loaded km
DEFAULT_BACKHAUL_SHARE = 0.25        # share of return kms that find a backhaul load
PLANT_HANDLING_PER_T = 60.0          # INR per tonne loading at plant
DEPOT_HANDLING_PER_T = 110.0         # INR per tonne re-handling at depot
PRIMARY_FREIGHT_PER_TKM = 1.15       # INR per tonne-km plant to depot (rail / bulk road blend)
MARKET_HIRE_PREMIUM = 0.18           # market-hired vehicles cost 18% more than contracted fleet
PLANT_LOADING_HOURS = 4.0
DEPOT_LOADING_HOURS = 2.0
STANDARD_UTILISATION = 0.90          # load factor used to convert truck cost to per-tonne tariff

# ---------------------------------------------------------------------------
# Commercial and finance assumptions
# ---------------------------------------------------------------------------
TRADE_PAYMENT_TERMS = [7, 15, 21, 30]
NON_TRADE_PAYMENT_TERMS = [30, 45, 60, 90]
ANNUAL_FINANCING_RATE = 0.10         # cost of working capital used for financing cost of credit
DEFAULT_LGD_TRADE = 0.45
DEFAULT_LGD_NON_TRADE = 0.35
DEFAULT_PD_CONVERSION = 0.18         # share of 30+ day late payers assumed to default within 12 months
HOLDING_COST_PCT_ANNUAL = 0.14       # inventory holding cost as share of inventory value per year
MIN_MARGIN_PCT_POLICY = 0.10         # minimum contribution margin threshold for pricing
MAX_DISCOUNT_PER_T_POLICY = 450.0    # approved maximum discount per tonne (simulation policy)
WORKING_DAYS_PER_YEAR = 300


@dataclass(frozen=True)
class DataSettings:
    """User-selectable data-generation settings (hashable for caching)."""

    seed: int = DEFAULT_SEED
    n_dealers: int = DEFAULT_DEALER_SUBSET
    n_weeks: int = DEFAULT_HISTORY_WEEKS
    n_institutional: int = DEFAULT_INSTITUTIONAL_CUSTOMERS
