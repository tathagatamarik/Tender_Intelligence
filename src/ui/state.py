"""Streamlit caching and session-state layer.

Expensive deterministic steps (data generation, model training) are cached as resources keyed
by the data settings. Scenario-dependent computations are cached by their parameter values, so
changing (for example) a freight rate re-runs only the dispatch optimisation, never the
unrelated models. Cached objects are treated as read-only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DataSettings
from src.data.synthetic_data import generate_dealer_population, generate_environment
from src.models.anomaly_detection import detect_trip_anomalies, offtake_early_warning
from src.models.credit_risk import apply_credit_policy, train_credit_model
from src.models.dealer_segmentation import receivables_position, segment_dealers, visit_priority
from src.models.demand_forecasting import train_demand_model
from src.models.eta_prediction import train_eta_model
from src.models.pricing_analysis import estimate_elasticities, market_opportunities
from src.models.tender_scoring import score_open_tenders, train_tender_model
from src.optimisation.freight_optimisation import FreightScenario, optimise_dispatch
from src.optimisation.inventory_optimisation import PlanningScenario, build_planning_demand, run_planning
from src.optimisation.replenishment import ReplenishmentSettings, recommend_replenishment
from src.services.order_economics import compute_order_economics, summarise_economics

Key = tuple  # (seed, n_dealers, n_weeks, n_institutional)


def settings() -> DataSettings:
    if "data_settings" not in st.session_state:
        st.session_state["data_settings"] = DataSettings()
    return st.session_state["data_settings"]


def key() -> Key:
    s = settings()
    return (s.seed, s.n_dealers, s.n_weeks, s.n_institutional)


def _s(k: Key) -> DataSettings:
    return DataSettings(seed=k[0], n_dealers=k[1], n_weeks=k[2], n_institutional=k[3])


# ---------------------------------------------------------------------------
# Data and trained models (resources)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Generating synthetic business environment...")
def env_for(k: Key):
    return generate_environment(_s(k))


def env():
    return env_for(key())


@st.cache_resource(show_spinner="Simulating the 49,300-partner dealer network...")
def population():
    return generate_dealer_population()


@st.cache_resource(show_spinner="Training dealer demand forecasting model...")
def demand_for(k: Key):
    return train_demand_model(env_for(k))


@st.cache_resource(show_spinner="Training late-payment risk model...")
def credit_for(k: Key):
    return train_credit_model(env_for(k))


@st.cache_resource(show_spinner="Training tender win-probability model...")
def tender_model_for(k: Key):
    return train_tender_model(env_for(k))


@st.cache_resource(show_spinner="Training ETA model...")
def eta_for(k: Key):
    return train_eta_model(env_for(k))


@st.cache_resource(show_spinner="Detecting shipment anomalies...")
def trip_anomalies_for(k: Key, contamination: float = 0.10, detour: float = 1.20, halt: float = 6.0):
    return detect_trip_anomalies(env_for(k), eta_for(k).model, contamination, detour, halt)


@st.cache_resource(show_spinner="Running offtake early-warning analysis...")
def offtake_for(k: Key, threshold: float = 0.80):
    e = env_for(k)
    cost = float(e.products["base_cost"].mean())
    return offtake_early_warning(e, attain_threshold=threshold, production_cost_per_t=cost)


@st.cache_resource(show_spinner="Segmenting dealers...")
def segments_for(k: Key):
    return segment_dealers(env_for(k), demand_for(k))


@st.cache_resource(show_spinner="Estimating price elasticities...")
def elasticities_for(k: Key):
    return estimate_elasticities(env_for(k).pricing_panel)


@st.cache_resource(show_spinner=False)
def receivables_for(k: Key):
    return receivables_position(env_for(k))


@st.cache_resource(show_spinner=False)
def order_economics_for(k: Key) -> pd.DataFrame:
    """Historical orders with contribution economics (no risk adjustment)."""
    return compute_order_economics(env_for(k).orders)


@st.cache_resource(show_spinner=False)
def commercial_baseline(k: Key) -> dict:
    o = order_economics_for(k)
    e = env_for(k)
    recent = o[o["order_date"] >= e.as_of - pd.Timedelta(days=365)]
    return summarise_economics(recent)


# ---------------------------------------------------------------------------
# Scenario computations (keyed by parameters)
# ---------------------------------------------------------------------------
def replenishment_settings() -> ReplenishmentSettings:
    return st.session_state.get("replenishment_settings", ReplenishmentSettings())


@st.cache_data(show_spinner="Calculating replenishment recommendations...")
def replenishment_for(k: Key, rs_repr: str, _rs: ReplenishmentSettings) -> pd.DataFrame:
    base = commercial_baseline(k)
    return recommend_replenishment(env_for(k), demand_for(k), _rs, receivables_for(k), base["net_price_per_t"],
                                   base["contribution_per_t"])


def replenishment(rs: ReplenishmentSettings | None = None) -> pd.DataFrame:
    rs = rs or replenishment_settings()
    return replenishment_for(key(), repr(rs), rs)


def visit_list(rs: ReplenishmentSettings | None = None) -> pd.DataFrame:
    return visit_priority(segments_for(key()), replenishment(rs), commercial_baseline(key())["contribution_per_t"])


def freight_scenario() -> FreightScenario:
    return st.session_state.get("freight_scenario", FreightScenario())


@st.cache_resource(show_spinner="Optimising dispatch plan...")
def freight_for(k: Key, sc_repr: str, _sc: FreightScenario):
    return optimise_dispatch(env_for(k), eta_for(k).model, _sc)


def freight(sc: FreightScenario | None = None):
    sc = sc or freight_scenario()
    return freight_for(key(), repr(sc), sc)


def credit_params() -> dict:
    return st.session_state.get("credit_params", {"pd_conversion": 0.18, "lgd_trade": 0.45, "lgd_corporate": 0.35,
                                                  "lgd_government": 0.15, "limit_multiplier": 1.0})


def pending_orders_value(k: Key) -> pd.DataFrame:
    oo = env_for(k).open_orders
    return oo.assign(order_value_inr=oo["qty_t"] * (oo["list_price_per_t"] - oo["discount_per_t"]))[["customer_id", "order_value_inr"]]


@st.cache_data(show_spinner="Applying credit policy...")
def credit_policy_for(k: Key, params_repr: str, params: dict) -> pd.DataFrame:
    cr = credit_for(k)
    return apply_credit_policy(cr.scored, env_for(k), params["pd_conversion"],
                               {"Trade": params["lgd_trade"], "Corporate": params["lgd_corporate"],
                                "Government": params["lgd_government"]},
                               params["limit_multiplier"], pending_orders_value(k))


def credit_policy(params: dict | None = None) -> pd.DataFrame:
    params = params or credit_params()
    return credit_policy_for(key(), repr(sorted(params.items())), params)


@st.cache_data(show_spinner="Scoring open tenders...")
def tenders_scored_for(k: Key, min_margin: float, pd_conversion: float) -> pd.DataFrame:
    return score_open_tenders(env_for(k), tender_model_for(k), credit_for(k).scored, min_margin, pd_conversion)


def tenders_scored() -> pd.DataFrame:
    return tenders_scored_for(key(), st.session_state.get("tender_min_margin", 0.08), credit_params()["pd_conversion"])


def planning_scenario() -> PlanningScenario:
    return st.session_state.get("planning_scenario", PlanningScenario())


@st.cache_resource(show_spinner="Optimising production and inventory plan...")
def planning_for(k: Key, sc_repr: str, _sc: PlanningScenario, rs_repr: str, _rs: ReplenishmentSettings,
                 tender_repr: str):
    e = env_for(k)
    demand = build_planning_demand(e, demand_for(k), replenishment_for(k, rs_repr, _rs), offtake_for(k),
                                   tenders_scored(), _sc)
    return run_planning(e, demand, _sc)


def planning(sc: PlanningScenario | None = None):
    sc = sc or planning_scenario()
    rs = replenishment_settings()
    tender_repr = repr((st.session_state.get("tender_min_margin", 0.08), credit_params()["pd_conversion"]))
    return planning_for(key(), repr(sc), sc, repr(rs), rs, tender_repr)


@st.cache_resource(show_spinner=False)
def pricing_opportunities_for(k: Key, min_margin: float, max_discount: float) -> pd.DataFrame:
    return market_opportunities(env_for(k).pricing_panel, elasticities_for(k), min_margin, max_discount)


def pricing_opportunities() -> pd.DataFrame:
    from src.config import MAX_DISCOUNT_PER_T_POLICY, MIN_MARGIN_PCT_POLICY

    return pricing_opportunities_for(key(), st.session_state.get("policy_min_margin", MIN_MARGIN_PCT_POLICY),
                                     st.session_state.get("policy_max_discount", MAX_DISCOUNT_PER_T_POLICY))


def all_results() -> dict:
    """Current-scenario results from every module (used by the dashboard and recommendations)."""
    k = key()
    return {
        "replenishment": replenishment(), "segments": segments_for(k), "visit_priority": visit_list(),
        "freight": freight(), "credit_policy": credit_policy(), "tenders": tenders_scored(),
        "planning": planning(), "offtake": offtake_for(k), "pricing_opps": pricing_opportunities(),
    }


def reset_scenarios() -> None:
    """Restore default scenario assumptions (keeps data settings and cached models)."""
    keep = {"data_settings"}
    for k in list(st.session_state.keys()):
        if k not in keep and not k.startswith("sb_"):
            del st.session_state[k]
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Global sidebar filters
# ---------------------------------------------------------------------------
def global_regions() -> list[str]:
    from src.config import REGIONS

    sel = st.session_state.get("sb_regions") or []
    return list(sel) if sel else list(REGIONS)


def global_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    from src.config import AS_OF_DATE

    rng = st.session_state.get("sb_date_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    end = pd.Timestamp(AS_OF_DATE)
    return end - pd.Timedelta(days=365), end
