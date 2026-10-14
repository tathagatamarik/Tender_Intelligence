import numpy as np
import pandas as pd

from src.models.anomaly_detection import cusum_change_point, detect_trip_anomalies, offtake_early_warning
from src.models.credit_risk import FEATURES, apply_credit_policy, build_features, build_labels, kaplan_meier
from src.models.demand_forecasting import horizon_weights, series_horizon_forecast
from src.models.metrics import forecast_metrics, wape
from src.models.pricing_analysis import discount_frontier, estimate_elasticities, simulate_scenario, suggest_discount_range
from src.models.tender_scoring import EXCLUDED_POST_OUTCOME, _design, eligibility_checklist, train_tender_model


def test_wape_handles_zero_demand():
    assert np.isnan(wape([0, 0], [1, 2]))
    assert wape([0, 10], [1, 9]) == 0.2
    m = forecast_metrics([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["Bias"] == 0


def test_demand_split_is_chronological_and_metrics_from_holdout(env, demand):
    assert demand.holdout["week_start"].nunique() == demand.holdout_weeks
    assert demand.holdout["week_start"].min() > env.dealer_weekly["week_start"].sort_values().unique()[-demand.holdout_weeks - 1]
    recomputed = forecast_metrics(demand.holdout["actual"], demand.holdout["model"])
    assert abs(recomputed["WAPE"] - demand.metrics.loc["Gradient boosting model", "WAPE"]) < 1e-12
    assert (demand.forecast["forecast"] >= 0).all()
    assert (demand.forecast["lower"] <= demand.forecast["forecast"]).all()


def test_horizon_conversion(demand):
    assert horizon_weights(7).sum() == 1 and horizon_weights(14).sum() == 2
    assert abs(horizon_weights(30).sum() - 30 / 7) < 1e-9
    f7, f30 = series_horizon_forecast(demand, 7), series_horizon_forecast(demand, 30)
    assert (f30["forecast_t"] >= f7["forecast_t"] - 1e-9).all()


def test_credit_probabilities_bounded_and_no_leakage(env, credit):
    p = credit.scored["p_late_90d"].dropna()
    assert p.between(0, 1).all()
    assert "latent_risk" not in FEATURES and "risk_drift" not in FEATURES
    t0 = env.as_of - pd.Timedelta(days=150)
    feats = build_features(env, t0)
    labels = build_labels(env, t0)
    assert set(labels.unique()) <= {0, 1}
    assert credit.metrics["ROC-AUC"].min() > 0.6


def test_ecl_formula(env, credit):
    pol = apply_credit_policy(credit.scored, env, pd_conversion=0.2)
    np.testing.assert_allclose(pol["ecl_inr"], pol["pd_12m"] * pol["lgd"] * pol["ead_inr"])
    assert pol["pd_12m"].between(0, 1).all()
    tight = apply_credit_policy(credit.scored, env, credit_limit_multiplier=0.5)
    assert (tight["recommended_action"] != "WITHIN LIMIT").sum() >= (pol["recommended_action"] != "WITHIN LIMIT").sum()


def test_kaplan_meier():
    s = kaplan_meier(np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1]), np.array([0, 1, 2, 4]))
    np.testing.assert_allclose(s, [1.0, 0.75, 0.5, 0.0])


def test_tender_model_excludes_post_outcome_fields(env):
    tm = train_tender_model(env)
    for col in EXCLUDED_POST_OUTCOME:
        assert not any(c.startswith(col) for c in tm.columns)
    p = tm.model.predict_proba(_design(env.tenders_open, tm.columns))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_eligibility_is_explainable(env):
    t = env.tenders_open.iloc[0].to_dict()
    t["required_documents"] = ["ISO 14001 Certificate", "BIS Licence", "Unknown Certificate"]
    t["min_turnover_cr"] = 5000
    c = eligibility_checklist(t, env.company_documents)
    status = dict(zip(c["Criterion"], c["Status"]))
    assert status["Minimum turnover"] == "FAIL"
    assert status["Document: ISO 14001 Certificate"] == "FAIL"   # expired in synthetic repository
    assert status["Document: Unknown Certificate"] == "FAIL"     # not held
    assert status["Document: BIS Licence"] == "PASS"
    assert c["Detail"].str.len().gt(0).all() and c["Evidence source"].str.len().gt(0).all()


def test_eta_beats_baseline(eta):
    assert eta.metrics.iloc[0]["MAE (hours)"] < eta.metrics.iloc[1]["MAE (hours)"]


def test_anomaly_and_offtake_detection(env, eta):
    an = detect_trip_anomalies(env, eta.model)
    assert an.metrics.loc["GPS rules (transparent)", "Recall"] > 0.6
    assert cusum_change_point(np.array([0.0, 0.0, 0.5, 0.5, 0.5])) is not None
    assert cusum_change_point(np.zeros(6)) is None
    off = offtake_early_warning(env)
    assert (off.projects["at_risk_t"] >= 0).all()
    assert off.metrics.loc["ACTION REQUIRED or WATCH", "Recall"] > 0.5


def test_pricing_elasticity_and_policy(env):
    el = estimate_elasticities(env.pricing_panel)
    assert (el["elasticity"] < 0).all()
    base = {"list_price": 7000, "discount": 200, "net_price": 6800, "competitor_price": 6900, "cost": 4300,
            "freight": 1200, "monthly_volume_t": 1000}
    same = simulate_scenario(base, -1.5)
    assert abs(same["monthly_volume_t"] - 1000) < 1e-9
    assert simulate_scenario(base, -1.5, 0.05)["monthly_volume_t"] < 1000
    fr = discount_frontier(base, -1.5, max_discount=450, min_margin=0.10)
    sug = suggest_discount_range(fr)
    assert sug is None or (0 <= sug["low"] <= sug["high"] <= 450 and sug["margin_pct"] >= 0.10)
