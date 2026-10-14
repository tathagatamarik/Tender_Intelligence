"""Late-payment risk model, expected credit loss and payment-timing survival analysis.

Target definition
-----------------
For a snapshot date T0, features are computed only from invoices and payments observable on or
before T0. The label is 1 if any invoice falling due in the following 90 days was paid more than
30 days late (or remained unpaid more than 30 days past due). Labels are therefore strictly in
the future relative to features (no leakage). The model is trained on the T0 = as-of minus 150
days snapshot and evaluated on customers held out from training.

Expected credit loss (SIMPLIFIED DEMONSTRATION FORMULA, NOT A REGULATORY ACCOUNTING MODEL):
    ECL = PD x LGD x EAD
    PD  = probability of 30+ day late payment x assumed default conversion rate
    LGD = assumed loss given default by customer class
    EAD = outstanding receivables + value of pending (unshipped) orders

No protected personal attributes are used. Features describe commercial payment behaviour only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_LGD_NON_TRADE, DEFAULT_LGD_TRADE, DEFAULT_PD_CONVERSION
from src.models.metrics import calibration_table, classification_metrics

FEATURES = ["avg_days_late_12m", "share_late30_12m", "late_trend", "overdue_ratio", "max_dpd",
            "credit_utilisation", "dso_proxy", "billing_trend", "payment_terms_days", "is_non_trade",
            "is_government", "is_developer", "log_billing_12m", "customer_concentration"]
FEATURE_LABELS = {
    "avg_days_late_12m": "Average days paid late (12 months)",
    "share_late30_12m": "Share of invoices paid 30+ days late",
    "late_trend": "Recent change in lateness (3m vs prior)",
    "overdue_ratio": "Overdue share of outstanding",
    "max_dpd": "Maximum days past due today",
    "credit_utilisation": "Credit-limit utilisation",
    "dso_proxy": "Days sales outstanding (proxy)",
    "billing_trend": "Billing trend (3m vs prior)",
    "payment_terms_days": "Contractual payment terms",
    "is_non_trade": "Non-trade customer",
    "is_government": "Government customer",
    "is_developer": "Real-estate developer segment",
    "log_billing_12m": "Billing volume (log, 12 months)",
    "customer_concentration": "Share of company billing",
}
LGD_BY_CLASS = {"Trade": DEFAULT_LGD_TRADE, "Corporate": DEFAULT_LGD_NON_TRADE, "Government": 0.15}
RISK_BANDS = [(0.0, 0.15, "Low"), (0.15, 0.35, "Medium"), (0.35, 0.60, "High"), (0.60, 1.01, "Very High")]


def risk_band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "Very High"


def build_features(env, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Customer features using only information observable at ``cutoff``."""
    inv = env.invoices[env.invoices["invoice_date"] <= cutoff].copy()
    paid_by_cutoff = inv["paid_date"].notna() & (inv["paid_date"] <= cutoff)
    inv["paid_obs"] = paid_by_cutoff
    inv["days_late_obs"] = np.where(paid_by_cutoff, (inv["paid_date"] - inv["due_date"]).dt.days, np.nan)
    inv["unpaid_dpd"] = np.where(~paid_by_cutoff & (inv["due_date"] < cutoff),
                                 (cutoff - inv["due_date"]).dt.days, 0)
    w12 = inv[inv["invoice_date"] > cutoff - pd.Timedelta(days=365)]
    w3 = inv[inv["invoice_date"] > cutoff - pd.Timedelta(days=90)]
    w_prior = w12[w12["invoice_date"] <= cutoff - pd.Timedelta(days=90)]
    g12 = w12.groupby("customer_id")
    late_any = (w12["days_late_obs"].fillna(0) > 30) | (w12["unpaid_dpd"] > 30)
    f = pd.DataFrame({
        "avg_days_late_12m": g12["days_late_obs"].apply(lambda s: s.clip(lower=0).mean()),
        "share_late30_12m": late_any.groupby(w12["customer_id"]).mean(),
        "billing_12m": g12["amount_inr"].sum(),
    })
    recent_late = w3.groupby("customer_id")["days_late_obs"].apply(lambda s: s.clip(lower=0).mean())
    prior_late = w_prior.groupby("customer_id")["days_late_obs"].apply(lambda s: s.clip(lower=0).mean())
    f["late_trend"] = recent_late.reindex(f.index).fillna(prior_late.reindex(f.index)) - prior_late.reindex(f.index)
    outstanding = inv[~paid_by_cutoff].groupby("customer_id")["amount_inr"].sum()
    overdue = inv[(~paid_by_cutoff) & (inv["due_date"] < cutoff)].groupby("customer_id")["amount_inr"].sum()
    f["outstanding_inr"] = outstanding.reindex(f.index).fillna(0)
    f["overdue_inr"] = overdue.reindex(f.index).fillna(0)
    f["overdue_ratio"] = np.where(f["outstanding_inr"] > 0, f["overdue_inr"] / f["outstanding_inr"], 0)
    f["max_dpd"] = inv.groupby("customer_id")["unpaid_dpd"].max().reindex(f.index).fillna(0)
    bill3 = w3.groupby("customer_id")["amount_inr"].sum().reindex(f.index).fillna(0)
    bill_prior = w_prior.groupby("customer_id")["amount_inr"].sum().reindex(f.index).fillna(0)
    f["billing_trend"] = np.where(bill_prior > 0, (bill3 / 3) / (bill_prior / 9) - 1, 0).clip(-1, 3)
    f["dso_proxy"] = np.where(bill3 > 0, f["outstanding_inr"] / (bill3 / 90), 120).clip(0, 365)
    f["customer_concentration"] = f["billing_12m"] / max(f["billing_12m"].sum(), 1)
    f["log_billing_12m"] = np.log1p(f["billing_12m"])
    f = f.reset_index().rename(columns={"index": "customer_id"})
    c = env.customers[["customer_id", "channel", "customer_segment", "payment_terms_days", "credit_limit_inr",
                       "lgd_class"]]
    f = c.merge(f, on="customer_id", how="left")
    f["credit_utilisation"] = np.where(f["credit_limit_inr"] > 0, f["outstanding_inr"].fillna(0) / f["credit_limit_inr"], 0)
    f["is_non_trade"] = (f["channel"] == "Non-Trade").astype(int)
    f["is_government"] = (f["customer_segment"] == "Government Department").astype(int)
    f["is_developer"] = (f["customer_segment"] == "Real Estate Developer").astype(int)
    for col in FEATURES:
        f[col] = f[col].astype(float).fillna(0.0)
    f["has_history"] = f["billing_12m"].fillna(0) > 0
    return f


def build_labels(env, cutoff: pd.Timestamp, window_days: int = 90, late_threshold: int = 30) -> pd.Series:
    inv = env.invoices
    win = inv[(inv["due_date"] >= cutoff) & (inv["due_date"] < cutoff + pd.Timedelta(days=window_days))]
    late = ((win["paid_date"] - win["due_date"]).dt.days > late_threshold) | \
        (win["paid_date"].isna() & ((env.as_of - win["due_date"]).dt.days > late_threshold))
    return late.groupby(win["customer_id"]).max().astype(int)


@dataclass
class CreditModelResult:
    champion_name: str
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    coefficients: pd.DataFrame
    scored: pd.DataFrame
    train_size: int
    test_size: int
    snapshot_date: pd.Timestamp
    test_frame: pd.DataFrame
    logistic: object
    champion: object


def train_credit_model(env, snapshot_offset_days: int = 150, random_state: int = 0) -> CreditModelResult:
    t0 = env.as_of - pd.Timedelta(days=snapshot_offset_days)
    X = build_features(env, t0)
    y = build_labels(env, t0)
    data = X[X["has_history"]].merge(y.rename("label"), left_on="customer_id", right_index=True, how="inner")
    if data["label"].nunique() < 2 or len(data) < 60:
        raise ValueError("Insufficient labelled customers to train the credit model.")
    train, test = train_test_split(data, test_size=0.3, random_state=random_state, stratify=data["label"])

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    logit.fit(train[FEATURES], train["label"])
    gbm = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                max_leaf_nodes=15, random_state=random_state),
                                 method="sigmoid", cv=3)
    gbm.fit(train[FEATURES], train["label"])
    p_log = logit.predict_proba(test[FEATURES])[:, 1]
    p_gbm = gbm.predict_proba(test[FEATURES])[:, 1]
    m_log = classification_metrics(test["label"], p_log, threshold=0.35)
    m_gbm = classification_metrics(test["label"], p_gbm, threshold=0.35)
    metrics = pd.DataFrame({"Logistic regression (transparent)": m_log,
                            "Gradient boosting (calibrated)": m_gbm}).T
    # Prefer the transparent model unless the challenger is materially better on ROC-AUC.
    use_gbm = (m_gbm["ROC-AUC"] - m_log["ROC-AUC"]) > 0.02
    champion = gbm if use_gbm else logit
    champion_name = "Gradient boosting (calibrated)" if use_gbm else "Logistic regression (transparent)"
    p_champ = p_gbm if use_gbm else p_log
    calib = calibration_table(test["label"], p_champ)

    lr = logit.named_steps["logisticregression"]
    coefs = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[f] for f in FEATURES],
                          "coefficient": lr.coef_[0]}).sort_values("coefficient", key=np.abs, ascending=False)

    scored = score_customers(env, champion, logit)
    test_frame = test[["customer_id", "label"]].assign(predicted=p_champ)
    return CreditModelResult(champion_name=champion_name, metrics=metrics, calibration=calib, coefficients=coefs,
                             scored=scored, train_size=len(train), test_size=len(test), snapshot_date=t0,
                             test_frame=test_frame, logistic=logit, champion=champion)


def explain_customer(logit_pipeline, row: pd.Series, top_n: int = 4) -> list[tuple[str, float]]:
    """Top feature contributions (standardised value x coefficient) from the transparent model."""
    scaler = logit_pipeline.named_steps["standardscaler"]
    lr = logit_pipeline.named_steps["logisticregression"]
    x = (row[FEATURES].astype(float).to_numpy() - scaler.mean_) / scaler.scale_
    contrib = x * lr.coef_[0]
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [(FEATURE_LABELS[FEATURES[i]], float(contrib[i])) for i in order]


def score_customers(env, champion, logit) -> pd.DataFrame:
    """Score all customers as of the simulation date."""
    f = build_features(env, env.as_of)
    f["p_late_90d"] = np.clip(champion.predict_proba(f[FEATURES])[:, 1], 0, 1)
    f.loc[~f["has_history"], "p_late_90d"] = np.nan
    f["risk_band"] = f["p_late_90d"].apply(lambda p: "No recent history" if pd.isna(p) else risk_band(p))
    drivers = []
    for _, r in f.iterrows():
        top = explain_customer(logit, r, 3)
        drivers.append("; ".join(f"{name} ({'raises' if v > 0 else 'lowers'} risk)" for name, v in top))
    f["key_risk_drivers"] = drivers
    return f


def apply_credit_policy(scored: pd.DataFrame, env, pd_conversion: float = DEFAULT_PD_CONVERSION,
                        lgd_overrides: dict | None = None, credit_limit_multiplier: float = 1.0,
                        pending_orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute PD, LGD, EAD, ECL, exposure after pending orders and recommended actions."""
    lgd_map = {**LGD_BY_CLASS, **(lgd_overrides or {})}
    s = scored.copy()
    s["p_late_90d_filled"] = s["p_late_90d"].fillna(s["p_late_90d"].median())
    s["pd_12m"] = (s["p_late_90d_filled"] * pd_conversion).clip(0, 1)
    s["lgd"] = s["lgd_class"].map(lgd_map).fillna(DEFAULT_LGD_TRADE)
    pend = pd.Series(0.0, index=s["customer_id"])
    if pending_orders is not None and len(pending_orders):
        pend = pending_orders.groupby("customer_id")["order_value_inr"].sum()
    s["pending_orders_inr"] = s["customer_id"].map(pend).fillna(0.0)
    s["ead_inr"] = s["outstanding_inr"].fillna(0) + s["pending_orders_inr"]
    s["ecl_inr"] = s["pd_12m"] * s["lgd"] * s["ead_inr"]
    s["scenario_credit_limit_inr"] = s["credit_limit_inr"] * credit_limit_multiplier
    s["exposure_after_orders_inr"] = s["ead_inr"]
    s["limit_headroom_inr"] = s["scenario_credit_limit_inr"] - s["exposure_after_orders_inr"]
    s["collection_priority_score"] = (s["overdue_inr"].fillna(0) * (0.5 + s["p_late_90d_filled"])
                                      * (1 + s["max_dpd"].clip(0, 180) / 60)).round(0)
    conditions = [
        (s["risk_band"].isin(["High", "Very High"])) & (s["limit_headroom_inr"] < 0),
        s["limit_headroom_inr"] < 0,
        (s["risk_band"] == "Very High") | (s["max_dpd"] > 60),
        (s["risk_band"] == "High") & (s["overdue_inr"] > 0),
    ]
    actions = ["ESCALATE ORDER RELEASE TO FINANCE", "CREDIT REVIEW - EXPOSURE ABOVE LIMIT",
               "COLLECTION INTERVENTION", "CREDIT REVIEW"]
    s["recommended_action"] = np.select(conditions, actions, "WITHIN LIMIT")
    return s


def kaplan_meier(durations: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival estimate S(t) evaluated on ``grid`` (implemented directly)."""
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    times = np.unique(d[e])
    surv, s = [], 1.0
    at = {}
    for t in times:
        n_risk = (d >= t).sum()
        n_event = ((d == t) & e).sum()
        if n_risk > 0:
            s *= 1 - n_event / n_risk
        at[t] = s
    keys = np.array(sorted(at))
    vals = np.array([at[k] for k in keys])
    for g in grid:
        idx = np.searchsorted(keys, g, side="right") - 1
        surv.append(vals[idx] if idx >= 0 else 1.0)
    return np.array(surv)


def payment_survival_by_band(env, scored: pd.DataFrame, horizon_days: int = 120) -> pd.DataFrame:
    """Probability an invoice is still unpaid t days after its due date, by current risk band."""
    inv = env.invoices[(env.invoices["due_date"] < env.as_of)
                       & (env.invoices["invoice_date"] >= env.as_of - pd.Timedelta(days=365))].copy()
    inv = inv.merge(scored[["customer_id", "risk_band"]], on="customer_id", how="left")
    paid = inv["paid_date"].notna()
    inv["duration"] = np.where(paid, (inv["paid_date"] - inv["due_date"]).dt.days,
                               (env.as_of - inv["due_date"]).dt.days).clip(min=0)
    inv["event"] = paid
    grid = np.arange(0, horizon_days + 1, 5)
    rows = []
    for band, g in inv.groupby("risk_band"):
        if len(g) < 20:
            continue
        s = kaplan_meier(g["duration"].to_numpy(), g["event"].to_numpy(), grid)
        rows.append(pd.DataFrame({"risk_band": band, "days_after_due": grid, "prob_unpaid": s}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["risk_band", "days_after_due", "prob_unpaid"])


def expected_collection_days(env, scored: pd.DataFrame) -> pd.Series:
    """Expected days from invoice to cash per customer = payment terms + historical mean lateness."""
    s = scored.set_index("customer_id")
    return (s["payment_terms_days"] + s["avg_days_late_12m"].clip(lower=0)).rename("expected_collection_days")
