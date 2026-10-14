"""Dealer-product weekly demand forecasting.

Approach
--------
* The target is normalised by each series' recent 13-week mean so one model generalises across
  dealers of very different sizes; predictions are rescaled afterwards.
* A single global gradient-boosting model (LightGBM when installed, otherwise scikit-learn
  HistGradientBoostingRegressor) is trained across all dealer-product series using lag,
  rolling-window, calendar and static features.
* Evaluation is strictly chronological: the last ``holdout_weeks`` weeks are held out and
  predicted one week ahead using only information available at that time.
* Two baselines are evaluated on the same holdout: a 4-week moving average and a seasonal
  naive forecast (same week last year).
* Future forecasts are produced recursively (each predicted week feeds the next week's lags)
  from a model refitted on the full history.
* Uncertainty: approximate prediction intervals use each series' holdout residual standard
  deviation scaled by sqrt(horizon). They are indicative, not guaranteed coverage.

Forecasts are estimates of sell-in demand, not guaranteed sales outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.metrics import forecast_metrics

FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag52", "roll4", "roll8", "roll13", "roll26", "std8", "nz8",
            "ly_same5", "ly_next4_vs_prev4", "woy_sin", "woy_cos", "level", "product_code", "region_code"]
MAX_HORIZON_WEEKS = 8
Z_90 = 1.645


def _make_regressor(random_state: int = 0):
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=150,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                             random_state=random_state, verbose=-1), "LightGBM gradient boosting"
    except Exception:  # pragma: no cover - fallback path
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15,
                                             min_samples_leaf=150, random_state=random_state), \
            "scikit-learn HistGradientBoosting"


def _features_at(Y: np.ndarray, t: int, woy_t: int, level: np.ndarray, prod_code: np.ndarray,
                 region_code: np.ndarray) -> np.ndarray:
    """Feature matrix (series x features) to predict column ``t`` using columns < t only."""
    n = Y.shape[0]

    def lag(k):
        return Y[:, t - k] if t - k >= 0 else np.zeros(n)

    w13 = Y[:, max(0, t - 13):t]
    w8 = Y[:, max(0, t - 8):t]
    w4 = Y[:, max(0, t - 4):t]
    roll13 = w13.mean(axis=1)
    roll26 = Y[:, max(0, t - 26):t].mean(axis=1)
    lag52 = Y[:, t - 52] if t - 52 >= 0 else roll13
    if t - 56 >= 0:
        # Same season last year (smoothed) and last year's seasonal momentum around this week
        ly_same5 = Y[:, t - 54:t - 49].mean(axis=1)
        ly_prev4 = Y[:, t - 56:t - 52].mean(axis=1)
        ly_next4 = Y[:, t - 52:t - 48].mean(axis=1)
        ly_ratio = (ly_next4 + 0.5) / (ly_prev4 + 0.5)
    else:
        ly_same5, ly_ratio = roll13, np.ones(n)
    ang = 2 * np.pi * woy_t / 52.0
    return np.column_stack([
        lag(1), lag(2), lag(3), lag(4), lag52, w4.mean(axis=1), w8.mean(axis=1), roll13, roll26, w8.std(axis=1),
        (w8 > 0).mean(axis=1), ly_same5, ly_ratio, np.full(n, np.sin(ang)), np.full(n, np.cos(ang)), level,
        prod_code, region_code,
    ])


SCALE_EPS = 0.5


def _scale(X: np.ndarray) -> np.ndarray:
    """Per-row scale (recent 13-week mean) used to normalise the target across series sizes."""
    return X[:, FEATURES.index("roll13")] + SCALE_EPS


def _predict(model, X: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(pd.DataFrame(X, columns=FEATURES)) * _scale(X), 0, None)


@dataclass
class DemandModelResult:
    model_name: str
    series: pd.DataFrame            # series metadata incl. residual sigma
    weeks: pd.DatetimeIndex
    holdout: pd.DataFrame           # long: series_id, week_start, actual, model, ma4, snaive
    metrics: pd.DataFrame           # rows: method; cols: MAE, RMSE, WAPE, Bias
    forecast: pd.DataFrame          # long: series_id, horizon, week_start, forecast, lower, upper
    feature_importance: pd.DataFrame
    holdout_weeks: int
    train_rows: int

    @property
    def model_wape(self) -> float:
        return float(self.metrics.loc["Gradient boosting model", "WAPE"])

    def aggregated_wape(self, level: str = "depot_id") -> float:
        """WAPE of the same holdout predictions aggregated to a planning level (e.g. depot x week)."""
        h = self.holdout.merge(self.series[["series_id", level]], on="series_id")
        g = h.groupby([level, "week_start"])[["actual", "model"]].sum()
        return forecast_metrics(g["actual"], g["model"])["WAPE"]

    @property
    def forecast_accuracy(self) -> float:
        """Forecast accuracy defined as 1 - WAPE on the chronological holdout."""
        return max(0.0, 1.0 - self.model_wape)


def _wide(env) -> tuple[np.ndarray, pd.DataFrame, pd.DatetimeIndex]:
    dw = env.dealer_weekly
    wide = dw.pivot_table(index="series_id", columns="week_start", values="qty_t", aggfunc="sum").fillna(0.0)
    series = env.dealer_series.set_index("series_id").loc[wide.index].reset_index()
    series = series.merge(env.dealers[["dealer_id", "region", "depot_id"]], on="dealer_id", how="left")
    return wide.to_numpy(dtype=float), series, pd.DatetimeIndex(wide.columns)


def _stack_training(Y, weeks, t_start, t_end, level, pcode, rcode):
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    X, y = [], []
    for t in range(t_start, t_end):
        X.append(_features_at(Y, t, woy[t], level, pcode, rcode))
        y.append(Y[:, t])
    return np.vstack(X), np.concatenate(y)


def train_demand_model(env, holdout_weeks: int = 12, random_state: int = 0) -> DemandModelResult:
    """Train, evaluate on a chronological holdout, and produce recursive future forecasts."""
    Y, series, weeks = _wide(env)
    n_s, n_w = Y.shape
    if n_w < holdout_weeks + 20:
        raise ValueError("Insufficient history for chronological evaluation.")
    cutoff = n_w - holdout_weeks
    t_start = 13
    pcode = series["product_id"].str[1:].astype(int).to_numpy(dtype=float)
    regions = sorted(series["region"].unique())
    rcode = series["region"].map({r: i for i, r in enumerate(regions)}).to_numpy(dtype=float)
    # Static level computed from training weeks only (prevents leakage into the holdout).
    level_train = Y[:, :cutoff].mean(axis=1)

    X_tr, y_tr = _stack_training(Y, weeks, t_start, cutoff, level_train, pcode, rcode)
    model, model_name = _make_regressor(random_state)
    model.fit(pd.DataFrame(X_tr, columns=FEATURES), y_tr / _scale(X_tr))

    # One-week-ahead holdout predictions
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    rows = []
    for t in range(cutoff, n_w):
        Xt = _features_at(Y, t, woy[t], level_train, pcode, rcode)
        pred = _predict(model, Xt)
        ma4 = Y[:, t - 4:t].mean(axis=1)
        snaive = Y[:, t - 52] if t - 52 >= 0 else ma4
        rows.append(pd.DataFrame({"series_id": series["series_id"], "week_start": weeks[t], "actual": Y[:, t],
                                  "model": pred, "ma4": ma4, "snaive": snaive}))
    holdout = pd.concat(rows, ignore_index=True)
    metrics = pd.DataFrame({
        "Gradient boosting model": forecast_metrics(holdout["actual"], holdout["model"]),
        "Baseline: 4-week moving average": forecast_metrics(holdout["actual"], holdout["ma4"]),
        "Baseline: seasonal naive (52 weeks)": forecast_metrics(holdout["actual"], holdout["snaive"]),
    }).T

    resid = holdout.assign(err=holdout["model"] - holdout["actual"]).groupby("series_id")["err"].std()
    series["sigma_weekly"] = series["series_id"].map(resid).fillna(0.0).to_numpy()

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:  # pragma: no cover
        imp = np.zeros(len(FEATURES))
    fi = pd.DataFrame({"feature": FEATURES, "importance": imp / max(imp.sum(), 1e-9)}).sort_values(
        "importance", ascending=False)

    # Refit on the full history for forward forecasts.
    level_full = Y.mean(axis=1)
    X_all, y_all = _stack_training(Y, weeks, t_start, n_w, level_full, pcode, rcode)
    final_model, _ = _make_regressor(random_state)
    final_model.fit(pd.DataFrame(X_all, columns=FEATURES), y_all / _scale(X_all))

    future_weeks = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=MAX_HORIZON_WEEKS, freq="W-MON")
    Y_ext = np.hstack([Y, np.zeros((n_s, MAX_HORIZON_WEEKS))])
    f_rows = []
    for h in range(MAX_HORIZON_WEEKS):
        t = n_w + h
        Xt = _features_at(Y_ext, t, int(future_weeks[h].isocalendar().week), level_full, pcode, rcode)
        pred = _predict(final_model, Xt)
        Y_ext[:, t] = pred
        sig = series["sigma_weekly"].to_numpy() * np.sqrt(h + 1)
        f_rows.append(pd.DataFrame({"series_id": series["series_id"], "horizon": h + 1,
                                    "week_start": future_weeks[h], "forecast": pred,
                                    "lower": np.clip(pred - Z_90 * sig, 0, None), "upper": pred + Z_90 * sig}))
    forecast = pd.concat(f_rows, ignore_index=True)
    return DemandModelResult(model_name=model_name, series=series, weeks=weeks, holdout=holdout, metrics=metrics,
                             forecast=forecast, feature_importance=fi, holdout_weeks=holdout_weeks,
                             train_rows=len(y_tr))


def horizon_weights(days: int) -> np.ndarray:
    """Weekly weights that convert weekly forecasts into a day-based horizon (7/14/30 days)."""
    full, rem = divmod(days, 7)
    w = np.zeros(MAX_HORIZON_WEEKS)
    w[:min(full, MAX_HORIZON_WEEKS)] = 1.0
    if rem and full < MAX_HORIZON_WEEKS:
        w[full] = rem / 7.0
    return w


def series_horizon_forecast(result: DemandModelResult, days: int) -> pd.DataFrame:
    """Total forecast per series over a day-based horizon with an approximate 90% interval."""
    w = horizon_weights(days)
    f = result.forecast.copy()
    f["w"] = f["horizon"].map(lambda h: w[h - 1])
    f["wf"] = f["forecast"] * f["w"]
    agg = f.groupby("series_id").agg(forecast_t=("wf", "sum")).reset_index()
    eff_weeks = w.sum()
    agg = agg.merge(result.series[["series_id", "dealer_id", "product_id", "region", "depot_id", "sigma_weekly"]],
                    on="series_id", how="left")
    spread = Z_90 * agg["sigma_weekly"] * np.sqrt(max(eff_weeks, 1e-9))
    agg["lower_t"] = (agg["forecast_t"] - spread).clip(lower=0)
    agg["upper_t"] = agg["forecast_t"] + spread
    agg["horizon_days"] = days
    return agg
