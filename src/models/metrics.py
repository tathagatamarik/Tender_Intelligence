"""Evaluation metrics computed from actual predictions (never hard-coded)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def wape(actual, predicted) -> float:
    """Weighted absolute percentage error = sum|e| / sum|y|. NaN when total actual demand is zero."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = np.abs(a).sum()
    return float(np.abs(a - p).sum() / denom) if denom > 0 else float("nan")


def forecast_metrics(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = p - a
    total = a.sum()
    return {
        "MAE": float(np.mean(np.abs(err))) if len(a) else float("nan"),
        "RMSE": float(np.sqrt(np.mean(err ** 2))) if len(a) else float("nan"),
        "WAPE": wape(a, p),
        "Bias": float(err.sum() / total) if total > 0 else float("nan"),
    }


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(proba, dtype=float), 0, 1)
    pred = (p >= threshold).astype(int)
    out = {
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier score": float(brier_score_loss(y, p)),
        "Positive rate (actual)": float(y.mean()) if len(y) else float("nan"),
        "Threshold": threshold,
    }
    out["ROC-AUC"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def calibration_table(y_true, proba, n_bins: int = 8) -> pd.DataFrame:
    """Observed event rate versus mean predicted probability by quantile bin."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(proba, dtype=float)})
    if df.empty:
        return pd.DataFrame(columns=["bin", "mean_predicted", "observed_rate", "count"])
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, len(df)), labels=False)
    out = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), count=("y", "size"))
    return out.reset_index()


def detection_metrics(y_true, y_flag) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    f = np.asarray(y_flag).astype(int)
    return {
        "Precision": float(precision_score(y, f, zero_division=0)),
        "Recall": float(recall_score(y, f, zero_division=0)),
        "F1": float(f1_score(y, f, zero_division=0)),
        "Flagged": int(f.sum()),
        "Actual positives": int(y.sum()),
    }
