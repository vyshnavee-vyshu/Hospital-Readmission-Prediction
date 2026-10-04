from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def confusion(y_true, y_pred) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "labels": ["Not Readmitted", "Readmitted"],
        "matrix": matrix.astype(int).tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def roc_points(y_true, y_proba) -> dict[str, Any]:
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "auc": None}
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = None
    # downsample long curves for the browser
    if len(fpr) > 120:
        idx = np.linspace(0, len(fpr) - 1, 120).astype(int)
        fpr = fpr[idx]
        tpr = tpr[idx]
    return {"fpr": fpr.astype(float).tolist(), "tpr": tpr.astype(float).tolist(), "auc": auc}


def feature_importance_rows(model, feature_names: list[str], column_groups: dict) -> list[dict[str, Any]]:
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    elif hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for item in model.calibrated_classifiers_:
            estimator = getattr(item, "estimator", None) or getattr(item, "base_estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                coefs.append(np.abs(np.asarray(estimator.coef_, dtype=float).ravel()))
        if coefs:
            values = np.mean(np.vstack(coefs), axis=0)

    if values is None or len(values) != len(feature_names):
        return []

    total = float(values.sum()) or 1.0
    numeric = set(column_groups.get("numeric", []))
    categorical = set(column_groups.get("categorical", []))
    high = set(column_groups.get("high_cardinality", []))

    rows = []
    for name, value in zip(feature_names, values):
        source = name
        if "__" in name:
            source = name.split("__", 1)[-1]
        base = source
        for suffix in ("_freq",):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if "=" in base:
            base = base.split("=", 1)[0]
        # sklearn OneHotEncoder names: original_value
        for prefix in list(numeric | categorical | high):
            if source.startswith(prefix) or base.startswith(prefix):
                base = prefix
                break
        if base in numeric:
            ftype = "Numeric"
        elif base in high:
            ftype = "High-cardinality"
        else:
            ftype = "Categorical"
        rows.append(
            {
                "feature": source.replace("numeric__", "").replace("categorical__", "").replace("high_cardinality__", ""),
                "importance": float(value / total),
                "type": ftype,
            }
        )
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows
