from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from app.paths import MODEL_KEYS

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def _linear_svm(random_state: int = 42):
    base = LinearSVC(max_iter=4000, dual="auto", class_weight="balanced", random_state=random_state)
    return CalibratedClassifierCV(estimator=base, cv=3)


def build_estimators(pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "svm": _linear_svm(),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=pos_weight,
            random_state=42,
        )
    return models


def fit_selected(model_keys: list[str], X_train, y_train) -> dict[str, Any]:
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = neg / pos
    catalog = build_estimators(pos_weight)
    fitted = {}
    errors = {}
    for key in model_keys:
        if key not in catalog:
            errors[key] = "Unknown model key."
            continue
        estimator = catalog[key]
        try:
            X_use, y_use = X_train, y_train
            if key == "svm" and len(X_train) > 6000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X_train), size=6000, replace=False)
                X_use = X_train[idx]
                y_use = y_train[idx]
            estimator.fit(X_use, y_use)
            fitted[key] = estimator
        except Exception as exc:
            errors[key] = str(exc)
    return {"models": fitted, "errors": errors, "labels": MODEL_KEYS}
