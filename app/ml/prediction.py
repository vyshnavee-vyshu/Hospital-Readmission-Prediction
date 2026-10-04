from __future__ import annotations

from typing import Any

import pandas as pd


def build_feature_row(payload: dict[str, Any], columns: list[str], schema: list[dict[str, Any]]) -> pd.DataFrame:
    lookup = {field["name"]: field for field in schema}
    row = {}
    for col in columns:
        field = lookup.get(col, {})
        value = payload.get(col, payload.get(field.get("label", ""), None))
        if value is None or value == "":
            if field.get("type") == "numeric":
                value = field.get("median", 0)
            else:
                options = field.get("options") or []
                value = options[0] if options else None
        if field.get("type") == "numeric":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = field.get("median", 0)
        row[col] = value
    return pd.DataFrame([row], columns=columns)


def predict_proba(model, X_transformed) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_transformed)[0, 1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_transformed)[0])
        proba = 1.0 / (1.0 + pow(2.718281828, -score))
    else:
        pred = int(model.predict(X_transformed)[0])
        proba = float(pred)
        return pred, proba
    pred = 1 if proba >= 0.5 else 0
    return pred, proba
