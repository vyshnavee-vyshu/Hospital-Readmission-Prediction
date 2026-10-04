from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from app.ml.evaluation import binary_metrics, confusion, feature_importance_rows, roc_points
from app.ml.prediction import build_feature_row, predict_proba
from app.ml.preprocessing import detect_target_column, map_target_series, prepare_training_frame
from app.ml.training import fit_selected
from app.paths import (
    DATA_DIR,
    DB_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_KEYS,
    PREPROCESSOR_PATH,
    TRAIN_COLUMNS_PATH,
)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATE_PATH = APP_DIR / "templates" / "index.html"

app = FastAPI(title="Hospital Readmission Prediction", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TrainRequest(BaseModel):
    models: list[str] | None = None


class PredictRequest(BaseModel):
    model: str | None = None
    features: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            feature_count INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {}


def save_metadata(data: dict[str, Any]) -> None:
    METADATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def current_csv_path(meta: dict[str, Any]) -> Path | None:
    name = meta.get("filename")
    if not name:
        return None
    path = DATA_DIR / name
    return path if path.exists() else None


def load_dataset(meta: dict[str, Any] | None = None) -> pd.DataFrame:
    meta = meta or load_metadata()
    path = current_csv_path(meta)
    if path is None:
        raise HTTPException(status_code=400, detail="No dataset has been uploaded.")
    return pd.read_csv(path)


def require_processed(meta: dict[str, Any]) -> None:
    if not meta.get("processed"):
        raise HTTPException(status_code=400, detail="Process the dataset before continuing.")
    if not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=400, detail="Processed artifacts are missing. Process the dataset again.")


def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


def load_train_columns() -> list[str]:
    return joblib.load(TRAIN_COLUMNS_PATH)


def model_path(key: str) -> Path:
    return MODEL_DIR / f"{key}.joblib"


def load_model(key: str):
    path = model_path(key)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"{MODEL_KEYS.get(key, key)} is not trained.")
    return joblib.load(path)


def trained_keys(meta: dict[str, Any]) -> list[str]:
    return [key for key in MODEL_KEYS if meta.get("results", {}).get(key) and model_path(key).exists()]


def pick_active_model(meta: dict[str, Any]) -> str | None:
    active = meta.get("active_model")
    available = trained_keys(meta)
    if active in available:
        return active
    best = None
    best_f1 = -1.0
    for key in available:
        f1 = meta.get("results", {}).get(key, {}).get("f1")
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best = key
    return best


def find_column(df: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for token in tokens:
        if token in lowered:
            return lowered[token]
    for key, original in lowered.items():
        if any(token in key for token in tokens):
            return original
    return None


def eda_payload(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    y = map_target_series(df[target_col])
    frame = df.loc[y.notna()].copy()
    y = y.loc[y.notna()].astype(int)
    payload: dict[str, Any] = {}

    counts = y.value_counts().to_dict()
    payload["readmission_distribution"] = {
        "labels": ["Not Readmitted", "Readmitted"],
        "values": [int(counts.get(0, 0)), int(counts.get(1, 0))],
    }

    age_col = find_column(frame, ("age",))
    if age_col:
        grouped = pd.crosstab(frame[age_col].astype(str), y)
        payload["age_distribution"] = {
            "labels": grouped.index.astype(str).tolist(),
            "not_readmitted": grouped[0].tolist() if 0 in grouped.columns else [0] * len(grouped),
            "readmitted": grouped[1].tolist() if 1 in grouped.columns else [0] * len(grouped),
        }
    else:
        payload["age_distribution"] = None

    gender_col = find_column(frame, ("gender", "sex"))
    if gender_col:
        grouped = pd.crosstab(frame[gender_col].astype(str), y)
        payload["gender_analysis"] = {
            "labels": grouped.index.astype(str).tolist(),
            "not_readmitted": grouped[0].tolist() if 0 in grouped.columns else [0] * len(grouped),
            "readmitted": grouped[1].tolist() if 1 in grouped.columns else [0] * len(grouped),
        }
    else:
        payload["gender_analysis"] = None

    los_col = find_column(frame, ("time_in_hospital", "length_of_stay", "los", "lengthofstay"))
    if los_col:
        nums = pd.to_numeric(frame[los_col], errors="coerce")
        valid = nums.notna()
        if valid.any():
            grouped = pd.DataFrame({"los": nums[valid], "y": y[valid]}).groupby("y")["los"].mean()
            payload["length_of_stay"] = {
                "labels": ["Not Readmitted", "Readmitted"],
                "values": [float(grouped.get(0, 0)), float(grouped.get(1, 0))],
            }
        else:
            payload["length_of_stay"] = None
    else:
        payload["length_of_stay"] = None

    med_col = find_column(frame, ("num_medications", "number_of_medications", "medications"))
    if med_col:
        nums = pd.to_numeric(frame[med_col], errors="coerce")
        valid = nums.notna()
        if valid.any():
            grouped = pd.DataFrame({"meds": nums[valid], "y": y[valid]}).groupby("y")["meds"].mean()
            payload["medication_analysis"] = {
                "labels": ["Not Readmitted", "Readmitted"],
                "values": [float(grouped.get(0, 0)), float(grouped.get(1, 0))],
            }
        else:
            payload["medication_analysis"] = None
    else:
        payload["medication_analysis"] = None

    missing = frame.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(18)
    payload["missing_values"] = {
        "labels": missing.index.tolist(),
        "values": [int(v) for v in missing.tolist()],
        "empty": bool(missing.empty),
    }

    numeric_df = frame.select_dtypes(include=[np.number]).copy()
    numeric_df["readmitted"] = y.to_numpy()
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        cols = list(corr.columns[:12])
        payload["correlation"] = {
            "features": cols,
            "matrix": corr.loc[cols, cols].fillna(0).round(3).values.tolist(),
        }
    else:
        payload["correlation"] = None

    payload["rows"] = int(len(frame))
    payload["target"] = target_col
    return payload


@app.get("/")
def root():
    return FileResponse(TEMPLATE_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    safe_name = Path(file.filename).name
    dest = DATA_DIR / "dataset.csv"
    dest.write_bytes(content)
    try:
        df = pd.read_csv(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV does not contain any rows.")

    target = detect_target_column(list(df.columns))
    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    mapping = None
    readmission_rate = None
    if target:
        try:
            y = map_target_series(df[target])
            mapping = {
                "column": target,
                "positive": "Readmitted (1)",
                "negative": "Not Readmitted (0)",
            }
            valid = y.dropna()
            if len(valid):
                readmission_rate = float(valid.mean())
        except ValueError:
            mapping = {"column": target, "error": "Target values could not be mapped yet."}

    meta = {
        "filename": "dataset.csv",
        "original_filename": safe_name,
        "uploaded_at": utc_now(),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns.astype(str)),
        "missing_values": missing_total,
        "duplicates": duplicates,
        "target_column": target,
        "target_mapping": mapping,
        "readmission_rate": readmission_rate,
        "processed": False,
        "results": {},
        "active_model": None,
        "input_schema": [],
        "feature_count": int(df.shape[1] - (1 if target else 0)),
    }
    for key in MODEL_KEYS:
        model_path(key).unlink(missing_ok=True)
    PREPROCESSOR_PATH.unlink(missing_ok=True)
    TRAIN_COLUMNS_PATH.unlink(missing_ok=True)
    save_metadata(meta)
    return {"info": meta}


@app.get("/api/dataset/info")
def dataset_info():
    meta = load_metadata()
    if not meta:
        return {"loaded": False}
    return {"loaded": True, **meta}


@app.get("/api/dataset/preview")
def dataset_preview(limit: int = 40):
    df = load_dataset()
    sample = df.head(max(1, min(limit, 100)))
    return {
        "columns": list(sample.columns.astype(str)),
        "rows": json.loads(sample.fillna("").to_json(orient="records")),
        "total_rows": int(len(df)),
    }


@app.post("/api/dataset/process")
def process_dataset():
    meta = load_metadata()
    df = load_dataset(meta)
    target = meta.get("target_column") or detect_target_column(list(df.columns))
    if not target:
        raise HTTPException(
            status_code=400,
            detail="No target column found. Include a column named readmitted, readmission, or target.",
        )
    try:
        prepared = prepare_training_frame(df, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    X = prepared["X"]
    y = prepared["y"]
    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="The target column must contain both classes after mapping.")
    if len(X) < 30:
        raise HTTPException(status_code=400, detail="At least 30 complete rows are required after processing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = prepared["preprocessor"]
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(list(X.columns), TRAIN_COLUMNS_PATH)
    split_path = MODEL_DIR / "split.joblib"
    joblib.dump(
        {
            "X_train": X_train_t,
            "X_test": X_test_t,
            "y_train": y_train.to_numpy(),
            "y_test": y_test.to_numpy(),
            "feature_names": feature_names,
        },
        split_path,
    )

    y_all = map_target_series(df[target]).dropna()
    meta.update(
        {
            "processed": True,
            "processed_at": utc_now(),
            "target_column": target,
            "target_mapping": prepared["target_mapping"],
            "rows_after_process": prepared["rows_after"],
            "duplicates_removed": prepared["duplicates_removed"],
            "dropped_columns": prepared["dropped_columns"],
            "numeric_features": prepared["column_groups"]["numeric"],
            "categorical_features": prepared["column_groups"]["categorical"],
            "high_cardinality_features": prepared["column_groups"]["high_cardinality"],
            "encoded_feature_count": len(feature_names),
            "input_schema": prepared["input_schema"],
            "feature_count": int(X.shape[1]),
            "readmission_rate": float(y_all.mean()) if len(y_all) else None,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "results": {},
            "active_model": None,
            "column_groups": prepared["column_groups"],
        }
    )
    for key in MODEL_KEYS:
        model_path(key).unlink(missing_ok=True)
    save_metadata(meta)
    return {"info": meta}


@app.post("/api/models/train")
def train_models(body: TrainRequest | None = None):
    meta = load_metadata()
    require_processed(meta)
    requested = (body.models if body and body.models else list(MODEL_KEYS.keys()))
    requested = [key for key in requested if key in MODEL_KEYS]
    if not requested:
        raise HTTPException(status_code=400, detail="Select at least one model to train.")

    split = joblib.load(MODEL_DIR / "split.joblib")
    outcome = fit_selected(requested, split["X_train"], split["y_train"])
    results = meta.get("results", {})
    for key, model in outcome["models"].items():
        joblib.dump(model, model_path(key))
        y_pred = model.predict(split["X_test"])
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(split["X_test"])[:, 1]
        else:
            y_proba = y_pred.astype(float)
        metrics = binary_metrics(split["y_test"], y_pred, y_proba)
        importance = feature_importance_rows(model, split["feature_names"], meta.get("column_groups", {}))
        results[key] = {
            **metrics,
            "trained": True,
            "trained_at": utc_now(),
            "display_name": MODEL_KEYS[key],
            "confusion": confusion(split["y_test"], y_pred),
            "roc": roc_points(split["y_test"], y_proba),
            "feature_importance": importance[:20],
        }
    for key, message in outcome["errors"].items():
        results[key] = {
            "trained": False,
            "error": message,
            "display_name": MODEL_KEYS.get(key, key),
        }

    meta["results"] = results
    meta["trained_at"] = utc_now()
    meta["active_model"] = pick_active_model(meta)
    save_metadata(meta)
    return {"results": results, "active_model": meta["active_model"], "errors": outcome["errors"]}


@app.get("/api/models/results")
def model_results():
    meta = load_metadata()
    return {
        "results": meta.get("results", {}),
        "active_model": pick_active_model(meta),
        "criterion": "F1-score",
        "criterion_note": "Highest F1-score in the current evaluation",
        "models": MODEL_KEYS,
    }


@app.get("/api/models/status")
def model_status():
    meta = load_metadata()
    loaded = bool(meta.get("filename") and current_csv_path(meta))
    processed = bool(meta.get("processed"))
    trained = bool(trained_keys(meta))
    active = pick_active_model(meta)
    active_metrics = meta.get("results", {}).get(active or "", {})
    return {
        "dataset_loaded": loaded,
        "dataset_processed": processed,
        "model_trained": trained,
        "filename": meta.get("original_filename"),
        "rows": meta.get("rows"),
        "columns": meta.get("columns"),
        "features": meta.get("feature_count"),
        "readmission_rate": meta.get("readmission_rate"),
        "active_model": active,
        "active_model_name": MODEL_KEYS.get(active or "", None),
        "f1": active_metrics.get("f1"),
        "roc_auc": active_metrics.get("roc_auc"),
        "target_column": meta.get("target_column"),
        "processed": processed,
    }


class ActiveModelRequest(BaseModel):
    model: str


@app.post("/api/models/select")
def select_model(body: ActiveModelRequest):
    meta = load_metadata()
    if body.model not in trained_keys(meta):
        raise HTTPException(status_code=400, detail="Train this model before selecting it.")
    meta["active_model"] = body.model
    save_metadata(meta)
    return {"active_model": body.model, "display_name": MODEL_KEYS[body.model]}


@app.get("/api/analytics/eda")
def analytics_eda():
    meta = load_metadata()
    df = load_dataset(meta)
    target = meta.get("target_column") or detect_target_column(list(df.columns))
    if not target:
        raise HTTPException(status_code=400, detail="A target column is required for analysis.")
    return eda_payload(df, target)


@app.get("/api/analytics/features")
def analytics_features():
    meta = load_metadata()
    active = pick_active_model(meta)
    if not active:
        raise HTTPException(status_code=400, detail="Train at least one model to inspect feature importance.")
    rows = meta.get("results", {}).get(active, {}).get("feature_importance") or []
    return {"model": active, "display_name": MODEL_KEYS[active], "features": rows}


@app.get("/api/analytics/roc")
def analytics_roc(model: str | None = None):
    meta = load_metadata()
    key = model or pick_active_model(meta)
    if not key:
        raise HTTPException(status_code=400, detail="Train a model before requesting an ROC curve.")
    roc = meta.get("results", {}).get(key, {}).get("roc")
    if not roc:
        raise HTTPException(status_code=400, detail="ROC data is not available for this model.")
    return {"model": key, "display_name": MODEL_KEYS[key], **roc}


@app.get("/api/analytics/confusion-matrix")
def analytics_confusion(model: str | None = None):
    meta = load_metadata()
    key = model or pick_active_model(meta)
    if not key:
        raise HTTPException(status_code=400, detail="Train a model before requesting a confusion matrix.")
    matrix = meta.get("results", {}).get(key, {}).get("confusion")
    if not matrix:
        raise HTTPException(status_code=400, detail="Confusion matrix is not available for this model.")
    return {"model": key, "display_name": MODEL_KEYS[key], **matrix}


@app.get("/api/overview")
def overview():
    meta = load_metadata()
    status = model_status()
    recent = []
    with db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, model_name, prediction, probability FROM predictions ORDER BY id DESC LIMIT 8"
        ).fetchall()
        recent = [dict(row) for row in rows]
    features = []
    active = pick_active_model(meta)
    if active:
        features = (meta.get("results", {}).get(active, {}).get("feature_importance") or [])[:8]
    comparison = []
    for key, label in MODEL_KEYS.items():
        item = meta.get("results", {}).get(key)
        if item and item.get("trained"):
            comparison.append({"key": key, "name": label, "f1": item.get("f1"), "roc_auc": item.get("roc_auc")})
    eda = None
    if status["dataset_loaded"] and meta.get("target_column"):
        try:
            eda = eda_payload(load_dataset(meta), meta["target_column"])
        except Exception:
            eda = None
    return {
        "status": status,
        "recent_predictions": recent,
        "top_features": features,
        "model_comparison": comparison,
        "distribution": None if not eda else eda["readmission_distribution"],
    }


@app.post("/api/predict")
def predict(body: PredictRequest):
    meta = load_metadata()
    require_processed(meta)
    key = body.model or pick_active_model(meta)
    if not key:
        raise HTTPException(status_code=400, detail="Train a model before generating a prediction.")
    model = load_model(key)
    preprocessor = load_preprocessor()
    columns = load_train_columns()
    row = build_feature_row(body.features, columns, meta.get("input_schema", []))
    transformed = preprocessor.transform(row)
    label, probability = predict_proba(model, transformed)
    created = utc_now()
    with db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (created_at, model_name, prediction, probability, feature_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (created, key, int(label), float(probability), int(len(columns))),
        )
        conn.commit()
        record_id = cursor.lastrowid
    return {
        "id": record_id,
        "timestamp": created,
        "model": key,
        "model_name": MODEL_KEYS[key],
        "prediction": int(label),
        "label": "Readmitted" if label == 1 else "Not Readmitted",
        "probability": float(probability),
        "disclaimer": "This is an estimated machine learning model output and is not a medical diagnosis or clinical recommendation.",
    }


@app.get("/api/predictions/history")
def prediction_history():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, created_at, model_name, prediction, probability FROM predictions ORDER BY id DESC LIMIT 200"
        ).fetchall()
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "timestamp": row["created_at"],
                "model": row["model_name"],
                "model_name": MODEL_KEYS.get(row["model_name"], row["model_name"]),
                "prediction": row["prediction"],
                "label": "Readmitted" if row["prediction"] == 1 else "Not Readmitted",
                "probability": row["probability"],
            }
        )
    return {"items": items}


@app.delete("/api/predictions/history/{item_id}")
def delete_prediction(item_id: int):
    with db() as conn:
        cursor = conn.execute("DELETE FROM predictions WHERE id = ?", (item_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction record not found.")
    return {"deleted": item_id}


@app.get("/api/schema")
def prediction_schema():
    meta = load_metadata()
    require_processed(meta)
    return {
        "fields": meta.get("input_schema", []),
        "active_model": pick_active_model(meta),
        "models": {key: MODEL_KEYS[key] for key in trained_keys(meta)},
    }

