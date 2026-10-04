from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.paths import TARGET_CANDIDATES


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categoricals as empirical frequencies."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.n_features_in_ = frame.shape[1]
        self.maps_ = []
        for i in range(self.n_features_in_):
            values = frame.iloc[:, i].astype(str)
            self.maps_.append(values.value_counts(normalize=True).to_dict())
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        out = np.zeros((len(frame), self.n_features_in_), dtype=float)
        for i in range(self.n_features_in_):
            values = frame.iloc[:, i].astype(str)
            out[:, i] = values.map(self.maps_[i]).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.array([f"{name}_freq" for name in names], dtype=object)


def detect_target_column(columns: list[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for key, original in lowered.items():
        if "readmit" in key:
            return original
    return None


def map_target_series(series: pd.Series) -> pd.Series:
    raw = series.copy()
    text = raw.astype(str).str.strip()
    lowered = text.str.lower()

    unique = set(lowered.dropna().unique())
    unique.discard("nan")
    unique.discard("none")
    unique.discard("")

    if unique <= {"0", "1"}:
        return pd.to_numeric(lowered, errors="coerce")
    if unique <= {"yes", "no"}:
        return lowered.map({"yes": 1, "no": 0})
    if unique <= {"true", "false"}:
        return lowered.map({"true": 1, "false": 0})
    if unique <= {"readmitted", "not readmitted"}:
        return lowered.map({"readmitted": 1, "not readmitted": 0})

    diabetes_tokens = {"no", ">30", "<30", "no.", "> 30", "< 30"}
    if unique & {"no", ">30", "<30"} or unique <= diabetes_tokens:
        return lowered.map(lambda v: 0 if v in {"no", "0", "nan", "none", ""} else 1)

    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.9:
        values = set(numeric.dropna().unique())
        if values <= {0, 1}:
            return numeric
        if values <= {0, 1, 2}:
            # treat any positive class as readmitted
            return (numeric > 0).astype(float)

    raise ValueError(
        "Could not map the target column to binary labels. "
        "Supported values: 0/1, Yes/No, True/False, or NO/>30/<30."
    )


def target_mapping_description(series: pd.Series) -> dict[str, Any]:
    text = series.astype(str).str.strip()
    unique = sorted({v for v in text.unique() if str(v).lower() not in {"nan", "none", ""}})
    mapping_rows = []
    mapped = map_target_series(series)
    for value in unique[:20]:
        mask = text == value
        assigned = mapped[mask].dropna()
        if assigned.empty:
            continue
        label = int(assigned.iloc[0])
        mapping_rows.append(
            {
                "raw": str(value),
                "encoded": label,
                "meaning": "Readmitted" if label == 1 else "Not Readmitted",
            }
        )
    return {
        "unique_raw": unique[:30],
        "mapping": mapping_rows,
        "positive_meaning": "Readmitted",
        "negative_meaning": "Not Readmitted",
    }


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric: list[str] = []
    low_card: list[str] = []
    high_card: list[str] = []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            continue
        if nunique == len(df) and series.dtype == object:
            continue

        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_ratio >= 0.9 and nunique > 8:
            numeric.append(col)
            continue
        if numeric_ratio >= 0.95 and nunique > 2:
            numeric.append(col)
            continue

        if nunique > 40:
            high_card.append(col)
        else:
            low_card.append(col)

    return {"numeric": numeric, "categorical": low_card, "high_cardinality": high_card}


def drop_unusable_columns(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    keep = []
    n = len(df)
    for col in df.columns:
        if col == target:
            keep.append(col)
            continue
        missing_rate = df[col].isna().mean()
        nunique = df[col].nunique(dropna=True)
        name = col.lower()
        is_id = any(token in name for token in ("encounter_id", "patient_nbr", "patient_id", "admission_id"))
        if missing_rate > 0.8 or nunique <= 1 or (is_id and nunique > n * 0.9) or (nunique == n):
            dropped.append(col)
            continue
        keep.append(col)
    return df[keep], dropped


def build_preprocessor(column_groups: dict[str, list[str]]) -> ColumnTransformer:
    transformers = []
    if column_groups["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                column_groups["numeric"],
            )
        )
    if column_groups["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                column_groups["categorical"],
            )
        )
    if column_groups["high_cardinality"]:
        transformers.append(
            (
                "high_cardinality",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("freq", FrequencyEncoder()),
                    ]
                ),
                column_groups["high_cardinality"],
            )
        )
    if not transformers:
        raise ValueError("No usable features were found after preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def infer_field_section(name: str) -> str:
    key = name.lower()
    if any(token in key for token in ("age", "gender", "sex", "race", "weight")):
        return "patient"
    if any(
        token in key
        for token in (
            "admission",
            "discharge",
            "time_in_hospital",
            "los",
            "length",
            "source",
            "payer",
        )
    ):
        return "admission"
    if any(
        token in key
        for token in (
            "diag",
            "diagnosis",
            "lab",
            "a1c",
            "glucose",
            "max_glu",
            "num_diagnos",
        )
    ):
        return "clinical"
    if any(
        token in key
        for token in (
            "inpatient",
            "outpatient",
            "emergency",
            "previous",
            "prior",
            "visit",
        )
    ):
        return "history"
    if any(
        token in key
        for token in (
            "med",
            "insulin",
            "change",
            "procedure",
            "diabetes",
            "drug",
            "treatment",
        )
    ):
        return "treatment"
    return "other"


def build_input_schema(df: pd.DataFrame, column_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    schema = []
    ordered = (
        column_groups["numeric"]
        + column_groups["categorical"]
        + column_groups["high_cardinality"]
    )
    for col in ordered:
        series = df[col]
        numeric = col in column_groups["numeric"]
        field: dict[str, Any] = {
            "name": col,
            "label": col.replace("_", " ").strip().title(),
            "type": "numeric" if numeric else "categorical",
            "section": infer_field_section(col),
        }
        if numeric:
            nums = pd.to_numeric(series, errors="coerce")
            field["min"] = float(nums.min()) if nums.notna().any() else None
            field["max"] = float(nums.max()) if nums.notna().any() else None
            field["median"] = float(nums.median()) if nums.notna().any() else 0.0
        else:
            counts = series.astype(str).value_counts().head(80)
            field["options"] = [str(v) for v in counts.index.tolist() if str(v).lower() != "nan"]
            field["default"] = field["options"][0] if field["options"] else ""
        schema.append(field)
    return schema


def prepare_training_frame(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    working = df.copy()
    working, dropped = drop_unusable_columns(working, target_column)
    y = map_target_series(working[target_column])
    valid = y.notna()
    working = working.loc[valid].copy()
    y = y.loc[valid].astype(int)
    working = working.drop(columns=[target_column])
    working = working.replace(["?", "Unknown/Invalid", "None", "NULL"], np.nan)
    before = len(working)
    duplicate_count = int(working.duplicated().sum())
    mask = ~working.duplicated()
    working = working.loc[mask].copy()
    y = y.loc[mask]
    groups = classify_columns(working)
    feature_frame = working[
        groups["numeric"] + groups["categorical"] + groups["high_cardinality"]
    ].copy()
    preprocessor = build_preprocessor(groups)
    schema = build_input_schema(feature_frame, groups)
    mapping = target_mapping_description(df[target_column])
    return {
        "X": feature_frame,
        "y": y,
        "preprocessor": preprocessor,
        "column_groups": groups,
        "input_schema": schema,
        "dropped_columns": dropped,
        "duplicates_removed": duplicate_count,
        "rows_after": int(len(feature_frame)),
        "rows_before": int(before + duplicate_count),
        "target_mapping": mapping,
    }
