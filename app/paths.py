from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR
ROOT_DIR = BASE_DIR.parent
DATA_DIR = APP_DIR / "data" / "uploads"
MODEL_DIR = APP_DIR / "models" / "saved"
DB_PATH = APP_DIR / "database" / "predictions.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

METADATA_PATH = MODEL_DIR / "metadata.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
TRAIN_COLUMNS_PATH = MODEL_DIR / "train_columns.joblib"

MODEL_KEYS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine",
}

TARGET_CANDIDATES = (
    "readmitted",
    "readmission",
    "readmit",
    "target",
    "label",
    "outcome",
    "readmission_flag",
)
