# Hospital Readmission Prediction Web Application

Local healthcare analytics workspace for uploading hospital encounter CSVs, processing them with scikit-learn pipelines, training classifiers, and scoring estimated readmission probability in the browser.

This is an academic demonstration. It is not a medical device and must not be used for clinical decisions.

## Features

- CSV upload and preview in the browser
- Target detection for `0/1`, `Yes/No`, `True/False`, and diabetes-style `NO` / `>30` / `<30`
- Duplicate handling, missing-value imputation, encoding, and scaling without fitting on the test split
- Exploratory charts from the uploaded file
- Logistic Regression, Decision Tree, Random Forest, XGBoost, and a calibrated linear SVM
- Hold-out Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, and ROC curve
- Active model selection by F1-score
- Dynamic prediction form based on processed features
- Prediction history in SQLite

## Architecture

A single FastAPI process serves HTML, CSS, JavaScript, and JSON APIs.

```
Browser  →  FastAPI (Uvicorn)  →  Pandas / Scikit-learn / XGBoost
                              →  Joblib model files
                              →  SQLite prediction history
```

## Technology stack

- Frontend: HTML5, CSS3, JavaScript, Chart.js, Fetch API
- Backend: Python, FastAPI, Uvicorn
- Machine learning: Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- Storage: local CSV uploads and SQLite

## Installation

You need Python 3.10 or later.

No Node.js, npm, Docker, or cloud services are required.

## Windows instructions

1. Extract `Hospital_Readmission_Prediction_Web_Application.zip`.
2. Double-click `run.bat`.
3. Wait until the terminal prints that the server is starting.
4. Open Chrome, Edge, or Firefox.
5. Go to [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Linux / macOS instructions

```bash
chmod +x run.sh
./run.sh
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## How to run (manual)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## How to upload a dataset

Open **Dataset**, drag a CSV onto the drop zone or use **Browse CSV**. Confirm rows, columns, missing values, duplicates, and the detected target column, then click **Process Dataset**.

## How to train models

Open **Model Lab**, keep the models you want selected, and click **Train Selected Models**. Metrics are computed on a stratified 20% hold-out split. The preprocessor is fitted on the training split only.

The default active model is the trained model with the highest F1-score in the current evaluation. You can choose another trained model with **Use as active model**.

## How to make predictions

After training, open **Prediction**. The form lists features used by the processed pipeline. Submit to score one encounter. History is stored without extra identifying fields.

## Dataset format

Provide a CSV with one row per encounter and a target column named `readmitted`, `readmission`, or `target`.

Typical optional columns include age, gender, admission type, discharge disposition, `time_in_hospital`, diagnoses, laboratory flags, prior visits, medication counts, and procedure counts. Columns that are missing from a file are simply omitted from analysis and the prediction form.

The application was designed around common hospital readmission tables, including the UCI Diabetes 130-US Hospitals style schema, but it will accept other CSVs with a mappable binary target.

## Target mapping

| Raw value | Encoded | Meaning |
| --- | --- | --- |
| `0`, `No`, `False`, `NO` | 0 | Not Readmitted |
| `1`, `Yes`, `True`, `>30`, `<30` | 1 | Readmitted |

## Troubleshooting

- **Python was not found**: install Python 3.10+ and enable “Add Python to PATH”.
- **Port 8000 already in use**: stop the other process or change the port in `run.bat` / `run.sh`.
- **Target column not found**: rename the label column to `readmitted`.
- **Training is slow**: SVM on large files uses a 6,000-row subsample; other models use the full training split.
- **XGBoost install error on Windows**: install the latest Visual C++ Redistributable, then run `run.bat` again.

## Healthcare disclaimer

Outputs are estimated machine learning scores from the uploaded sample. They are not diagnoses, risk scores approved for care, or clinical recommendations.
