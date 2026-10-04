import json
import random
import urllib.error
import urllib.request
from io import BytesIO

import pandas as pd

BASE = "http://127.0.0.1:8000"


def req(path, method="GET", data=None, headers=None, files=None):
    url = BASE + path
    if files:
        field, filename, content, ctype = files
        boundary = "----WebBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    else:
        raw = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(url, data=raw, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                request.add_header(k, v)
    try:
        with urllib.request.urlopen(request, timeout=180) as resp:
            payload = resp.read()
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


rng = random.Random(42)
rows = []
for i in range(180):
    inpatient = rng.randint(0, 6)
    los = rng.randint(1, 14)
    rec = {
        "encounter_id": 1000 + i,
        "age": rng.choice(["[50-60)", "[60-70)", "[70-80)", "[80-90)"]),
        "gender": rng.choice(["Female", "Male"]),
        "admission_type_id": rng.choice(["Emergency", "Urgent", "Elective"]),
        "discharge_disposition_id": rng.choice(["Home", "SNF", "Home Health"]),
        "time_in_hospital": los,
        "num_lab_procedures": rng.randint(10, 80),
        "num_procedures": rng.randint(0, 6),
        "num_medications": rng.randint(1, 25),
        "number_outpatient": rng.randint(0, 5),
        "number_emergency": rng.randint(0, 4),
        "number_inpatient": inpatient,
        "diag_1": rng.choice(["250.00", "414.01", "428.0", "486", "599.0"]),
        "number_diagnoses": rng.randint(1, 9),
        "A1Cresult": rng.choice(["None", "Norm", ">7", ">8"]),
        "insulin": rng.choice(["No", "Steady", "Up", "Down"]),
        "diabetesMed": rng.choice(["Yes", "No"]),
        "change": rng.choice(["Ch", "No"]),
    }
    rec["readmitted"] = ">30" if (inpatient >= 2 or los >= 10 or rng.random() < 0.2) else "NO"
    if rng.random() < 0.08:
        rec["readmitted"] = "<30"
    rows.append(rec)
csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()

status, body = req("/api/health")
assert status == 200, body
status, body = req("/")
assert status == 200 and b"Hospital Readmission Prediction" in body
status, body = req("/api/dataset/upload", files=("file", "hospital.csv", csv_bytes, "text/csv"))
print("upload", status, json.loads(body)["info"]["target_column"])
assert status == 200
status, body = req("/api/dataset/preview")
assert status == 200
status, body = req("/api/dataset/process", method="POST")
print("process", status, body[:400])
assert status == 200, body
status, body = req("/api/analytics/eda")
assert status == 200
print("eda ok")
status, body = req("/api/models/train", method="POST", data={})
print("train", status, body[:800] if status != 200 else json.loads(body)["active_model"])
assert status == 200, body
results = json.loads(body)
for key, value in results["results"].items():
    print(key, value.get("trained"), value.get("f1"), value.get("error"))
status, body = req("/api/analytics/features")
print("features", status, len(json.loads(body).get("features", [])))
assert req("/api/analytics/roc")[0] == 200
assert req("/api/analytics/confusion-matrix")[0] == 200
schema = json.loads(req("/api/schema")[1])
feats = {}
for field in schema["fields"]:
    if field["type"] == "numeric":
        feats[field["name"]] = field.get("median") or 0
    else:
        feats[field["name"]] = (field.get("options") or [""])[0]
status, body = req("/api/predict", method="POST", data={"features": feats})
print("predict", status, body[:400])
assert status == 200, body
history = json.loads(req("/api/predictions/history")[1])["items"]
assert history
pid = history[0]["id"]
assert req(f"/api/predictions/history/{pid}", method="DELETE")[0] == 200
assert req("/api/overview")[0] == 200
print("ALL API TESTS PASSED")
