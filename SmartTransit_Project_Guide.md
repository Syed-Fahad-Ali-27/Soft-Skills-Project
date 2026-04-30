# SmartTransit — Project Workflow & Step-by-Step Guide

**Team:** Syed Fahad Ali · Leon Jose Mathew · Mohammad Omar Sharieff  
**Stack:** Python · FastAPI · Scikit-learn · XGBoost · TensorFlow/Keras · STIB/MIVB GTFS API

---

## Repository Structure

```
smarttransit/
├── main                    ← stable, always working
├── branch-fahad            ← Data pipeline & feature engineering
├── branch-leon             ← ML model training & evaluation
└── branch-omar             ← FastAPI backend + frontend dashboard
```

**Rule:** no one pushes directly to `main`. Every feature is a pull request from your branch, reviewed by at least one teammate before merging.

---

## Phase 0 — Project Setup (All, Day 1)

Everyone does this once together.

**0.1 Clone the repo and set up a virtual environment**

```bash
git clone https://github.com/<your-org>/smarttransit.git
cd smarttransit
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**0.2 Create `requirements.txt` (commit to main)**

```
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
tensorflow
requests
python-dotenv
joblib
pytest
httpx
```

**0.3 Create a `.env` file (never commit this — add to `.gitignore`)**

```
STIB_API_KEY=your_key_here
```

Get your API key from: https://data.stib-mivb.brussels/pages/home/

**0.4 Agree on a shared folder structure (commit to main)**

```
smarttransit/
├── data/
│   ├── raw/          ← downloaded GTFS files, API responses
│   ├── processed/    ← cleaned CSVs ready for training
│   └── README.md     ← document what each file contains
├── models/           ← saved .pkl and .h5 model files
├── notebooks/        ← Jupyter notebooks for exploration
├── src/
│   ├── pipeline/     ← Fahad's data code
│   ├── models/       ← Leon's training code
│   └── api/          ← Omar's FastAPI code
├── frontend/         ← Omar's dashboard
├── tests/            ← unit tests
└── main.py           ← FastAPI entry point
```

---

## Phase 1 — Data Pipeline (Branch: branch-fahad)

**Goal:** Produce a clean, analysis-ready dataset of historical + live STIB data.

### Step 1.1 — Download GTFS Static data

GTFS Static gives you the timetable backbone: routes, stops, scheduled times.

```python
# src/pipeline/download_gtfs.py
import requests, zipfile, io, os

GTFS_URL = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/gtfs-files-production/exports/gtfs"

def download_gtfs_static(dest_folder="data/raw/gtfs_static"):
    os.makedirs(dest_folder, exist_ok=True)
    r = requests.get(GTFS_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest_folder)
    print(f"Extracted {len(z.namelist())} files to {dest_folder}")

if __name__ == "__main__":
    download_gtfs_static()
```

Key files you'll get: `stop_times.txt`, `trips.txt`, `routes.txt`, `stops.txt`, `calendar.txt`.

### Step 1.2 — Poll GTFS Realtime (live delay data)

The realtime endpoint gives vehicle positions and trip updates every 20–30 seconds. Run this on a schedule to build up a historical delay dataset.

```python
# src/pipeline/fetch_realtime.py
import requests, json, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STIB_API_KEY")

WAITING_TIME_URL = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/waiting-time-rt-production/records"
VEHICLE_POS_URL  = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/vehicle-position-rt-production/records"

def fetch_waiting_times(limit=100):
    params = {"limit": limit, "apikey": API_KEY}
    r = requests.get(WAITING_TIME_URL, params=params)
    return r.json().get("results", [])

def save_snapshot(data, folder="data/raw/realtime"):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/snapshot_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} records → {path}")

if __name__ == "__main__":
    data = fetch_waiting_times()
    save_snapshot(data)
```

**Run this every 30 seconds for a few days** to accumulate enough delay observations. Use a simple loop or a cron job.

### Step 1.3 — Fetch Traveller Info (strike alerts)

```python
# src/pipeline/fetch_alerts.py
import requests, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STIB_API_KEY")

ALERT_URL = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/travellers-information-rt-production/records"

def fetch_alerts():
    params = {"limit": 100, "apikey": API_KEY}
    r = requests.get(ALERT_URL, params=params)
    return r.json().get("results", [])
```

### Step 1.4 — Merge & Feature Engineer

Combine static schedule data with observed realtime data to compute the target variable: **delay in minutes**.

```python
# src/pipeline/build_dataset.py
import pandas as pd
import json, glob, os

def load_realtime_snapshots(folder="data/raw/realtime"):
    records = []
    for path in glob.glob(f"{folder}/*.json"):
        with open(path) as f:
            records.extend(json.load(f))
    return pd.DataFrame(records)

def compute_delay(df):
    # Waiting time columns vary by API version — adapt field names as needed
    df["scheduled_time"] = pd.to_datetime(df["expectedArrivalTime"], errors="coerce")
    df["actual_time"]    = pd.to_datetime(df["observedTime"], errors="coerce")
    df["delay_minutes"]  = (df["actual_time"] - df["scheduled_time"]).dt.total_seconds() / 60
    return df

def add_time_features(df):
    df["hour"]       = df["scheduled_time"].dt.hour
    df["day_of_week"]= df["scheduled_time"].dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"]    = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df

def build_dataset():
    df = load_realtime_snapshots()
    df = compute_delay(df)
    df = add_time_features(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/delay_dataset.csv", index=False)
    print(f"Dataset: {len(df)} rows → data/processed/delay_dataset.csv")

if __name__ == "__main__":
    build_dataset()
```

**Key features to engineer:**

| Feature | Description |
|---|---|
| `hour` | Hour of departure (0–23) |
| `day_of_week` | 0=Monday to 6=Sunday |
| `is_peak` | 1 during morning/evening rush |
| `is_weekend` | 1 on Sat/Sun |
| `route_id` | Encoded line number |
| `stop_id` | Encoded stop identifier |
| `delay_minutes` | **Target variable** |
| `is_strike_day` | 1 if a strike alert was active |
| `rolling_avg_delay` | 15-min rolling average delay on this route |

**Deliverable from this branch:** `data/processed/delay_dataset.csv` committed and a short `data/README.md` explaining all columns.

---

## Phase 2 — ML Model Training (Branch: branch-leon)

**Goal:** Train, evaluate, and save three models. Produce a clear comparison.

### Step 2.1 — Shared training setup

```python
# src/models/train_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES = ["hour", "day_of_week", "is_peak", "is_weekend",
            "route_id_enc", "stop_id_enc", "is_strike_day", "rolling_avg_delay"]
TARGET   = "delay_minutes"

def load_data(path="data/processed/delay_dataset.csv"):
    df = pd.read_csv(path).dropna(subset=FEATURES + [TARGET])
    le_route = LabelEncoder()
    le_stop  = LabelEncoder()
    df["route_id_enc"] = le_route.fit_transform(df["route_id"].astype(str))
    df["stop_id_enc"]  = le_stop.fit_transform(df["stop_id"].astype(str))
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 2.2 — Random Forest (baseline)

```python
# src/models/train_rf.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from train_utils import load_data

X_train, X_test, y_train, y_test = load_data()

rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
print(f"RF  MAE: {mean_absolute_error(y_test, preds):.2f} min")
print(f"RF  R²:  {r2_score(y_test, preds):.3f}")

joblib.dump(rf, "models/random_forest.pkl")
```

### Step 2.3 — XGBoost (primary model)

```python
# src/models/train_xgb.py
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from train_utils import load_data

X_train, X_test, y_train, y_test = load_data()

xgb_model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=50)

preds = xgb_model.predict(X_test)
print(f"XGB MAE: {mean_absolute_error(y_test, preds):.2f} min")
print(f"XGB R²:  {r2_score(y_test, preds):.3f}")

joblib.dump(xgb_model, "models/xgboost.pkl")
```

### Step 2.4 — LSTM (time-series model)

The LSTM needs sequences: for each stop, feed it the last N delay observations as input.

```python
# src/models/train_lstm.py
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

SEQ_LEN = 10   # use last 10 observations to predict next delay

def make_sequences(series, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

df = pd.read_csv("data/processed/delay_dataset.csv").dropna(subset=["delay_minutes"])
df = df.sort_values("scheduled_time")

scaler = MinMaxScaler()
delays = scaler.fit_transform(df[["delay_minutes"]]).flatten()

X, y = make_sequences(delays)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mae")
model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))

model.save("models/lstm.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")
```

### Step 2.5 — Compare models & pick the best

Create `notebooks/model_comparison.ipynb` with a summary table:

| Model | MAE (min) | R² | Training time | Notes |
|---|---|---|---|---|
| Random Forest | ? | ? | Fast | Good interpretability |
| XGBoost | ? | ? | Fast | Best for tabular data |
| LSTM | ? | ? | Slow | Best for sequences |

Use XGBoost as the **primary production model** (lowest MAE expected). Keep LSTM for the "delay propagation" feature.

**Deliverable:** `models/xgboost.pkl`, `models/random_forest.pkl`, `models/lstm.h5` all committed.

---

## Phase 3 — Backend & Frontend (Branch: branch-omar)

**Goal:** A working FastAPI server and interactive dashboard.

### Step 3.1 — FastAPI app structure

```
src/api/
├── main.py          ← app entry point
├── routers/
│   ├── predict.py   ← /predict endpoint
│   ├── routes.py    ← /routes endpoint (STIB lines)
│   └── delays.py    ← /delays endpoint (history)
└── schemas.py       ← Pydantic request/response models
```

### Step 3.2 — Pydantic schemas

```python
# src/api/schemas.py
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    route_id: str
    stop_id: str
    hour: int
    day_of_week: int
    is_strike_day: int = 0

class PredictionResponse(BaseModel):
    route_id: str
    stop_id: str
    predicted_delay_minutes: float
    confidence: str       # "low" | "medium" | "high"
```

### Step 3.3 — Prediction endpoint

```python
# src/api/routers/predict.py
from fastapi import APIRouter
import joblib, numpy as np
from ..schemas import PredictionRequest, PredictionResponse

router = APIRouter()
model  = joblib.load("models/xgboost.pkl")

ROUTE_ENC = {}  # populate from training label encoder
STOP_ENC  = {}

@router.post("/predict", response_model=PredictionResponse)
def predict_delay(req: PredictionRequest):
    route_enc = ROUTE_ENC.get(req.route_id, 0)
    stop_enc  = STOP_ENC.get(req.stop_id, 0)
    is_peak   = int(req.hour in [7, 8, 9, 17, 18, 19])
    is_weekend= int(req.day_of_week in [5, 6])

    features = np.array([[req.hour, req.day_of_week, is_peak,
                          is_weekend, route_enc, stop_enc,
                          req.is_strike_day, 0.0]])
    delay = float(model.predict(features)[0])
    confidence = "high" if delay < 3 else "medium" if delay < 8 else "low"
    return PredictionResponse(
        route_id=req.route_id,
        stop_id=req.stop_id,
        predicted_delay_minutes=round(delay, 1),
        confidence=confidence
    )
```

### Step 3.4 — Main app

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import predict, routes, delays

app = FastAPI(title="SmartTransit API", version="1.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(predict.router, tags=["Predictions"])
app.include_router(routes.router, tags=["Routes"])
app.include_router(delays.router, tags=["Delays"])

@app.get("/")
def root():
    return {"message": "SmartTransit API is running"}
```

Run with: `uvicorn main:app --reload`  
Interactive docs at: `http://localhost:8000/docs`

### Step 3.5 — Frontend dashboard (HTML/JS or React)

Create `frontend/index.html` with at minimum:

1. **Route selector** — dropdown of STIB lines
2. **Stop selector** — stops on that line
3. **Time picker** — departure hour
4. **Strike toggle** — checkbox for strike day
5. **Predict button** — calls `POST /predict`
6. **Result card** — shows predicted delay + confidence badge
7. **Delay heatmap** — chart of delay by hour across the week

```html
<!-- frontend/index.html (simplified) -->
<!DOCTYPE html>
<html>
<head>
  <title>SmartTransit</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>SmartTransit — Delay Predictor</h1>
  <label>Route: <input id="route" value="21"></label>
  <label>Stop:  <input id="stop"  value="1234"></label>
  <label>Hour:  <input id="hour"  type="number" min="0" max="23" value="8"></label>
  <label><input id="strike" type="checkbox"> Strike day</label>
  <button onclick="predict()">Predict delay</button>
  <div id="result"></div>

  <script>
    async function predict() {
      const body = {
        route_id: document.getElementById("route").value,
        stop_id:  document.getElementById("stop").value,
        hour:     parseInt(document.getElementById("hour").value),
        day_of_week: new Date().getDay(),
        is_strike_day: document.getElementById("strike").checked ? 1 : 0
      };
      const res  = await fetch("http://localhost:8000/predict", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body)
      });
      const data = await res.json();
      document.getElementById("result").innerHTML =
        `<b>${data.predicted_delay_minutes} min delay</b> (confidence: ${data.confidence})`;
    }
  </script>
</body>
</html>
```

---

## Phase 4 — Integration & Testing (All, final week)

### Step 4.1 — Write tests

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_returns_delay():
    res = client.post("/predict", json={
        "route_id": "21", "stop_id": "1234",
        "hour": 8, "day_of_week": 1, "is_strike_day": 0
    })
    assert res.status_code == 200
    assert "predicted_delay_minutes" in res.json()
    assert res.json()["predicted_delay_minutes"] >= 0
```

Run with: `pytest tests/ -v`

### Step 4.2 — Merge order

```
branch-fahad → main         (data pipeline first, others need the data)
branch-leon  → main         (models depend on data)
branch-omar  → main         (API depends on models)
```

Use GitHub pull requests. Each PR needs one reviewer approval.

### Step 4.3 — Final checklist before presentation

- [ ] `data/processed/delay_dataset.csv` exists and is documented
- [ ] All three model files in `models/`
- [ ] `uvicorn main:app` starts without errors
- [ ] `/docs` shows all endpoints with correct schemas
- [ ] Frontend connects to the live API and shows predictions
- [ ] `pytest` passes all tests
- [ ] `notebooks/model_comparison.ipynb` shows MAE comparison table
- [ ] `README.md` has installation and run instructions

---

## Division of Work Summary

| Task | Owner | Branch |
|---|---|---|
| GTFS static download & parsing | Fahad | branch-fahad |
| Realtime API polling script | Fahad | branch-fahad |
| Feature engineering & dataset build | Fahad | branch-fahad |
| Random Forest training | Leon | branch-leon |
| XGBoost training & tuning | Leon | branch-leon |
| LSTM training & evaluation | Leon | branch-leon |
| Model comparison notebook | Leon | branch-leon |
| FastAPI app & endpoints | Omar | branch-omar |
| Pydantic schemas & validation | Omar | branch-omar |
| Frontend dashboard | Omar | branch-omar |
| Tests | All | respective branches |
| README & documentation | All | main |

---

## Useful Resources

- STIB open data portal: https://data.stib-mivb.brussels/pages/home/
- GTFS spec: https://gtfs.org/documentation/overview/
- FastAPI docs: https://fastapi.tiangolo.com
- XGBoost docs: https://xgboost.readthedocs.io
- Keras LSTM guide: https://keras.io/api/layers/recurrent_layers/lstm/
